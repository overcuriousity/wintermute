"""
Shared utilities for LLM output parsing and embedding calls.

LLMs frequently wrap JSON in markdown code fences or add prose around it.
The helpers here centralise the extraction logic so individual modules don't
each implement their own slightly-different variant.

Embedding helpers (``embed``, ``embed_batch``, ``make_content_id``) are used
by both ``memory_store`` and ``skill_store``.
"""

import hashlib
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


def strip_fences(text: str) -> str:
    """Remove markdown code fences (```json … ```) from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"(?:\r?\n)?```\s*$", "", text)
    return text.strip()


def parse_json_from_llm(text: str, expected_type: type) -> Any:
    """Extract a JSON value of *expected_type* from LLM output.

    Tries in order:
    1. Direct JSON parse.
    2. Strip markdown code fences, then parse.
    3. Scan for the outermost delimiters (``{…}`` or ``[…]``), then parse.

    For ``expected_type=dict`` a bare single-element list wrapping a dict is
    also accepted (some models wrap objects in an array).

    Raises ``ValueError`` if no valid JSON of the expected type is found.
    Raises ``TypeError`` if *expected_type* is not ``dict`` or ``list``.
    """
    if expected_type not in (dict, list):
        raise TypeError(
            f"parse_json_from_llm expected_type must be dict or list, got {expected_type!r}"
        )
    if expected_type not in (dict, list):
        raise TypeError(
            f"parse_json_from_llm: expected_type must be dict or list, got {expected_type!r}"
        )
    type_name = "object" if expected_type is dict else "array"
    open_char, close_char = ("{", "}") if expected_type is dict else ("[", "]")

    text = text.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, expected_type):
            return result
        # dict: accept a single-element list wrapping a dict
        if (
            expected_type is dict
            and isinstance(result, list)
            and result
            and isinstance(result[0], dict)
        ):
            return result[0]
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences, then parse
    fenced = strip_fences(text)
    try:
        result = json.loads(fenced)
        if isinstance(result, expected_type):
            return result
    except json.JSONDecodeError:
        pass

    # 3. Outermost delimiter scan
    start = text.find(open_char)
    end = text.rfind(close_char)
    if start != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No JSON {type_name} found in response: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def make_content_id(text: str) -> str:
    """Deterministic UUID from text content (SHA-256 → UUID v5-style).

    Used by both memory_store and skill_store.
    """
    h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    # Format as UUID: 8-4-4-4-12
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# Interaction logging
# ---------------------------------------------------------------------------

def log_store_interaction(timestamp: float, action: str, input_text: str,
                          output_text: str, status: str = "ok",
                          llm: str = "", session: str = "system:store") -> None:
    """Log a store interaction to the database interaction_log.

    Used by memory_store and skill_store for audit trailing of embedding
    calls and backend operations.  Never raises — logging failures must
    not break store operations.
    """
    try:
        from wintermute.infra import database
        database.save_interaction_log(
            timestamp, action, session,
            llm, input_text[:2000], output_text[:2000], status,
        )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str], embed_cfg: dict, model: str,
                url: str, headers: dict) -> list[list[float]]:
    """Send a single HTTP request for a batch of (already-prefixed) texts.

    Raises on error after up to 3 retries for transient 5xx/network failures.
    On 5xx responses the server error body is logged prominently to aid diagnosis
    (e.g. "input too large to process" from LiteLLM batch-size limits).
    """
    import httpx

    payload: dict = {"input": texts, "model": model}
    if embed_cfg.get("send_dimensions"):
        dimensions = embed_cfg.get("dimensions")
        if dimensions:
            payload["dimensions"] = dimensions

    max_retries = 3
    t0 = time.time()
    status = "ok"
    last_exc: Exception | None = None
    input_summary = f"{len(texts)} texts, model={model}"

    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            result = [item["embedding"] for item in items]
            output_summary = f"{len(result)} vectors, {len(result[0])} dims" if result else "empty"
            if attempt > 1:
                status = f"ok (retry {attempt - 1})"
            log_store_interaction(t0, "embedding", input_summary, output_summary, status, llm=model)
            return result
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout,
                httpx.WriteTimeout, httpx.PoolTimeout, ConnectionError, OSError) as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code < 500:
                    # 4xx — not retryable, log and raise immediately.
                    status = f"error {exc.response.status_code}: {exc}"
                    log_store_interaction(t0, "embedding", input_summary, status, status, llm=model)
                    raise
                # 5xx — log the server error body so the cause is visible.
                try:
                    body = exc.response.json()
                except Exception:
                    body = exc.response.text
                logger.warning(
                    "Embedding endpoint returned HTTP %d (attempt %d/%d). "
                    "Server response: %s. "
                    "Hint: if the error mentions 'batch size' or 'too large', "
                    "set memory.embeddings.batch_size in config.yaml to a smaller value.",
                    exc.response.status_code, attempt, max_retries, body,
                )
            last_exc = exc
            if attempt < max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s
                logger.warning("Embedding attempt %d/%d failed (%s), retrying in %.1fs",
                               attempt, max_retries, exc, backoff)
                time.sleep(backoff)
            else:
                status = f"error: {exc}"
                log_store_interaction(t0, "embedding", input_summary, status, status, llm=model)
                raise
        except Exception as exc:
            status = f"error: {exc}"
            log_store_interaction(t0, "embedding", input_summary, status, status, llm=model)
            raise
    raise last_exc  # type: ignore[misc]


def embed(texts: list[str], embed_cfg: dict, task: str = "document") -> list[list[float]]:
    """Call an OpenAI-compatible embeddings endpoint.

    Uses httpx (sync) — callers in async context should run via executor.

    *task* is ``"query"`` (search) or ``"document"`` (upsert/index).
    Prefix is auto-detected for known models (e.g. EmbeddingGemma) or
    can be overridden via ``query_prefix`` / ``document_prefix`` in config.

    Texts are sent in sub-batches of ``batch_size`` (in the embeddings
    config, default 32) to avoid hitting server-side physical batch
    token limits (e.g. LiteLLM's per-request token cap).
    """
    endpoint = embed_cfg.get("endpoint", "").rstrip("/")
    model = embed_cfg.get("model", "text-embedding-3-small")
    api_key = embed_cfg.get("api_key", "") or None
    if not endpoint:
        raise RuntimeError("embeddings.endpoint is not configured")

    # --- task-type prefix handling ---
    _AUTO_PREFIXES: dict[str, dict[str, str]] = {
        "gemma": {"query": "search_query: ", "document": "search_document: "},
    }
    query_prefix = embed_cfg.get("query_prefix", "")
    document_prefix = embed_cfg.get("document_prefix", "")
    if not query_prefix and not document_prefix:
        model_lower = model.lower()
        for key, prefixes in _AUTO_PREFIXES.items():
            if key in model_lower:
                query_prefix = prefixes["query"]
                document_prefix = prefixes["document"]
                break
    prefix = query_prefix if task == "query" else document_prefix
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]

    url = f"{endpoint}/embeddings"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # --- per-text truncation to stay within server token limits ---
    max_chars: int = int(embed_cfg.get("max_text_chars", 2000))
    if max_chars > 0:
        truncated = []
        for t in texts:
            if len(t) > max_chars:
                logger.debug("Truncating embedding input from %d to %d chars", len(t), max_chars)
                truncated.append(t[:max_chars])
            else:
                truncated.append(t)
        texts = truncated

    batch_size: int = int(embed_cfg.get("batch_size", 32))
    if len(texts) <= batch_size:
        return embed_batch(texts, embed_cfg, model, url, headers)

    # Split into sub-batches and concatenate results.
    logger.debug("Embedding %d texts in batches of %d", len(texts), batch_size)
    result: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        result.extend(embed_batch(chunk, embed_cfg, model, url, headers))
    return result
