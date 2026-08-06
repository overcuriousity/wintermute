"""Local ONNX embedding fallback for zero-config memory.

Used only when ``memory.embeddings.endpoint`` is unset.  A configured
endpoint always wins; this module never silently substitutes for a
configured-but-failing endpoint, because mixing vector dimensions would
corrupt an existing store.

The model (all-MiniLM-L6-v2, 384-dimensional) and its tokenizer are fetched
over HTTPS on first use and cached under ``data/.embedding_cache/``.  We
fetch by URL rather than depend on ``huggingface_hub`` to keep the
dependency budget flat.
"""

import logging
import threading
from pathlib import Path

from wintermute.infra.paths import DATA_DIR

logger = logging.getLogger(__name__)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384
MAX_SEQ_LENGTH = 256

CACHE_DIR: Path = DATA_DIR / ".embedding_cache"
_MODEL_PATH = CACHE_DIR / "model.onnx"
_TOKENIZER_PATH = CACHE_DIR / "tokenizer.json"

_BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main"
_MODEL_URL = f"{_BASE_URL}/onnx/model.onnx"
_TOKENIZER_URL = f"{_BASE_URL}/tokenizer.json"

_lock = threading.Lock()
_session = None  # onnxruntime.InferenceSession
_tokenizer = None  # tokenizers.Tokenizer
_available: "bool | None" = None


def provider_name() -> str:
    """Identifier persisted alongside stored vectors for the dimension guard."""
    return "local:all-MiniLM-L6-v2"


def _import_backend():
    """Import the optional extra.  Raises ImportError when it is absent."""
    import onnxruntime
    import tokenizers

    return onnxruntime, tokenizers


def reset_cache() -> None:
    """Drop memoised availability and loaded model state (used by tests)."""
    global _session, _tokenizer, _available
    with _lock:
        _session = None
        _tokenizer = None
        _available = None


def is_available() -> bool:
    """True when the optional extra is importable.  Never raises, never
    downloads."""
    global _available
    if _available is None:
        try:
            _import_backend()
            _available = True
        except ImportError:
            _available = False
    return _available


def is_model_cached() -> bool:
    """True when both model artefacts are already on disk."""
    return _MODEL_PATH.is_file() and _TOKENIZER_PATH.is_file()


def _download(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading embedding artefact %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
    tmp.rename(dest)


def _ensure_loaded():
    """Load (downloading on first use) the ONNX session and tokenizer."""
    global _session, _tokenizer
    if _session is not None and _tokenizer is not None:
        return _session, _tokenizer

    if not is_available():
        raise RuntimeError(
            "Local embeddings are not installed. Either configure "
            "memory.embeddings.endpoint in config.yaml, or install the "
            "optional extra: uv sync --extra local-embeddings"
        )

    onnxruntime, tokenizers = _import_backend()

    with _lock:
        if _session is not None and _tokenizer is not None:
            return _session, _tokenizer
        try:
            if not _MODEL_PATH.is_file():
                _download(_MODEL_URL, _MODEL_PATH)
            if not _TOKENIZER_PATH.is_file():
                _download(_TOKENIZER_URL, _TOKENIZER_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch the local embedding model from {_BASE_URL}. "
                f"The first use requires outbound HTTPS. Original error: {exc}"
            ) from exc

        _tokenizer = tokenizers.Tokenizer.from_file(str(_TOKENIZER_PATH))
        _tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        _tokenizer.enable_padding(length=None)
        _session = onnxruntime.InferenceSession(
            str(_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        logger.info("Local embedding model loaded (%s, %d-dim)", MODEL_ID, DIMENSIONS)
    return _session, _tokenizer


def embed_local(texts: list[str]) -> list[list[float]]:
    """Embed *texts* with the cached local model.

    Returns one L2-normalised 384-float vector per input, using mean pooling
    over the token embeddings — the pooling all-MiniLM-L6-v2 was trained with.
    """
    if not texts:
        return []

    import numpy as np

    session, tokenizer = _ensure_loaded()
    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    expected = {i.name for i in session.get_inputs()}
    if "token_type_ids" in expected:
        inputs["token_type_ids"] = np.zeros_like(input_ids)
    inputs = {k: v for k, v in inputs.items() if k in expected}

    token_embeddings = session.run(None, inputs)[0]  # (batch, seq, 384)

    # Mean pooling over non-padding tokens.
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = summed / counts

    # L2 normalise so cosine similarity reduces to a dot product.
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    normalised = pooled / norms

    return normalised.astype(np.float32).tolist()
