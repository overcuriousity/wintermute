"""
Progressive Tool Disclosure (#178)

Embedding-based classifier that determines which tool tiers to expose per
turn.  Pre-computes tier label embeddings at startup; at runtime, embeds the
user message and computes cosine similarity to decide which tiers are relevant.

Falls back to all tools on any error (generous default).
"""

import asyncio
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

DEFAULT_TIERS: dict[int, list[str]] = {
    0: ["append_memory", "read_file", "write_file"],
    1: ["search_web", "fetch_url", "execute_shell", "send_file"],
    2: ["worker_delegation", "task", "skill", "query_telemetry", "restart_self"],
}

ALL_TIER_INDICES = set(DEFAULT_TIERS.keys())

# Descriptive labels per tier used for embedding similarity.
TIER_LABELS: dict[int, str] = {
    1: (
        "search the web, look something up online, fetch a URL, download a page, "
        "run a shell command, execute code, send a file, upload, research, "
        "find information, browse, investigate"
    ),
    2: (
        "delegate work to a background worker, spawn a sub-session, create a task, "
        "manage tasks, schedule, orchestrate, multi-step workflow, complex project, "
        "plan and coordinate, manage skills, query telemetry, restart, "
        "break this into subtasks"
    ),
}

# ---------------------------------------------------------------------------
# Module state — populated by init_tier_embeddings()
# ---------------------------------------------------------------------------

_tier_embeddings: dict[int, list[float]] = {}
_initialized: bool = False


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def init_tier_embeddings(embed_cfg: dict) -> None:
    """Pre-compute tier label embeddings at startup (one-time cost)."""
    global _tier_embeddings, _initialized
    from wintermute.infra import llm_utils

    labels = list(TIER_LABELS.values())
    tier_indices = list(TIER_LABELS.keys())
    try:
        embeddings = await asyncio.to_thread(
            llm_utils.embed, labels, embed_cfg, "document",
        )
        _tier_embeddings = dict(zip(tier_indices, embeddings))
        _initialized = True
        logger.info("Tool disclosure: pre-computed embeddings for %d tiers", len(_tier_embeddings))
    except Exception as exc:
        logger.warning("Tool disclosure: failed to compute tier embeddings, "
                       "will fall back to all tools: %s", exc)
        _tier_embeddings = {}
        _initialized = False


async def classify_intent(
    message: str,
    embed_cfg: dict,
    threshold: float = 0.3,
) -> set[int]:
    """Classify which tier indices are relevant for *message*.

    Always includes tier 0.  Returns all tiers on any error.
    """
    if not _initialized or not _tier_embeddings:
        return ALL_TIER_INDICES

    if not message or not message.strip():
        return {0}

    from wintermute.infra import llm_utils

    try:
        msg_embeddings = await asyncio.to_thread(
            llm_utils.embed, [message], embed_cfg, "query",
        )
        msg_vec = msg_embeddings[0]
    except Exception as exc:
        logger.warning("Tool disclosure: embedding failed, returning all tiers: %s", exc)
        return ALL_TIER_INDICES

    tiers = {0}
    for tier_idx, tier_vec in _tier_embeddings.items():
        sim = _cosine_similarity(msg_vec, tier_vec)
        logger.debug("Tool disclosure: tier %d similarity = %.3f (threshold %.3f)",
                     tier_idx, sim, threshold)
        if sim >= threshold:
            tiers.add(tier_idx)

    return tiers


def get_disclosed_tool_names(
    tiers: set[int],
    tier_config: Optional[dict[int, list[str]]] = None,
) -> set[str]:
    """Resolve tier indices to a set of tool names."""
    config = tier_config or DEFAULT_TIERS
    names: set[str] = set()
    for tier_idx in tiers:
        names.update(config.get(tier_idx, []))
    return names
