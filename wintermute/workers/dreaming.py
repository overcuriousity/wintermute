"""
Dreaming – Nightly Memory Consolidation

Biologically-inspired multi-phase memory consolidation system.

Phases:
  Housekeeping (always):
    - dedup: Merge near-duplicate memory entries
    - contradiction: Resolve conflicting memory entries
    - stale_pruning: Remove old/unaccessed entries
    - task_consolidation: Review/clean task items
    - skill_consolidation: Deduplicate and condense skill files

  Creative (gated by conditions):
    - association: REM-inspired cross-domain insight discovery
    - schema: NREM-inspired episodic→semantic generalisation
    - prediction: Behavioural/temporal pattern extraction

Uses a single-pass similarity matrix shared across phases to avoid O(n²)
recomputation.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from wintermute.infra import database
from wintermute.infra import data_versioning
from wintermute.infra import memory_store
from wintermute.infra import prompt_loader
from wintermute.infra.llm_utils import parse_json_from_llm
from wintermute.infra.memory_io import read_text_safe

if TYPE_CHECKING:
    from wintermute.core.types import BackendPool
    from wintermute.infra.event_bus import EventBus

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DreamingConfig:
    hour: int = 1
    minute: int = 0
    timezone: str = "UTC"


@dataclass
class PhaseResult:
    """Result of a single dreaming phase."""
    phase_name: str = ""
    items_processed: int = 0
    summary: str = ""
    error: str = ""


@dataclass
class DreamReport:
    """Aggregated results from a full dream cycle."""
    phases_run: list[str] = field(default_factory=list)
    results: list[PhaseResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class SimilarityData:
    """Pre-computed similarity data shared across phases (single matrix)."""
    entries: list[dict] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    dedup_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    contradiction_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    schema_clusters: list[list[int]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

async def _consolidate(pool: "BackendPool",
                        label: str, prompt_template: str, content: str,
                        json_mode: bool = False,
                        max_tokens: int = 2048,
                        **extra_vars: str) -> str:
    """Call the LLM to consolidate a single memory component.

    When *json_mode* is True, passes ``response_format={"type": "json_object"}``
    to the API so the model is constrained to output valid JSON rather than
    relying solely on prompt instructions.
    """
    prompt = prompt_template
    if "{content}" in prompt:
        prompt = prompt.format(content=content, **extra_vars)
    else:
        prompt = prompt + "\n\n" + content
    call_kwargs: dict = {"messages": [{"role": "user", "content": prompt}],
                         "max_tokens_override": max_tokens}
    if json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}
    response = await pool.call(**call_kwargs)
    if not response.content:
        logger.warning("Dreaming (%s): LLM returned empty response", label)
        logger.debug("Empty response: %s", response)
        return ""
    result = response.content.strip()
    try:
        await database.async_call(
            database.save_interaction_log,
            _time.time(), "dreaming", f"system:dreaming:{label}",
            pool.last_used,
            prompt[:2000], result[:2000], "ok",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to log dreaming interaction for %s", label, exc_info=True)
    logger.debug("Dreaming: %s consolidated (%d -> %d chars)", label, len(content), len(result))
    return result




def _validate_dreaming_output(
    text: str, phase: str, parsed: dict, context: dict,
    *, cfg: dict | None = None,
) -> tuple[bool, str]:
    """Programmatic structural validation for creative-phase outputs.

    Returns ``(True, "")`` on pass or ``(False, reason)`` on reject.
    Fail-open: exceptions inside validation allow the entry through.
    Pass *cfg* to avoid redundant config file reads inside tight loops.
    """
    try:
        # ── Common checks ──────────────────────────────────────────
        if cfg is None:
            cfg = _load_dreaming_config()
        default_min = 20 if phase == "prediction" else 15
        min_len = cfg.get("confidence_min_text_length", default_min)
        if len(text) < min_len:
            return False, f"too_short ({len(text)}<{min_len})"

        # 3+ consecutive identical words → hallucination loop
        words = text.split()
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return False, "consecutive_repeat"

        # JSON / markdown fragments in output
        if '{"' in text or "```" in text or text.lstrip().startswith("##"):
            return False, "formatting_fragment"

        # ── Association-specific ───────────────────────────────────
        if phase == "association":
            source_indices = parsed.get("source_indices", [])
            seed_texts = context.get("seed_texts", [])
            if seed_texts and isinstance(source_indices, list):
                if len(source_indices) < 2:
                    return False, "too_few_sources"
                if any(
                    not isinstance(idx, int) or idx < 0 or idx >= len(seed_texts)
                    for idx in source_indices
                ):
                    return False, "invalid_source_index"
            # Near-substring of a single seed → model copied input
            if seed_texts:
                from difflib import SequenceMatcher
                for st in seed_texts:
                    if st and SequenceMatcher(None, text.lower(), st.lower()).ratio() > 0.6:
                        return False, "near_copy_of_seed"

        # ── Schema-specific ────────────────────────────────────────
        elif phase == "schema":
            cluster_texts = context.get("cluster_texts", [])
            # Schema should generalise, not copy a single member
            for ct in cluster_texts:
                if ct and text.strip().lower() in ct.lower():
                    return False, "substring_of_cluster_member"
            # Force-downgrade replaces if confidence isn't high
            if parsed.get("replaces_entries") and parsed.get("confidence") != "high":
                parsed["replaces_entries"] = False

        # ── Prediction-specific ────────────────────────────────────
        elif phase == "prediction":
            # Input-parrot detection
            activity = context.get("activity_summary", "")
            if activity:
                from difflib import SequenceMatcher
                max_overlap = cfg.get("confidence_max_input_overlap", 0.7)
                if SequenceMatcher(None, text.lower(), activity.lower()).ratio() > max_overlap:
                    return False, "parrots_input"
            # Duplicate detection
            existing = context.get("existing_texts", set())
            if text.strip().lower() in existing:
                return False, "duplicate_prediction"
            # Temporal suffix validation
            pred_type = parsed.get("type", "behavioral")
            if pred_type == "temporal" and "||" in text:
                import re as _re
                hour_match = _re.findall(r'\|\|hours?=([\d,\-]+)\|\|', text)
                day_match = _re.findall(r'\|\|days?=([\w,\-]+)\|\|', text)
                valid_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
                malformed = False
                for hm in hour_match:
                    for h in hm.replace("-", ",").split(","):
                        if h and (not h.isdigit() or not (0 <= int(h) <= 23)):
                            malformed = True
                for dm in day_match:
                    for d in dm.split(","):
                        if d.strip().lower() not in valid_days:
                            malformed = True
                if malformed:
                    parsed["type"] = "behavioral"  # downgrade, don't reject
    except Exception:
        logger.debug("Dreaming output validation error (fail-open)", exc_info=True)

    return True, ""


def _load_dreaming_config() -> dict:
    """Load dreaming-specific config from config.yaml with defaults."""
    defaults = {
        # Housekeeping
        "dedup_similarity_threshold": 0.80,
        "stale_days": 90,
        "stale_min_access": 3,
        "working_set_size": 50,
        # Association (REM)
        "association_min_new_memories": 20,
        "association_max_insights": 5,
        "association_days_interval": 7,
        "association_seed_count": 8,
        "association_sim_low": 0.3,
        "association_sim_high": 0.6,
        # Schema (NREM)
        "schema_min_store_size": 50,
        "schema_days_interval": 14,
        "schema_merge_trigger": 10,
        "schema_cluster_threshold": 0.6,
        "schema_max_clusters": 5,
        # Prediction
        "prediction_min_outcomes": 50,
        "prediction_days_interval": 7,
        "prediction_lookback_days": 30,
        "prediction_max_predictions": 3,
    }
    try:
        cfg_path = Path("config.yaml")
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                full = yaml.safe_load(f) or {}
            dreaming_cfg = full.get("memory", {}).get("dreaming", {})
            if dreaming_cfg:
                defaults.update(dreaming_cfg)
    except Exception:  # noqa: BLE001
        logger.debug("Could not load dreaming config, using defaults")
    return defaults


def _union_find_clusters(similarities: list[tuple[int, int, float]],
                         n: int, threshold: float) -> list[list[int]]:
    """Union-find clustering from pairwise similarities above threshold."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, sim in similarities:
        if sim >= threshold:
            union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)
    return [members for members in clusters.values() if len(members) > 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Single-pass similarity computation
# ═══════════════════════════════════════════════════════════════════════════════

async def _compute_similarity_data(cfg: dict) -> SimilarityData:
    """Compute similarity data for all phases.

    When Qdrant is the backend, uses search_batch for O(n*k) neighbor
    lookups instead of computing the full O(n^2) pairwise matrix in Python.
    Falls back to the numpy matrix for local_vector backends.

    Returns a SimilarityData with:
      - dedup_pairs: (i, j, sim) where sim >= dedup threshold
      - contradiction_pairs: (i, j, sim) in the suspicious range [0.5, threshold)
      - schema_clusters: clusters of entries with sim >= schema_cluster_threshold
    """
    dedup_threshold = cfg["dedup_similarity_threshold"]
    schema_threshold = cfg.get("schema_cluster_threshold", 0.6)

    # ─── Qdrant-native path: use search_batch for O(n·k) ───────────────
    if memory_store.is_qdrant_backend():
        all_entries = await asyncio.to_thread(memory_store.get_all_with_vectors)
        if not all_entries:
            return SimilarityData()

        sd = SimilarityData(
            entries=all_entries,
            texts=[e["text"] for e in all_entries],
            ids=[e["id"] for e in all_entries],
        )
        n = len(all_entries)
        if n < 2:
            return sd

        # Fetch neighbors for every entry in a single batched call.
        # limit=50 gives enough neighbors to cover dedup, contradiction,
        # and schema clustering without a full n^2 matrix.
        neighbor_limit = min(n - 1, 50)
        neighbors_map = await asyncio.to_thread(
            memory_store.search_neighbors_batch,
            sd.ids, limit=neighbor_limit, score_threshold=schema_threshold,
        )

        if not neighbors_map:
            # search_batch failed — fall through to numpy path below.
            logger.debug("Dreaming: Qdrant search_batch returned empty, "
                         "falling back to numpy")
        else:
            # Build index lookup.
            id_to_idx = {eid: i for i, eid in enumerate(sd.ids)}
            seen_pairs: set[tuple[int, int]] = set()
            schema_pairs: list[tuple[int, int, float]] = []

            for eid, hits in neighbors_map.items():
                i = id_to_idx.get(eid)
                if i is None:
                    continue
                for h in hits:
                    j = id_to_idx.get(h["id"])
                    if j is None or i == j:
                        continue
                    pair = (min(i, j), max(i, j))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    s = h["score"]
                    if s >= dedup_threshold:
                        sd.dedup_pairs.append((pair[0], pair[1], s))
                    elif 0.5 <= s < dedup_threshold:
                        sd.contradiction_pairs.append((pair[0], pair[1], s))
                    if s >= schema_threshold:
                        schema_pairs.append((pair[0], pair[1], s))

            # Cap contradiction pairs at top 20.
            sd.contradiction_pairs.sort(key=lambda x: x[2], reverse=True)
            sd.contradiction_pairs = sd.contradiction_pairs[:20]

            # Build schema clusters.
            if schema_pairs:
                raw = _union_find_clusters(schema_pairs, n, schema_threshold)
                sd.schema_clusters = [c for c in raw if len(c) >= 3]

            return sd

    # ─── Numpy fallback path: full O(n^2) matrix ───────────────────────
    import numpy as np

    all_entries = await asyncio.to_thread(memory_store.get_all_with_vectors)
    if not all_entries:
        return SimilarityData()

    entries_with_vecs = [e for e in all_entries if e.get("vector")]
    if not entries_with_vecs:
        return SimilarityData(entries=all_entries,
                              texts=[e["text"] for e in all_entries],
                              ids=[e.get("id", "") for e in all_entries])

    sd = SimilarityData(
        entries=entries_with_vecs,
        texts=[e["text"] for e in entries_with_vecs],
        ids=[e["id"] for e in entries_with_vecs],
    )
    n = len(entries_with_vecs)
    if n < 2:
        return sd

    vecs = np.array([e["vector"] for e in entries_with_vecs], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs_normed = vecs / norms

    # Single matrix computation (the expensive O(n²) operation).
    sim_matrix = vecs_normed @ vecs_normed.T

    schema_pairs_np: list[tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if s >= dedup_threshold:
                sd.dedup_pairs.append((i, j, s))
            elif 0.5 <= s < dedup_threshold:
                sd.contradiction_pairs.append((i, j, s))
            if s >= schema_threshold:
                schema_pairs_np.append((i, j, s))

    # Cap contradiction pairs at top 20 most similar.
    sd.contradiction_pairs.sort(key=lambda x: x[2], reverse=True)
    sd.contradiction_pairs = sd.contradiction_pairs[:20]

    # Build schema clusters (groups of 3+ semantically related entries).
    if schema_pairs_np:
        raw_clusters = _union_find_clusters(schema_pairs_np, n, schema_threshold)
        sd.schema_clusters = [c for c in raw_clusters if len(c) >= 3]

    return sd


# ═══════════════════════════════════════════════════════════════════════════════
# Housekeeping phases
# ═══════════════════════════════════════════════════════════════════════════════

async def _phase_dedup(pool: "BackendPool", cfg: dict,
                       sim_data: SimilarityData) -> PhaseResult:
    """Phase: merge near-duplicate entries using union-find clusters."""
    result = PhaseResult(phase_name="dedup")
    threshold = cfg["dedup_similarity_threshold"]

    if not sim_data.dedup_pairs:
        result.summary = "no duplicates found"
        return result

    clusters = _union_find_clusters(sim_data.dedup_pairs, len(sim_data.texts), threshold)
    if not clusters:
        result.summary = "no clusters formed"
        return result

    dedup_prompt = prompt_loader.load("DREAM_DEDUP_PROMPT.txt")
    merged_count = 0

    for cluster in clusters:
        cluster_texts = [sim_data.texts[idx] for idx in cluster]
        cluster_ids = [sim_data.ids[idx] for idx in cluster]
        content = "\n---\n".join(cluster_texts)
        try:
            merged_text = await _consolidate(pool, "dedup_merge", dedup_prompt, content)
            if merged_text:
                await asyncio.to_thread(memory_store.bulk_delete, cluster_ids)
                await asyncio.to_thread(
                    memory_store.add, merged_text, None, "dreaming_merge"
                )
                merged_count += 1
                logger.debug("Dreaming: merged cluster of %d entries", len(cluster))
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: failed to merge cluster")

    result.items_processed = merged_count
    result.summary = f"merged {merged_count} clusters"
    logger.info("Dreaming phase dedup: %s", result.summary)
    return result


async def _phase_contradiction(pool: "BackendPool", cfg: dict,
                               sim_data: SimilarityData) -> PhaseResult:
    """Phase: detect and resolve contradictory memory pairs."""
    result = PhaseResult(phase_name="contradiction")

    if not sim_data.contradiction_pairs:
        result.summary = "no contradiction candidates"
        return result

    contra_prompt = prompt_loader.load("DREAM_CONTRADICTION_PROMPT.txt")
    resolved_count = 0

    for i, j, _ in sim_data.contradiction_pairs:
        try:
            raw = await _consolidate(
                pool, "contradiction",
                contra_prompt.replace("{entry_1}", sim_data.texts[i])
                             .replace("{entry_2}", sim_data.texts[j]),
                "",
            )
            decision = parse_json_from_llm(raw, dict)
            action = decision.get("action", "")
            if action == "keep_first":
                await asyncio.to_thread(memory_store.delete, sim_data.ids[j])
                resolved_count += 1
            elif action == "keep_second":
                await asyncio.to_thread(memory_store.delete, sim_data.ids[i])
                resolved_count += 1
            elif action == "merge" and decision.get("result"):
                await asyncio.to_thread(
                    memory_store.bulk_delete, [sim_data.ids[i], sim_data.ids[j]]
                )
                await asyncio.to_thread(
                    memory_store.add, decision["result"], None, "dreaming_merge"
                )
                resolved_count += 1
        except (_json.JSONDecodeError, Exception):  # noqa: BLE001
            logger.debug("Dreaming: contradiction resolution failed for pair",
                         exc_info=True)

    result.items_processed = resolved_count
    result.summary = f"resolved {resolved_count} contradictions"
    logger.info("Dreaming phase contradiction: %s", result.summary)
    return result


async def _phase_stale_pruning(pool: "BackendPool", cfg: dict,
                               sim_data: SimilarityData) -> PhaseResult:
    """Phase: remove old entries with low access counts."""
    result = PhaseResult(phase_name="stale_pruning")
    stale = await asyncio.to_thread(
        memory_store.get_stale, cfg["stale_days"], cfg["stale_min_access"]
    )
    # Protect user-explicit memories and dreaming schemas from pruning.
    prune_ids = [
        e["id"] for e in stale
        if e.get("source") not in ("user_explicit", "dreaming_schema")
    ]
    if prune_ids:
        deleted = await asyncio.to_thread(memory_store.bulk_delete, prune_ids)
        result.items_processed = deleted
        result.summary = f"pruned {deleted} stale entries"
    else:
        result.summary = "no stale entries"
    logger.info("Dreaming phase stale_pruning: %s", result.summary)
    return result



async def _phase_task_consolidation(pool: "BackendPool", cfg: dict,
                                    sim_data: SimilarityData) -> PhaseResult:
    """Phase: review and clean up task database items."""
    result = PhaseResult(phase_name="task_consolidation")
    task_items = await database.async_call(database.list_tasks, "active")
    if not task_items:
        result.summary = "no active tasks"
        return result

    task_prompt = prompt_loader.load("DREAM_TASK_PROMPT.txt")
    formatted = "\n".join(
        f"[P{it['priority']}] #{it['id']}: {it['content']}"
        + (f" [{it['schedule_desc']}]" if it.get("schedule_desc") else "")
        for it in task_items
    )
    raw = await _consolidate(pool, "tasks", task_prompt, formatted)
    applied = 0
    if raw:
        try:
            actions = parse_json_from_llm(raw, list)
        except ValueError as exc:
            logger.warning("Dreaming: tasks LLM returned non-JSON: %s", exc)
            actions = []
        for act in actions:
            a = act.get("action")
            aid = act.get("id")
            if a == "complete" and aid is not None:
                # Remove the scheduler job first so the task cannot fire again
                # after being marked completed.
                from wintermute.workers import scheduler_thread as _sched_mod
                if _sched_mod._instance is not None:
                    try:
                        _sched_mod._instance.remove_job(str(aid))
                    except Exception:
                        logger.warning("Dreaming: failed to remove scheduler job %s", aid, exc_info=True)
                await database.async_call(
                    database.complete_task, str(aid),
                    reason="Completed via dreaming consolidation",
                )
                applied += 1
            elif a == "update" and aid is not None:
                kwargs = {}
                if "content" in act:
                    kwargs["content"] = act["content"]
                if "priority" in act:
                    kwargs["priority"] = int(act["priority"])
                if kwargs:
                    await database.async_call(
                        database.update_task, str(aid), **kwargs,
                    )
                    applied += 1
    result.items_processed = applied
    result.summary = f"applied {applied} task actions"
    logger.info("Dreaming phase task_consolidation: %s", result.summary)
    return result


async def _phase_skill_consolidation(pool: "BackendPool", cfg: dict,
                                     sim_data: SimilarityData) -> PhaseResult:
    """Phase: deduplicate and condense skills via skill_store."""
    from wintermute.infra import skill_store

    result = PhaseResult(phase_name="skill_consolidation")

    try:
        all_skills = skill_store.get_all()
    except Exception:
        logger.warning("Dreaming: skill_store.get_all() failed", exc_info=True)
        result.summary = "skill_store unavailable"
        return result

    if not all_skills:
        result.summary = "no skills"
        return result

    # Build name -> record map.
    skills: dict[str, dict] = {s["name"]: s for s in all_skills}

    # Auto-retire unused skills (>90 days without reads, <2 accesses).
    try:
        stale = skill_store.get_stale(max_age_days=90, min_access=2)
        for rec in stale:
            name = rec["name"]
            skill_store.delete(name)
            skills.pop(name, None)
            logger.info("Dreaming: retired unused skill '%s'", name)
    except Exception:
        logger.debug("Dreaming: skill retirement failed", exc_info=True)

    if not skills:
        result.summary = "all skills retired"
        return result

    # Deduplication.
    merged_skills = 0
    if len(skills) > 1:
        try:
            dedup_prompt = prompt_loader.load("DREAM_SKILLS_DEDUP_PROMPT.txt")
            formatted = "\n\n".join(
                f"=== {name} ===\n{rec.get('summary', '')}\n{rec.get('documentation', '')}"
                for name, rec in skills.items()
            )
            raw = await _consolidate(pool, "skills_dedup", dedup_prompt, formatted)
            actions = parse_json_from_llm(raw, list)
            for act in actions:
                action = act.get("action")
                name = act.get("file", "")
                if action == "delete":
                    if name in skills:
                        skill_store.delete(name)
                        del skills[name]
                        merged_skills += 1
                        logger.info("Dreaming: deleted duplicate skill '%s'", name)
                elif action == "merge":
                    target = act.get("into", "")
                    content = act.get("content", "").strip()
                    if not target or not content:
                        continue
                    # Split merged content into summary (first line) and documentation (rest).
                    summary_line, _, rest = content.partition("\n")
                    merge_summary = summary_line.strip()
                    merge_doc = rest.lstrip("\n").strip()
                    if skill_store.exists(target):
                        skill_store.update(target, summary=merge_summary, documentation=merge_doc)
                    else:
                        skill_store.add(target, merge_summary, merge_doc)
                    if target in skills:
                        skills[target]["summary"] = merge_summary
                        skills[target]["documentation"] = merge_doc
                    if name != target and name in skills:
                        skill_store.delete(name)
                        del skills[name]
                    merged_skills += 1
        except ValueError as exc:
            logger.warning("Dreaming: skill dedup non-JSON: %s", exc)
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: skill dedup failed")

    # Condense each surviving skill (less aggressive — only if doc > 600 chars).
    condensed = 0
    try:
        condense_template = prompt_loader.load("DREAM_SKILLS_CONDENSATION_PROMPT.txt")
    except FileNotFoundError:
        logger.warning("Dreaming: DREAM_SKILLS_CONDENSATION_PROMPT.txt not found, skipping condensation")
        condense_template = None
    if condense_template is None:
        result.items_processed = merged_skills
        result.summary = f"merged {merged_skills}, condensed 0 skills (template missing)"
        logger.info("Dreaming phase skill_consolidation: %s", result.summary)
        return result
    for name, rec in list(skills.items()):
        doc = rec.get("documentation", "")
        if len(doc) < 600:
            continue  # skip short skills — no need to condense
        try:
            # Pass full skill text (summary + doc) to the condense prompt
            # so the model can produce a complete condensed version.
            summary = rec.get("summary", "")
            full_content = f"{summary}\n\n{doc}".strip() if summary else doc
            prompt = condense_template.format(skill_name=name, content=full_content)
            response = await pool.call(
                messages=[{"role": "user", "content": prompt}],
                max_tokens_override=600,
            )
            if response.content is None:
                continue
            condensed_text = (response.content or "").strip()
            if condensed_text:
                # Split output back into summary (first line) and documentation.
                cond_first, _, cond_rest = condensed_text.partition("\n")
                cond_summary = cond_first.strip()
                cond_doc = cond_rest.lstrip("\n").strip()
                skill_store.update(name, summary=cond_summary, documentation=cond_doc)
                condensed += 1
                try:
                    await database.async_call(
                        database.save_interaction_log,
                        _time.time(), "dreaming", f"system:dreaming:skill:{name}",
                        pool.last_used, prompt[:2000], condensed_text[:2000], "ok",
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to log skill consolidation for %s", name, exc_info=True)
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: failed to condense skill '%s'", name)

    result.items_processed = merged_skills + condensed
    result.summary = f"merged {merged_skills}, condensed {condensed} skills"
    logger.info("Dreaming phase skill_consolidation: %s", result.summary)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Creative phase: Associative Discovery (REM-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

async def _phase_association(pool: "BackendPool", cfg: dict,
                             sim_data: SimilarityData) -> PhaseResult:
    """REM-inspired phase: find non-obvious cross-domain connections.

    Gate conditions:
    - >= association_min_new_memories new memories since last run, AND
    - >= association_days_interval days since last run.
    """
    result = PhaseResult(phase_name="association")

    # Check gate conditions.
    state = await database.async_call(
        database.get_dreaming_phase_state, "association"
    )
    last_run = state["last_run_at"] if state else 0
    now = _time.time()
    days_since = (now - last_run) / 86400 if last_run else float("inf")

    if days_since < cfg["association_days_interval"]:
        result.summary = (
            f"skipped: only {days_since:.1f}d since last "
            f"(need {cfg['association_days_interval']})"
        )
        return result

    since_ts = last_run if last_run else (now - 30 * 86400)
    new_count = await database.async_call(
        database.count_memories_added_since, since_ts
    )
    if new_count < cfg["association_min_new_memories"]:
        result.summary = (
            f"skipped: only {new_count} new memories "
            f"(need {cfg['association_min_new_memories']})"
        )
        return result

    # Select seed entries: prefer diverse set via Qdrant recommend or random.
    seed_count = cfg["association_seed_count"]
    sim_low = cfg["association_sim_low"]
    sim_high = cfg["association_sim_high"]

    if memory_store.is_qdrant_backend() and len(sim_data.ids) >= seed_count:
        # Use Qdrant recommend: seed with recent entries as positives,
        # older diverse entries as negatives to push toward novelty.
        try:
            recent = sim_data.ids[:3]  # Most recent entries as positives.
            # Use a few high-access entries as negatives to bias toward
            # less-accessed (surprising) memories.
            negative_candidates = [
                sim_data.ids[idx] for idx in range(len(sim_data.ids) - 1, -1, -1)
                if sim_data.ids[idx] not in recent
            ][:3]
            seeds = await asyncio.to_thread(
                memory_store.recommend, recent, negative_candidates,
                seed_count, score_threshold=sim_low,
            )
        except Exception:
            logger.debug("Dreaming: Qdrant recommend failed, falling back",
                         exc_info=True)
            seeds = []
    else:
        seeds = []

    if not seeds and sim_data.texts:
        # Fallback: pick entries in the mid-similarity range.
        mid_pairs = [
            (i, j, s) for i, j, s in sim_data.contradiction_pairs
            if sim_low <= s <= sim_high
        ]
        if not mid_pairs:
            # Broaden: use any available pairs.
            mid_pairs = sim_data.contradiction_pairs[:seed_count]

        seen_indices: set[int] = set()
        for i, j, _ in mid_pairs:
            seen_indices.add(i)
            seen_indices.add(j)
            if len(seen_indices) >= seed_count:
                break
        seeds = [
            {"id": sim_data.ids[idx], "text": sim_data.texts[idx]}
            for idx in list(seen_indices)[:seed_count]
        ]

    if not seeds:
        result.summary = "no suitable seeds found"
        return result

    # Format seed texts for LLM.
    seed_texts = []
    for s in seeds:
        text = s.get("text", "") if isinstance(s, dict) else str(s)
        seed_texts.append(text)

    content = "\n---\n".join(
        f"[{k}] {text}" for k, text in enumerate(seed_texts)
    )

    assoc_prompt = prompt_loader.load("DREAM_ASSOCIATION_PROMPT.txt")
    max_insights = cfg["association_max_insights"]
    insights_created = 0
    collected_ids: list[str] = []

    try:
        raw = await _consolidate(
            pool, "association", assoc_prompt, content,
            json_mode=True, max_tokens=1024,
        )
        parsed = parse_json_from_llm(raw, dict)
        if parsed.get("no_insight", False):
            result.summary = "LLM found no cross-domain insights"
            return result

        insights = parsed.get("insights", [])
        for insight in insights:
            if insights_created >= max_insights:
                break
            confidence = insight.get("confidence", "low")
            if confidence not in ("high", "medium"):
                continue
            text = insight.get("text", "").strip()
            valid, reason = _validate_dreaming_output(
                text, "association", insight, {"seed_texts": seed_texts}, cfg=cfg,
            )
            if not valid:
                logger.info("Dreaming association rejected: %s", reason)
                continue
            if text:
                eid = await asyncio.to_thread(
                    memory_store.add, text, None, "dreaming_association"
                )
                collected_ids.append(eid)
                insights_created += 1
                logger.info("Dreaming association insight: %s", text[:100])
    except Exception:  # noqa: BLE001
        logger.exception("Dreaming: association phase LLM failed")
        result.error = "LLM call failed"
        result.summary = "LLM call failed"

    # Record outside the try block so already-added IDs are tracked even on
    # partial failure (e.g. embedding backend error midway through the loop).
    if collected_ids:
        await database.async_call(
            database.record_dreaming_entries, "association",
            list(dict.fromkeys(collected_ids)),
        )

    if result.error:
        return result

    result.items_processed = insights_created
    result.summary = f"created {insights_created} insights from {len(seeds)} seeds"
    await database.async_call(
        database.update_dreaming_phase_state, "association",
        insights_created, result.summary,
    )
    logger.info("Dreaming phase association: %s", result.summary)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Creative phase: Schema Abstraction (NREM slow-wave inspired)
# ═══════════════════════════════════════════════════════════════════════════════

async def _phase_schema(pool: "BackendPool", cfg: dict,
                        sim_data: SimilarityData) -> PhaseResult:
    """NREM-inspired phase: generalise episodic memories into schemas.

    Gate conditions:
    - memory store has >= schema_min_store_size entries, AND
    - >= schema_days_interval days since last run, OR
    - housekeeping just merged >= schema_merge_trigger clusters.
    """
    result = PhaseResult(phase_name="schema")

    # Check gate conditions.
    state = await database.async_call(
        database.get_dreaming_phase_state, "schema"
    )
    last_run = state["last_run_at"] if state else 0
    now = _time.time()
    days_since = (now - last_run) / 86400 if last_run else float("inf")

    store_count = await asyncio.to_thread(memory_store.count)
    dedup_state = await database.async_call(
        database.get_dreaming_phase_state, "dedup"
    )
    recent_merges = dedup_state["items_processed"] if dedup_state else 0

    if store_count < cfg["schema_min_store_size"]:
        result.summary = (
            f"skipped: only {store_count} entries "
            f"(need {cfg['schema_min_store_size']})"
        )
        return result

    time_gate = days_since >= cfg["schema_days_interval"]
    merge_gate = recent_merges >= cfg["schema_merge_trigger"]
    if not time_gate and not merge_gate:
        result.summary = (
            f"skipped: {days_since:.1f}d since last "
            f"(need {cfg['schema_days_interval']}), "
            f"{recent_merges} recent merges "
            f"(need {cfg['schema_merge_trigger']})"
        )
        return result

    if not sim_data.schema_clusters:
        result.summary = "no clusters of size >= 3"
        return result

    schema_prompt = prompt_loader.load("DREAM_SCHEMA_PROMPT.txt")
    max_clusters = cfg["schema_max_clusters"]
    schemas_formed = 0
    schema_collected_ids: list[str] = []

    # Sort clusters by size (largest first), cap at max.
    clusters = sorted(sim_data.schema_clusters, key=len, reverse=True)[:max_clusters]

    for cluster in clusters:
        cluster_texts = [sim_data.texts[idx] for idx in cluster]
        cluster_ids = [sim_data.ids[idx] for idx in cluster]

        # Check if an existing schema already covers this cluster.
        has_existing_schema = any(
            sim_data.entries[idx].get("source") == "dreaming_schema"
            for idx in cluster
        )

        content = "\n---\n".join(
            f"[{k}] {text}" for k, text in enumerate(cluster_texts)
        )

        if has_existing_schema:
            content = (
                "(Note: one or more entries below are existing schemas. "
                "Refine rather than create a new separate schema.)\n" + content
            )

        try:
            raw = await _consolidate(
                pool, "schema", schema_prompt, content,
                json_mode=True, max_tokens=1024,
            )
            parsed = parse_json_from_llm(raw, dict)
            schema_text = parsed.get("schema", "").strip()
            confidence = parsed.get("confidence", "medium")

            if not schema_text:
                continue
            if confidence not in ("high", "medium"):
                continue

            valid, reason = _validate_dreaming_output(
                schema_text, "schema", parsed,
                {"cluster_texts": cluster_texts}, cfg=cfg,
            )
            if not valid:
                logger.info("Dreaming schema rejected: %s", reason)
                continue

            # Re-read replaces after validation (validator may have downgraded it).
            replaces = parsed.get("replaces_entries", False)

            # Store the schema.
            eid = await asyncio.to_thread(
                memory_store.add, schema_text, None, "dreaming_schema"
            )
            schema_collected_ids.append(eid)
            schemas_formed += 1
            logger.info("Dreaming schema: %s (replaces=%s)",
                        schema_text[:100], replaces)

            # If high confidence and replaces, remove the originals.
            if replaces and confidence == "high":
                # Don't delete entries that are themselves schemas (lineage).
                delete_ids = [
                    cid for cid, idx in zip(cluster_ids, cluster)
                    if sim_data.entries[idx].get("source") != "dreaming_schema"
                ]
                if delete_ids:
                    await asyncio.to_thread(
                        memory_store.bulk_delete, delete_ids
                    )
                    logger.info(
                        "Dreaming schema: replaced %d episodic entries",
                        len(delete_ids),
                    )
        except Exception:  # noqa: BLE001
            logger.debug("Dreaming: schema extraction failed for cluster",
                         exc_info=True)

    if schema_collected_ids:
        await database.async_call(
            database.record_dreaming_entries, "schema",
            list(dict.fromkeys(schema_collected_ids)),
        )
    result.items_processed = schemas_formed
    result.summary = f"formed {schemas_formed} schemas from {len(clusters)} clusters"
    await database.async_call(
        database.update_dreaming_phase_state, "schema",
        schemas_formed, result.summary,
    )
    logger.info("Dreaming phase schema: %s", result.summary)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Creative phase: Predictive Pattern Extraction
# ═══════════════════════════════════════════════════════════════════════════════

async def _phase_prediction(pool: "BackendPool", cfg: dict,
                            sim_data: SimilarityData) -> PhaseResult:
    """Extract behavioural/temporal patterns from operational history.

    Gate conditions:
    - >= prediction_min_outcomes sub-session outcomes since last run, AND
    - >= prediction_days_interval days since last run.
    """
    result = PhaseResult(phase_name="prediction")

    # Check gate conditions.
    state = await database.async_call(
        database.get_dreaming_phase_state, "prediction"
    )
    last_run = state["last_run_at"] if state else 0
    now = _time.time()
    days_since = (now - last_run) / 86400 if last_run else float("inf")

    if days_since < cfg["prediction_days_interval"]:
        result.summary = (
            f"skipped: only {days_since:.1f}d since last "
            f"(need {cfg['prediction_days_interval']})"
        )
        return result

    lookback = cfg["prediction_lookback_days"] * 86400
    since = now - lookback
    outcomes = await database.async_call(
        database.get_outcomes_since, since, None, 200
    )

    if len(outcomes) < cfg["prediction_min_outcomes"]:
        result.summary = (
            f"skipped: only {len(outcomes)} outcomes "
            f"(need {cfg['prediction_min_outcomes']})"
        )
        return result

    # Gather statistics for the activity summary.
    status_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    hour_counts: dict[int, int] = {}
    dow_counts: dict[int, int] = {}
    duration_sum = 0.0
    duration_n = 0

    for o in outcomes:
        st = o.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1
        if o.get("duration_seconds"):
            duration_sum += o["duration_seconds"]
            duration_n += 1
        if o.get("tools_used"):
            try:
                tools = (
                    _json.loads(o["tools_used"])
                    if isinstance(o["tools_used"], str)
                    else o["tools_used"]
                )
                for t in tools:
                    tool_counts[t] = tool_counts.get(t, 0) + 1
            except Exception:
                pass
        ts = o.get("timestamp", 0)
        if ts:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            hour_counts[dt.hour] = hour_counts.get(dt.hour, 0) + 1
            dow_counts[dt.weekday()] = dow_counts.get(dt.weekday(), 0) + 1

    # Conversation summaries for topic patterns.
    summaries = await database.async_call(
        database.get_summaries_since, since, 20
    )
    summary_texts = [s["content"][:200] for s in summaries] if summaries else []

    # Tool usage stats.
    tool_stats = await database.async_call(
        database.get_tool_usage_stats, since
    )

    # Build compact activity summary.
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines = [
        f"Period: {cfg['prediction_lookback_days']} days",
        f"Sub-sessions: {len(outcomes)} total, by status: {status_counts}",
        (f"Avg duration: {duration_sum / duration_n:.0f}s"
         if duration_n else "No duration data"),
        f"Activity by hour: {dict(sorted(hour_counts.items()))}",
        (f"Activity by day: "
         f"{', '.join(f'{dow_names[k]}={v}' for k, v in sorted(dow_counts.items()))}"),
        f"Top tools: {tool_stats[:10]}",
    ]
    if summary_texts:
        lines.append("Recent conversation topics:")
        for idx, st in enumerate(summary_texts[:5]):
            lines.append(f"  [{idx}] {st}")

    content = "\n".join(lines)

    # Call LLM for pattern synthesis.
    pred_prompt = prompt_loader.load("DREAM_PREDICTION_PROMPT.txt")
    max_predictions = cfg["prediction_max_predictions"]
    pred_collected_ids: list[str] = []

    try:
        raw = await _consolidate(
            pool, "prediction", pred_prompt, content,
            json_mode=True, max_tokens=1024,
        )
        parsed = parse_json_from_llm(raw, dict)
        predictions = parsed.get("predictions", [])

        # Pre-fetch existing prediction texts for duplicate detection.
        _existing_raw = await asyncio.to_thread(
            memory_store.get_by_source, "dreaming_prediction", 100, False,
        )
        _pred_tag_re = re.compile(r'^\[prediction:[^\]]+\]\s*', re.IGNORECASE)
        existing_pred_texts: set[str] = set()
        for _e in _existing_raw:
            _raw = (_e.get("text", "") or "").strip()
            # Stored predictions are tagged "[prediction:<type>] <text>"; strip
            # the tag so duplicate detection compares plain text (as the
            # validator receives it).
            _normalized = _pred_tag_re.sub("", _raw).strip().lower()
            if _normalized:
                existing_pred_texts.add(_normalized)

        predictions_added = 0
        for pred in predictions:
            if predictions_added >= max_predictions:
                break
            confidence = pred.get("confidence", "low")
            if confidence not in ("high", "medium"):
                continue
            text = pred.get("text", "").strip()
            # Normalize temporal suffix BEFORE duplicate detection so the
            # comparison uses the same canonical form as stored entries.
            # e.g. "foo||hours=1||days=mon||" → "foo ||hours=1|| ||days=mon||"
            if "||" in text:
                suffix_match = re.search(r'(?:\s*\|\|\w+=[\w,\-]+\|\|)+\s*$', text)
                if suffix_match:
                    suffix_parts = re.findall(r'\|\|(\w+=[\w,\-]+)\|\|', suffix_match.group(0))
                    if suffix_parts:
                        normalized_suffix = " ".join(f"||{p}||" for p in suffix_parts)
                        base_text = text[:suffix_match.start()].rstrip()
                        text = f"{base_text} {normalized_suffix}" if base_text else normalized_suffix
            pred_type = pred.get("type", "behavioral")
            valid, reason = _validate_dreaming_output(
                text, "prediction", pred,
                {"activity_summary": content, "existing_texts": existing_pred_texts},
                cfg=cfg,
            )
            if not valid:
                logger.info("Dreaming prediction rejected: %s", reason)
                continue
            # Re-read type — validator may have downgraded temporal → behavioral.
            pred_type = pred.get("type", "behavioral")
            if text:
                tagged = f"[prediction:{pred_type}] {text}"
                entry_id = await asyncio.to_thread(
                    memory_store.add, tagged, None, "dreaming_prediction"
                )
                # Track accuracy for the new prediction.
                try:
                    await database.async_call(
                        database.upsert_prediction, entry_id, tagged, pred_type
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("Dreaming: prediction accuracy upsert failed", exc_info=True)
                # Update duplicate-detection set so later iterations catch repeats.
                existing_pred_texts.add(text.strip().lower())
                pred_collected_ids.append(entry_id)
                predictions_added += 1
                logger.info("Dreaming prediction: %s", text[:100])
        result.items_processed = predictions_added
        result.summary = f"added {predictions_added} predictions"
    except Exception:  # noqa: BLE001
        logger.exception("Dreaming: prediction phase LLM failed")
        result.summary = "LLM call failed"
        result.error = "LLM call failed"

    # Record outside the try block so already-added IDs are tracked even on
    # partial failure (e.g. embedding backend error midway through the loop).
    if pred_collected_ids:
        unique_pred_ids = list(dict.fromkeys(pred_collected_ids))
        await database.async_call(
            database.record_dreaming_entries, "prediction", unique_pred_ids,
        )

    # Validate old predictions: prune unaccessed ones older than 30 days,
    # promote frequently-accessed ones (>= 5 accesses) to dreaming_schema.
    try:
        stale_predictions = await asyncio.to_thread(
            memory_store.get_stale, 30, 1
        )
        pred_to_prune = [
            e["id"] for e in stale_predictions
            if e.get("source") == "dreaming_prediction"
        ]
        if pred_to_prune:
            pruned = await asyncio.to_thread(
                memory_store.bulk_delete, pred_to_prune
            )
            # Mark pruned predictions as retired in accuracy tracking.
            if pruned == len(pred_to_prune):
                for pid in pred_to_prune:
                    try:
                        await database.async_call(database.retire_prediction, pid)
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "Dreaming: failed to retire prediction %s after pruning",
                            pid,
                            exc_info=True,
                        )
            else:
                logger.warning(
                    "Dreaming prediction: bulk_delete deleted %d of %d requested; "
                    "skipping retirement to avoid inconsistent accuracy tracking",
                    pruned,
                    len(pred_to_prune),
                )
            result.summary += f", pruned {pruned} stale predictions"
            logger.info("Dreaming prediction: pruned %d stale predictions",
                        pruned)
    except Exception:  # noqa: BLE001
        logger.debug("Dreaming: prediction pruning failed", exc_info=True)

    # Promote validated predictions (high access count) to schemas.
    promoted = 0
    try:
        # Get prediction-source entries that are well-accessed.
        all_entries = await asyncio.to_thread(
            memory_store.get_by_source, "dreaming_prediction", 200, False
        )
        for entry in all_entries:
            if (entry.get("source") == "dreaming_prediction"
                    and entry.get("access_count", 0) >= 5):
                # Re-store as schema, delete the prediction version.
                text = entry.get("text", "")
                if text.startswith("[prediction:"):
                    # Strip the prediction tag for the schema version.
                    bracket_end = text.find("] ")
                    if bracket_end != -1:
                        text = text[bracket_end + 2:]
                await asyncio.to_thread(
                    memory_store.add, text, None, "dreaming_schema"
                )
                await asyncio.to_thread(
                    memory_store.delete, entry["id"]
                )
                # Retire the prediction in accuracy tracking (promoted, not pruned).
                try:
                    await database.async_call(database.retire_prediction, entry["id"])
                except Exception:  # noqa: BLE001
                    pass
                promoted += 1
                logger.info("Dreaming: promoted prediction to schema: %s",
                            text[:100])
        if promoted:
            result.summary += f", promoted {promoted} to schema"
    except Exception:  # noqa: BLE001
        logger.debug("Dreaming: prediction promotion failed", exc_info=True)

    await database.async_call(
        database.update_dreaming_phase_state, "prediction",
        result.items_processed, result.summary,
    )
    logger.info("Dreaming phase prediction: %s", result.summary)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Dream Cycle Runner
# ═══════════════════════════════════════════════════════════════════════════════

async def _check_survival(cfg: dict) -> dict[str, bool]:
    """Check dreaming entry survival rates and decide which phases to run.

    Returns ``{phase_name: should_run}`` for each creative phase.
    Phases with insufficient data default to enabled.
    """
    threshold = cfg.get("quality_survival_threshold", 0.3)
    min_cycles = cfg.get("quality_min_cycles", 3)
    lookback = cfg.get("quality_lookback_cycles", 5)
    force_enable = set(cfg.get("quality_force_enable_phases", []))

    result: dict[str, bool] = {}
    for phase in ("association", "schema", "prediction"):
        if phase in force_enable:
            result[phase] = True
            continue
        try:
            # Check unchecked rows and update survival counts.
            unchecked = await database.async_call(
                database.get_unchecked_dreaming_entries, phase,
            )
            if unchecked:
                # Decode all entry IDs and issue a single batched lookup to
                # avoid one backend round-trip per row.
                row_entry_ids: dict = {
                    row["id"]: _json.loads(row["entry_ids"]) for row in unchecked
                }
                all_entry_ids = list(dict.fromkeys(
                    eid for ids in row_entry_ids.values() for eid in ids
                ))
                surviving_set: set[str] = set()
                if all_entry_ids:
                    surviving_set = await asyncio.to_thread(
                        memory_store.exists_batch, all_entry_ids,
                    )
                for row in unchecked:
                    ids = row_entry_ids.get(row["id"], [])
                    survived = sum(1 for eid in ids if eid in surviving_set)
                    await database.async_call(
                        database.update_dreaming_survival, row["id"], survived,
                    )

            # Compute rate and decide.
            rate_info = await database.async_call(
                database.get_phase_survival_rate, phase, lookback,
            )
            rate = rate_info[0] if rate_info is not None else None
            checked_count = rate_info[1] if rate_info is not None else 0
            if rate is not None and rate < threshold:
                if checked_count >= min_cycles or rate == 0.0:
                    logger.warning(
                        "Dreaming: auto-disabling phase '%s' "
                        "(survival rate %.1f%% < %.1f%% threshold)",
                        phase, rate * 100, threshold * 100,
                    )
                    result[phase] = False
                    continue
            result[phase] = True
        except Exception:  # noqa: BLE001
            logger.debug("Dreaming: survival check failed for %s", phase, exc_info=True)
            result[phase] = True  # fail-open
    return result


async def run_dream_cycle(
    pool: "BackendPool",
    event_bus: "EventBus | None" = None,
) -> DreamReport:
    """Run a full nightly dreaming cycle.

    Phases execute in order: housekeeping -> association -> schema -> prediction.
    Each phase is independent — a failure in one does not abort the others.
    """
    report = DreamReport()
    cfg = _load_dreaming_config()

    # Create Qdrant snapshot before housekeeping for rollback safety.
    if memory_store.is_qdrant_backend():
        try:
            snap_name = await asyncio.to_thread(memory_store.create_snapshot)
            if snap_name:
                logger.info("Dreaming: Qdrant snapshot: %s", snap_name)
                # Record in dreaming_state for provenance.
                await database.async_call(
                    database.update_dreaming_phase_state,
                    "snapshot", 0, snap_name,
                )
        except Exception:  # noqa: BLE001
            logger.debug("Dreaming: snapshot creation failed", exc_info=True)

    # Pre-compute shared similarity data (single matrix for all phases).
    sim_data = SimilarityData()
    try:
        sim_data = await _compute_similarity_data(cfg)
    except Exception:  # noqa: BLE001
        logger.exception("Dreaming: similarity computation failed")
        report.errors.append("similarity computation failed")

    # Define phase roster: housekeeping first, then creative.
    housekeeping_phases: list[tuple[str, Any]] = [
        ("dedup", lambda: _phase_dedup(pool, cfg, sim_data)),
        ("contradiction", lambda: _phase_contradiction(pool, cfg, sim_data)),
        ("stale_pruning", lambda: _phase_stale_pruning(pool, cfg, sim_data)),
        ("task_consolidation", lambda: _phase_task_consolidation(pool, cfg, sim_data)),
        ("skill_consolidation", lambda: _phase_skill_consolidation(pool, cfg, sim_data)),
    ]

    creative_phases: list[tuple[str, Any]] = [
        ("association", lambda: _phase_association(pool, cfg, sim_data)),
        ("schema", lambda: _phase_schema(pool, cfg, sim_data)),
        ("prediction", lambda: _phase_prediction(pool, cfg, sim_data)),
    ]

    # Run housekeeping first (deletions may affect survival checks).
    phases: list[tuple[str, Any]] = list(housekeeping_phases)

    # Execute housekeeping, then check survival to gate creative phases.
    for phase_name, phase_fn in phases:
        logger.info("Dreaming: starting phase '%s'", phase_name)
        if event_bus:
            event_bus.emit("dreaming.phase_started", phase=phase_name)
        try:
            phase_result = await phase_fn()
            report.results.append(phase_result)
            report.phases_run.append(phase_name)
            if event_bus:
                event_bus.emit(
                    "dreaming.phase_completed",
                    phase=phase_name,
                    items=phase_result.items_processed,
                    summary=phase_result.summary,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Dreaming: phase '%s' failed: %s", phase_name, exc)
            report.errors.append(f"{phase_name}: {exc}")
            report.results.append(PhaseResult(
                phase_name=phase_name, error=str(exc),
                summary=f"failed: {exc}",
            ))

    # Check quality metrics to gate creative phases.
    should_run = await _check_survival(cfg)

    # Execute creative phases, filtered by survival check.
    for phase_name, phase_fn in creative_phases:
        if not should_run.get(phase_name, True):
            logger.info("Dreaming: skipping phase '%s' (auto-disabled by quality metrics)", phase_name)
            report.results.append(PhaseResult(
                phase_name=phase_name,
                summary="auto-disabled (low survival rate)",
            ))
            continue
        logger.info("Dreaming: starting phase '%s'", phase_name)
        if event_bus:
            event_bus.emit("dreaming.phase_started", phase=phase_name)
        try:
            phase_result = await phase_fn()
            report.results.append(phase_result)
            report.phases_run.append(phase_name)
            if event_bus:
                event_bus.emit(
                    "dreaming.phase_completed",
                    phase=phase_name,
                    items=phase_result.items_processed,
                    summary=phase_result.summary,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Dreaming: phase '%s' failed: %s", phase_name, exc)
            report.errors.append(f"{phase_name}: {exc}")
            report.results.append(PhaseResult(
                phase_name=phase_name, error=str(exc),
                summary=f"failed: {exc}",
            ))

    # Auto-commit all changes.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, data_versioning.auto_commit, "dreaming: nightly consolidation",
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# DreamingLoop (scheduler)
# ═══════════════════════════════════════════════════════════════════════════════

class DreamingLoop:
    """Asyncio task that fires ``run_dream_cycle`` once per night.

    Calculates seconds until the configured target time and sleeps until
    then — no APScheduler dependency, no persistence needed.
    """

    # Number of memory appends before considering an early consolidation.
    _EARLY_TRIGGER_THRESHOLD = 50

    # Minimum seconds between early skill consolidation runs to avoid spamming.
    _SKILLS_COOLDOWN_SECONDS = 3600  # 1 hour

    def __init__(self, config: DreamingConfig,
                 pool: "BackendPool",
                 event_bus: "EventBus | None" = None) -> None:
        self._cfg = config
        self._pool = pool
        self._event_bus = event_bus
        self._running = False
        self._memory_append_count = 0
        self._event_bus_subs: list[str] = []
        self._last_consolidation: float = 0.0
        self._firing: bool = False

    async def _on_memory_appended(self, event) -> None:
        """Track memory appends; trigger early consolidation if threshold exceeded."""
        self._memory_append_count += 1
        if self._memory_append_count >= self._EARLY_TRIGGER_THRESHOLD:
            if self._firing:
                logger.debug("Dreaming: memory threshold reached but consolidation in progress")
                return
            logger.info(
                "Dreaming: %d memory appends — triggering early consolidation",
                self._memory_append_count,
            )
            self._memory_append_count = 0
            await self._fire()

    async def _on_skills_oversized(self, event) -> None:
        """Trigger early dreaming cycle when skill TOC exceeds size limit."""
        if self._firing:
            logger.debug("Dreaming: skills oversized but consolidation in progress, skipping")
            return
        now = _time.monotonic()
        if now - self._last_consolidation < self._SKILLS_COOLDOWN_SECONDS:
            logger.debug("Dreaming: skills oversized but cooldown active, skipping")
            return
        logger.info("Dreaming: skills TOC oversized — triggering early consolidation")
        await self._fire()

    async def run(self) -> None:
        self._running = True
        if self._event_bus:
            sub_id = self._event_bus.subscribe(
                "memory.appended", self._on_memory_appended,
                debounce_ms=5000,
            )
            self._event_bus_subs.append(sub_id)
            sub_id = self._event_bus.subscribe(
                "skills.oversized", self._on_skills_oversized,
                debounce_ms=30_000,
            )
            self._event_bus_subs.append(sub_id)
        target = dt_time(self._cfg.hour, self._cfg.minute)
        logger.info(
            "Dreaming loop started (target=%02d:%02d %s, model=%s)",
            target.hour, target.minute, self._cfg.timezone,
            self._pool.primary.model,
        )
        while self._running:
            delay = self._seconds_until(target, self._cfg.timezone)
            logger.debug("Dreaming: next run in %.0f s", delay)
            await asyncio.sleep(delay)
            if not self._running:
                break
            await self._fire()

    def stop(self) -> None:
        self._running = False
        if self._event_bus:
            for sub_id in self._event_bus_subs:
                self._event_bus.unsubscribe(sub_id)
            self._event_bus_subs.clear()

    async def _fire(self) -> None:
        if self._firing:
            logger.debug("Dreaming: consolidation already in progress, skipping")
            return
        self._firing = True
        try:
            if not self._pool.enabled:
                logger.error("Dreaming: no backends configured, skipping")
                return
            logger.info(
                "Dreaming: starting consolidation cycle (model=%s)",
                self._pool.primary.model,
            )
            if self._event_bus:
                self._event_bus.emit("dreaming.started")
            try:
                report = await run_dream_cycle(
                    pool=self._pool, event_bus=self._event_bus,
                )
                if self._event_bus:
                    self._event_bus.emit(
                        "dreaming.completed",
                        phases=report.phases_run,
                        errors=report.errors,
                        results=[
                            {"phase": r.phase_name, "items": r.items_processed,
                             "summary": r.summary}
                            for r in report.results
                        ],
                    )
                logger.info(
                    "Dreaming: consolidation cycle complete (%d phases, %d errors)",
                    len(report.phases_run), len(report.errors),
                )
                self._last_consolidation = _time.monotonic()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Dreaming: consolidation cycle failed: %s", exc)
        finally:
            self._firing = False

    @staticmethod
    def _seconds_until(target: dt_time, tz_name: str = "UTC") -> float:
        """Return seconds from now until the next occurrence of *target* time."""
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = timezone.utc
        now = datetime.now(tz)
        candidate = now.replace(
            hour=target.hour, minute=target.minute,
            second=0, microsecond=0,
        )
        if candidate <= now:
            candidate += timedelta(days=1)
        return (candidate - now).total_seconds()
