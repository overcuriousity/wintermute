"""
Dreaming – Nightly Memory Consolidation

Runs as an autonomous asyncio task that fires at a configurable hour each
night to review and prune the persistent memory components (MEMORIES.txt,
tasks DB items) without user interaction.  Uses a direct API call (no tool
loop, no thread history) so it never interferes with ongoing conversations.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re as _re
import time as _time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from wintermute.infra import database
from wintermute.infra import data_versioning
from wintermute.infra import memory_store
from wintermute.infra import prompt_assembler
from wintermute.infra import prompt_loader

if TYPE_CHECKING:
    from wintermute.core.llm_thread import BackendPool
    from wintermute.infra.event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class DreamingConfig:
    hour: int = 1
    minute: int = 0
    timezone: str = "UTC"


async def _consolidate(pool: "BackendPool",
                        label: str, prompt_template: str, content: str,
                        json_mode: bool = False,
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
                         "max_tokens_override": 2048}
    if json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}
    response = await pool.call(**call_kwargs)
    if not response.choices:
        logger.warning("Dreaming (%s): LLM returned empty choices", label)
        logger.debug("Empty choices raw response: %s", response)
        return ""
    result = (response.choices[0].message.content or "").strip()
    try:
        await database.async_call(
            database.save_interaction_log,
            _time.time(), "dreaming", f"system:dreaming:{label}",
            pool.last_used,
            prompt[:2000], result[:2000], "ok",
        )
    except Exception:
        pass
    logger.debug("Dreaming: %s consolidated (%d -> %d chars)", label, len(content), len(result))
    return result


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from text, tolerating prose wrappers."""
    text = text.strip()
    try:
        result = _json.loads(text)
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
    except _json.JSONDecodeError:
        pass
    fenced = _re.sub(r"^```(?:json)?\s*", "", text, flags=_re.MULTILINE)
    fenced = _re.sub(r"```\s*$", "", fenced, flags=_re.MULTILINE).strip()
    try:
        result = _json.loads(fenced)
        if isinstance(result, dict):
            return result
    except _json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            result = _json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except _json.JSONDecodeError:
            pass
    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


def _extract_json_array(text: str) -> list:
    """Extract the first JSON array from text, tolerating prose wrappers.

    Tries in order:
    1. Direct parse (model returned clean JSON)
    2. Strip ```json ... ``` or ``` ... ``` fences
    3. Find the first '[' ... last ']' substring and parse that
    Raises ValueError if nothing parses.
    """
    text = text.strip()
    # 1. Direct parse
    try:
        result = _json.loads(text)
        if isinstance(result, list):
            return result
    except _json.JSONDecodeError:
        pass
    # 2. Strip markdown code fences
    fenced = _re.sub(r"^```(?:json)?\s*", "", text, flags=_re.MULTILINE)
    fenced = _re.sub(r"```\s*$", "", fenced, flags=_re.MULTILINE).strip()
    try:
        result = _json.loads(fenced)
        if isinstance(result, list):
            return result
    except _json.JSONDecodeError:
        pass
    # 3. Grab outermost [...] span
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            result = _json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except _json.JSONDecodeError:
            pass
    raise ValueError(f"No JSON array found in response: {text[:200]!r}")


async def _consolidate_skills(pool: "BackendPool") -> None:
    """Deduplicate and condense all skill .md files in data/skills/."""
    skills_dir = prompt_assembler.SKILLS_DIR
    if not skills_dir.exists():
        logger.debug("Dreaming: skills dir missing, skipping")
        return

    skill_files = sorted(skills_dir.glob("*.md"))
    if not skill_files:
        logger.debug("Dreaming: no skill files, skipping")
        return

    # stem -> (path, content)
    skills: dict[str, tuple[Path, str]] = {}
    for f in skill_files:
        content = f.read_text(encoding="utf-8").strip()
        if content:
            skills[f.stem] = (f, content)

    if not skills:
        return

    # ── Step 0: Auto-retire unused skills ────────────────────────────────────
    try:
        from wintermute.workers import skill_stats
        unused = skill_stats.get_unused_skills(days=90)
        if unused:
            archive_dir = skills_dir / ".archive"
            archive_dir.mkdir(exist_ok=True)
            for name in unused:
                src = skills_dir / f"{name}.md"
                if src.exists():
                    src.rename(archive_dir / f"{name}.md")
                    skills.pop(name, None)
                    skill_stats.remove_skill(name)
                    logger.info("Dreaming: retired unused skill '%s' → .archive/", name)
    except Exception:
        logger.debug("Dreaming: skill retirement failed", exc_info=True)

    if not skills:
        return

    # ── Step 1: Deduplication ────────────────────────────────────────────────
    if len(skills) > 1:
        try:
            dedup_prompt = prompt_loader.load("DREAM_SKILLS_DEDUP_PROMPT.txt")
            formatted = "\n\n".join(
                f"=== {name} ===\n{content}"
                for name, (_, content) in skills.items()
            )
            raw = await _consolidate(pool, "skills_dedup", dedup_prompt, formatted)
            actions = _extract_json_array(raw)
            for act in actions:
                action = act.get("action")
                name = act.get("file", "")
                if action == "delete":
                    if name in skills:
                        skills[name][0].unlink()
                        logger.info("Dreaming: deleted duplicate skill '%s'", name)
                        del skills[name]
                elif action == "merge":
                    target = act.get("into", "")
                    content = act.get("content", "").strip()
                    if not target or not content:
                        logger.warning("Dreaming: merge action missing 'into' or 'content' for '%s'", name)
                        continue
                    # Write merged content to the target file.
                    target_path = skills_dir / f"{target}.md"
                    target_path.write_text(content, encoding="utf-8")
                    logger.info("Dreaming: merged skill '%s' into '%s'", name, target)
                    # Update in-memory entry for the target so condensation uses merged content.
                    skills[target] = (target_path, content)
                    # Remove the source file if it differs from the target.
                    if name != target and name in skills:
                        skills[name][0].unlink()
                        del skills[name]
        except (ValueError, _json.JSONDecodeError) as exc:
            logger.warning("Dreaming: skill dedup returned non-JSON, skipping dedup: %s", exc)
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: skill dedup failed")

    # ── Step 2: Condense each surviving skill ────────────────────────────────
    condense_template = prompt_loader.load("DREAM_SKILLS_CONDENSATION_PROMPT.txt")
    for name, (fpath, content) in list(skills.items()):
        try:
            prompt = condense_template.format(skill_name=name, content=content)
            response = await pool.call(
                messages=[{"role": "user", "content": prompt}],
                max_tokens_override=600,
            )
            if not response.choices:
                logger.warning("Dreaming: skill condensation returned empty choices for '%s'", name)
                logger.debug("Empty choices raw response: %s", response)
                continue
            result = (response.choices[0].message.content or "").strip()
            if result:
                fpath.write_text(result, encoding="utf-8")
                logger.info(
                    "Dreaming: condensed skill '%s' (%d -> %d chars)",
                    name, len(content), len(result),
                )
                try:
                    await database.async_call(
                        database.save_interaction_log,
                        _time.time(), "dreaming", f"system:dreaming:skill:{name}",
                        pool.last_used, prompt[:2000], result[:2000], "ok",
                    )
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: failed to condense skill '%s'", name)


def _load_dreaming_config() -> dict:
    """Load dreaming-specific config from config.yaml with defaults."""
    defaults = {
        "dedup_similarity_threshold": 0.85,
        "stale_days": 90,
        "stale_min_access": 3,
        "working_set_size": 50,
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


async def _vector_dream_cycle(pool: "BackendPool") -> None:
    """4-phase vector-native dreaming: dedup, contradictions, stale pruning, export."""
    cfg = _load_dreaming_config()
    threshold = cfg["dedup_similarity_threshold"]

    # ── Phase 1: Deduplication clustering ──────────────────────────────
    logger.info("Dreaming phase 1: deduplication clustering (threshold=%.2f)", threshold)
    all_entries = await asyncio.to_thread(memory_store.get_all_with_vectors)
    if not all_entries:
        logger.debug("Dreaming: vector store empty, skipping")
        return

    # Filter out entries without vectors (FTS5 backend returns []).
    entries_with_vecs = [e for e in all_entries if e.get("vector")]
    if not entries_with_vecs:
        logger.info("Dreaming: no vectors available, falling back to working set export only")
    else:
        import numpy as np
        texts = [e["text"] for e in entries_with_vecs]
        ids = [e["id"] for e in entries_with_vecs]
        vecs = np.array([e["vector"] for e in entries_with_vecs], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs_normed = vecs / norms

        # Compute cosine similarity matrix.
        sim_matrix = vecs_normed @ vecs_normed.T

        # Collect above-threshold pairs for union-find.
        pairs: list[tuple[int, int, float]] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                s = float(sim_matrix[i, j])
                if s >= threshold:
                    pairs.append((i, j, s))

        clusters = _union_find_clusters(pairs, len(texts), threshold)
        merged_count = 0
        dedup_prompt = prompt_loader.load("DREAM_DEDUP_PROMPT.txt")

        for cluster in clusters:
            cluster_texts = [texts[idx] for idx in cluster]
            cluster_ids = [ids[idx] for idx in cluster]
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

        if merged_count:
            logger.info("Dreaming phase 1: merged %d clusters", merged_count)

        # ── Phase 2: Contradiction detection ───────────────────────────
        logger.info("Dreaming phase 2: contradiction detection")
        # Re-fetch after dedup.
        all_entries = await asyncio.to_thread(memory_store.get_all_with_vectors)
        entries_with_vecs = [e for e in all_entries if e.get("vector")]

        if len(entries_with_vecs) >= 2:
            texts = [e["text"] for e in entries_with_vecs]
            ids = [e["id"] for e in entries_with_vecs]
            vecs = np.array([e["vector"] for e in entries_with_vecs], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs_normed = vecs / norms
            sim_matrix = vecs_normed @ vecs_normed.T

            # Find pairs in the "suspicious" similarity range.
            contradiction_pairs: list[tuple[int, int, float]] = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    s = float(sim_matrix[i, j])
                    if 0.5 <= s < threshold:
                        contradiction_pairs.append((i, j, s))

            # Cap at top 20 most similar.
            contradiction_pairs.sort(key=lambda x: x[2], reverse=True)
            contradiction_pairs = contradiction_pairs[:20]

            contra_prompt = prompt_loader.load("DREAM_CONTRADICTION_PROMPT.txt")
            resolved_count = 0
            for i, j, _ in contradiction_pairs:
                try:
                    raw = await _consolidate(
                        pool, "contradiction",
                        contra_prompt.replace("{entry_1}", texts[i]).replace("{entry_2}", texts[j]),
                        "",
                    )
                    decision = _extract_json_object(raw)
                    action = decision.get("action", "")
                    if action == "keep_first":
                        await asyncio.to_thread(memory_store.delete, ids[j])
                        resolved_count += 1
                    elif action == "keep_second":
                        await asyncio.to_thread(memory_store.delete, ids[i])
                        resolved_count += 1
                    elif action == "merge" and decision.get("result"):
                        await asyncio.to_thread(memory_store.bulk_delete, [ids[i], ids[j]])
                        await asyncio.to_thread(
                            memory_store.add, decision["result"], None, "dreaming_merge"
                        )
                        resolved_count += 1
                except (_json.JSONDecodeError, Exception):  # noqa: BLE001
                    logger.debug("Dreaming: contradiction resolution failed for pair", exc_info=True)

            if resolved_count:
                logger.info("Dreaming phase 2: resolved %d contradictions", resolved_count)

    # ── Phase 3: Stale pruning ─────────────────────────────────────────
    logger.info("Dreaming phase 3: stale pruning (days=%d, min_access=%d)",
                cfg["stale_days"], cfg["stale_min_access"])
    stale = await asyncio.to_thread(
        memory_store.get_stale, cfg["stale_days"], cfg["stale_min_access"]
    )
    # Protect user-explicit memories from pruning.
    prune_ids = [e["id"] for e in stale if e.get("source") != "user_explicit"]
    if prune_ids:
        deleted = await asyncio.to_thread(memory_store.bulk_delete, prune_ids)
        logger.info("Dreaming phase 3: pruned %d stale entries", deleted)

    # ── Phase 4: Working set export ────────────────────────────────────
    logger.info("Dreaming phase 4: exporting working set to MEMORIES.txt (size=%d)",
                cfg["working_set_size"])
    top = await asyncio.to_thread(memory_store.get_top_accessed, cfg["working_set_size"])
    if top:
        working_set = "\n".join(e["text"] for e in top if e.get("text"))
        # Write MEMORIES.txt directly — do NOT call update_memories() which
        # would replace_all() in the vector store and destroy metadata.
        prompt_assembler.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with prompt_assembler._memories_lock:
            prompt_assembler.MEMORIES_FILE.write_text(working_set, encoding="utf-8")
        import threading
        threading.Thread(
            target=data_versioning.auto_commit, args=("dreaming: working set export",),
            daemon=True,
        ).start()
        logger.info("Dreaming: MEMORIES.txt exported as working set (%d entries)", len(top))


async def run_dream_cycle(pool: "BackendPool") -> None:
    """
    Run a full nightly consolidation pass over MEMORIES.txt and task DB items.

    Skips any component that is empty or missing.  Each component is
    consolidated independently so a failure in one does not abort the other.
    """
    if memory_store.is_vector_enabled():
        try:
            await _vector_dream_cycle(pool)
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: vector dream cycle failed")
    else:
        memories_snapshot = prompt_assembler._read(prompt_assembler.MEMORIES_FILE)
        if memories_snapshot:
            try:
                mem_prompt = prompt_loader.load("DREAM_MEMORIES_PROMPT.txt")
                _snap_lines = [l for l in memories_snapshot.splitlines() if l.strip()]
                consolidated = await _consolidate(
                    pool, "MEMORIES.txt", mem_prompt, memories_snapshot,
                    source="MEMORIES.txt flat file",
                    entry_count=str(len(_snap_lines)),
                )
                if consolidated:
                    prompt_assembler.merge_consolidated_memories(
                        memories_snapshot, consolidated,
                    )
                    logger.info("Dreaming: MEMORIES.txt updated (%d chars)", len(consolidated))
            except Exception:  # noqa: BLE001
                logger.exception("Dreaming: failed to consolidate MEMORIES.txt")
        else:
            logger.debug("Dreaming: MEMORIES.txt empty or missing, skipping")

    task_items = await database.async_call(database.list_tasks, "active")
    if task_items:
        try:
            task_prompt = prompt_loader.load("DREAM_TASK_PROMPT.txt")
            formatted = "\n".join(
                f"[P{it['priority']}] #{it['id']}: {it['content']}"
                + (f" [{it['schedule_desc']}]" if it.get("schedule_desc") else "")
                for it in task_items
            )
            raw = await _consolidate(pool, "tasks", task_prompt, formatted)
            if raw:
                try:
                    actions = _extract_json_array(raw)
                except (ValueError, _json.JSONDecodeError) as exc:
                    logger.warning("Dreaming: tasks LLM returned non-JSON, skipping: %s", exc)
                    actions = []
                applied = 0
                for act in actions:
                    a = act.get("action")
                    aid = act.get("id")
                    if a == "complete" and aid is not None:
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
                logger.info("Dreaming: applied %d task actions", applied)
            await database.async_call(database.delete_old_completed_tasks, 30)
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: failed to consolidate tasks")
    else:
        logger.debug("Dreaming: no active tasks, skipping")

    try:
        await _consolidate_skills(pool)
    except Exception:  # noqa: BLE001
        logger.exception("Dreaming: failed to consolidate skills")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, data_versioning.auto_commit, "dreaming: nightly consolidation",
    )


class DreamingLoop:
    """Asyncio task that fires ``run_dream_cycle`` once per night.

    Calculates seconds until the configured target time and sleeps until
    then — no APScheduler dependency, no persistence needed.
    """

    # Number of memory appends before considering an early consolidation.
    _EARLY_TRIGGER_THRESHOLD = 50

    def __init__(self, config: DreamingConfig,
                 pool: "BackendPool",
                 event_bus: "Optional[EventBus]" = None) -> None:
        self._cfg = config
        self._pool = pool
        self._event_bus = event_bus
        self._running = False
        self._memory_append_count = 0
        self._event_bus_subs: list[str] = []

    async def _on_memory_appended(self, event) -> None:
        """Track memory appends; trigger early consolidation if threshold exceeded."""
        self._memory_append_count += 1
        if self._memory_append_count >= self._EARLY_TRIGGER_THRESHOLD:
            logger.info("Dreaming: %d memory appends — triggering early consolidation",
                         self._memory_append_count)
            self._memory_append_count = 0
            await self._fire()

    async def run(self) -> None:
        self._running = True
        if self._event_bus:
            sub_id = self._event_bus.subscribe("memory.appended", self._on_memory_appended,
                                                debounce_ms=5000)
            self._event_bus_subs.append(sub_id)
        target = dt_time(self._cfg.hour, self._cfg.minute)
        logger.info("Dreaming loop started (target=%02d:%02d %s, model=%s)",
                     target.hour, target.minute, self._cfg.timezone,
                     self._pool.primary.model)
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
        if not self._pool.enabled:
            logger.error("Dreaming: no backends configured, skipping")
            return
        logger.info("Dreaming: starting nightly consolidation (model=%s)",
                     self._pool.primary.model)
        if self._event_bus:
            self._event_bus.emit("dreaming.started")
        try:
            await run_dream_cycle(pool=self._pool)
            if self._event_bus:
                self._event_bus.emit("dreaming.completed")
            logger.info("Dreaming: nightly consolidation complete")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Dreaming: nightly consolidation failed: %s", exc)

    @staticmethod
    def _seconds_until(target: dt_time, tz_name: str = "UTC") -> float:
        """Return seconds from now until the next occurrence of *target* time in *tz_name*."""
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = timezone.utc
        now = datetime.now(tz)
        candidate = now.replace(hour=target.hour, minute=target.minute,
                                second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return (candidate - now).total_seconds()
