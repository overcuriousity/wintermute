"""
Dreaming – Nightly Memory Consolidation

Runs as an autonomous asyncio task that fires at a configurable hour each
night to review and prune the persistent memory components (MEMORIES.txt,
agenda DB items) without user interaction.  Uses a direct API call (no tool
loop, no thread history) so it never interferes with ongoing conversations.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time as _time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import TYPE_CHECKING

from wintermute.infra import database
from wintermute.infra import data_versioning
from wintermute.infra import memory_store
from wintermute.infra import prompt_assembler
from wintermute.infra import prompt_loader

if TYPE_CHECKING:
    from wintermute.core.llm_thread import BackendPool

logger = logging.getLogger(__name__)


@dataclass
class DreamingConfig:
    hour: int = 1
    minute: int = 0
    timezone: str = "UTC"


async def _consolidate(pool: "BackendPool",
                        label: str, prompt_template: str, content: str) -> str:
    """Call the LLM to consolidate a single memory component."""
    prompt = prompt_template
    if "{content}" in prompt:
        prompt = prompt.format(content=content)
    else:
        prompt = prompt + "\n\n" + content
    response = await pool.call(
        messages=[{"role": "user", "content": prompt}],
        max_tokens_override=2048,
    )
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

    # ── Step 1: Deduplication ────────────────────────────────────────────────
    if len(skills) > 1:
        try:
            dedup_prompt = prompt_loader.load("DREAM_SKILLS_DEDUP_PROMPT.txt")
            formatted = "\n\n".join(
                f"=== {name} ===\n{content}"
                for name, (_, content) in skills.items()
            )
            raw = await _consolidate(pool, "skills_dedup", dedup_prompt, formatted)
            actions = _json.loads(raw)
            for act in actions:
                if act.get("action") == "delete":
                    name = act.get("file", "")
                    if name in skills:
                        skills[name][0].unlink()
                        logger.info("Dreaming: deleted duplicate skill '%s'", name)
                        del skills[name]
        except _json.JSONDecodeError:
            logger.warning("Dreaming: skill dedup returned non-JSON, skipping dedup")
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


async def run_dream_cycle(pool: "BackendPool") -> None:
    """
    Run a full nightly consolidation pass over MEMORIES.txt and agenda DB items.

    Skips any component that is empty or missing.  Each component is
    consolidated independently so a failure in one does not abort the other.
    """
    if memory_store.is_vector_enabled():
        all_entries = memory_store.get_all()
        if all_entries:
            try:
                mem_prompt = prompt_loader.load("DREAM_MEMORIES_PROMPT.txt")
                memories_text = "\n".join(e["text"] for e in all_entries)
                consolidated = await _consolidate(
                    pool, "MEMORIES.txt", mem_prompt, memories_text,
                )
                if consolidated:
                    entries = [l.strip() for l in consolidated.strip().splitlines() if l.strip()]
                    memory_store.replace_all(entries)
                    # Keep MEMORIES.txt as git-versioned backup.
                    prompt_assembler.MEMORIES_FILE.write_text(
                        consolidated, encoding="utf-8",
                    )
                    logger.info("Dreaming: vector store + MEMORIES.txt updated (%d entries)",
                                len(entries))
            except Exception:  # noqa: BLE001
                logger.exception("Dreaming: failed to consolidate vector memories")
        else:
            logger.debug("Dreaming: vector store empty, skipping memory consolidation")
    else:
        memories_snapshot = prompt_assembler._read(prompt_assembler.MEMORIES_FILE)
        if memories_snapshot:
            try:
                mem_prompt = prompt_loader.load("DREAM_MEMORIES_PROMPT.txt")
                consolidated = await _consolidate(
                    pool, "MEMORIES.txt", mem_prompt, memories_snapshot,
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

    agenda_items = await database.async_call(database.list_agenda_items, "active")
    if agenda_items:
        try:
            agenda_prompt = prompt_loader.load("DREAM_AGENDA_PROMPT.txt")
            formatted = "\n".join(
                f"[P{it['priority']}] #{it['id']}: {it['content']}"
                for it in agenda_items
            )
            raw = await _consolidate(pool, "agenda", agenda_prompt, formatted)
            if raw:
                try:
                    actions = _json.loads(raw)
                except _json.JSONDecodeError:
                    logger.warning("Dreaming: agenda LLM returned non-JSON, skipping")
                    actions = []
                applied = 0
                for act in actions:
                    a = act.get("action")
                    aid = act.get("id")
                    if a == "complete" and aid is not None:
                        await database.async_call(database.complete_agenda_item, int(aid))
                        applied += 1
                    elif a == "update" and aid is not None:
                        kwargs = {}
                        if "content" in act:
                            kwargs["content"] = act["content"]
                        if "priority" in act:
                            kwargs["priority"] = int(act["priority"])
                        if kwargs:
                            await database.async_call(database.update_agenda_item, int(aid), **kwargs)
                            applied += 1
                logger.info("Dreaming: applied %d agenda actions", applied)
            await database.async_call(database.delete_old_completed_agenda, 30)
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: failed to consolidate agenda")
    else:
        logger.debug("Dreaming: no active agenda items, skipping")

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

    def __init__(self, config: DreamingConfig,
                 pool: "BackendPool") -> None:
        self._cfg = config
        self._pool = pool
        self._running = False

    async def run(self) -> None:
        self._running = True
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

    async def _fire(self) -> None:
        if not self._pool.enabled:
            logger.error("Dreaming: no backends configured, skipping")
            return
        logger.info("Dreaming: starting nightly consolidation (model=%s)",
                     self._pool.primary.model)
        try:
            await run_dream_cycle(pool=self._pool)
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
