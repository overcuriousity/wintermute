"""
Dreaming – Nightly Memory Consolidation

Runs as an autonomous asyncio task that fires at a configurable hour each
night to review and prune the persistent memory components (MEMORIES.txt,
PULSE.txt) without user interaction.  Uses a direct API call (no tool loop,
no thread history) so it never interferes with ongoing conversations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from wintermute import prompt_assembler

if TYPE_CHECKING:
    from wintermute.llm_thread import BackendPool

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DREAM_MEMORIES_PROMPT_FILE = DATA_DIR / "DREAM_MEMORIES_PROMPT.txt"
DREAM_PULSE_PROMPT_FILE    = DATA_DIR / "DREAM_PULSE_PROMPT.txt"


@dataclass
class DreamingConfig:
    hour: int = 1
    minute: int = 0

_DEFAULT_MEMORIES_PROMPT = """\
Below is the current content of MEMORIES.txt — the long-term memory store for \
a personal AI assistant. Memories are appended throughout the day and may \
contain duplicates or near-duplicates.

Your task is to consolidate it:
- Merge duplicate and near-duplicate entries into single concise statements.
- Remove entries that are clearly outdated or contradicted by newer entries.
- Preserve all distinct, useful facts exactly.
- Keep the result as short as possible without losing information.
- Maintain a flat, scannable structure (one fact per line or short paragraph).

Return ONLY the consolidated MEMORIES.txt content, with no preamble or \
explanation.

--- MEMORIES.txt ---
{content}
"""

_DEFAULT_PULSE_PROMPT = """\
Below is the current content of PULSE.txt — the active pulse working memory \
(ongoing goals, recurring tasks) for a personal AI assistant.

Your task is to consolidate it:
- Remove completed or clearly stale items.
- Merge duplicate or overlapping goals.
- Keep all genuinely active tasks.
- Return the result as a concise, well-structured list.

Return ONLY the consolidated PULSE.txt content, with no preamble or \
explanation.

--- PULSE.txt ---
{content}
"""


def _load_prompt(path: Path, default: str) -> str:
    """Load a prompt template from a file, falling back to the built-in default."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
    return default


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
    result = (response.choices[0].message.content or "").strip()
    logger.debug("Dreaming: %s consolidated (%d -> %d chars)", label, len(content), len(result))
    return result


async def run_dream_cycle(pool: "BackendPool") -> None:
    """
    Run a full nightly consolidation pass over MEMORIES.txt and PULSE.txt.

    Skips any component that is empty or missing.  Each component is
    consolidated independently so a failure in one does not abort the other.
    """
    memories = prompt_assembler._read(prompt_assembler.MEMORIES_FILE)
    if memories:
        try:
            mem_prompt = _load_prompt(DREAM_MEMORIES_PROMPT_FILE, _DEFAULT_MEMORIES_PROMPT)
            consolidated = await _consolidate(
                pool, "MEMORIES.txt", mem_prompt, memories,
            )
            if consolidated:
                prompt_assembler.update_memories(consolidated)
                logger.info("Dreaming: MEMORIES.txt updated (%d chars)", len(consolidated))
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: failed to consolidate MEMORIES.txt")
    else:
        logger.debug("Dreaming: MEMORIES.txt empty or missing, skipping")

    pulse = prompt_assembler._read(prompt_assembler.PULSE_FILE)
    if pulse:
        try:
            pulse_prompt = _load_prompt(DREAM_PULSE_PROMPT_FILE, _DEFAULT_PULSE_PROMPT)
            consolidated = await _consolidate(
                pool, "PULSE.txt", pulse_prompt, pulse,
            )
            if consolidated:
                prompt_assembler.update_pulse(consolidated)
                logger.info("Dreaming: PULSE.txt updated (%d chars)", len(consolidated))
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: failed to consolidate PULSE.txt")
    else:
        logger.debug("Dreaming: PULSE.txt empty or missing, skipping")


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
        logger.info("Dreaming loop started (target=%02d:%02d, model=%s)",
                     target.hour, target.minute, self._pool.primary.model)
        while self._running:
            delay = self._seconds_until(target)
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
    def _seconds_until(target: dt_time) -> float:
        """Return seconds from now until the next occurrence of *target* time (UTC)."""
        now = datetime.now(timezone.utc)
        candidate = now.replace(hour=target.hour, minute=target.minute,
                                second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return (candidate - now).total_seconds()
