"""
Dreaming – Nightly Memory Consolidation

Runs as an autonomous asyncio task that fires at a configurable hour each
night to review and prune the persistent memory components (MEMORIES.txt,
PULSE.txt) without user interaction.  Uses a direct API call (no tool loop,
no thread history) so it never interferes with ongoing conversations.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Optional

from openai import AsyncOpenAI

from wintermute import prompt_assembler

logger = logging.getLogger(__name__)


@dataclass
class DreamingConfig:
    hour: int = 1
    minute: int = 0
    model: Optional[str] = None  # None = fall back to compaction_model, then main model

_MEMORIES_PROMPT = """\
Below is the current content of MEMORIES.txt — the long-term memory store for \
a personal AI assistant.

Your task is to consolidate it:
- Remove duplicate or outdated entries.
- Merge closely related facts into single concise statements.
- Preserve all distinct, useful facts exactly.
- Keep the result as short as possible without losing information.

Return ONLY the consolidated MEMORIES.txt content, with no preamble or \
explanation.

--- MEMORIES.txt ---
{content}
"""

_PULSE_PROMPT = """\
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


async def _consolidate(client: AsyncOpenAI, model: str,
                        label: str, prompt_template: str, content: str) -> str:
    """Call the LLM to consolidate a single memory component."""
    prompt = prompt_template.format(content=content)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    result = (response.choices[0].message.content or "").strip()
    logger.debug("Dreaming: %s consolidated (%d -> %d chars)", label, len(content), len(result))
    return result


async def run_dream_cycle(client: AsyncOpenAI, model: str) -> None:
    """
    Run a full nightly consolidation pass over MEMORIES.txt and PULSE.txt.

    Skips any component that is empty or missing.  Each component is
    consolidated independently so a failure in one does not abort the other.
    """
    memories = prompt_assembler._read(prompt_assembler.MEMORIES_FILE)
    if memories:
        try:
            consolidated = await _consolidate(
                client, model, "MEMORIES.txt", _MEMORIES_PROMPT, memories
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
            consolidated = await _consolidate(
                client, model, "PULSE.txt", _PULSE_PROMPT, pulse
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
                 llm_client: AsyncOpenAI,
                 llm_model: str,
                 compaction_model: Optional[str] = None) -> None:
        self._cfg = config
        self._client = llm_client
        # Resolve model: dreaming config override > compaction_model > main model
        self._model = config.model or compaction_model or llm_model
        self._running = False

    async def run(self) -> None:
        self._running = True
        target = dt_time(self._cfg.hour, self._cfg.minute)
        logger.info("Dreaming loop started (target=%02d:%02d, model=%s)",
                     target.hour, target.minute, self._model)
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
        if not self._model:
            logger.error("Dreaming: no model configured, skipping")
            return
        logger.info("Dreaming: starting nightly consolidation (model=%s)", self._model)
        try:
            await run_dream_cycle(client=self._client, model=self._model)
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
