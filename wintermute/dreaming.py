"""
Dreaming – Nightly Memory Consolidation

Called by the scheduler at a configurable hour to review and prune the
persistent memory components (MEMORIES.txt, PULSE.txt) without user
interaction.  Uses a direct API call (no tool loop, no thread history) so it
never interferes with ongoing conversations.
"""

import logging
from pathlib import Path

from openai import AsyncOpenAI

from wintermute import prompt_assembler

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

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
                client, model, "PULSE.txt", _PULSE_PROMPT, heartbeats
            )
            if consolidated:
                prompt_assembler.update_pulse(consolidated)
                logger.info("Dreaming: PULSE.txt updated (%d chars)", len(consolidated))
        except Exception:  # noqa: BLE001
            logger.exception("Dreaming: failed to consolidate PULSE.txt")
    else:
        logger.debug("Dreaming: PULSE.txt empty or missing, skipping")
