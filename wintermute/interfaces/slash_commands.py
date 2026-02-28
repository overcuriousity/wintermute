"""
Shared Slash-Command Handler

Dispatches user-typed /commands that are common to all interfaces (Matrix,
web).  Each interface calls ``dispatch()`` with the raw text and a
``send_fn`` callback; if the text matches a known command the handler runs
it and returns ``True``, otherwise returns ``False`` so the caller can
forward the message to the LLM.

Interface-specific commands (``/verify-session``, ``/kimi-auth``) remain
in their respective interface modules.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.core.llm_thread import LLMThread
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.infra.thread_config import ThreadConfigManager

logger = logging.getLogger(__name__)

# Type alias for the async send callback each interface provides.
SendFn = Callable[[str], Awaitable[None]]


class SlashCommandHandler:
    """Stateless dispatcher for slash commands shared across interfaces.

    Parameters
    ----------
    llm : LLMThread
        The main LLM thread (conversation history, compaction, etc.).
    sub_sessions : SubSessionManager | None
        Background sub-session manager (may be ``None`` at startup).
    thread_config_manager : ThreadConfigManager | None
        Per-thread configuration manager.
    dreaming_loop : object | None
        The DreamingLoop instance (optional).
    memory_harvest : object | None
        The MemoryHarvestLoop instance (optional).
    scheduler : object | None
        The SchedulerThread instance (optional).
    reflection_loop : object | None
        The ReflectionLoop instance (optional).
    self_model : object | None
        The SelfModelProfiler instance (optional).
    update_checker : object | None
        The UpdateChecker instance (optional).
    """

    def __init__(
        self,
        llm: LLMThread,
        sub_sessions: Optional[SubSessionManager] = None,
        thread_config_manager: Optional[ThreadConfigManager] = None,
        dreaming_loop: object = None,
        memory_harvest: object = None,
        scheduler: object = None,
        reflection_loop: object = None,
        self_model: object = None,
        update_checker: object = None,
    ) -> None:
        self._llm = llm
        self._sub_sessions = sub_sessions
        self._thread_config_manager = thread_config_manager
        self._dreaming_loop = dreaming_loop
        self._memory_harvest = memory_harvest
        self._scheduler = scheduler
        self._reflection_loop = reflection_loop
        self._self_model = self_model
        self._update_checker = update_checker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(self, text: str, thread_id: str, send_fn: SendFn) -> bool:
        """Try to handle *text* as a slash command.

        Returns ``True`` if the command was handled, ``False`` if the text
        is not a recognized command and should be forwarded to the LLM.
        """
        if not text.startswith("/"):
            return False

        if text == "/new":
            await self._cmd_new(thread_id, send_fn)
            return True
        if text == "/compact":
            await self._cmd_compact(thread_id, send_fn)
            return True
        if text == "/tasks":
            await self._cmd_tasks(send_fn)
            return True
        if text == "/status":
            await self._cmd_status(thread_id, send_fn)
            return True
        if text == "/dream":
            await self._cmd_dream(thread_id, send_fn)
            return True
        if text == "/reflect":
            await self._cmd_reflect(thread_id, send_fn)
            return True
        if text == "/rebuild-index":
            await self._cmd_rebuild_index(send_fn)
            return True
        if text == "/memory-stats":
            await self._cmd_memory_stats(send_fn)
            return True
        if text.startswith("/config"):
            await self._cmd_config(text, thread_id, send_fn)
            return True
        if text == "/commands":
            await self._cmd_commands(send_fn)
            return True

        return False

    # ------------------------------------------------------------------
    # Individual command implementations
    # ------------------------------------------------------------------

    async def _cmd_new(self, thread_id: str, send_fn: SendFn) -> None:
        await self._llm.reset_session(thread_id)
        await send_fn("Session reset. Starting fresh conversation.")
        from wintermute.infra import prompt_loader
        try:
            seed_prompt = prompt_loader.load_seed(self._llm.seed_language)
            await self._llm.enqueue_system_event(seed_prompt, thread_id)
        except Exception:  # noqa: BLE001
            logger.exception("Seed after /new failed (non-fatal)")

    async def _cmd_compact(self, thread_id: str, send_fn: SendFn) -> None:
        before = self._llm.get_token_budget(thread_id)
        await self._llm.force_compact(thread_id)
        after = self._llm.get_token_budget(thread_id)
        await send_fn(
            f"Context compacted.\n"
            f"Before: {before['total_used']} tokens ({before['msg_count']} msgs, {before['pct']}%)\n"
            f"After: {after['total_used']} tokens ({after['msg_count']} msgs, {after['pct']}%)"
        )

    async def _cmd_tasks(self, send_fn: SendFn) -> None:
        from wintermute import tools as tool_module
        result = tool_module.execute_tool("task", {"action": "list"})
        await send_fn(f"Tasks:\n```json\n{result}\n```")

    async def _cmd_status(self, thread_id: str, send_fn: SendFn) -> None:
        from wintermute.infra import database as db, prompt_assembler

        lines = ["**Wintermute Status**"]

        # --- LLM backends ---
        lines.append("\n**LLM Backends**")

        def _fmt_pool(label, pool):
            if not pool or not pool.enabled:
                return f"{label}: disabled"
            chain = " → ".join(f"`{b[0].model}`" for b in pool._backends)
            ctx = pool.primary.context_size // 1024
            return f"{label}: {chain} ({ctx}k ctx)"

        lines.append(_fmt_pool("Main", self._llm.main_pool))
        lines.append(_fmt_pool("Compaction", self._llm.compaction_pool))
        lines.append(_fmt_pool("Turing Protocol", self._llm.turing_protocol_pool))
        lines.append(_fmt_pool("NL Translation", self._llm.nl_translation_pool))
        if self._sub_sessions:
            lines.append(_fmt_pool("Sub-sessions", self._sub_sessions._pool))
        if self._dreaming_loop:
            lines.append(_fmt_pool("Dreaming", self._dreaming_loop._pool))

        # --- Context budget for current thread ---
        try:
            budget = self._llm.get_token_budget(thread_id)
            lines.append(
                f"\n**Context** (thread: `{thread_id}`)\n"
                f"{budget['total_used']:,} / {budget['total_limit']:,} tokens"
                f" ({budget['pct']}%) — {budget['msg_count']} messages"
            )
        except Exception:  # noqa: BLE001
            pass

        # Queue depth
        qsize = self._llm.queue_size
        if qsize:
            lines.append(f"Queue: {qsize} item(s) pending")

        # --- Memory & knowledge ---
        lines.append("\n**Memory & Knowledge**")

        def _read_memory_and_skills():
            mem = prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or ""
            skill_paths = sorted(prompt_assembler.SKILLS_DIR.glob("*.md")) if prompt_assembler.SKILLS_DIR.exists() else []
            return mem, skill_paths

        mem_text, skill_paths = await asyncio.to_thread(_read_memory_and_skills)
        mem_lines = mem_text.count("\n") + (1 if mem_text.strip() else 0)
        lines.append(f"MEMORIES.txt: {mem_lines} lines ({len(mem_text):,} chars)")
        if skill_paths:
            lines.append(f"Skills ({len(skill_paths)}):")
            for p in skill_paths:
                lines.append(f"- {p.stem}")
        else:
            lines.append("Skills: none")

        # Active tasks
        try:
            task_items = await db.async_call(db.list_tasks, "active")
            if task_items:
                lines.append(f"Tasks ({len(task_items)} active):")
                for item in task_items:
                    content = (item["content"] or "")[:80]
                    prio = item.get("priority", "?")
                    sched = f" [{item['schedule_desc']}]" if item.get("schedule_desc") else ""
                    lines.append(f"- [P{prio}] #{item['id']}: {content}{sched}")
            else:
                lines.append("Tasks: none")
        except Exception:  # noqa: BLE001
            pass

        # --- Background loops ---
        lines.append("\n**Background Loops**")
        if self._dreaming_loop:
            state = "running" if self._dreaming_loop._running else "stopped"
            dl_cfg = self._dreaming_loop._cfg
            lines.append(
                f"Dreaming: {state} (nightly at {dl_cfg.hour:02d}:{dl_cfg.minute:02d} UTC,"
                f" model: `{self._dreaming_loop._pool.primary.model}`)"
            )
        if self._memory_harvest:
            mh = self._memory_harvest
            state = "running" if getattr(mh, "_running", False) else "stopped"
            threshold = getattr(getattr(mh, "_cfg", None), "message_threshold", "?")
            pending = sum(mh._msg_counts.values()) if hasattr(mh, "_msg_counts") else 0
            in_flight = len(mh._in_flight) if hasattr(mh, "_in_flight") else 0
            extra = f" (threshold: {threshold} msgs"
            if pending:
                extra += f", {pending} pending"
            if in_flight:
                extra += f", {in_flight} in-flight"
            extra += ")"
            lines.append(f"Memory harvest: {state}{extra}")
        if self._scheduler:
            jobs = self._scheduler.list_jobs()
            if jobs:
                lines.append(f"Scheduler jobs ({len(jobs)} active):")
                for j in jobs:
                    lines.append(f"- {j.get('id', '?')}: next {j.get('next_run', '?')}")
            else:
                lines.append("Scheduler jobs: none")

        # --- Reflection loop ---
        if self._reflection_loop:
            rl = self._reflection_loop
            state = "running" if getattr(rl, "_running", False) else "stopped"
            cfg = rl._cfg
            pending = getattr(rl, "_completed_count", 0)
            lines.append(
                f"Reflection: {state} (batch every {cfg.batch_threshold} completions,"
                f" {pending}/{cfg.batch_threshold} pending, failure_limit={cfg.consecutive_failure_limit})"
            )

        # --- Self-model ---
        if self._self_model:
            sm = self._self_model
            summary = sm.get_summary()
            last_updated = sm._state.get("last_updated")
            last_changes = sm._state.get("last_tuning_changes", [])
            ts_str = ""
            if last_updated:
                from datetime import datetime as _dt, timezone as _tz
                ts_str = " (updated " + _dt.fromtimestamp(last_updated, tz=_tz.utc).strftime("%Y-%m-%d %H:%M UTC") + ")"
            lines.append(f"\n**Self-Model**{ts_str}")
            if summary:
                lines.append(summary)
            else:
                lines.append("No summary yet (runs with next reflection cycle)")
            if last_changes:
                lines.append("Last tuning: " + "; ".join(last_changes))

        # --- Update checker ---
        if self._update_checker:
            uc = self._update_checker
            state = "running" if uc._running else "stopped"
            lines.append("\n**Updates**")
            lines.append(f"Update checker: {state} (every {uc._config.interval_hours}h)")
            try:
                msg = await uc.check()
                lines.append(f"Status: {msg}" if msg else "Status: up-to-date")
            except Exception:
                logger.debug("Update check in /status failed", exc_info=True)
                cached = uc.last_result
                if cached:
                    lines.append(f"Status: {cached} (cached)")
                else:
                    lines.append("Status: check failed")

        # --- Sub-sessions ---
        lines.append("\n**Sub-sessions**")
        if self._sub_sessions:
            active = self._sub_sessions.list_active()
            if active:
                lines.append(f"{len(active)} running:")
                for s in active:
                    lines.append(f"- `{s['session_id']}` [{s['status']}] {s['objective'][:80]}")
            else:
                lines.append("None active")
            workflows = self._sub_sessions.list_workflows()
            running_wfs = [w for w in workflows if w["status"] == "running"]
            if running_wfs:
                lines.append(f"\n{len(running_wfs)} active workflow(s):")
                for w in running_wfs:
                    nodes_summary = ", ".join(
                        f"{n['node_id']}[{n['status']}]" for n in w["nodes"]
                    )
                    lines.append(f"- `{w['workflow_id']}`: {nodes_summary}")
        else:
            lines.append("Not available")

        await send_fn("\n".join(lines))

    async def _cmd_dream(self, thread_id: str, send_fn: SendFn) -> None:
        from wintermute.workers import dreaming
        from wintermute.infra import prompt_assembler

        if not self._dreaming_loop:
            await send_fn("Dreaming loop not available.")
            return

        dl = self._dreaming_loop
        from wintermute.infra import database as db

        def _snapshot_memory_skills():
            mem_len = len(prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or "")
            skill_files = sorted(prompt_assembler.SKILLS_DIR.glob("*.md")) if prompt_assembler.SKILLS_DIR.exists() else []
            skills_size = sum(f.stat().st_size for f in skill_files)
            return mem_len, len(skill_files), skills_size

        mem_before, skills_count_before, skills_size_before = await asyncio.to_thread(_snapshot_memory_skills)
        tasks_before = len(await db.async_call(db.list_tasks, "active"))

        await send_fn("Starting dream cycle...")
        try:
            report = await dreaming.run_dream_cycle(pool=dl._pool)
        except Exception as exc:
            await send_fn(f"Dream cycle failed: {exc}")
            return

        mem_after, skills_count_after, skills_size_after = await asyncio.to_thread(_snapshot_memory_skills)
        tasks_after = len(await db.async_call(db.list_tasks, "active"))

        # Build phase summary from DreamReport.
        phase_lines = []
        for r in report.results:
            status = "\u2713" if not r.error else "\u2717"
            phase_lines.append(f"  {status} {r.phase_name}: {r.summary}")
        phases_text = "\n".join(phase_lines) if phase_lines else "  (no phases ran)"
        errors_text = f"\nErrors: {', '.join(report.errors)}" if report.errors else ""

        await send_fn(
            f"Dream cycle complete ({len(report.phases_run)} phases).\n"
            f"MEMORIES.txt: {mem_before} -> {mem_after} chars\n"
            f"Tasks: {tasks_before} -> {tasks_after} active\n"
            f"Skills: {skills_count_before} -> {skills_count_after} files, "
            f"{skills_size_before} -> {skills_size_after} bytes\n"
            f"Phases:\n{phases_text}{errors_text}"
        )

    async def _cmd_reflect(self, thread_id: str, send_fn: SendFn) -> None:
        if not self._reflection_loop:
            await send_fn("Reflection loop not available.")
            return

        rl = self._reflection_loop
        if not rl._cfg.enabled:
            await send_fn("Reflection loop is disabled by config.")
            return

        await send_fn("Running reflection cycle...")
        try:
            findings = await rl._run_rules()
            if findings and rl._pool and rl._pool.enabled:
                await rl._run_analysis(findings)
            if rl._self_model:
                await rl._self_model.update(findings)
            rl._checked_failures.clear()
        except Exception as exc:
            await send_fn(f"Reflection cycle failed: {exc}")
            return

        lines = [f"Reflection cycle complete. {len(findings)} finding(s)."]
        for f in findings:
            action = f" → {f.action_taken}" if f.action_taken else ""
            lines.append(f"- [{f.severity.upper()}] {f.rule}: {f.detail[:120]}{action}")

        if self._self_model:
            sm_summary = self._self_model.get_summary()
            if sm_summary:
                lines.append(f"\n**Self-Assessment updated:**\n{sm_summary}")
            tuning = self._self_model._state.get("last_tuning_changes", [])
            if tuning:
                lines.append("Tuning changes: " + "; ".join(tuning))

        await send_fn("\n".join(lines))

    async def _cmd_rebuild_index(self, send_fn: SendFn) -> None:
        from wintermute.infra import database as db, memory_store
        if not memory_store.is_vector_enabled():
            await send_fn("Vector memory is not enabled (backend: flat_file).")
            return
        await send_fn("Rebuilding memory index...")
        try:
            await db.async_call(memory_store.rebuild)
            st = await db.async_call(memory_store.stats)
            await send_fn(
                f"Memory index rebuilt.\n```json\n{_json.dumps(st, indent=2)}\n```"
            )
        except Exception as exc:
            await send_fn(f"Rebuild failed: {exc}")

    async def _cmd_memory_stats(self, send_fn: SendFn) -> None:
        from wintermute.infra import database as db, memory_store
        try:
            st = await db.async_call(memory_store.stats)
            await send_fn(
                f"**Memory Store**\n```json\n{_json.dumps(st, indent=2)}\n```"
            )
        except Exception as exc:
            await send_fn(f"Failed to get memory stats: {exc}")

    async def _cmd_config(self, text: str, thread_id: str, send_fn: SendFn) -> None:
        mgr = self._thread_config_manager
        if mgr is None:
            await send_fn("Thread configuration is not available.")
            return

        parts = text.strip().split(None, 2)  # ["/config", key?, value?]

        # /config — show resolved config with sources
        if len(parts) == 1:
            resolved = mgr.resolve_as_dict(thread_id)
            lines = [f"**Configuration** (thread: `{thread_id}`)\n"]
            for key, info in resolved.items():
                src_tag = f" _(from {info['source']})_" if info["source"] != "default" else ""
                lines.append(f"- `{key}`: **{info['value']}**{src_tag}")
            backends = mgr.get_available_backends()
            if backends:
                lines.append(f"\nAvailable backends: {', '.join(f'`{b}`' for b in sorted(backends))}")
            await send_fn("\n".join(lines))
            return

        # /config reset [<key>]
        if parts[1] == "reset":
            if len(parts) == 3:
                key = parts[2]
                current = mgr.get(thread_id)
                if current is None or getattr(current, key, None) is None:
                    await send_fn(f"No override set for `{key}` on this thread.")
                    return
                try:
                    mgr.set(thread_id, key, None)
                    resolved = mgr.resolve(thread_id)
                    new_val = getattr(resolved, key, "?")
                    await send_fn(
                        f"Override for `{key}` removed. Effective value: **{new_val}**"
                    )
                except (ValueError, AttributeError) as exc:
                    await send_fn(f"Error: {exc}")
            else:
                mgr.reset(thread_id)
                await send_fn("All per-thread overrides removed.")
            return

        # /config <key> <value>
        if len(parts) < 3:
            await send_fn(
                "Usage: `/config <key> <value>` or `/config reset [<key>]`\n"
                "Keys: `backend_name`, `session_timeout_minutes`, `sub_sessions_enabled`, `system_prompt_mode`"
            )
            return

        key, value = parts[1], parts[2]
        try:
            mgr.set(thread_id, key, value)
            resolved = mgr.resolve(thread_id)
            new_val = getattr(resolved, key, "?")
            await send_fn(f"`{key}` set to **{new_val}** for this thread.")
        except (ValueError, TypeError) as exc:
            await send_fn(f"Error: {exc}")

    async def _cmd_commands(self, send_fn: SendFn) -> None:
        await send_fn(
            "**Wintermute — Slash Commands**\n\n"
            "**Conversation**\n"
            "- `/new` — Wipe history and start a fresh session (also cancels running sub-sessions)\n"
            "- `/compact` — Force context compaction now; shows before/after token counts\n\n"
            "**Autonomy**\n"
            "- `/tasks` — List all active tasks\n"
            "- `/dream` — Run a dream cycle (memory consolidation + task pruning)\n"
            "- `/reflect` — Trigger a reflection cycle; shows findings and self-model update\n\n"
            "**Memory**\n"
            "- `/memory-stats` — Show memory store backend, entry count, and status\n"
            "- `/rebuild-index` — Rebuild the vector memory index from MEMORIES.txt\n\n"
            "**Configuration**\n"
            "- `/config` — Show current resolved config for this thread\n"
            "- `/config <key> <value>` — Set a per-thread override (keys: backend_name, session_timeout_minutes, sub_sessions_enabled, system_prompt_mode)\n"
            "- `/config reset` — Remove all per-thread overrides\n"
            "- `/config reset <key>` — Remove a single per-thread override\n\n"
            "**System**\n"
            "- `/status` — Show runtime status: models, token budget, memory, loops, sub-sessions\n"
            "- `/commands` — Show this list\n\n"
            "**Interface-Specific** *(may not be available in all clients)*\n"
            "- `/verify-session` — Matrix: verify your encryption session with the bot\n"
            "- `/kimi-auth` — Matrix: authenticate Kimi integration for this account"
        )
