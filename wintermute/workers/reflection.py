"""
Reflection Cycle — Feedback Loop for Autonomous Self-Improvement

Three-tier architecture (optimized for token poverty):

1. Rule engine (zero LLM cost) — programmatic pattern detection on DB/event
   data.  Auto-applies simple actions (pause failing tasks, flag stale skills).
   Runs on every trigger event.

2. LLM analysis (cheap, one-shot) — direct pool.call() with a prose prompt
   summarising recent findings.  Only runs when the rule engine finds something
   interesting.

3. Sub-session mutations (expensive, rare) — spawned only when the LLM
   analysis recommends a creative change (e.g. rewriting a skill).

Triggers (event-driven, no polling):
  - sub_session.failed  → immediate rule-engine check
  - sub_session.completed count threshold → batch analysis

All findings are logged to interaction_log (action="reflection_rule" /
"reflection_analysis") and emitted as events for the /debug SSE stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time as _time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from wintermute.infra import database

if TYPE_CHECKING:
    from wintermute.core.types import BackendPool
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.infra.event_bus import EventBus

logger = logging.getLogger(__name__)

# Hardcoded fallback prompt used when REFLECTION_ANALYSIS.txt is missing.
_DEFAULT_ANALYSIS_PROMPT = """\
You are reviewing the recent operational history of an AI assistant system.

## Rule Engine Findings
{findings}

## Recent Failed Sub-Sessions
{failed_sessions}

## Active Tasks
{active_tasks}

Based on this data, provide brief observations:
1. What patterns do you see in the failures?
2. Are any tasks misconfigured or need adjustment?
3. Should any skills be updated or retired?

After your observations, end your response with this JSON block on its own line:
{{"skill_actions": []}}

If you recommend a skill change, add a brief English instruction per action:
{{"skill_actions": ["Review and update data/skills/example.md - correlates with 3 failures"]}}\
"""


def _extract_skill_actions(text: str) -> list[str]:
    """Extract skill_actions from a JSON block appended to the LLM response.

    Looks for the last ``{"skill_actions": [...]}`` object in the text.
    Returns an empty list if none is found or the JSON is malformed.
    Language-neutral: works regardless of the prose language.
    """
    # Match the last JSON object containing skill_actions anywhere in the text.
    # The block may appear at the end with preceding whitespace/newlines.
    matches = re.findall(r'\{[^{}]*"skill_actions"\s*:\s*\[[^\]]*\][^{}]*\}', text)
    if not matches:
        return []
    try:
        parsed = json.loads(matches[-1])
        actions = parsed.get("skill_actions", [])
        if isinstance(actions, list):
            return [str(a) for a in actions if a]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


@dataclass
class ReflectionConfig:
    enabled: bool = True
    batch_threshold: int = 10            # trigger batch analysis every N completions
    consecutive_failure_limit: int = 3   # auto-pause after N consecutive failures
    lookback_seconds: int = 86400        # 24h window for pattern detection
    min_result_length: int = 50          # below this = "no meaningful output" (stale check)
    main_turn_batch_threshold: int = 15  # trigger reflection after N main-thread turns
    synthesis_min_cluster_size: int = 3  # min sessions per tool-set cluster
    synthesis_min_outcomes: int = 20     # min completed outcomes before synthesis runs


@dataclass
class ReflectionFinding:
    rule: str              # "consecutive_failures" | "timeout_pattern" | "stale_task" | "skill_correlation"
    severity: str          # "warning" | "action_taken"
    subject_type: str      # "task" | "skill" | "sub_session"
    subject_id: str        # task_id, skill filename, or session_id
    detail: str            # human-readable description
    action_taken: str = "" # "" if warning-only, or "paused_task" etc.


class ReflectionLoop:
    """Asyncio task that closes the observe→reflect→adapt feedback loop."""

    def __init__(
        self,
        config: ReflectionConfig,
        sub_session_manager: "Optional[SubSessionManager]" = None,
        pool: "Optional[BackendPool]" = None,
        event_bus: "Optional[EventBus]" = None,
    ) -> None:
        self._cfg = config
        self._sub_sessions = sub_session_manager
        self._pool = pool
        self._event_bus = event_bus
        self._running = False
        self._completed_count = 0
        self._check_event = asyncio.Event()
        self._event_bus_subs: list[str] = []
        # Session IDs that have already triggered an immediate rule check so
        # we don't double-fire when the same failure is picked up by the batch.
        self._checked_failures: set[str] = set()
        self._self_model: object | None = None
        self._main_turn_count: int = 0
        # Timestamp of last synthesis run (epoch seconds). 0.0 is a sentinel
        # meaning "never run", intentionally allowing the first synthesis after
        # construction (or after a loop restart) to bypass the cooldown check.
        self._last_synthesis_ts: float = 0.0

    def inject_self_model(self, profiler) -> None:
        """Set the SelfModelProfiler instance (called once at startup)."""
        self._self_model = profiler

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        if not self._cfg.enabled:
            logger.info("[reflection] Reflection loop disabled by config")
            return

        self._running = True

        if self._event_bus:
            sub_id = self._event_bus.subscribe(
                "sub_session.failed", self._on_sub_session_failed
            )
            self._event_bus_subs.append(sub_id)
            sub_id = self._event_bus.subscribe(
                "sub_session.completed", self._on_sub_session_completed
            )
            self._event_bus_subs.append(sub_id)
            sub_id = self._event_bus.subscribe(
                "main_thread.turn_completed", self._on_main_turn_completed
            )
            self._event_bus_subs.append(sub_id)

        logger.info(
            "[reflection] Reflection loop started (batch_threshold=%d, "
            "failure_limit=%d, lookback=%ds)",
            self._cfg.batch_threshold,
            self._cfg.consecutive_failure_limit,
            self._cfg.lookback_seconds,
        )

        # Fallback poll: 6h.  Events trigger checks much sooner.
        fallback_poll = 6 * 3600

        while self._running:
            try:
                await asyncio.wait_for(self._check_event.wait(), timeout=fallback_poll)
            except asyncio.TimeoutError:
                pass
            self._check_event.clear()
            if not self._running:
                break
            try:
                await self._cycle()
            except Exception:  # noqa: BLE001
                logger.exception("[reflection] Error during reflection cycle")

    def stop(self) -> None:
        self._running = False
        self._check_event.set()
        if self._event_bus:
            for sub_id in self._event_bus_subs:
                self._event_bus.unsubscribe(sub_id)
            self._event_bus_subs.clear()

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    async def _on_sub_session_failed(self, event) -> None:
        session_id = event.data.get("session_id", "")
        self._checked_failures.add(session_id)
        self._check_event.set()

    async def _on_sub_session_completed(self, event) -> None:
        self._completed_count += 1
        if self._completed_count >= self._cfg.batch_threshold:
            self._completed_count = 0
            self._check_event.set()

    async def _on_main_turn_completed(self, event) -> None:
        self._main_turn_count += 1
        if self._main_turn_count >= self._cfg.main_turn_batch_threshold:
            self._main_turn_count = 0
            self._check_event.set()

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def _cycle(self) -> None:
        findings = await self._run_rules()
        if findings and self._pool and self._pool.enabled:
            await self._run_analysis(findings)
        if self._pool and self._pool.enabled and self._sub_sessions:
            if _time.time() - self._last_synthesis_ts >= 86400:
                await self._run_synthesis()
        if self._self_model:
            await self._self_model.update(findings)
        # Flush accumulated skill read counts to YAML.
        try:
            from wintermute.workers import skill_stats
            skill_stats.flush()
        except Exception:
            logger.debug("[reflection] Failed to flush skill_stats", exc_info=True)
        self._checked_failures.clear()

    # ------------------------------------------------------------------
    # Tier 1: Rule engine
    # ------------------------------------------------------------------

    async def _run_rules(self) -> list[ReflectionFinding]:
        since = _time.time() - self._cfg.lookback_seconds
        try:
            outcomes = await database.async_call(
                database.get_outcomes_since, since, limit=500
            )
            active_tasks = await database.async_call(
                database.list_tasks, "active"
            )
        except Exception:
            logger.exception("[reflection] Rule engine: failed to query DB")
            return []

        findings: list[ReflectionFinding] = []

        # Rule 1: Consecutive failures for scheduled tasks.
        tasks_with_task_id = {
            o["task_id"]
            for o in outcomes
            if o.get("task_id")
        }
        for task_id in tasks_with_task_id:
            try:
                streak = await database.async_call(
                    database.get_task_failure_streak, task_id
                )
            except Exception:
                continue
            if streak >= self._cfg.consecutive_failure_limit:
                # Find the task to check its status.
                task = await database.async_call(database.get_task, task_id)
                if task and task.get("status") == "active":
                    # Auto-pause the task.
                    paused = await database.async_call(database.pause_task, task_id)
                    action = "paused_task" if paused else "already_paused"
                    finding = ReflectionFinding(
                        rule="consecutive_failures",
                        severity="action_taken",
                        subject_type="task",
                        subject_id=task_id,
                        detail=(
                            f"Task {task_id!r} ({task.get('content', '')[:80]}) "
                            f"failed {streak} consecutive times — auto-paused."
                        ),
                        action_taken=action,
                    )
                    findings.append(finding)
                    if paused:
                        logger.warning(
                            "[reflection] Auto-paused task %s after %d consecutive failures",
                            task_id, streak,
                        )
                        if self._event_bus:
                            self._event_bus.emit(
                                "reflection.finding",
                                rule="consecutive_failures",
                                severity="action_taken",
                                subject_id=task_id,
                                action_taken=action,
                            )

        # Rule 2: Timeout pattern (3+ consecutive timeouts).
        # Group outcomes by task_id once instead of re-querying per task.
        outcomes_by_task: dict[str, list] = {}
        for o in outcomes:
            tid = o.get("task_id")
            if tid:
                outcomes_by_task.setdefault(tid, []).append(o)

        for task_id in tasks_with_task_id:
            task = await database.async_call(database.get_task, task_id)
            if not task or task.get("status") != "active":
                continue
            task_outcomes = outcomes_by_task.get(task_id, [])
            if len(task_outcomes) >= 3:
                last_three = task_outcomes[:3]
                if all(o["status"] == "timeout" for o in last_three):
                    finding = ReflectionFinding(
                        rule="timeout_pattern",
                        severity="warning",
                        subject_type="task",
                        subject_id=task_id,
                        detail=(
                            f"Task {task_id!r} timed out 3+ consecutive times. "
                            f"Consider a longer timeout or simpler ai_prompt."
                        ),
                    )
                    findings.append(finding)
                    if self._event_bus:
                        self._event_bus.emit(
                            "reflection.finding",
                            rule="timeout_pattern",
                            severity="warning",
                            subject_id=task_id,
                            action_taken="",
                        )

        # Rule 3: Stale task (runs produced no meaningful output).
        for task in active_tasks:
            task_id = task.get("id", "")
            run_count = task.get("run_count") or 0
            last_summary = task.get("last_result_summary") or ""
            if (
                run_count > 0
                and task.get("ai_prompt")
                and len(last_summary) < self._cfg.min_result_length
            ):
                finding = ReflectionFinding(
                    rule="stale_task",
                    severity="warning",
                    subject_type="task",
                    subject_id=task_id,
                    detail=(
                        f"Task {task_id!r} has {run_count} run(s) but produced "
                        f"very short output ({len(last_summary)} chars). "
                        f"May need a better ai_prompt."
                    ),
                )
                findings.append(finding)
                if self._event_bus:
                    self._event_bus.emit(
                        "reflection.finding",
                        rule="stale_task",
                        severity="warning",
                        subject_id=task_id,
                        action_taken="",
                    )

        # Rule 4: Skill failure correlation.
        failed_outcomes = [o for o in outcomes if o["status"] in ("failed", "timeout")]
        failed_session_ids = {o["session_id"] for o in failed_outcomes}

        if failed_session_ids:
            # Query interaction_log for read_file calls in failed sessions.
            skill_fail_counts: dict[str, int] = {}
            try:
                log_entries = await database.async_call(
                    database.get_interaction_log,
                    limit=1000,
                    action_filter="tool_call",
                )
            except Exception:
                log_entries = []

            for entry in log_entries:
                session = entry.get("session", "")
                if session not in failed_session_ids:
                    continue
                raw = entry.get("input", "")
                # Look for read_file calls on data/skills/ paths.
                if "read_file" in raw and "data/skills/" in raw:
                    # Extract skill filename from the input.
                    # Capture stem only (no .md extension) for consistent key lookup.
                    matches = re.findall(r'data/skills/([^\s"\']+)\.md', raw)
                    for skill_name in matches:
                        skill_fail_counts[skill_name] = skill_fail_counts.get(skill_name, 0) + 1

            # Enrich with lifetime stats from skill_stats.
            try:
                from wintermute.workers import skill_stats
                all_stats = skill_stats.get_all()
            except Exception:
                all_stats = {}

            for skill_name, fail_count in skill_fail_counts.items():
                if fail_count >= 3:
                    # Build enrichment suffix from lifetime stats.
                    extra = ""
                    sstat = all_stats.get(skill_name, {})
                    if sstat:
                        total = sstat.get("sessions_loaded", 0)
                        failures = sstat.get("failure_count", 0)
                        rate = round(failures / total * 100) if total else 0
                        extra = f" Lifetime: {total} sessions, {rate}% failure rate."
                    finding = ReflectionFinding(
                        rule="skill_failure_correlation",
                        severity="warning",
                        subject_type="skill",
                        subject_id=skill_name,
                        detail=(
                            f"Skill '{skill_name}' was loaded in {fail_count} failed "
                            f"sub-sessions within the lookback window."
                            f"{extra} Consider reviewing or updating this skill."
                        ),
                    )
                    findings.append(finding)
                    if self._event_bus:
                        self._event_bus.emit(
                            "reflection.skill_flagged",
                            skill_name=skill_name,
                            failure_count=fail_count,
                        )

        # Log all findings to interaction_log.
        for finding in findings:
            try:
                await database.async_call(
                    database.save_interaction_log,
                    _time.time(),
                    "reflection_rule",
                    "system:reflection",
                    "rule_engine",
                    finding.rule,
                    json.dumps({
                        "severity": finding.severity,
                        "subject_type": finding.subject_type,
                        "subject_id": finding.subject_id,
                        "detail": finding.detail,
                        "action_taken": finding.action_taken,
                    }),
                    "ok",
                )
            except Exception:
                logger.debug("[reflection] Failed to log finding", exc_info=True)

        if findings:
            logger.info(
                "[reflection] Rule engine: %d finding(s) — %s",
                len(findings),
                ", ".join(f.rule for f in findings),
            )

        return findings

    # ------------------------------------------------------------------
    # Tier 2: LLM analysis
    # ------------------------------------------------------------------

    async def _run_analysis(self, findings: list[ReflectionFinding]) -> None:
        """Run one-shot LLM analysis on the rule engine findings."""
        since = _time.time() - self._cfg.lookback_seconds
        try:
            failed_outcomes = await database.async_call(
                database.get_outcomes_since, since, status_filter="failed", limit=20
            )
            active_tasks = await database.async_call(database.list_tasks, "active")
        except Exception:
            logger.exception("[reflection] LLM analysis: failed to query DB")
            return

        # Build compact summaries for the prompt.
        findings_text = "\n".join(
            f"- [{f.severity.upper()}] {f.rule}: {f.detail}"
            + (f" → {f.action_taken}" if f.action_taken else "")
            for f in findings
        )

        failed_text = "\n".join(
            f"- {o['session_id']}: {o['objective'][:120]} "
            f"[{o['status']}, {o.get('duration_seconds', '?')}s]"
            for o in failed_outcomes[:10]
        ) or "(none)"

        tasks_text = "\n".join(
            f"- {t['id']}: {t['content'][:80]}"
            + (f" [runs={t.get('run_count', 0)}]" if t.get('run_count') else "")
            for t in active_tasks[:15]
        ) or "(none)"

        # Build skill stats summary for enrichment.
        skill_stats_text = "(unavailable)"
        try:
            from wintermute.workers import skill_stats
            all_stats = skill_stats.get_all()
            if all_stats:
                lines = []
                for sname, sdata in sorted(all_stats.items()):
                    total = sdata.get("sessions_loaded", 0)
                    fails = sdata.get("failure_count", 0)
                    rate = round(fails / total * 100) if total else 0
                    lines.append(
                        f"- {sname}: reads={sdata.get('read_count', 0)}, "
                        f"sessions={total}, fail_rate={rate}%, "
                        f"version={sdata.get('version', 1)}"
                    )
                skill_stats_text = "\n".join(lines)
            else:
                skill_stats_text = "(no skill stats yet)"
        except Exception:
            pass

        # Load prompt template, fall back to hardcoded.
        try:
            from wintermute.infra import prompt_loader
            prompt_text = prompt_loader.load(
                "REFLECTION_ANALYSIS.txt",
                findings=findings_text,
                failed_sessions=failed_text,
                active_tasks=tasks_text,
                skill_stats=skill_stats_text,
            )
        except FileNotFoundError:
            prompt_text = _DEFAULT_ANALYSIS_PROMPT.format(
                findings=findings_text,
                failed_sessions=failed_text,
                active_tasks=tasks_text,
            )
        except Exception:
            logger.exception("[reflection] Failed to load REFLECTION_ANALYSIS.txt")
            return

        try:
            response = await self._pool.call(
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens_override=512,
            )
        except Exception:
            logger.exception("[reflection] LLM analysis call failed")
            return

        if not response.choices:
            logger.warning("[reflection] LLM analysis returned empty choices")
            return

        analysis_text = response.choices[0].message.content or ""
        model_name = getattr(self._pool, "last_used", "unknown")

        logger.info("[reflection] LLM analysis complete (%d chars)", len(analysis_text))

        # Log to interaction_log.
        try:
            await database.async_call(
                database.save_interaction_log,
                _time.time(),
                "reflection_analysis",
                "system:reflection",
                model_name or "unknown",
                f"{len(findings)} finding(s)",
                analysis_text[:2000],
                "ok",
            )
        except Exception:
            logger.debug("[reflection] Failed to log analysis", exc_info=True)

        # Extract structured skill_actions from the JSON block at the end of
        # the response.  Language-neutral: the JSON structure is always the
        # same regardless of what language the prose observations are in.
        skill_actions = _extract_skill_actions(analysis_text)

        if self._event_bus:
            self._event_bus.emit(
                "reflection.analysis_completed",
                findings_count=len(findings),
                actions_recommended=len(skill_actions),
            )

        # Tier 3: spawn one mutation sub-session per recommended action
        # (capped at 1 per cycle to avoid runaway mutations).
        if skill_actions and self._sub_sessions:
            await self._spawn_mutation(skill_actions[0])

    # ------------------------------------------------------------------
    # Tier 3: Sub-session mutation
    # ------------------------------------------------------------------

    async def _spawn_mutation(self, objective: str) -> None:
        """Spawn a constrained sub-session to act on a SKILL_ACTION recommendation."""
        if self._sub_sessions is None:
            return
        logger.info("[reflection] Spawning skill mutation sub-session")

        # Enumerate existing skill files so the LLM doesn't waste rounds
        # trying to discover them (it has read_file but no directory listing).
        from pathlib import Path
        skills_dir = Path("data/skills")
        existing_files: list[str] = []
        if skills_dir.exists():
            existing_files = sorted(
                f.name for f in skills_dir.glob("*.md")
            )
        if existing_files:
            listing = "Existing skill files: " + ", ".join(existing_files)
        else:
            listing = "No skill files exist yet."

        try:
            session_id = self._sub_sessions.spawn(
                objective=(
                    "You are performing a skill maintenance task based on a "
                    "reflection cycle recommendation.\n\n"
                    + objective
                    + f"\n\n{listing}"
                    + "\n\nUse read_file to examine any relevant skill files before "
                    "making changes.  Use add_skill to create or update skills. "
                    "Keep changes minimal and justified."
                ),
                tool_names=["read_file", "add_skill", "append_memory"],
                system_prompt_mode="none",
                pool=self._pool,
                parent_thread_id=None,   # fire-and-forget
                skip_tp_on_exit=True,
                max_rounds=5,
            )
            logger.info("[reflection] Skill mutation sub-session spawned: %s", session_id)
        except Exception:
            logger.exception("[reflection] Failed to spawn mutation sub-session")

    # ------------------------------------------------------------------
    # Tier 4: Pattern-to-skill synthesis
    # ------------------------------------------------------------------

    async def _run_synthesis(self) -> None:
        """Detect recurring successful patterns and propose new skills."""
        since = _time.time() - self._cfg.lookback_seconds

        try:
            outcomes = await database.async_call(
                database.get_outcomes_since, since,
                status_filter="completed", limit=100,
            )
        except Exception:
            logger.debug("[reflection] Synthesis: failed to query outcomes", exc_info=True)
            return

        if len(outcomes) < self._cfg.synthesis_min_outcomes:
            return

        # Cluster by tool set.
        clusters: dict[frozenset[str], list[dict]] = {}
        for o in outcomes:
            raw_tools = o.get("tools_used") or "[]"
            try:
                tools = frozenset(json.loads(raw_tools))
            except (json.JSONDecodeError, TypeError):
                continue
            # Intentional: we ignore sessions that did not use any tools.
            # Synthesis here is focused on reusable tool workflows; tool-less
            # completions are not clustered. Revisit if we ever want to learn
            # from purely conversational or non-tool runs.
            if not tools:
                continue
            clusters.setdefault(tools, []).append(o)

        # Filter: keep clusters with enough sessions.
        min_size = self._cfg.synthesis_min_cluster_size
        viable = {k: v for k, v in clusters.items() if len(v) >= min_size}
        if not viable:
            return

        # Deduplicate: skip clusters whose tool set is already covered by a skill.
        from pathlib import Path
        skills_dir = Path("data/skills")
        existing_skill_texts: dict[str, str] = {}
        if skills_dir.exists():
            for sf in skills_dir.glob("*.md"):
                try:
                    existing_skill_texts[sf.stem] = sf.read_text(encoding="utf-8")
                except Exception:
                    pass

        novel_clusters: dict[frozenset[str], list[dict]] = {}
        for tool_set, sessions in viable.items():
            # Check if any existing skill mentions all tools in the set.
            # Use word-boundary matching to avoid false positives where a tool
            # name is a substring of another (e.g. "shell" matching "execute_shell").
            covered = False
            for _skill_name, skill_text in existing_skill_texts.items():
                if all(
                    re.search(r'\b' + re.escape(t) + r'\b', skill_text)
                    for t in tool_set
                ):
                    covered = True
                    break
            if not covered:
                novel_clusters[tool_set] = sessions

        if not novel_clusters:
            return

        # Build prompt for LLM synthesis.
        pattern_lines = []
        for i, (tool_set, sessions) in enumerate(list(novel_clusters.items())[:5], 1):
            objectives = [
                (s.get("objective") or "")[:100] for s in sessions[:5]
            ]
            pattern_lines.append(
                f"Pattern {i}: tools={sorted(tool_set)}, "
                f"{len(sessions)} sessions\n"
                f"  Example objectives:\n"
                + "\n".join(f"  - {obj}" for obj in objectives)
            )
        patterns_text = "\n\n".join(pattern_lines)

        try:
            from wintermute.infra import prompt_loader
            prompt_text = prompt_loader.load(
                "SKILL_SYNTHESIS.txt",
                patterns=patterns_text,
            )
        except FileNotFoundError:
            prompt_text = (
                "You are analyzing recurring successful patterns in an AI assistant's "
                "sub-session history. Propose reusable skills from these patterns.\n\n"
                f"{patterns_text}\n\n"
                'Respond with JSON: {"skill_proposals": [{"name": "...", "summary": "...", "procedure": "..."}]}\n'
                "Only propose skills that would genuinely save effort on future similar tasks."
            )
        except Exception:
            logger.exception("[reflection] Failed to load SKILL_SYNTHESIS.txt")
            return

        try:
            response = await self._pool.call(
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens_override=1024,
            )
        except Exception:
            logger.exception("[reflection] Synthesis LLM call failed")
            return

        if not response.choices:
            return

        text = response.choices[0].message.content or ""

        # Extract skill_proposals from JSON using brace-depth tracking to
        # correctly handle nested objects within the proposals array.
        proposals = []
        skill_key = '"skill_proposals"'
        idx = text.find(skill_key)
        if idx != -1:
            start = text.rfind("{", 0, idx)
            if start != -1:
                depth = 0
                end = None
                for pos in range(start, len(text)):
                    ch = text[pos]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = pos + 1
                            break
                if end is not None:
                    candidate = text[start:end]
                    try:
                        parsed = json.loads(candidate)
                        proposals = parsed.get("skill_proposals", [])
                    except (json.JSONDecodeError, ValueError):
                        pass

        if not proposals:
            logger.info("[reflection] Synthesis: no skill proposals returned")
            return

        self._last_synthesis_ts = _time.time()
        logger.info("[reflection] Synthesis: %d skill proposal(s)", len(proposals))

        # Log to interaction_log.
        try:
            await database.async_call(
                database.save_interaction_log,
                _time.time(),
                "reflection_synthesis",
                "system:reflection",
                "synthesis",
                f"{len(novel_clusters)} pattern(s)",
                text[:2000],
                "ok",
            )
        except Exception:
            pass

        if self._event_bus:
            self._event_bus.emit(
                "reflection.synthesis_completed",
                patterns_count=len(novel_clusters),
                proposals_count=len(proposals),
            )

        # Spawn one mutation per proposal (cap at 2 per cycle).
        for proposal in proposals[:2]:
            name = proposal.get("name", "unnamed")
            summary = proposal.get("summary", "").strip()
            procedure = proposal.get("procedure", "")
            if name and procedure:
                # Build objective string, omitting summary if it's empty/whitespace.
                objective = f"Create skill '{name}'"
                if summary:
                    objective += f": {summary}"
                objective += f"\n\nProcedure:\n{procedure}"
                await self._spawn_mutation(objective)
