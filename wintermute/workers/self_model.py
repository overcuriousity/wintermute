"""
Self-Model — Operational Self-Awareness and Auto-Tuning

Aggregates metrics from sub_session_outcomes, interaction_log, and event bus
history to build an operational profile.  A cached LLM-generated prose summary
is injected into the main-thread system prompt so the assistant has awareness
of its own performance characteristics.

Auto-tuning adjusts internal parameters (sub-session timeout, memory harvest
threshold) within configured safe bounds based on observed patterns.

Not a standalone asyncio task — runs inside the reflection cycle via
``update()`` which is called after rule-engine findings.
"""

from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import yaml

from wintermute.infra import database
from wintermute.infra import data_versioning

if TYPE_CHECKING:
    from wintermute.core.types import BackendPool
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.infra.event_bus import EventBus
    from wintermute.workers.memory_harvest import MemoryHarvestLoop

logger = logging.getLogger(__name__)


@dataclass
class SelfModelConfig:
    enabled: bool = True
    yaml_path: str = "data/self_model.yaml"
    sub_session_timeout_range: tuple[int, int] = (120, 900)
    memory_harvest_threshold_range: tuple[int, int] = (5, 50)
    summary_max_chars: int = 300


class SelfModelProfiler:
    """Collects operational metrics, auto-tunes parameters, and caches a prose summary."""

    def __init__(
        self,
        config: SelfModelConfig,
        pool: Optional[BackendPool] = None,
        event_bus: Optional[EventBus] = None,
        sub_session_manager: Optional[SubSessionManager] = None,
        memory_harvest_loop: Optional[MemoryHarvestLoop] = None,
    ) -> None:
        self._cfg = config
        self._pool = pool
        self._event_bus = event_bus
        self._sub_sessions = sub_session_manager
        self._harvest = memory_harvest_loop
        self._yaml_path = Path(self._cfg.yaml_path)
        self._summary: str = ""
        self._state: dict = {}
        self._load()

    @property
    def yaml_path(self) -> Path:
        """Path to the persisted YAML metrics file."""
        return self._yaml_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load persisted state from YAML on init."""
        try:
            if self._yaml_path.exists():
                self._state = yaml.safe_load(self._yaml_path.read_text(encoding="utf-8")) or {}
                self._summary = self._state.get("summary", "")
                logger.info("[self_model] Loaded state from %s", self._yaml_path)
        except Exception:
            logger.debug("[self_model] Failed to load YAML state", exc_info=True)
            self._state = {}

    def _save(self) -> None:
        """Write state to YAML and auto-commit."""
        try:
            self._yaml_path.parent.mkdir(parents=True, exist_ok=True)
            self._yaml_path.write_text(
                yaml.dump(self._state, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
            data_versioning.commit_async("self_model: update")
        except Exception:
            logger.debug("[self_model] Failed to save YAML state", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, findings: list) -> None:
        """Called by reflection._cycle() after rule engine runs."""
        if not self._cfg.enabled:
            return
        try:
            metrics = await self._collect_metrics()
            # Pre-fetch recent harvest logs asynchronously to avoid blocking in _auto_tune.
            recent_harvests = None
            try:
                recent_harvests = await database.async_call(
                    database.get_interaction_log,
                    limit=5, action_filter="memory_harvest",
                )
            except Exception:
                pass
            changes = self._auto_tune(metrics, recent_harvests=recent_harvests)
            summary = await self._generate_summary(metrics, changes)
            self._summary = summary

            self._state["metrics"] = metrics
            self._state["summary"] = summary
            self._state["last_updated"] = _time.time()
            if changes:
                self._state["last_tuning_changes"] = changes
            self._save()

            logger.info(
                "[self_model] Updated — %d metric(s), %d tuning change(s)",
                len(metrics), len(changes),
            )
        except Exception:
            logger.exception("[self_model] Error during update")

    def get_summary(self) -> str:
        """Return the cached prose summary (no LLM call)."""
        return self._summary

    # ------------------------------------------------------------------
    # Metrics collection
    # ------------------------------------------------------------------

    async def _collect_metrics(self) -> dict:
        """Gather metrics from DB and event bus."""
        metrics: dict = {}

        # Sub-session outcome stats.
        try:
            stats = await database.async_call(database.get_outcome_stats)
            metrics["sub_session_total"] = stats.get("total", 0)
            metrics["sub_session_by_status"] = stats.get("by_status", {})
            metrics["avg_duration_seconds"] = stats.get("avg_duration_seconds")
            metrics["avg_tool_calls"] = stats.get("avg_tool_calls")
            metrics["timeout_rate_pct"] = stats.get("timeout_rate_pct", 0)
        except Exception:
            logger.debug("[self_model] Failed to collect outcome stats", exc_info=True)

        # Top tools used.
        try:
            tool_stats = await database.async_call(
                database.get_tool_usage_stats,
                _time.time() - 86400,
            )
            metrics["top_tools"] = tool_stats[:5]
        except Exception:
            logger.debug("[self_model] Failed to collect tool usage stats", exc_info=True)
            metrics["top_tools"] = []

        # Compaction and harvest counts from interaction_log.
        try:
            compaction_count = await database.async_call(
                database.count_interaction_log,
                action_filter="compaction",
            )
            harvest_count = await database.async_call(
                database.count_interaction_log,
                action_filter="memory_harvest",
            )
            metrics["compaction_count"] = compaction_count
            metrics["harvest_count"] = harvest_count
        except Exception:
            logger.debug("[self_model] Failed to collect log counts", exc_info=True)

        # Inference stats from interaction_log.
        try:
            inference_count = await database.async_call(
                database.count_interaction_log,
                action_filter="inference_completed",
            )
            metrics["inference_count"] = inference_count
        except Exception:
            logger.debug("[self_model] Failed to collect inference count", exc_info=True)

        # Skill stats.
        try:
            from wintermute.workers import skill_stats
            all_stats = skill_stats.get_all()
            metrics["skill_count"] = len(all_stats)
            metrics["skill_total_reads"] = sum(s.get("read_count", 0) for s in all_stats.values())
            metrics["skills_unused_90d"] = len(skill_stats.get_unused_skills(days=90))
        except Exception:
            logger.debug("[self_model] Failed to collect skill stats", exc_info=True)

        return metrics

    # ------------------------------------------------------------------
    # Auto-tuning
    # ------------------------------------------------------------------

    def _auto_tune(self, metrics: dict, recent_harvests: list | None = None) -> list[str]:
        """Apply heuristic tuning rules, return list of changes made.

        *recent_harvests* is an optional pre-fetched list of interaction_log
        entries (passed from the async caller to avoid blocking DB calls).
        """
        changes: list[str] = []
        timeout_rate = metrics.get("timeout_rate_pct", 0)
        avg_duration = metrics.get("avg_duration_seconds")
        lo_timeout, hi_timeout = self._cfg.sub_session_timeout_range
        lo_harvest, hi_harvest = self._cfg.memory_harvest_threshold_range

        # --- Sub-session timeout tuning ---
        if self._sub_sessions is not None:
            current_timeout = self._sub_sessions.default_timeout

            if timeout_rate > 30 and current_timeout < hi_timeout:
                new_timeout = min(current_timeout + 60, hi_timeout)
                self._sub_sessions.default_timeout = new_timeout
                changes.append(f"Increased sub-session timeout {current_timeout}s → {new_timeout}s (timeout_rate={timeout_rate}%)")
                logger.info("[self_model] %s", changes[-1])

            elif timeout_rate < 5 and avg_duration and avg_duration < current_timeout * 0.4 and current_timeout > lo_timeout:
                new_timeout = max(current_timeout - 60, lo_timeout)
                self._sub_sessions.default_timeout = new_timeout
                changes.append(f"Decreased sub-session timeout {current_timeout}s → {new_timeout}s (timeout_rate={timeout_rate}%, avg_duration={avg_duration:.0f}s)")
                logger.info("[self_model] %s", changes[-1])

        # --- Memory harvest threshold tuning ---
        if self._harvest is not None:
            current_threshold = self._harvest.message_threshold

            if self._harvest.has_backlog() and current_threshold > lo_harvest:
                # Harvest can't keep up — decrease threshold to trigger sooner.
                new_threshold = max(current_threshold - 5, lo_harvest)
                self._harvest.message_threshold = new_threshold
                changes.append(f"Decreased harvest threshold {current_threshold} → {new_threshold} (backlog detected)")
                logger.info("[self_model] %s", changes[-1])
            elif current_threshold < hi_harvest and recent_harvests is not None:
                # Check recent harvests for low yield → increase threshold.
                try:
                    low_yield = (
                        len(recent_harvests) >= 3
                        and all(len(h.get("output", "")) < 50 for h in recent_harvests)
                    )
                    if low_yield:
                        new_threshold = min(current_threshold + 5, hi_harvest)
                        self._harvest.message_threshold = new_threshold
                        changes.append(f"Increased harvest threshold {current_threshold} → {new_threshold} (low yield)")
                        logger.info("[self_model] %s", changes[-1])
                except Exception:
                    pass

        return changes

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    async def _generate_summary(self, metrics: dict, changes: list[str]) -> str:
        """Generate a short prose summary via one LLM call."""
        if not self._pool or not self._pool.enabled:
            return self._format_fallback_summary(metrics, changes)

        try:
            from wintermute.infra import prompt_loader
            prompt_text = prompt_loader.load(
                "SELF_MODEL_SUMMARY.txt",
                metrics=json.dumps(metrics, default=str),
                changes="\n".join(changes) if changes else "(none)",
                max_chars=self._cfg.summary_max_chars,
            )
        except FileNotFoundError:
            return self._format_fallback_summary(metrics, changes)
        except Exception:
            logger.debug("[self_model] Failed to load summary prompt", exc_info=True)
            return self._format_fallback_summary(metrics, changes)

        try:
            response = await self._pool.call(
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens_override=256,
            )
            if response.content is not None:
                text = (response.content or "").strip()
                return text[:self._cfg.summary_max_chars]
        except Exception:
            logger.debug("[self_model] LLM summary call failed", exc_info=True)

        return self._format_fallback_summary(metrics, changes)

    @staticmethod
    def _format_fallback_summary(metrics: dict, changes: list[str]) -> str:
        """Produce a simple text summary without an LLM call."""
        parts = []
        total = metrics.get("sub_session_total", 0)
        if total:
            timeout_pct = metrics.get("timeout_rate_pct", 0)
            avg_dur = metrics.get("avg_duration_seconds")
            parts.append(
                f"{total} sub-sessions tracked ({timeout_pct}% timeout rate"
                + (f", avg {avg_dur:.0f}s" if avg_dur else "")
                + ")"
            )
        top_tools = metrics.get("top_tools", [])
        if top_tools:
            tool_str = ", ".join(f"{name}({count})" for name, count in top_tools[:3])
            parts.append(f"Top tools: {tool_str}")
        if changes:
            parts.append("Recent tuning: " + "; ".join(changes))
        return ". ".join(parts) if parts else ""
