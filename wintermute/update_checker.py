"""
Update Checker Loop

Periodically checks for new commits on the upstream git remote and notifies
via Matrix when updates are available.  Also exposes a one-shot ``check()``
method used by the ``/status`` command.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

# Repo root: one level up from this file's directory (wintermute/).
_REPO_DIR = str(Path(__file__).resolve().parents[1])

_GIT_TIMEOUT = 60  # seconds


@dataclass
class UpdateCheckerConfig:
    enabled: bool = True
    interval_hours: int = 24
    remote_url: str = ""  # empty → use 'origin'


class UpdateCheckerLoop:
    """Runs as an asyncio task, periodically checking for upstream updates."""

    def __init__(
        self,
        config: UpdateCheckerConfig,
        broadcast_fn: Callable[..., Awaitable[None]],
        matrix_rooms: list[str],
    ) -> None:
        self._config = config
        self._broadcast = broadcast_fn
        self._matrix_rooms = matrix_rooms
        self._running = False
        self._last_notified_head: Optional[str] = None
        self._last_result: Optional[str] = None  # cached for /status

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        interval = self._config.interval_hours * 3600
        logger.info("Update checker started (interval=%dh)", self._config.interval_hours)
        if not self._matrix_rooms:
            logger.warning("Update checker: no Matrix rooms configured — notifications will not be sent")
        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                break
            try:
                msg = await self.check()
                if msg:
                    await self._notify(msg)
            except Exception:
                logger.exception("Update check failed")

    def stop(self) -> None:
        self._running = False

    @property
    def last_result(self) -> Optional[str]:
        """Return the cached result from the last periodic check."""
        return self._last_result

    async def check(self) -> Optional[str]:
        """Check for upstream updates.

        Returns a human-readable message if updates are available,
        or ``None`` if the local repo is up-to-date.
        """
        remote = await self._resolve_remote()
        if not remote:
            return None

        branch = await self._current_branch()
        if not branch or branch == "HEAD":
            logger.warning("Detached HEAD — update check skipped")
            return None

        # Fetch latest from remote
        ok = await self._git("fetch", remote)
        if not ok:
            return None

        local_head = await self._git_output("rev-parse", "HEAD")
        remote_head = await self._git_output("rev-parse", f"{remote}/{branch}")
        if not local_head or not remote_head:
            return None

        if local_head == remote_head:
            self._last_result = None
            return None

        count = await self._git_output(
            "rev-list", "--count", f"HEAD..{remote}/{branch}"
        )
        n = int(count) if count and count.isdigit() else "?"
        msg = (
            f"Update available: {n} new commit(s) on {remote}/{branch}. "
            f"Run 'git pull' to update."
        )
        self._last_result = msg
        self._last_notified_head = remote_head
        return msg

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _resolve_remote(self) -> Optional[str]:
        if self._config.remote_url:
            url = self._config.remote_url
            existing = await self._git_output("remote", "get-url", "wintermute-upstream")
            if existing and existing.strip() == url:
                return "wintermute-upstream"
            if existing:
                logger.info("Update checker: changing wintermute-upstream URL to %s", url)
                await self._git("remote", "set-url", "wintermute-upstream", url)
            else:
                await self._git("remote", "add", "wintermute-upstream", url)
            return "wintermute-upstream"
        # Default: use 'origin'
        url = await self._git_output("remote", "get-url", "origin")
        return "origin" if url else None

    async def _current_branch(self) -> Optional[str]:
        return await self._git_output("rev-parse", "--abbrev-ref", "HEAD")

    async def _notify(self, message: str) -> None:
        # Dedup: skip if we already notified for this remote HEAD.
        if self._last_notified_head and self._last_notified_head == getattr(self, "_prev_notified_head", None):
            return
        self._prev_notified_head = self._last_notified_head

        for room_id in self._matrix_rooms:
            try:
                await self._broadcast(message, room_id)
            except Exception:
                logger.debug("Failed to send update notification to %s", room_id)

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    async def _git(self, *args: str) -> bool:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=_REPO_DIR,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning("git %s timed out after %ds", " ".join(args), _GIT_TIMEOUT)
            return False
        if proc.returncode != 0:
            logger.warning("git %s failed: %s", " ".join(args), stderr.decode().strip())
            return False
        return True

    async def _git_output(self, *args: str) -> Optional[str]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=_REPO_DIR,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning("git %s timed out after %ds", " ".join(args), _GIT_TIMEOUT)
            return None
        if proc.returncode != 0:
            logger.debug("git %s failed: %s", " ".join(args), stderr.decode().strip())
            return None
        return stdout.decode().strip()
