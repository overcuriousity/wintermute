"""Syscall-style interface for tool execution.

Provides structured request/result types that formalize the tool
execution contract, replacing loose positional arguments with a
typed ``SyscallRequest`` and returning a ``SyscallResult`` with
timing, category, and error metadata.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from wintermute.core.tool_deps import ToolDeps

logger = logging.getLogger(__name__)


@dataclass
class SyscallRequest:
    """Structured request for a single tool invocation."""

    name: str
    inputs: dict
    thread_id: str | None = None
    nesting_depth: int = 0
    parent_thread_id: str | None = None
    tool_deps: ToolDeps | None = None
    allowed_categories: frozenset[str] | None = None  # None = all


@dataclass
class SyscallResult:
    """Structured result from a tool invocation."""

    success: bool
    data: str  # JSON string
    error: str | None = None
    tool_name: str = ""
    duration_ms: float = 0.0
    category: str = ""
