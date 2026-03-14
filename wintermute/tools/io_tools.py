"""File I/O and shell execution tool implementations."""

import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from wintermute.core.tool_deps import ToolDeps

logger = logging.getLogger(__name__)


def tool_execute_shell(inputs: dict, **_kw) -> str:
    command = inputs["command"]
    timeout = int(inputs.get("timeout", 30))
    logger.info("execute_shell: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return json.dumps({
            "stdout":    result.stdout,
            "stderr":    result.stderr,
            "exit_code": result.returncode,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("execute_shell failed")
        return json.dumps({"error": str(exc)})


def tool_read_file(inputs: dict, **_kw) -> str:
    path = Path(inputs["path"])
    try:
        result = json.dumps({"content": path.read_text(encoding="utf-8")})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except (OSError, UnicodeDecodeError) as exc:
        return json.dumps({"error": str(exc)})
    return result


def tool_write_file(inputs: dict, **_kw) -> str:
    path = Path(inputs["path"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inputs["content"], encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(path)})
    except OSError as exc:
        return json.dumps({"error": str(exc)})


def tool_send_file(inputs: dict, *, tool_deps: Optional["ToolDeps"] = None,
                   thread_id: Optional[str] = None, **_kw) -> str:
    """Validate a file and emit a ``send_file`` event for frontends to handle."""
    import mimetypes

    path = Path(inputs["path"])
    if not path.is_file():
        return json.dumps({"error": f"File not found or not a regular file: {path}"})

    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    file_size = path.stat().st_size
    caption = inputs.get("caption", "")

    if tool_deps and tool_deps.event_bus:
        tool_deps.event_bus.emit(
            "send_file",
            path=str(path.resolve()),
            filename=path.name,
            mime_type=mime_type,
            file_size=file_size,
            caption=caption,
            thread_id=thread_id or "",
        )
        logger.info("send_file: emitted event for %s (%s, %d bytes)", path.name, mime_type, file_size)
        return json.dumps({
            "status": "ok",
            "path": str(path.resolve()),
            "filename": path.name,
            "mime_type": mime_type,
            "file_size": file_size,
        })

    logger.warning("send_file: no event_bus available, file not delivered")
    return json.dumps({"error": "No delivery channel available (event_bus not configured)"})
