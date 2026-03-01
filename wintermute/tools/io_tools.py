"""File I/O and shell execution tool implementations."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from wintermute.infra import prompt_assembler

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
    # Track skill reads for skill_stats (only canonical data/skills/*.md).
    try:
        skills_dir = prompt_assembler.SKILLS_DIR.resolve()
        if path.suffix == ".md" and path.resolve().parent == skills_dir:
            from wintermute.workers import skill_stats
            skill_stats.record_read(path.stem)
    except Exception:
        pass
    return result


def tool_write_file(inputs: dict, **_kw) -> str:
    path = Path(inputs["path"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inputs["content"], encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(path)})
    except OSError as exc:
        return json.dumps({"error": str(exc)})
