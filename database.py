"""
SQLite database operations for conversation history.
The APScheduler job store uses its own SQLite file (scheduler.db).
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONVERSATION_DB = Path("data/conversation.db")


def init_db() -> None:
    """Create tables if they don't exist."""
    CONVERSATION_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                token_count INTEGER,
                archived    INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL    NOT NULL,
                content   TEXT    NOT NULL
            )
        """)
        conn.commit()
    logger.debug("Database initialised at %s", CONVERSATION_DB)


def save_message(role: str, content: str, token_count: Optional[int] = None) -> int:
    """Insert a message and return its row id."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        cur = conn.execute(
            "INSERT INTO messages (timestamp, role, content, token_count) VALUES (?, ?, ?, ?)",
            (time.time(), role, content, token_count),
        )
        conn.commit()
        return cur.lastrowid


def load_active_messages() -> list[dict]:
    """Return all non-archived messages ordered by id."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        rows = conn.execute(
            "SELECT id, timestamp, role, content, token_count "
            "FROM messages WHERE archived=0 ORDER BY id ASC"
        ).fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "role": r[2], "content": r[3], "token_count": r[4]}
        for r in rows
    ]


def archive_messages(before_id: int) -> None:
    """Mark messages with id <= before_id as archived (kept for audit trail)."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute("UPDATE messages SET archived=1 WHERE id <= ?", (before_id,))
        conn.commit()


def save_summary(content: str) -> None:
    """Persist a compaction summary."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute(
            "INSERT INTO summaries (timestamp, content) VALUES (?, ?)",
            (time.time(), content),
        )
        conn.commit()


def load_latest_summary() -> Optional[str]:
    """Return the most recent compaction summary, or None."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        row = conn.execute(
            "SELECT content FROM summaries ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


def clear_active_messages() -> None:
    """Archive all active messages (used by /new command)."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute("UPDATE messages SET archived=1 WHERE archived=0")
        conn.commit()
