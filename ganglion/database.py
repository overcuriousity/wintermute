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
    """Create tables if they don't exist, and run migrations."""
    CONVERSATION_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                token_count INTEGER,
                archived    INTEGER NOT NULL DEFAULT 0,
                thread_id   TEXT    NOT NULL DEFAULT 'default'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL    NOT NULL,
                content   TEXT    NOT NULL,
                thread_id TEXT    NOT NULL DEFAULT 'default'
            )
        """)
        conn.commit()

        # Migration: add thread_id column if missing
        _migrate_add_thread_id(conn, "messages")
        _migrate_add_thread_id(conn, "summaries")

    logger.debug("Database initialised at %s", CONVERSATION_DB)


def _migrate_add_thread_id(conn: sqlite3.Connection, table: str) -> None:
    """Add thread_id column to table if it doesn't exist."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if "thread_id" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN thread_id TEXT NOT NULL DEFAULT 'default'")
        conn.commit()
        logger.info("Migrated %s: added thread_id column", table)


def save_message(role: str, content: str, thread_id: str = "default",
                 token_count: Optional[int] = None) -> int:
    """Insert a message and return its row id."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        cur = conn.execute(
            "INSERT INTO messages (timestamp, role, content, token_count, thread_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), role, content, token_count, thread_id),
        )
        conn.commit()
        return cur.lastrowid


def load_active_messages(thread_id: str = "default") -> list[dict]:
    """Return all non-archived messages for a thread, ordered by id."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        rows = conn.execute(
            "SELECT id, timestamp, role, content, token_count "
            "FROM messages WHERE archived=0 AND thread_id=? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "role": r[2], "content": r[3], "token_count": r[4]}
        for r in rows
    ]


def archive_messages(before_id: int, thread_id: str = "default") -> None:
    """Mark messages with id <= before_id as archived for a specific thread."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute(
            "UPDATE messages SET archived=1 WHERE id <= ? AND thread_id=?",
            (before_id, thread_id),
        )
        conn.commit()


def save_summary(content: str, thread_id: str = "default") -> None:
    """Persist a compaction summary for a thread."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute(
            "INSERT INTO summaries (timestamp, content, thread_id) VALUES (?, ?, ?)",
            (time.time(), content, thread_id),
        )
        conn.commit()


def load_latest_summary(thread_id: str = "default") -> Optional[str]:
    """Return the most recent compaction summary for a thread, or None."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        row = conn.execute(
            "SELECT content FROM summaries WHERE thread_id=? ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    return row[0] if row else None


def clear_active_messages(thread_id: str = "default") -> None:
    """Archive all active messages for a thread (used by /new command)."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute(
            "UPDATE messages SET archived=1 WHERE archived=0 AND thread_id=?",
            (thread_id,),
        )
        conn.commit()


def get_active_thread_ids() -> list[str]:
    """Return distinct thread_ids that have non-archived messages."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM messages WHERE archived=0"
        ).fetchall()
    return [r[0] for r in rows]
