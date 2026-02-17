"""
SQLite database operations for conversation history and pulse.
The APScheduler job store uses its own SQLite file (scheduler.db).
"""

import sqlite3
import time
import logging
from importlib import resources
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONVERSATION_DB = Path("data/conversation.db")

# All migration files in order.  Add new entries at the end.
_MIGRATIONS = [
    "001_initial_schema.sql",
    "002_add_pulse.sql",
    "003_add_interaction_log.sql",
    "004_add_thread_id.sql",
    "005_add_raw_output.sql",
]


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending SQL migrations from wintermute/migrations/."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name       TEXT PRIMARY KEY,
            applied_at REAL NOT NULL
        )
    """)
    conn.commit()

    applied = {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}

    # Bootstrap: existing DB already has the full schema from inline CREATE TABLE
    if not applied:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "messages" in tables:
            now = time.time()
            for name in _MIGRATIONS:
                conn.execute(
                    "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                    (name, now),
                )
            conn.commit()
            logger.info("Bootstrap: marked %d existing migrations as applied", len(_MIGRATIONS))
            return

    migration_pkg = resources.files("wintermute.migrations")
    for name in _MIGRATIONS:
        if name in applied:
            continue
        sql = (migration_pkg / name).read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            (name, time.time()),
        )
        conn.commit()
        logger.info("Applied migration: %s", name)


def init_db() -> None:
    """Create tables if they don't exist, and run migrations."""
    CONVERSATION_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CONVERSATION_DB) as conn:
        run_migrations(conn)
    _migrate_pulse_from_file()
    logger.debug("Database initialised at %s", CONVERSATION_DB)


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
    """Mark messages with id <= before_id as archived for a specific thread.

    Also deletes archived messages older than 30 days to prevent unbounded
    DB growth. Archived messages are never queried operationally — the
    compaction summary preserves their information.
    """
    cutoff = time.time() - 30 * 86400  # 30 days
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute(
            "UPDATE messages SET archived=1 WHERE id <= ? AND thread_id=?",
            (before_id, thread_id),
        )
        deleted = conn.execute(
            "DELETE FROM messages WHERE archived=1 AND timestamp < ? AND thread_id=?",
            (cutoff, thread_id),
        ).rowcount
        conn.commit()
    if deleted:
        logger.info("Purged %d archived messages older than 30 days", deleted)


def save_summary(content: str, thread_id: str = "default") -> None:
    """Persist a compaction summary for a thread.

    Keeps only the latest summary per thread to prevent unbounded growth.
    Old summaries are superseded by the chained compaction approach.
    """
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.execute(
            "DELETE FROM summaries WHERE thread_id=?",
            (thread_id,),
        )
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


def get_recently_active_thread_ids(since: float) -> list[str]:
    """Return thread_ids with at least one non-archived message after `since` (Unix timestamp)."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM messages "
            "WHERE archived=0 AND timestamp > ?",
            (since,),
        ).fetchall()
    return [r[0] for r in rows]


def get_thread_stats(thread_id: str = "default") -> dict:
    """Return message count and estimated token usage for a thread's active messages."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(COALESCE(token_count, LENGTH(content)/4)), 0) "
            "FROM messages WHERE archived=0 AND thread_id=?",
            (thread_id,),
        ).fetchone()
    return {"msg_count": row[0], "token_used": int(row[1])}


# ---------------------------------------------------------------------------
# Pulse CRUD
# ---------------------------------------------------------------------------

def add_pulse_item(content: str, priority: int = 5, thread_id: str | None = None) -> int:
    """Insert a new active pulse item. Returns the row id."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        cur = conn.execute(
            "INSERT INTO pulse (content, status, priority, created, thread_id) "
            "VALUES (?, 'active', ?, ?, ?)",
            (content, priority, time.time(), thread_id),
        )
        conn.commit()
        return cur.lastrowid


def complete_pulse_item(item_id: int) -> bool:
    """Mark a pulse item as completed. Returns True if a row was updated."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        n = conn.execute(
            "UPDATE pulse SET status='completed', updated=? WHERE id=?",
            (time.time(), item_id),
        ).rowcount
        conn.commit()
    return n > 0


def update_pulse_item(item_id: int, **kwargs) -> bool:
    """Update fields on a pulse item. Supported: content, priority, status."""
    allowed = {"content", "priority", "status"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    updates["updated"] = time.time()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [item_id]
    with sqlite3.connect(CONVERSATION_DB) as conn:
        n = conn.execute(f"UPDATE pulse SET {set_clause} WHERE id=?", values).rowcount
        conn.commit()
    return n > 0


def list_pulse_items(status: str = "active") -> list[dict]:
    """Return pulse items filtered by status, ordered by priority then id."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.row_factory = sqlite3.Row
        if status == "all":
            rows = conn.execute(
                "SELECT id, content, status, priority, created, updated, thread_id "
                "FROM pulse ORDER BY priority ASC, id ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, content, status, priority, created, updated, thread_id "
                "FROM pulse WHERE status=? ORDER BY priority ASC, id ASC",
                (status,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_active_pulse_text() -> str:
    """Compact formatted string of active pulse items for system prompt injection."""
    items = list_pulse_items("active")
    if not items:
        return ""
    return "\n".join(f"[P{it['priority']}] #{it['id']}: {it['content']}" for it in items)


def delete_old_completed_pulse(days: int = 30) -> int:
    """Delete completed pulse items older than *days*. Returns count deleted."""
    cutoff = time.time() - days * 86400
    with sqlite3.connect(CONVERSATION_DB) as conn:
        n = conn.execute(
            "DELETE FROM pulse WHERE status='completed' AND created < ?",
            (cutoff,),
        ).rowcount
        conn.commit()
    if n:
        logger.info("Purged %d completed pulse items older than %d days", n, days)
    return n


def _migrate_pulse_from_file() -> None:
    """One-time migration: import PULSE.txt content into the DB."""
    pulse_file = Path("data/PULSE.txt")
    if not pulse_file.exists():
        return
    try:
        text = pulse_file.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if not text:
        pulse_file.rename(pulse_file.with_suffix(".txt.migrated"))
        return
    # Skip if it's just the default placeholder
    if "no active pulse" in text.lower():
        pulse_file.rename(pulse_file.with_suffix(".txt.migrated"))
        logger.info("PULSE.txt was empty placeholder, renamed to .migrated")
        return
    # Parse line by line — each non-empty, non-header line becomes an item
    now = time.time()
    items = []
    for line in text.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    if not items:
        # Single blob
        items = [text]
    with sqlite3.connect(CONVERSATION_DB) as conn:
        for item in items:
            conn.execute(
                "INSERT INTO pulse (content, status, priority, created) VALUES (?, 'active', 5, ?)",
                (item, now),
            )
        conn.commit()
    pulse_file.rename(pulse_file.with_suffix(".txt.migrated"))
    logger.info("Migrated %d pulse items from PULSE.txt to DB", len(items))


# ---------------------------------------------------------------------------
# Interaction Log CRUD
# ---------------------------------------------------------------------------

def save_interaction_log(timestamp: float, action: str, session: str,
                         llm: str, input_text: str, output_text: str,
                         status: str = "ok",
                         raw_output: Optional[str] = None) -> int:
    """Insert an interaction log entry and return its row id."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        cur = conn.execute(
            "INSERT INTO interaction_log (timestamp, action, session, llm, input, output, status, raw_output) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, action, session, llm, input_text, output_text, status, raw_output),
        )
        conn.commit()
        return cur.lastrowid


def get_interaction_log(limit: int = 200, offset: int = 0,
                        session_filter: Optional[str] = None) -> list[dict]:
    """Return interaction log entries, newest first. No truncation."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.row_factory = sqlite3.Row
        if session_filter:
            rows = conn.execute(
                "SELECT * FROM interaction_log WHERE session=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (session_filter, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM interaction_log ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows]


def get_interaction_log_entry(entry_id: int) -> Optional[dict]:
    """Return a single interaction log entry by id, or None."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM interaction_log WHERE id=?", (entry_id,)
        ).fetchone()
    return dict(row) if row else None


def count_interaction_log(session_filter: Optional[str] = None) -> int:
    """Return total count of interaction log entries."""
    with sqlite3.connect(CONVERSATION_DB) as conn:
        if session_filter:
            row = conn.execute(
                "SELECT COUNT(*) FROM interaction_log WHERE session=?",
                (session_filter,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM interaction_log").fetchone()
    return row[0]
