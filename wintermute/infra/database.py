"""
SQLite database operations for conversation history and agenda.
The APScheduler job store uses its own SQLite file (scheduler.db).
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONVERSATION_DB = Path("data/conversation.db")


def thread_has_messages(thread_id: str = "default") -> bool:
    """Return True if the thread has any non-archived messages."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM messages WHERE thread_id=? AND archived=0 LIMIT 1",
            (thread_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    """Open a WAL-mode connection with a 5-second busy timeout."""
    conn = sqlite3.connect(CONVERSATION_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def run_migrations(conn: sqlite3.Connection) -> None:
    """Ensure all tables and columns exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            token_count INTEGER,
            archived    INTEGER NOT NULL DEFAULT 0,
            thread_id   TEXT    NOT NULL DEFAULT 'default'
        );
        CREATE TABLE IF NOT EXISTS summaries (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL    NOT NULL,
            content   TEXT    NOT NULL,
            thread_id TEXT    NOT NULL DEFAULT 'default'
        );
        CREATE TABLE IF NOT EXISTS agenda (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            content   TEXT    NOT NULL,
            status    TEXT    NOT NULL DEFAULT 'active',
            priority  INTEGER NOT NULL DEFAULT 5,
            created   REAL    NOT NULL,
            updated   REAL,
            thread_id TEXT
        );
        CREATE TABLE IF NOT EXISTS interaction_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  REAL    NOT NULL,
            action     TEXT    NOT NULL,
            session    TEXT    NOT NULL,
            llm        TEXT    NOT NULL,
            input      TEXT    NOT NULL,
            output     TEXT    NOT NULL,
            status     TEXT    NOT NULL DEFAULT 'ok',
            raw_output TEXT
        );
    """)
    conn.commit()


def init_db() -> None:
    """Create tables if they don't exist, and run migrations."""
    CONVERSATION_DB.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        run_migrations(conn)
    _migrate_agenda_from_file()
    logger.debug("Database initialised at %s", CONVERSATION_DB)


def save_message(role: str, content: str, thread_id: str = "default",
                 token_count: Optional[int] = None) -> int:
    """Insert a message and return its row id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (timestamp, role, content, token_count, thread_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), role, content, token_count, thread_id),
        )
        conn.commit()
        return cur.lastrowid


def load_active_messages(thread_id: str = "default") -> list[dict]:
    """Return all non-archived messages for a thread, ordered by id."""
    with _connect() as conn:
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
    with _connect() as conn:
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
    with _connect() as conn:
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
    with _connect() as conn:
        row = conn.execute(
            "SELECT content FROM summaries WHERE thread_id=? ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    return row[0] if row else None



def clear_active_messages(thread_id: str = "default") -> None:
    """Archive all active messages for a thread (used by /new command)."""
    with _connect() as conn:
        conn.execute(
            "UPDATE messages SET archived=1 WHERE archived=0 AND thread_id=?",
            (thread_id,),
        )
        conn.commit()


def get_active_thread_ids() -> list[str]:
    """Return distinct thread_ids that have non-archived messages."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM messages WHERE archived=0"
        ).fetchall()
    return [r[0] for r in rows]


def get_recently_active_thread_ids(since: float) -> list[str]:
    """Return thread_ids with at least one non-archived message after `since` (Unix timestamp)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM messages "
            "WHERE archived=0 AND timestamp > ?",
            (since,),
        ).fetchall()
    return [r[0] for r in rows]


def get_thread_stats(thread_id: str = "default") -> dict:
    """Return message count and estimated token usage for a thread's active messages."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(COALESCE(token_count, LENGTH(content)/4)), 0) "
            "FROM messages WHERE archived=0 AND thread_id=?",
            (thread_id,),
        ).fetchone()
    return {"msg_count": row[0], "token_used": int(row[1])}


# ---------------------------------------------------------------------------
# Agenda CRUD
# ---------------------------------------------------------------------------

def add_agenda_item(content: str, priority: int = 5, thread_id: str | None = None) -> int:
    """Insert a new active agenda item. Returns the row id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO agenda (content, status, priority, created, thread_id) "
            "VALUES (?, 'active', ?, ?, ?)",
            (content, priority, time.time(), thread_id),
        )
        conn.commit()
        return cur.lastrowid


def complete_agenda_item(item_id: int, thread_id: Optional[str] = None) -> bool:
    """Mark a agenda item as completed. Returns True if a row was updated.

    When *thread_id* is given the item must belong to that thread (ownership guard).
    """
    with _connect() as conn:
        if thread_id:
            n = conn.execute(
                "UPDATE agenda SET status='completed', updated=? WHERE id=? AND thread_id=?",
                (time.time(), item_id, thread_id),
            ).rowcount
        else:
            n = conn.execute(
                "UPDATE agenda SET status='completed', updated=? WHERE id=?",
                (time.time(), item_id),
            ).rowcount
        conn.commit()
    return n > 0


def update_agenda_item(item_id: int, thread_id: Optional[str] = None, **kwargs) -> bool:
    """Update fields on a agenda item. Supported: content, priority, status.

    When *thread_id* is given the item must belong to that thread (ownership guard).
    """
    allowed = {"content", "priority", "status"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    updates["updated"] = time.time()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    if thread_id:
        values = list(updates.values()) + [item_id, thread_id]
        where = "WHERE id=? AND thread_id=?"
    else:
        values = list(updates.values()) + [item_id]
        where = "WHERE id=?"
    with _connect() as conn:
        n = conn.execute(f"UPDATE agenda SET {set_clause} {where}", values).rowcount
        conn.commit()
    return n > 0


def list_agenda_items(status: str = "active", thread_id: Optional[str] = None) -> list[dict]:
    """Return agenda items filtered by status, ordered by priority then id.

    When *thread_id* is given, only items belonging to that thread are returned.
    """
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        if status == "all" and not thread_id:
            rows = conn.execute(
                "SELECT id, content, status, priority, created, updated, thread_id "
                "FROM agenda ORDER BY priority ASC, id ASC"
            ).fetchall()
        elif status == "all" and thread_id:
            rows = conn.execute(
                "SELECT id, content, status, priority, created, updated, thread_id "
                "FROM agenda WHERE thread_id=? ORDER BY priority ASC, id ASC",
                (thread_id,),
            ).fetchall()
        elif thread_id:
            rows = conn.execute(
                "SELECT id, content, status, priority, created, updated, thread_id "
                "FROM agenda WHERE status=? AND thread_id=? ORDER BY priority ASC, id ASC",
                (status, thread_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, content, status, priority, created, updated, thread_id "
                "FROM agenda WHERE status=? ORDER BY priority ASC, id ASC",
                (status,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_active_agenda_text(thread_id: Optional[str] = None) -> str:
    """Compact formatted string of active agenda items for system prompt injection.

    When *thread_id* is given, only items belonging to that thread are included.
    """
    items = list_agenda_items("active", thread_id=thread_id)
    if not items:
        return ""
    return "\n".join(f"[P{it['priority']}] #{it['id']}: {it['content']}" for it in items)


def get_agenda_thread_ids() -> list[tuple[str, int]]:
    """Return (thread_id, count) pairs for active agenda items with non-NULL thread_id."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT thread_id, COUNT(*) FROM agenda "
            "WHERE status='active' AND thread_id IS NOT NULL "
            "GROUP BY thread_id",
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def delete_old_completed_agenda(days: int = 30) -> int:
    """Delete completed agenda items older than *days*. Returns count deleted."""
    cutoff = time.time() - days * 86400
    with _connect() as conn:
        n = conn.execute(
            "DELETE FROM agenda WHERE status='completed' AND created < ?",
            (cutoff,),
        ).rowcount
        conn.commit()
    if n:
        logger.info("Purged %d completed agenda items older than %d days", n, days)
    return n


def _migrate_agenda_from_file() -> None:
    """One-time migration: import AGENDA.txt content into the DB."""
    agenda_file = Path("data/AGENDA.txt")
    if not agenda_file.exists():
        return
    try:
        text = agenda_file.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if not text:
        agenda_file.rename(agenda_file.with_suffix(".txt.migrated"))
        return
    # Skip if it's just the default placeholder
    if "no active agenda" in text.lower():
        agenda_file.rename(agenda_file.with_suffix(".txt.migrated"))
        logger.info("AGENDA.txt was empty placeholder, renamed to .migrated")
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
    with _connect() as conn:
        for item in items:
            conn.execute(
                "INSERT INTO agenda (content, status, priority, created) VALUES (?, 'active', 5, ?)",
                (item, now),
            )
        conn.commit()
    agenda_file.rename(agenda_file.with_suffix(".txt.migrated"))
    logger.info("Migrated %d agenda items from AGENDA.txt to DB", len(items))


# ---------------------------------------------------------------------------
# Interaction Log CRUD
# ---------------------------------------------------------------------------

def load_harvest_state() -> dict[str, int]:
    """Return {thread_id: max_message_id} from the last harvest run per thread."""
    import re
    result: dict[str, int] = {}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT session, input FROM interaction_log WHERE action = 'memory_harvest'"
        ).fetchall()
    for session, input_text in rows:
        # session = "harvest:<thread_id>", input = "thread=<id> msgs=<n> max_id=<N>"
        thread_id = session.removeprefix("harvest:")
        m = re.search(r"max_id=(\d+)", input_text or "")
        if m:
            result[thread_id] = max(result.get(thread_id, 0), int(m.group(1)))
    return result


def save_interaction_log(timestamp: float, action: str, session: str,
                         llm: str, input_text: str, output_text: str,
                         status: str = "ok",
                         raw_output: Optional[str] = None) -> int:
    """Insert an interaction log entry and return its row id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO interaction_log (timestamp, action, session, llm, input, output, status, raw_output) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, action, session, llm, input_text, output_text, status, raw_output),
        )
        conn.commit()
        return cur.lastrowid


def get_interaction_log(limit: int = 200, offset: int = 0,
                        session_filter: Optional[str] = None,
                        action_filter: Optional[str] = None,
                        before_id: Optional[int] = None,
                        after_id: Optional[int] = None) -> list[dict]:
    """Return interaction log entries. Newest first unless after_id is set (then ASC)."""
    conditions: list[str] = []
    params: list = []
    if session_filter:
        conditions.append("session=?")
        params.append(session_filter)
    if action_filter:
        conditions.append("action=?")
        params.append(action_filter)
    if before_id is not None:
        conditions.append("id < ?")
        params.append(before_id)
    if after_id is not None:
        conditions.append("id > ?")
        params.append(after_id)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    order = "ORDER BY id ASC" if after_id is not None else "ORDER BY id DESC"
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM interaction_log {where} {order} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    return [dict(r) for r in rows]


def get_interaction_log_max_id() -> int:
    """Return the highest id in interaction_log, or 0 if empty."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM interaction_log"
        ).fetchone()
    return row[0]


def get_interaction_log_entry(entry_id: int) -> Optional[dict]:
    """Return a single interaction log entry by id, or None."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM interaction_log WHERE id=?", (entry_id,)
        ).fetchone()
    return dict(row) if row else None


def count_interaction_log(session_filter: Optional[str] = None,
                          action_filter: Optional[str] = None) -> int:
    """Return total count of interaction log entries."""
    conditions: list[str] = []
    params: list = []
    if session_filter:
        conditions.append("session=?")
        params.append(session_filter)
    if action_filter:
        conditions.append("action=?")
        params.append(action_filter)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    with _connect() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) FROM interaction_log {where}", params
        ).fetchone()
    return row[0]
