"""
SQLite database operations for conversation history and agenda.
The APScheduler job store uses its own SQLite file (scheduler.db).

All public functions are synchronous (they use sqlite3 directly).
Async callers MUST use ``await async_call(fn, *args, **kwargs)`` to
avoid blocking the event loop (SQLite's busy_timeout can stall for
up to 5 seconds under write contention).
"""

import asyncio
import json
import sqlite3
import time
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CONVERSATION_DB = Path("data/conversation.db")


async def async_call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous database function in a thread.

    All async code that touches the database should use this wrapper
    to prevent SQLite I/O (especially ``busy_timeout`` waits) from
    blocking the asyncio event loop.

    Usage::

        rows = await database.async_call(database.load_active_messages, thread_id)
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


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
        CREATE TABLE IF NOT EXISTS sub_session_outcomes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT NOT NULL,
            workflow_id         TEXT,
            timestamp           REAL NOT NULL,
            objective           TEXT NOT NULL,
            system_prompt_mode  TEXT NOT NULL,
            tools_available     TEXT,
            tools_used          TEXT,
            tool_call_count     INTEGER,
            duration_seconds    REAL,
            timeout_value       INTEGER,
            turing_verdict      TEXT,
            status              TEXT NOT NULL,
            result_length       INTEGER,
            nesting_depth       INTEGER,
            continuation_count  INTEGER,
            backend_used        TEXT,
            objective_embedding BLOB
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
    """Return {thread_id: max_message_id} from the last *successful* harvest per thread."""
    import re
    result: dict[str, int] = {}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT session, input FROM interaction_log "
            "WHERE action = 'memory_harvest' AND status = 'ok'"
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


# ---------------------------------------------------------------------------
# Sub-session Outcome Tracking
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "this", "that", "not", "has", "had", "have", "will", "can", "do",
    "does", "did", "its", "all", "into", "also", "than", "then",
})


def save_sub_session_outcome(**fields) -> int:
    """Insert a sub-session outcome row and return its row id.

    Optionally embeds the objective for vector similarity search if the
    memory store's vector backend is active.
    """
    objective = fields.get("objective", "")

    # Try to embed the objective for vector search.
    embedding_blob = None
    try:
        from wintermute.infra import memory_store
        if memory_store.is_vector_enabled() and memory_store._config:
            embed_cfg = memory_store._config.get("embeddings", {})
            if embed_cfg.get("endpoint"):
                vectors = memory_store._embed([objective], embed_cfg)
                if vectors and vectors[0]:
                    import struct
                    embedding_blob = struct.pack(f"{len(vectors[0])}f", *vectors[0])
    except Exception as exc:
        logger.debug("Could not embed outcome objective: %s", exc)

    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO sub_session_outcomes "
            "(session_id, workflow_id, timestamp, objective, system_prompt_mode, "
            "tools_available, tools_used, tool_call_count, duration_seconds, "
            "timeout_value, turing_verdict, status, result_length, nesting_depth, "
            "continuation_count, backend_used, objective_embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fields.get("session_id"),
                fields.get("workflow_id"),
                fields.get("timestamp", time.time()),
                objective,
                fields.get("system_prompt_mode", "minimal"),
                json.dumps(fields["tools_available"]) if fields.get("tools_available") else None,
                json.dumps(fields["tools_used"]) if fields.get("tools_used") else None,
                fields.get("tool_call_count"),
                fields.get("duration_seconds"),
                fields.get("timeout_value"),
                fields.get("turing_verdict"),
                fields.get("status", "unknown"),
                fields.get("result_length"),
                fields.get("nesting_depth"),
                fields.get("continuation_count"),
                fields.get("backend_used"),
                embedding_blob,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_similar_outcomes(objective: str, limit: int = 5) -> list[dict]:
    """Find past sub-session outcomes similar to the given objective.

    Uses vector similarity if available, falls back to keyword LIKE matching.
    """
    # Try vector search first.
    try:
        from wintermute.infra import memory_store
        if memory_store.is_vector_enabled() and memory_store._config:
            embed_cfg = memory_store._config.get("embeddings", {})
            if embed_cfg.get("endpoint"):
                results = _vector_search_outcomes(objective, embed_cfg, limit)
                if results:
                    return results
    except Exception as exc:
        logger.debug("Vector outcome search failed, falling back to keyword: %s", exc)

    return _keyword_search_outcomes(objective, limit)


def _vector_search_outcomes(objective: str, embed_cfg: dict, limit: int) -> list[dict]:
    """Search outcomes by cosine similarity of objective embeddings."""
    import struct
    from wintermute.infra.memory_store import _embed

    query_vec = _embed([objective], embed_cfg, task="query")
    if not query_vec or not query_vec[0]:
        return []
    qv = query_vec[0]

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sub_session_outcomes WHERE objective_embedding IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()

    if not rows:
        return []

    scored = []
    dim = len(qv)
    for row in rows:
        blob = row["objective_embedding"]
        if not blob or len(blob) != dim * 4:
            continue
        vec = struct.unpack(f"{dim}f", blob)
        # Cosine similarity.
        dot = sum(a * b for a, b in zip(qv, vec))
        mag_q = sum(a * a for a in qv) ** 0.5
        mag_v = sum(a * a for a in vec) ** 0.5
        if mag_q == 0 or mag_v == 0:
            continue
        sim = dot / (mag_q * mag_v)
        if sim > 0.5:
            scored.append((sim, dict(row)))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, row_dict in scored[:limit]:
        row_dict.pop("objective_embedding", None)
        row_dict["similarity"] = round(sim, 3)
        results.append(row_dict)
    return results


def _keyword_search_outcomes(objective: str, limit: int) -> list[dict]:
    """Search outcomes by keyword LIKE matching on objective text."""
    words = [w.lower() for w in objective.split() if len(w) > 3 and w.lower() not in _STOPWORDS]
    if not words:
        return []

    # Use at most 5 keywords to keep the query reasonable.
    words = words[:5]
    conditions = " OR ".join("objective LIKE ?" for _ in words)
    params = [f"%{w}%" for w in words]

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM sub_session_outcomes WHERE {conditions} "
            "ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        d.pop("objective_embedding", None)
        results.append(d)
    return results


def get_outcome_stats() -> dict:
    """Return aggregate sub-session outcome statistics."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sub_session_outcomes").fetchone()[0]
        by_status = conn.execute(
            "SELECT status, COUNT(*) FROM sub_session_outcomes GROUP BY status"
        ).fetchall()
        avg_duration = conn.execute(
            "SELECT AVG(duration_seconds) FROM sub_session_outcomes WHERE duration_seconds IS NOT NULL"
        ).fetchone()[0]
        avg_tool_calls = conn.execute(
            "SELECT AVG(tool_call_count) FROM sub_session_outcomes WHERE tool_call_count IS NOT NULL"
        ).fetchone()[0]
        timeout_rate_row = conn.execute(
            "SELECT COUNT(*) FROM sub_session_outcomes WHERE status='timeout'"
        ).fetchone()[0]
    return {
        "total": total,
        "by_status": {r[0]: r[1] for r in by_status},
        "avg_duration_seconds": round(avg_duration, 1) if avg_duration else None,
        "avg_tool_calls": round(avg_tool_calls, 1) if avg_tool_calls else None,
        "timeout_rate_pct": round(timeout_rate_row * 100 / total) if total else 0,
    }


def get_outcomes_page(
    limit: int = 200,
    offset: int = 0,
    status_filter: Optional[str] = None,
) -> tuple[list[dict], int, dict]:
    """Return a page of sub-session outcomes plus totals and aggregate stats."""
    where = "WHERE status = ?" if status_filter else ""
    params: list = [status_filter] if status_filter else []

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT id, session_id, workflow_id, timestamp, objective, system_prompt_mode, "
            f"tools_used, tool_call_count, duration_seconds, timeout_value, turing_verdict, "
            f"status, result_length, nesting_depth, continuation_count, backend_used "
            f"FROM sub_session_outcomes {where} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) FROM sub_session_outcomes {where}", params
        ).fetchone()[0]

    entries = [dict(r) for r in rows]
    stats = get_outcome_stats()
    return entries, total, stats
