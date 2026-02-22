"""
Vector-indexed memory retrieval for Wintermute.

Provides three backends for memory storage and retrieval:
  - flat_file  — wraps current MEMORIES.txt behavior (default, zero-config)
  - fts5       — SQLite FTS5 keyword search with BM25 ranking
  - qdrant     — Qdrant vector DB with embedding-based semantic search

Module-level singleton pattern (like database.py, prompt_assembler.py).
Call ``init(config)`` once at startup; all other functions use the active backend.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
MEMORIES_FILE = DATA_DIR / "MEMORIES.txt"
FTS5_DB_PATH = DATA_DIR / "memory_index.db"

# Module-level singleton state.
_backend: MemoryBackend | None = None
_config: dict = {}


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryBackend(Protocol):
    def init(self) -> None: ...
    def add(self, entry: str, entry_id: str | None = None) -> str: ...
    def search(self, query: str, top_k: int, threshold: float) -> list[dict]: ...
    def get_all(self) -> list[dict]: ...
    def replace_all(self, entries: list[str]) -> None: ...
    def delete(self, entry_id: str) -> bool: ...
    def count(self) -> int: ...
    def stats(self) -> dict: ...
    def rebuild(self) -> None: ...


# ---------------------------------------------------------------------------
# Flat-file backend (default — no ranking, returns all memories)
# ---------------------------------------------------------------------------

class FlatFileBackend:
    """Wraps current MEMORIES.txt behavior.  search() returns all memories."""

    def init(self) -> None:
        logger.info("Memory backend: flat_file (no vector indexing)")

    def add(self, entry: str, entry_id: str | None = None) -> str:
        # File writes are handled by prompt_assembler; this is a no-op.
        return entry_id or _make_id(entry)

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        # Return all memories — no ranking.
        return self.get_all()

    def get_all(self) -> list[dict]:
        try:
            text = MEMORIES_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return []
        if not text:
            return []
        return [
            {"id": _make_id(line), "text": line, "score": 1.0}
            for line in text.splitlines() if line.strip()
        ]

    def replace_all(self, entries: list[str]) -> None:
        pass  # File writes handled by prompt_assembler.

    def delete(self, entry_id: str) -> bool:
        return False

    def count(self) -> int:
        try:
            text = MEMORIES_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return 0
        return len([l for l in text.splitlines() if l.strip()]) if text else 0

    def stats(self) -> dict:
        return {"backend": "flat_file", "count": self.count()}

    def rebuild(self) -> None:
        pass  # Nothing to rebuild.


# ---------------------------------------------------------------------------
# FTS5 backend (SQLite keyword search with BM25)
# ---------------------------------------------------------------------------

class FTS5Backend:
    """SQLite FTS5 full-text search with BM25 scoring."""

    def __init__(self) -> None:
        self._db_path = FTS5_DB_PATH
        self._lock = threading.Lock()

    def init(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memories_meta ("
                "  entry_id TEXT PRIMARY KEY,"
                "  text TEXT NOT NULL,"
                "  created_at REAL NOT NULL"
                ")"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
                "USING fts5(entry_id, text, content=memories_meta, content_rowid=rowid)"
            )
            # Triggers to keep FTS in sync with meta table.
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories_meta BEGIN
                    INSERT INTO memories_fts(rowid, entry_id, text)
                    VALUES (new.rowid, new.entry_id, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories_meta BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, entry_id, text)
                    VALUES ('delete', old.rowid, old.entry_id, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories_meta BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, entry_id, text)
                    VALUES ('delete', old.rowid, old.entry_id, old.text);
                    INSERT INTO memories_fts(rowid, entry_id, text)
                    VALUES (new.rowid, new.entry_id, new.text);
                END;
            """)
            conn.commit()
            conn.close()
        logger.info("Memory backend: fts5 (SQLite FTS5 at %s)", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def add(self, entry: str, entry_id: str | None = None) -> str:
        eid = entry_id or _make_id(entry)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO memories_meta (entry_id, text, created_at) "
                    "VALUES (?, ?, ?)",
                    (eid, entry.strip(), time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        return eid

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        if not query.strip():
            return self.get_all()[:top_k]
        # Escape FTS5 special chars and build simple OR query.
        tokens = query.split()
        fts_query = " OR ".join(f'"{t}"' for t in tokens[:20] if t.strip())
        if not fts_query:
            return self.get_all()[:top_k]
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT m.entry_id, m.text, rank "
                    "FROM memories_fts f "
                    "JOIN memories_meta m ON f.rowid = m.rowid "
                    "WHERE memories_fts MATCH ? "
                    "ORDER BY rank "
                    "LIMIT ?",
                    (fts_query, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                # Malformed query — fall back to returning all.
                logger.debug("FTS5 query failed, returning all memories")
                rows = []
            finally:
                conn.close()
        if not rows:
            return self.get_all()[:top_k]
        # BM25 rank is negative (more negative = better match).
        return [
            {"id": r[0], "text": r[1], "score": -r[2]}
            for r in rows
        ]

    def get_all(self) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text FROM memories_meta ORDER BY created_at"
                ).fetchall()
            finally:
                conn.close()
        return [{"id": r[0], "text": r[1], "score": 1.0} for r in rows]

    def replace_all(self, entries: list[str]) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM memories_meta")
                now = time.time()
                conn.executemany(
                    "INSERT INTO memories_meta (entry_id, text, created_at) VALUES (?, ?, ?)",
                    [(_make_id(e), e.strip(), now) for e in entries if e.strip()],
                )
                conn.commit()
            finally:
                conn.close()
        logger.info("FTS5: replaced all entries (%d)", len(entries))

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM memories_meta WHERE entry_id = ?", (entry_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM memories_meta").fetchone()[0]
            finally:
                conn.close()

    def stats(self) -> dict:
        return {
            "backend": "fts5",
            "count": self.count(),
            "db_path": str(self._db_path),
        }

    def rebuild(self) -> None:
        """Drop and re-import from MEMORIES.txt."""
        try:
            text = MEMORIES_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            text = ""
        entries = [l.strip() for l in text.splitlines() if l.strip()] if text else []
        self.replace_all(entries)
        logger.info("FTS5: rebuilt index from MEMORIES.txt (%d entries)", len(entries))


# ---------------------------------------------------------------------------
# Qdrant backend (vector semantic search)
# ---------------------------------------------------------------------------

class QdrantBackend:
    """Qdrant vector DB with embedding-based semantic search."""

    def __init__(self, config: dict) -> None:
        self._qdrant_cfg = config.get("qdrant", {})
        self._embed_cfg = config.get("embeddings", {})
        self._url = self._qdrant_cfg.get("url", "http://localhost:6333")
        self._collection = self._qdrant_cfg.get("collection", "wintermute_memories")
        self._dimensions = self._embed_cfg.get("dimensions", 1536)
        self._client: Any = None
        self._lock = threading.Lock()

    def init(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=self._url, timeout=30)
        # Create collection if it doesn't exist.
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Qdrant: created collection %r (%d dims)",
                        self._collection, self._dimensions)
        logger.info("Memory backend: qdrant (url=%s, collection=%s)",
                     self._url, self._collection)

    def add(self, entry: str, entry_id: str | None = None) -> str:
        from qdrant_client.models import PointStruct

        eid = entry_id or _make_id(entry)
        vectors = _embed([entry], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        with self._lock:
            self._client.upsert(
                collection_name=self._collection,
                points=[PointStruct(
                    id=eid,
                    vector=vectors[0],
                    payload={"text": entry.strip(), "created_at": time.time()},
                )],
            )
        return eid

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        vectors = _embed([query], self._embed_cfg)
        if not vectors:
            logger.warning("Qdrant: embedding failed for query, falling back to get_all")
            return self.get_all()[:top_k]
        with self._lock:
            results = self._client.query_points(
                collection_name=self._collection,
                query=vectors[0],
                limit=top_k,
                score_threshold=threshold,
            ).points
        return [
            {"id": str(r.id), "text": r.payload.get("text", ""), "score": r.score}
            for r in results
        ]

    def get_all(self) -> list[dict]:
        with self._lock:
            result = self._client.scroll(
                collection_name=self._collection,
                limit=10000,
                with_payload=True,
            )
        points = result[0] if result else []
        return [
            {"id": str(p.id), "text": p.payload.get("text", ""), "score": 1.0}
            for p in points
        ]

    def replace_all(self, entries: list[str]) -> None:
        from qdrant_client.models import PointStruct

        entries = [e.strip() for e in entries if e.strip()]
        if not entries:
            with self._lock:
                # Delete all points.
                self._client.delete_collection(self._collection)
            self.init()  # Recreate empty collection.
            return

        # Batch embed.
        vectors = _embed(entries, self._embed_cfg)
        if not vectors or len(vectors) != len(entries):
            raise RuntimeError(
                f"Embedding mismatch: {len(entries)} entries but {len(vectors) if vectors else 0} vectors"
            )

        points = [
            PointStruct(
                id=_make_id(entry),
                vector=vec,
                payload={"text": entry, "created_at": time.time()},
            )
            for entry, vec in zip(entries, vectors)
        ]

        with self._lock:
            # Recreate collection to ensure clean state.
            self._client.delete_collection(self._collection)
        self.init()
        # Upsert in batches of 100.
        for i in range(0, len(points), 100):
            batch = points[i:i + 100]
            with self._lock:
                self._client.upsert(
                    collection_name=self._collection,
                    points=batch,
                )
        logger.info("Qdrant: replaced all entries (%d)", len(entries))

    def delete(self, entry_id: str) -> bool:
        from qdrant_client.models import PointIdsList

        with self._lock:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[entry_id]),
            )
        return True

    def count(self) -> int:
        with self._lock:
            info = self._client.get_collection(self._collection)
        return info.points_count

    def stats(self) -> dict:
        with self._lock:
            info = self._client.get_collection(self._collection)
        return {
            "backend": "qdrant",
            "count": info.points_count,
            "url": self._url,
            "collection": self._collection,
            "dimensions": self._dimensions,
            "status": str(info.status),
        }

    def rebuild(self) -> None:
        """Drop collection and re-import from MEMORIES.txt."""
        try:
            text = MEMORIES_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            text = ""
        entries = [l.strip() for l in text.splitlines() if l.strip()] if text else []
        self.replace_all(entries)
        logger.info("Qdrant: rebuilt index from MEMORIES.txt (%d entries)", len(entries))


# ---------------------------------------------------------------------------
# Embedding helper (used by QdrantBackend)
# ---------------------------------------------------------------------------

def _embed(texts: list[str], embed_cfg: dict) -> list[list[float]]:
    """Call an OpenAI-compatible embeddings endpoint.

    Uses httpx (sync) — callers in async context should run via executor.
    """
    import httpx

    endpoint = embed_cfg.get("endpoint", "").rstrip("/")
    model = embed_cfg.get("model", "text-embedding-3-small")
    if not endpoint:
        raise RuntimeError("memory.embeddings.endpoint is not configured")

    url = f"{endpoint}/embeddings"
    payload = {"input": texts, "model": model}
    dimensions = embed_cfg.get("dimensions")
    if dimensions:
        payload["dimensions"] = dimensions

    resp = httpx.post(url, json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    # Sort by index to preserve order.
    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    """Deterministic ID from text content (SHA-256 prefix)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

def init(config: dict) -> None:
    """Select and initialize the memory backend.

    If the configured backend fails to init, falls back to flat_file.
    On first run with a vector backend, imports MEMORIES.txt into the index.
    """
    global _backend, _config
    _config = dict(config)
    backend_name = config.get("backend", "flat_file")

    try:
        if backend_name == "fts5":
            _backend = FTS5Backend()
        elif backend_name == "qdrant":
            _backend = QdrantBackend(config)
        else:
            _backend = FlatFileBackend()
        _backend.init()
    except Exception as exc:
        logger.error("Memory backend %r failed to init: %s — falling back to flat_file",
                      backend_name, exc)
        _backend = FlatFileBackend()
        _backend.init()
        return

    # Cold-boot: if vector backend is empty and MEMORIES.txt has content, import.
    if backend_name in ("fts5", "qdrant"):
        try:
            if _backend.count() == 0:
                text = MEMORIES_FILE.read_text(encoding="utf-8").strip()
                if text:
                    entries = [l.strip() for l in text.splitlines() if l.strip()]
                    if entries:
                        logger.info("Cold-boot: importing %d memories from MEMORIES.txt into %s",
                                     len(entries), backend_name)
                        _backend.replace_all(entries)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.error("Cold-boot import failed: %s", exc)


def search(query: str, top_k: int | None = None, threshold: float | None = None) -> list[dict]:
    """Search memories by relevance.  Uses configured defaults for top_k/threshold."""
    if top_k is None:
        top_k = get_top_k()
    if threshold is None:
        threshold = get_threshold()
    return _backend.search(query, top_k, threshold)


def add(entry: str, entry_id: str | None = None) -> str:
    return _backend.add(entry, entry_id)


def get_all() -> list[dict]:
    return _backend.get_all()


def replace_all(entries: list[str]) -> None:
    _backend.replace_all(entries)


def delete(entry_id: str) -> bool:
    return _backend.delete(entry_id)


def count() -> int:
    return _backend.count()


def stats() -> dict:
    return _backend.stats()


def rebuild() -> None:
    _backend.rebuild()


def is_vector_enabled() -> bool:
    """True when the active backend is not flat_file."""
    return not isinstance(_backend, FlatFileBackend)


def get_top_k() -> int:
    return _config.get("top_k", 10)


def get_threshold() -> float:
    return _config.get("score_threshold", 0.3)
