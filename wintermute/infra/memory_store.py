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
    def add(self, entry: str, entry_id: str | None = None, source: str = "unknown") -> str: ...
    def search(self, query: str, top_k: int, threshold: float) -> list[dict]: ...
    def get_all(self) -> list[dict]: ...
    def replace_all(self, entries: list[str]) -> None: ...
    def delete(self, entry_id: str) -> bool: ...
    def count(self) -> int: ...
    def stats(self) -> dict: ...
    def rebuild(self) -> None: ...
    def get_all_with_vectors(self) -> list[dict]: ...
    def get_stale(self, max_age_days: int, min_access: int) -> list[dict]: ...
    def bulk_delete(self, entry_ids: list[str]) -> int: ...
    def get_top_accessed(self, limit: int) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Flat-file backend (default — no ranking, returns all memories)
# ---------------------------------------------------------------------------

class FlatFileBackend:
    """Wraps current MEMORIES.txt behavior.  search() returns all memories."""

    def init(self) -> None:
        logger.info("Memory backend: flat_file (no vector indexing)")

    def add(self, entry: str, entry_id: str | None = None, source: str = "unknown") -> str:
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

    def get_all_with_vectors(self) -> list[dict]:
        return []

    def get_stale(self, max_age_days: int, min_access: int) -> list[dict]:
        return []

    def bulk_delete(self, entry_ids: list[str]) -> int:
        return 0

    def get_top_accessed(self, limit: int) -> list[dict]:
        return self.get_all()


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
            # Inline migrations for metadata columns.
            for col, default in [
                ("last_accessed REAL", "0"),
                ("access_count INTEGER", "0"),
                ("source TEXT", "'unknown'"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE memories_meta ADD COLUMN {col} DEFAULT {default}")
                except sqlite3.OperationalError:
                    pass  # Column already exists.
            conn.commit()
            conn.close()
        logger.info("Memory backend: fts5 (SQLite FTS5 at %s)", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def add(self, entry: str, entry_id: str | None = None, source: str = "unknown") -> str:
        eid = entry_id or _make_id(entry)
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO memories_meta "
                    "(entry_id, text, created_at, last_accessed, access_count, source) "
                    "VALUES (?, ?, ?, ?, 0, ?) "
                    "ON CONFLICT(entry_id) DO UPDATE SET text = excluded.text",
                    (eid, entry.strip(), now, now, source),
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
                    "SELECT m.entry_id, m.text, bm25(memories_fts) AS score "
                    "FROM memories_fts f "
                    "JOIN memories_meta m ON f.rowid = m.rowid "
                    "WHERE memories_fts MATCH ? "
                    "ORDER BY score "
                    "LIMIT ?",
                    (fts_query, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                logger.debug("FTS5 query failed, returning all memories")
                rows = []
            finally:
                conn.close()
        if not rows:
            return self.get_all()[:top_k]
        hits = [
            {"id": r[0], "text": r[1], "score": -r[2]}
            for r in rows
        ]
        # Track access for returned results.
        self._track_access([h["id"] for h in hits])
        return hits

    def _track_access(self, entry_ids: list[str]) -> None:
        if not entry_ids:
            return
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "UPDATE memories_meta SET last_accessed = ?, access_count = access_count + 1 "
                    "WHERE entry_id = ?",
                    [(now, eid) for eid in entry_ids],
                )
                conn.commit()
            finally:
                conn.close()

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
        now = time.time()
        cleaned = [(e.strip(), _make_id(e)) for e in entries if e.strip()]
        new_ids = [eid for _, eid in cleaned]
        with self._lock:
            conn = self._connect()
            try:
                # Delete entries no longer present.
                if new_ids:
                    placeholders = ",".join("?" for _ in new_ids)
                    conn.execute(
                        f"DELETE FROM memories_meta WHERE entry_id NOT IN ({placeholders})",
                        new_ids,
                    )
                else:
                    conn.execute("DELETE FROM memories_meta")
                # Upsert: insert new entries, update text for existing (preserve metadata).
                conn.executemany(
                    "INSERT INTO memories_meta "
                    "(entry_id, text, created_at, last_accessed, access_count, source) "
                    "VALUES (?, ?, ?, ?, 0, 'dreaming') "
                    "ON CONFLICT(entry_id) DO UPDATE SET text = excluded.text",
                    [(eid, text, now, now) for text, eid in cleaned],
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

    def get_all_with_vectors(self) -> list[dict]:
        return []  # FTS5 has no vectors.

    def get_stale(self, max_age_days: int, min_access: int) -> list[dict]:
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text, created_at, last_accessed, access_count, source "
                    "FROM memories_meta WHERE last_accessed < ? AND access_count < ?",
                    (cutoff, min_access),
                ).fetchall()
            finally:
                conn.close()
        return [
            {"id": r[0], "text": r[1], "created_at": r[2],
             "last_accessed": r[3], "access_count": r[4], "source": r[5]}
            for r in rows
        ]

    def bulk_delete(self, entry_ids: list[str]) -> int:
        if not entry_ids:
            return 0
        placeholders = ",".join("?" for _ in entry_ids)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"DELETE FROM memories_meta WHERE entry_id IN ({placeholders})",
                    entry_ids,
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def get_top_accessed(self, limit: int) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text FROM memories_meta "
                    "ORDER BY access_count DESC, last_accessed DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                conn.close()
        return [{"id": r[0], "text": r[1], "score": 1.0} for r in rows]


# ---------------------------------------------------------------------------
# Local vector backend (numpy cosine similarity, persisted in SQLite BLOBs)
# ---------------------------------------------------------------------------

LOCAL_VECTOR_DB_PATH = DATA_DIR / "local_vectors.db"


class LocalVectorBackend:
    """numpy cosine similarity over vectors stored in SQLite BLOBs.

    Requires an OpenAI-compatible embeddings endpoint (same config key as
    QdrantBackend) but does **not** require any external vector service.
    """

    def __init__(self, config: dict) -> None:
        self._embed_cfg = config.get("embeddings", {})
        self._db_path = LOCAL_VECTOR_DB_PATH
        self._lock = threading.Lock()

    def init(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS local_vectors ("
                "  entry_id TEXT PRIMARY KEY,"
                "  text TEXT NOT NULL,"
                "  vector BLOB NOT NULL,"
                "  created_at REAL NOT NULL"
                ")"
            )
            # Inline migrations for metadata columns.
            for col, default in [
                ("last_accessed REAL", "0"),
                ("access_count INTEGER", "0"),
                ("source TEXT", "'unknown'"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE local_vectors ADD COLUMN {col} DEFAULT {default}")
                except sqlite3.OperationalError:
                    pass  # Column already exists.
            conn.commit()
            conn.close()
        logger.info("Memory backend: local_vector (SQLite+numpy at %s)", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _vec_to_blob(vec: list[float]) -> bytes:
        import numpy as np
        return np.array(vec, dtype=np.float32).tobytes()

    def add(self, entry: str, entry_id: str | None = None, source: str = "unknown") -> str:
        eid = entry_id or _make_id(entry)
        vectors = _embed([entry], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        blob = self._vec_to_blob(vectors[0])
        t0 = time.time()
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO local_vectors "
                    "(entry_id, text, vector, created_at, last_accessed, access_count, source) "
                    "VALUES (?, ?, ?, ?, ?, 0, ?) "
                    "ON CONFLICT(entry_id) DO UPDATE SET "
                    "text = excluded.text, vector = excluded.vector",
                    (eid, entry.strip(), blob, now, now, source),
                )
                conn.commit()
            finally:
                conn.close()
        _log_interaction(t0, "local_vector_add", entry[:200], f"id={eid}", llm="local_vector")
        return eid

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        import numpy as np

        t0 = time.time()
        vectors = _embed([query], self._embed_cfg, task="query")
        if not vectors:
            logger.warning("LocalVector: embedding failed for query, falling back to get_all")
            return self.get_all()[:top_k]
        q_vec = np.array(vectors[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return self.get_all()[:top_k]
        q_vec = q_vec / q_norm

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text, vector FROM local_vectors ORDER BY created_at"
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        results: list[dict] = []
        for entry_id, text, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            score = float(np.dot(q_vec, vec / norm))
            if score >= threshold:
                results.append({"id": entry_id, "text": text, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        hits = results[:top_k]
        # Track access for returned results.
        self._track_access([h["id"] for h in hits])
        _log_interaction(
            t0, "local_vector_search", query[:200],
            f"{len(hits)} hits (top_k={top_k}, threshold={threshold})",
            llm="local_vector",
        )
        return hits

    def _track_access(self, entry_ids: list[str]) -> None:
        if not entry_ids:
            return
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "UPDATE local_vectors SET last_accessed = ?, access_count = access_count + 1 "
                    "WHERE entry_id = ?",
                    [(now, eid) for eid in entry_ids],
                )
                conn.commit()
            finally:
                conn.close()

    def get_all(self) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text FROM local_vectors ORDER BY created_at"
                ).fetchall()
            finally:
                conn.close()
        return [{"id": r[0], "text": r[1], "score": 1.0} for r in rows]

    def replace_all(self, entries: list[str]) -> None:
        t0 = time.time()
        entries = [e.strip() for e in entries if e.strip()]
        if not entries:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute("DELETE FROM local_vectors")
                    conn.commit()
                finally:
                    conn.close()
            _log_interaction(t0, "local_vector_replace_all", "0 entries", "cleared", llm="local_vector")
            return

        vectors = _embed(entries, self._embed_cfg)
        if not vectors or len(vectors) != len(entries):
            raise RuntimeError(
                f"Embedding mismatch: {len(entries)} entries but "
                f"{len(vectors) if vectors else 0} vectors"
            )

        now = time.time()
        rows = [
            (_make_id(entry), entry, self._vec_to_blob(vec), now, now)
            for entry, vec in zip(entries, vectors)
        ]
        new_ids = [r[0] for r in rows]
        with self._lock:
            conn = self._connect()
            try:
                # Delete entries no longer present.
                if new_ids:
                    placeholders = ",".join("?" for _ in new_ids)
                    conn.execute(
                        f"DELETE FROM local_vectors WHERE entry_id NOT IN ({placeholders})",
                        new_ids,
                    )
                else:
                    conn.execute("DELETE FROM local_vectors")
                # Upsert: insert new, update text+vector for existing (preserve metadata).
                conn.executemany(
                    "INSERT INTO local_vectors "
                    "(entry_id, text, vector, created_at, last_accessed, access_count, source) "
                    "VALUES (?, ?, ?, ?, ?, 0, 'dreaming') "
                    "ON CONFLICT(entry_id) DO UPDATE SET "
                    "text = excluded.text, vector = excluded.vector",
                    rows,
                )
                conn.commit()
            finally:
                conn.close()
        logger.info("LocalVector: replaced all entries (%d)", len(entries))
        _log_interaction(
            t0, "local_vector_replace_all", f"{len(entries)} entries",
            f"upserted {len(entries)} rows", llm="local_vector",
        )

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM local_vectors WHERE entry_id = ?", (entry_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM local_vectors").fetchone()[0]
            finally:
                conn.close()

    def stats(self) -> dict:
        return {
            "backend": "local_vector",
            "count": self.count(),
            "db_path": str(self._db_path),
        }

    def rebuild(self) -> None:
        """Drop and re-embed from MEMORIES.txt."""
        try:
            text = MEMORIES_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            text = ""
        entries = [l.strip() for l in text.splitlines() if l.strip()] if text else []
        self.replace_all(entries)
        logger.info("LocalVector: rebuilt index from MEMORIES.txt (%d entries)", len(entries))

    def get_all_with_vectors(self) -> list[dict]:
        import numpy as np
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text, vector, created_at, last_accessed, "
                    "access_count, source FROM local_vectors ORDER BY created_at"
                ).fetchall()
            finally:
                conn.close()
        results = []
        for r in rows:
            vec = np.frombuffer(r[2], dtype=np.float32).tolist()
            results.append({
                "id": r[0], "text": r[1], "vector": vec, "created_at": r[3],
                "last_accessed": r[4], "access_count": r[5], "source": r[6],
            })
        return results

    def get_stale(self, max_age_days: int, min_access: int) -> list[dict]:
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text, created_at, last_accessed, access_count, source "
                    "FROM local_vectors WHERE last_accessed < ? AND access_count < ?",
                    (cutoff, min_access),
                ).fetchall()
            finally:
                conn.close()
        return [
            {"id": r[0], "text": r[1], "created_at": r[2],
             "last_accessed": r[3], "access_count": r[4], "source": r[5]}
            for r in rows
        ]

    def bulk_delete(self, entry_ids: list[str]) -> int:
        if not entry_ids:
            return 0
        placeholders = ",".join("?" for _ in entry_ids)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"DELETE FROM local_vectors WHERE entry_id IN ({placeholders})",
                    entry_ids,
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def get_top_accessed(self, limit: int) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text FROM local_vectors "
                    "ORDER BY access_count DESC, last_accessed DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                conn.close()
        return [{"id": r[0], "text": r[1], "score": 1.0} for r in rows]


# ---------------------------------------------------------------------------
# Qdrant backend (vector semantic search)
# ---------------------------------------------------------------------------

class QdrantBackend:
    """Qdrant vector DB with embedding-based semantic search."""

    def __init__(self, config: dict) -> None:
        self._qdrant_cfg = config.get("qdrant", {})
        self._embed_cfg = config.get("embeddings", {})
        self._url = self._qdrant_cfg.get("url", "http://localhost:6333")
        self._api_key = self._qdrant_cfg.get("api_key", "") or None
        self._collection = self._qdrant_cfg.get("collection", "wintermute_memories")
        self._dimensions = self._embed_cfg.get("dimensions", 1536)
        self._client: Any = None
        self._lock = threading.Lock()

    def init(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from urllib.parse import urlparse

        # Parse URL into host/port/https components — qdrant-client's url=
        # parameter doesn't work reliably with HTTPS endpoints.
        parsed = urlparse(self._url)
        use_https = parsed.scheme == "https"
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if use_https else 6333)

        self._client = QdrantClient(
            host=host, port=port, https=use_https,
            api_key=self._api_key, timeout=30,
            prefer_grpc=False,
        )
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
        # Create payload indexes for efficient filtered queries (dreaming, stale detection).
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtered queries.

        Indexes: last_accessed (range), access_count (range), source (keyword),
        created_at (range).  Idempotent — Qdrant silently ignores duplicates.
        """
        try:
            from qdrant_client.models import PayloadSchemaType
            for field, schema in [
                ("last_accessed", PayloadSchemaType.FLOAT),
                ("access_count", PayloadSchemaType.INTEGER),
                ("source", PayloadSchemaType.KEYWORD),
                ("created_at", PayloadSchemaType.FLOAT),
            ]:
                try:
                    self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception:
                    pass  # Index may already exist.
            logger.debug("Qdrant: payload indexes ensured")
        except Exception:
            logger.debug("Qdrant: payload index creation skipped", exc_info=True)

    def recommend(
        self,
        positive_ids: list[str],
        negative_ids: list[str] | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Use Qdrant's recommend API to find memories related to positive
        examples but distant from negative examples.

        Returns list of dicts with id, text, score.
        Falls back to empty list if recommend is unavailable.
        """
        try:
            from qdrant_client.models import RecommendRequest
            with self._lock:
                results = self._client.recommend(
                    collection_name=self._collection,
                    positive=positive_ids,
                    negative=negative_ids or [],
                    limit=limit,
                    score_threshold=score_threshold,
                )
            return [
                {"id": str(r.id), "text": r.payload.get("text", ""), "score": r.score}
                for r in results
            ]
        except Exception as exc:
            logger.debug("Qdrant recommend() failed: %s", exc)
            return []

    def create_snapshot(self) -> str:
        """Create a Qdrant collection snapshot for rollback.

        Returns the snapshot name, or empty string on failure.
        """
        try:
            with self._lock:
                info = self._client.create_snapshot(
                    collection_name=self._collection,
                )
            name = getattr(info, "name", str(info)) if info else ""
            logger.info("Qdrant: snapshot created: %s", name)
            return name
        except Exception as exc:
            logger.warning("Qdrant: snapshot creation failed: %s", exc)
            return ""

    def search_neighbors_batch(
        self,
        entry_ids: list[str],
        limit: int = 20,
        score_threshold: float = 0.0,
    ) -> dict[str, list[dict]]:
        """Find nearest neighbors for multiple entries in a single Qdrant call.

        Returns a dict mapping each entry_id to its list of neighbor dicts
        (id, text, score).  Uses search_batch for efficiency: O(n*k) instead
        of the O(n^2) full pairwise matrix.
        """
        try:
            from qdrant_client.models import SearchRequest
            # Fetch vectors for the requested entries.
            with self._lock:
                points = self._client.retrieve(
                    collection_name=self._collection,
                    ids=entry_ids,
                    with_vectors=True,
                    with_payload=True,
                )
            if not points:
                return {}

            id_to_vec = {str(p.id): p.vector for p in points if p.vector}

            requests = []
            order = []  # Track which id each request corresponds to.
            for eid in entry_ids:
                vec = id_to_vec.get(eid)
                if vec is None:
                    continue
                requests.append(SearchRequest(
                    vector=vec,
                    limit=limit + 1,  # +1 to exclude self.
                    score_threshold=score_threshold,
                    with_payload=True,
                ))
                order.append(eid)

            if not requests:
                return {}

            with self._lock:
                batch_results = self._client.search_batch(
                    collection_name=self._collection,
                    requests=requests,
                )

            result: dict[str, list[dict]] = {}
            for eid, hits in zip(order, batch_results):
                neighbors = [
                    {"id": str(h.id), "text": h.payload.get("text", ""),
                     "score": h.score, "source": h.payload.get("source", "unknown")}
                    for h in hits
                    if str(h.id) != eid  # Exclude self.
                ][:limit]
                result[eid] = neighbors
            return result
        except Exception as exc:
            logger.debug("Qdrant search_neighbors_batch failed: %s", exc)
            return {}

    def add(self, entry: str, entry_id: str | None = None, source: str = "unknown") -> str:
        from qdrant_client.models import PointStruct

        eid = entry_id or _make_id(entry)
        vectors = _embed([entry], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        t0 = time.time()
        now = time.time()
        # Try to preserve existing metadata (access_count, last_accessed, source, created_at).
        existing_payload: dict = {}
        try:
            with self._lock:
                pts = self._client.retrieve(
                    collection_name=self._collection,
                    ids=[eid], with_payload=True, with_vectors=False,
                )
            if pts:
                existing_payload = pts[0].payload or {}
        except Exception as exc:  # noqa: BLE001
            # Point may not exist yet — use defaults.  Log unexpected failures.
            if "not found" not in str(exc).lower():
                logger.debug("Failed to retrieve existing metadata for point %s: %s", eid, exc)
        payload = {
            "text": entry.strip(),
            "created_at": existing_payload.get("created_at", now),
            "last_accessed": existing_payload.get("last_accessed", now),
            "access_count": existing_payload.get("access_count", 0),
            "source": existing_payload.get("source", source),
        }
        with self._lock:
            self._client.upsert(
                collection_name=self._collection,
                points=[PointStruct(
                    id=eid,
                    vector=vectors[0],
                    payload=payload,
                )],
            )
        _log_interaction(t0, "qdrant_add", entry[:200], f"id={eid}", llm="qdrant")
        return eid

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        vectors = _embed([query], self._embed_cfg, task="query")
        if not vectors:
            logger.warning("Qdrant: embedding failed for query, falling back to get_all")
            return self.get_all()[:top_k]
        t0 = time.time()
        with self._lock:
            results = self._client.query_points(
                collection_name=self._collection,
                query=vectors[0],
                limit=top_k,
                score_threshold=threshold,
            ).points
        hits = [
            {"id": str(r.id), "text": r.payload.get("text", ""), "score": r.score}
            for r in results
        ]
        # Track access for returned results.
        self._track_access([h["id"] for h in hits])
        _log_interaction(
            t0, "qdrant_search", query[:200],
            f"{len(hits)} hits (top_k={top_k}, threshold={threshold})",
            llm="qdrant",
        )
        return hits

    def _track_access(self, entry_ids: list[str]) -> None:
        if not entry_ids:
            return
        now = time.time()
        for eid in entry_ids:
            try:
                with self._lock:
                    self._client.set_payload(
                        collection_name=self._collection,
                        payload={"last_accessed": now},
                        points=[eid],
                    )
            except Exception:  # noqa: BLE001
                pass  # Best-effort access tracking.

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

        t0 = time.time()
        entries = [e.strip() for e in entries if e.strip()]
        if not entries:
            with self._lock:
                # Delete all points.
                self._client.delete_collection(self._collection)
            self.init()  # Recreate empty collection.
            _log_interaction(t0, "qdrant_replace_all", "0 entries", "collection cleared", llm="qdrant")
            return

        # Batch embed.
        vectors = _embed(entries, self._embed_cfg)
        if not vectors or len(vectors) != len(entries):
            raise RuntimeError(
                f"Embedding mismatch: {len(entries)} entries but {len(vectors) if vectors else 0} vectors"
            )

        now = time.time()
        new_ids = [_make_id(entry) for entry in entries]

        # Read existing metadata before replacing, so we can carry it over.
        # Paginate to handle collections larger than any single scroll page.
        existing_meta: dict[str, dict] = {}
        try:
            offset = None
            while True:
                with self._lock:
                    result = self._client.scroll(
                        collection_name=self._collection,
                        limit=1000,
                        offset=offset,
                        with_payload=True,
                    )
                if not result:
                    break
                for p in result[0]:
                    existing_meta[str(p.id)] = p.payload or {}
                offset = result[1] if len(result) > 1 else None
                if not offset:
                    break
        except Exception:  # noqa: BLE001
            pass  # Fresh collection or scroll failed — use defaults.

        points = []
        for entry, vec, eid in zip(entries, vectors, new_ids):
            old = existing_meta.get(eid, {})
            points.append(PointStruct(
                id=eid,
                vector=vec,
                payload={
                    "text": entry,
                    "created_at": old.get("created_at", now),
                    "last_accessed": old.get("last_accessed", now),
                    "access_count": old.get("access_count", 0),
                    "source": old.get("source", "dreaming"),
                },
            ))

        from qdrant_client.models import Filter, HasIdCondition
        # Delete points no longer present (avoids delete_collection race window).
        with self._lock:
            if new_ids:
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=Filter(
                        must_not=[HasIdCondition(has_id=new_ids)],
                    ),
                )
            else:
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
        _log_interaction(
            t0, "qdrant_replace_all", f"{len(entries)} entries",
            f"upserted {len(points)} points", llm="qdrant",
        )

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

    def get_all_with_vectors(self) -> list[dict]:
        with self._lock:
            result = self._client.scroll(
                collection_name=self._collection,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
        points = result[0] if result else []
        return [
            {
                "id": str(p.id),
                "text": p.payload.get("text", ""),
                "vector": p.vector,
                "created_at": p.payload.get("created_at", 0),
                "last_accessed": p.payload.get("last_accessed", 0),
                "access_count": p.payload.get("access_count", 0),
                "source": p.payload.get("source", "unknown"),
            }
            for p in points
        ]

    def get_stale(self, max_age_days: int, min_access: int) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, Range
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            result = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="last_accessed", range=Range(lt=cutoff)),
                    FieldCondition(key="access_count", range=Range(lt=min_access)),
                ]),
                limit=10000,
                with_payload=True,
            )
        points = result[0] if result else []
        return [
            {
                "id": str(p.id), "text": p.payload.get("text", ""),
                "created_at": p.payload.get("created_at", 0),
                "last_accessed": p.payload.get("last_accessed", 0),
                "access_count": p.payload.get("access_count", 0),
                "source": p.payload.get("source", "unknown"),
            }
            for p in points
        ]

    def bulk_delete(self, entry_ids: list[str]) -> int:
        if not entry_ids:
            return 0
        from qdrant_client.models import PointIdsList
        with self._lock:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=entry_ids),
            )
        return len(entry_ids)

    def get_top_accessed(self, limit: int) -> list[dict]:
        with self._lock:
            result = self._client.scroll(
                collection_name=self._collection,
                limit=10000,
                with_payload=True,
            )
        points = result[0] if result else []
        # Sort client-side (Qdrant scroll doesn't support ORDER BY).
        points.sort(
            key=lambda p: (p.payload.get("access_count", 0), p.payload.get("last_accessed", 0)),
            reverse=True,
        )
        return [
            {"id": str(p.id), "text": p.payload.get("text", ""), "score": 1.0}
            for p in points[:limit]
        ]


# ---------------------------------------------------------------------------
# Embedding helper (used by QdrantBackend)
# ---------------------------------------------------------------------------

def _embed(texts: list[str], embed_cfg: dict, task: str = "document") -> list[list[float]]:
    """Call an OpenAI-compatible embeddings endpoint.

    Uses httpx (sync) — callers in async context should run via executor.

    *task* is ``"query"`` (search) or ``"document"`` (upsert/index).
    Prefix is auto-detected for known models (e.g. EmbeddingGemma) or
    can be overridden via ``query_prefix`` / ``document_prefix`` in config.
    """
    import httpx

    endpoint = embed_cfg.get("endpoint", "").rstrip("/")
    model = embed_cfg.get("model", "text-embedding-3-small")
    api_key = embed_cfg.get("api_key", "") or None
    if not endpoint:
        raise RuntimeError("memory.embeddings.endpoint is not configured")

    # --- task-type prefix handling ---
    _AUTO_PREFIXES: dict[str, dict[str, str]] = {
        "gemma": {"query": "search_query: ", "document": "search_document: "},
    }
    query_prefix = embed_cfg.get("query_prefix", "")
    document_prefix = embed_cfg.get("document_prefix", "")
    if not query_prefix and not document_prefix:
        model_lower = model.lower()
        for key, prefixes in _AUTO_PREFIXES.items():
            if key in model_lower:
                query_prefix = prefixes["query"]
                document_prefix = prefixes["document"]
                break
    prefix = query_prefix if task == "query" else document_prefix
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]

    url = f"{endpoint}/embeddings"
    payload = {"input": texts, "model": model}
    # Only send dimensions if explicitly opted in — most non-OpenAI
    # endpoints reject this parameter.
    if embed_cfg.get("send_dimensions"):
        dimensions = embed_cfg.get("dimensions")
        if dimensions:
            payload["dimensions"] = dimensions

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    max_retries = 3
    t0 = time.time()
    status = "ok"
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            # Sort by index to preserve order.
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            result = [item["embedding"] for item in items]
            output_summary = f"{len(result)} vectors, {len(result[0])} dims" if result else "empty"
            if attempt > 1:
                status = f"ok (retry {attempt - 1})"
            input_summary = f"{len(texts)} texts, model={model}"
            _log_interaction(t0, "embedding", input_summary, output_summary, status, llm=model)
            return result
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout,
                httpx.WriteTimeout, httpx.PoolTimeout, ConnectionError, OSError) as exc:
            # Only retry on transient errors (5xx, timeouts, connection issues).
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                status = f"error: {exc}"
                output_summary = status
                input_summary = f"{len(texts)} texts, model={model}"
                _log_interaction(t0, "embedding", input_summary, output_summary, status, llm=model)
                raise
            last_exc = exc
            if attempt < max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
                logger.warning("Embedding attempt %d/%d failed (%s), retrying in %.1fs",
                               attempt, max_retries, exc, backoff)
                time.sleep(backoff)
            else:
                status = f"error: {exc}"
                output_summary = status
                input_summary = f"{len(texts)} texts, model={model}"
                _log_interaction(t0, "embedding", input_summary, output_summary, status, llm=model)
                raise
        except Exception as exc:
            status = f"error: {exc}"
            output_summary = status
            input_summary = f"{len(texts)} texts, model={model}"
            _log_interaction(t0, "embedding", input_summary, output_summary, status, llm=model)
            raise
    raise last_exc  # type: ignore[misc]  # unreachable, satisfies type checker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    """Deterministic UUID from text content (SHA-256 → UUID v5-style)."""
    h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    # Format as UUID: 8-4-4-4-12
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _log_interaction(timestamp: float, action: str, input_text: str,
                     output_text: str, status: str = "ok",
                     llm: str = "") -> None:
    """Log a memory store interaction to the database interaction_log."""
    try:
        from wintermute.infra import database
        database.save_interaction_log(
            timestamp, action, "system:memory_store",
            llm, input_text[:2000], output_text[:2000], status,
        )
    except Exception:  # noqa: BLE001
        pass  # Never let logging failures break memory operations.


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
    backend_name = config.get("backend", "local_vector")

    try:
        if backend_name == "fts5":
            _backend = FTS5Backend()
        elif backend_name == "local_vector":
            _backend = LocalVectorBackend(config)
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
    if backend_name in ("fts5", "local_vector", "qdrant"):
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
    """Search memories by relevance.  Uses configured defaults for top_k/threshold.

    Returns an empty list on transient errors (network, embedding failures)
    so callers can degrade gracefully.
    """
    if top_k is None:
        top_k = get_top_k()
    if threshold is None:
        threshold = get_threshold()
    try:
        return _backend.search(query, top_k, threshold)
    except Exception as exc:
        logger.warning("memory_store.search failed, returning empty results: %s", exc)
        return []


def add(entry: str, entry_id: str | None = None, source: str = "unknown") -> str:
    return _backend.add(entry, entry_id, source=source)


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


def get_all_with_vectors() -> list[dict]:
    return _backend.get_all_with_vectors()


def get_stale(max_age_days: int, min_access: int) -> list[dict]:
    return _backend.get_stale(max_age_days, min_access)


def bulk_delete(entry_ids: list[str]) -> int:
    return _backend.bulk_delete(entry_ids)


def get_top_accessed(limit: int) -> list[dict]:
    return _backend.get_top_accessed(limit)


def recommend(
    positive_ids: list[str],
    negative_ids: list[str] | None = None,
    limit: int = 10,
    score_threshold: float | None = None,
) -> list[dict]:
    """Use Qdrant recommend API if available, else return empty list."""
    if isinstance(_backend, QdrantBackend):
        return _backend.recommend(positive_ids, negative_ids, limit, score_threshold)
    return []


def is_qdrant_backend() -> bool:
    """True when the active backend is QdrantBackend."""
    return isinstance(_backend, QdrantBackend)


def create_snapshot() -> str:
    """Create a Qdrant snapshot. Returns snapshot name or empty string."""
    if isinstance(_backend, QdrantBackend):
        return _backend.create_snapshot()
    return ""


def search_neighbors_batch(
    entry_ids: list[str],
    limit: int = 20,
    score_threshold: float = 0.0,
) -> dict[str, list[dict]]:
    """Find nearest neighbors for multiple entries via Qdrant search_batch."""
    if isinstance(_backend, QdrantBackend):
        return _backend.search_neighbors_batch(entry_ids, limit, score_threshold)
    return {}


def get_embed_config() -> dict:
    """Return the embeddings config dict (for external callers needing to embed)."""
    return _config.get("embeddings", {})


def is_vector_enabled() -> bool:
    """True when the active backend is not flat_file (and has been initialized)."""
    if _backend is None:
        return False
    return not isinstance(_backend, FlatFileBackend)


def get_top_k() -> int:
    return _config.get("top_k", 10)


def get_threshold() -> float:
    return _config.get("score_threshold", 0.3)
