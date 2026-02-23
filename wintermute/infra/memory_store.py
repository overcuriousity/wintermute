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

    def add(self, entry: str, entry_id: str | None = None) -> str:
        eid = entry_id or _make_id(entry)
        vectors = _embed([entry], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        blob = self._vec_to_blob(vectors[0])
        t0 = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO local_vectors "
                    "(entry_id, text, vector, created_at) VALUES (?, ?, ?, ?)",
                    (eid, entry.strip(), blob, time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        _log_interaction(t0, "local_vector_add", entry[:200], f"id={eid}", llm="local_vector")
        return eid

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        import numpy as np

        t0 = time.time()
        vectors = _embed([query], self._embed_cfg)
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
        _log_interaction(
            t0, "local_vector_search", query[:200],
            f"{len(hits)} hits (top_k={top_k}, threshold={threshold})",
            llm="local_vector",
        )
        return hits

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
            (_make_id(entry), entry, self._vec_to_blob(vec), now)
            for entry, vec in zip(entries, vectors)
        ]
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM local_vectors")
                conn.executemany(
                    "INSERT INTO local_vectors (entry_id, text, vector, created_at) "
                    "VALUES (?, ?, ?, ?)",
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

    def add(self, entry: str, entry_id: str | None = None) -> str:
        from qdrant_client.models import PointStruct

        eid = entry_id or _make_id(entry)
        vectors = _embed([entry], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        t0 = time.time()
        with self._lock:
            self._client.upsert(
                collection_name=self._collection,
                points=[PointStruct(
                    id=eid,
                    vector=vectors[0],
                    payload={"text": entry.strip(), "created_at": time.time()},
                )],
            )
        _log_interaction(t0, "qdrant_add", entry[:200], f"id={eid}", llm="qdrant")
        return eid

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        vectors = _embed([query], self._embed_cfg)
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
        _log_interaction(
            t0, "qdrant_search", query[:200],
            f"{len(hits)} hits (top_k={top_k}, threshold={threshold})",
            llm="qdrant",
        )
        return hits

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
    api_key = embed_cfg.get("api_key", "") or None
    if not endpoint:
        raise RuntimeError("memory.embeddings.endpoint is not configured")

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

    t0 = time.time()
    status = "ok"
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to preserve order.
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        result = [item["embedding"] for item in items]
        output_summary = f"{len(result)} vectors, {len(result[0])} dims" if result else "empty"
    except Exception as exc:
        status = f"error: {exc}"
        output_summary = status
        raise
    finally:
        input_summary = f"{len(texts)} texts, model={model}"
        _log_interaction(t0, "embedding", input_summary, output_summary, status, llm=model)
    return result


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
