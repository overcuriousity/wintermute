"""
Ranked memory retrieval for Wintermute.

Provides two embedding-based backends for memory storage and retrieval:
  - local_vector — SQLite + numpy cosine similarity (default)
  - qdrant       — Qdrant vector DB with embedding-based semantic search

An OpenAI-compatible embeddings endpoint is **required**.

Module-level singleton pattern (like database.py, prompt_assembler.py).
Call ``init(config)`` once at startup; all other functions use the active backend.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time

from typing import Any, Protocol, runtime_checkable

from wintermute.infra.llm_utils import (
    embed as _embed,

    log_store_interaction as _log_interaction_impl,
    make_content_id as _make_id,
)

from wintermute.infra.paths import DATA_DIR

logger = logging.getLogger(__name__)

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
    def search(self, query: str, top_k: int, threshold: float,
               *, bump_access: bool = True) -> list[dict]: ...
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
    def get_by_source(self, source: str, limit: int = 50, bump_access: bool = True) -> list[dict]: ...
    def track_access(self, entry_ids: list[str]) -> None: ...
    def promote_source(self, entry_id: str, new_source: str) -> None: ...


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

    def search(self, query: str, top_k: int, threshold: float,
               *, bump_access: bool = True) -> list[dict]:
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
        if bump_access:
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
                # Use SET-DIFFERENCE delete in batches to stay under SQLite's
                # parameter limit (~999 variables per statement).
                if new_ids:
                    cur = conn.execute("SELECT entry_id FROM local_vectors")
                    existing_ids = {row[0] for row in cur.fetchall()}
                    new_ids_set = set(new_ids)
                    to_delete = list(existing_ids - new_ids_set)
                    for i in range(0, len(to_delete), 900):
                        batch = to_delete[i : i + 900]
                        placeholders = ",".join("?" for _ in batch)
                        conn.execute(
                            f"DELETE FROM local_vectors WHERE entry_id IN ({placeholders})",
                            batch,
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
        """Re-embed all entries, preserving entry_id and metadata."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text FROM local_vectors"
                ).fetchall()
            finally:
                conn.close()
        if not rows:
            return
        texts = [r[1] for r in rows]
        vectors = _embed(texts, self._embed_cfg)
        if not vectors or len(vectors) != len(rows):
            raise RuntimeError("Embedding mismatch during rebuild")
        with self._lock:
            conn = self._connect()
            try:
                for (eid, _text), vec in zip(rows, vectors):
                    conn.execute(
                        "UPDATE local_vectors SET vector = ? WHERE entry_id = ?",
                        (self._vec_to_blob(vec), eid),
                    )
                conn.commit()
            finally:
                conn.close()
        logger.info("LocalVector: rebuilt vectors in place (%d entries)", len(rows))

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

    def get_by_source(self, source: str, limit: int = 50, bump_access: bool = True) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, text, created_at, last_accessed, access_count, source "
                    "FROM local_vectors WHERE source = ? ORDER BY created_at DESC LIMIT ?",
                    (source, limit),
                ).fetchall()
            finally:
                conn.close()
        hits = [
            {"id": r[0], "text": r[1], "created_at": r[2],
             "last_accessed": r[3], "access_count": r[4], "source": r[5]}
            for r in rows
        ]
        if bump_access:
            self._track_access([h["id"] for h in hits])
        return hits

    def track_access(self, entry_ids: list[str]) -> None:
        self._track_access(entry_ids)

    def promote_source(self, entry_id: str, new_source: str) -> None:
        """Upgrade source tag on *entry_id* if *new_source* has higher priority."""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT source FROM local_vectors WHERE entry_id = ?",
                        (entry_id,),
                    ).fetchone()
                    if row and _source_rank(new_source) < _source_rank(row[0]):
                        conn.execute(
                            "UPDATE local_vectors SET source = ? WHERE entry_id = ?",
                            (new_source, entry_id),
                        )
                        conn.commit()
                finally:
                    conn.close()
        except Exception:
            logger.debug("Source promotion failed for %s", entry_id, exc_info=True)


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
            from qdrant_client.models import QueryRequest
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
                requests.append(QueryRequest(
                    query=vec,
                    limit=limit + 1,  # +1 to exclude self.
                    score_threshold=score_threshold,
                    with_payload=True,
                ))
                order.append(eid)

            if not requests:
                return {}

            with self._lock:
                batch_results = self._client.query_batch_points(
                    collection_name=self._collection,
                    requests=requests,
                )

            result: dict[str, list[dict]] = {}
            for eid, resp in zip(order, batch_results):
                neighbors = [
                    {"id": str(h.id), "text": h.payload.get("text", ""),
                     "score": h.score, "source": h.payload.get("source", "unknown")}
                    for h in resp.points
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
        # Retrieve existing metadata and upsert under a single lock acquisition to
        # prevent a concurrent add() from overwriting metadata with stale data.
        # retrieve() returns [] for a missing point — no exception for "not found".
        with self._lock:
            existing_payload: dict = {}
            pts = self._client.retrieve(
                collection_name=self._collection,
                ids=[eid], with_payload=True, with_vectors=False,
            )
            if pts:
                existing_payload = pts[0].payload or {}
            payload = {
                "text": entry.strip(),
                "created_at": existing_payload.get("created_at", now),
                "last_accessed": existing_payload.get("last_accessed", now),
                "access_count": existing_payload.get("access_count", 0),
                "source": existing_payload.get("source", source),
            }
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

    def search(self, query: str, top_k: int, threshold: float,
               *, bump_access: bool = True) -> list[dict]:
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
        if bump_access:
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
        try:
            with self._lock:
                # Batch-retrieve current access counts in one call.
                pts = self._client.retrieve(
                    collection_name=self._collection,
                    ids=entry_ids,
                    with_payload=["access_count"],
                )
                count_map = {
                    str(p.id): int((p.payload or {}).get("access_count") or 0)
                    for p in pts
                }
                for pt in pts:
                    eid = str(pt.id)
                    current_count = count_map.get(eid, 0)
                    try:
                        self._client.set_payload(
                            collection_name=self._collection,
                            payload={"last_accessed": now, "access_count": current_count + 1},
                            points=[eid],
                        )
                    except Exception:  # noqa: BLE001
                        pass  # Best-effort per-point.
        except Exception:  # noqa: BLE001
            pass  # Best-effort access tracking.

    def get_all(self) -> list[dict]:
        points = []
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
            points.extend(result[0])
            offset = result[1] if len(result) > 1 else None
            if offset is None:
                break
        return [
            {"id": str(p.id), "text": p.payload.get("text", ""), "score": 1.0}
            for p in points
        ]

    def replace_all(self, entries: list[str]) -> None:
        from qdrant_client.models import Filter, HasIdCondition, PointStruct

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

        # Hold a single lock for the entire retrieve-metadata → delete → upsert
        # critical section so no concurrent add() / _track_access() can interleave
        # and have its metadata overwritten by stale values from existing_meta.
        with self._lock:
            # Fetch existing metadata only for the incoming IDs via batched
            # retrieve() — cheaper than scrolling the whole collection.
            existing_meta: dict[str, dict] = {}
            try:
                for i in range(0, len(new_ids), 256):
                    pts = self._client.retrieve(
                        collection_name=self._collection,
                        ids=new_ids[i : i + 256],
                        with_payload=True,
                        with_vectors=False,
                    )
                    for p in pts or []:
                        existing_meta[str(p.id)] = p.payload or {}
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Qdrant: failed to read existing metadata in replace_all; "
                    "aborting to avoid metadata loss",
                    exc_info=True,
                )
                raise

            points = [
                PointStruct(
                    id=eid,
                    vector=vec,
                    payload={
                        "text": entry,
                        "created_at": existing_meta.get(eid, {}).get("created_at", now),
                        "last_accessed": existing_meta.get(eid, {}).get("last_accessed", now),
                        "access_count": existing_meta.get(eid, {}).get("access_count", 0),
                        "source": existing_meta.get(eid, {}).get("source", "dreaming"),
                    },
                )
                for entry, vec, eid in zip(entries, vectors, new_ids)
            ]

            # Delete points no longer present (avoids delete_collection race window).
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
                self._client.upsert(
                    collection_name=self._collection,
                    points=points[i : i + 100],
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
        """Re-embed all entries, preserving point IDs and metadata."""
        from qdrant_client.models import PointStruct

        all_entries = self.get_all_with_vectors()
        if not all_entries:
            return
        texts = [e["text"] for e in all_entries]
        vectors = _embed(texts, self._embed_cfg)
        if not vectors or len(vectors) != len(all_entries):
            raise RuntimeError("Embedding mismatch during rebuild")
        # Upsert with new vectors but preserve all existing payload.
        points = [
            PointStruct(
                id=e["id"],
                vector=vec,
                payload={
                    "text": e["text"],
                    "created_at": e.get("created_at", 0),
                    "last_accessed": e.get("last_accessed", 0),
                    "access_count": e.get("access_count", 0),
                    "source": e.get("source", "unknown"),
                },
            )
            for e, vec in zip(all_entries, vectors)
        ]
        with self._lock:
            for i in range(0, len(points), 100):
                self._client.upsert(
                    collection_name=self._collection,
                    points=points[i : i + 100],
                )
        logger.info("Qdrant: rebuilt vectors in place (%d entries)", len(all_entries))

    def get_all_with_vectors(self) -> list[dict]:
        points = []
        offset = None
        while True:
            with self._lock:
                result = self._client.scroll(
                    collection_name=self._collection,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
            if not result:
                break
            points.extend(result[0])
            offset = result[1] if len(result) > 1 else None
            if offset is None:
                break
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
        scroll_filter = Filter(must=[
            FieldCondition(key="last_accessed", range=Range(lt=cutoff)),
            FieldCondition(key="access_count", range=Range(lt=min_access)),
        ])
        points = []
        offset = None
        while True:
            with self._lock:
                result = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                )
            if not result:
                break
            points.extend(result[0])
            offset = result[1] if len(result) > 1 else None
            if offset is None:
                break
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

    def get_by_source(self, source: str, limit: int = 50, bump_access: bool = True) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        scroll_filter = Filter(must=[
            FieldCondition(key="source", match=MatchValue(value=source)),
        ])
        points: list = []
        offset = None
        while len(points) < limit:
            with self._lock:
                result = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=scroll_filter,
                    limit=min(limit - len(points), 1000),
                    offset=offset,
                    with_payload=True,
                )
            if not result:
                break
            points.extend(result[0])
            offset = result[1] if len(result) > 1 else None
            if offset is None:
                break
        hits = sorted(
            [
                {"id": str(p.id), "text": p.payload.get("text", ""),
                 "created_at": p.payload.get("created_at", 0),
                 "last_accessed": p.payload.get("last_accessed", 0),
                 "access_count": p.payload.get("access_count", 0),
                 "source": p.payload.get("source", "unknown")}
                for p in points
            ],
            key=lambda h: h["created_at"],
            reverse=True,
        )[:limit]
        if bump_access:
            self._track_access([h["id"] for h in hits])
        return hits

    def track_access(self, entry_ids: list[str]) -> None:
        self._track_access(entry_ids)

    def promote_source(self, entry_id: str, new_source: str) -> None:
        """Upgrade source tag on *entry_id* if *new_source* has higher priority."""
        try:
            with self._lock:
                pts = self._client.retrieve(
                    collection_name=self._collection,
                    ids=[entry_id], with_payload=True, with_vectors=False,
                )
                if pts:
                    existing_source = pts[0].payload.get("source", "unknown")
                    if _source_rank(new_source) < _source_rank(existing_source):
                        self._client.set_payload(
                            collection_name=self._collection,
                            payload={"source": new_source},
                            points=[entry_id],
                        )
        except Exception:
            logger.debug("Source promotion failed for %s", entry_id, exc_info=True)


# ---------------------------------------------------------------------------
# Helpers — thin wrappers around llm_utils shared functions
# ---------------------------------------------------------------------------

def _log_interaction(timestamp: float, action: str, input_text: str,
                     output_text: str, status: str = "ok",
                     llm: str = "") -> None:
    """Log a memory store interaction (delegates to llm_utils)."""
    _log_interaction_impl(timestamp, action, input_text, output_text,
                          status, llm=llm, session="system:memory_store")


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

def init(config: dict) -> None:
    """Select and initialize the memory backend.

    Requires an OpenAI-compatible embeddings endpoint.  The legacy ``fts5``
    backend has been removed — if ``backend: "fts5"`` is still present in
    ``config.yaml``, startup will fail with a clear migration message.
    """
    global _backend, _config
    _config = dict(config)

    embeddings_cfg = config.get("embeddings", {})
    has_embeddings = bool(embeddings_cfg.get("endpoint"))
    backend_name = config.get("backend", "local_vector")

    # ── Fail-fast for removed FTS5 backend ──────────────────────────────
    if backend_name == "fts5":
        raise ValueError(
            "The 'fts5' memory backend has been removed. "
            "An embedding-based backend (local_vector or qdrant) is now required.\n"
            "  1. Configure memory.embeddings.endpoint in config.yaml "
            "(any OpenAI-compatible /v1/embeddings endpoint).\n"
            "  2. Set memory.backend to 'local_vector' (default) or 'qdrant'.\n"
            "  Existing FTS5 memories will NOT be lost — the old "
            "data/memory_index.db file is preserved and can be exported manually."
        )

    if backend_name not in ("local_vector", "qdrant"):
        raise ValueError(
            f"Unknown memory backend {backend_name!r}. "
            f"Supported backends: local_vector, qdrant"
        )

    if not has_embeddings:
        raise ValueError(
            "memory.embeddings.endpoint is required. "
            "Configure an OpenAI-compatible /v1/embeddings endpoint in config.yaml.\n"
            "  Example:\n"
            "    memory:\n"
            "      embeddings:\n"
            "        endpoint: \"http://localhost:8080/v1\"\n"
            "        model: \"text-embedding-3-small\"\n"
            "        dimensions: 1536"
        )

    if backend_name == "local_vector":
        _backend = LocalVectorBackend(config)
    elif backend_name == "qdrant":
        _backend = QdrantBackend(config)
    _backend.init()

    logger.info("Memory backend initialized: %s (%d entries)", backend_name, _backend.count())


def search(query: str, top_k: int | None = None, threshold: float | None = None,
           *, bump_access: bool = True) -> list[dict]:
    """Search memories by relevance.  Uses configured defaults for top_k/threshold.

    Set *bump_access* to False for internal lookups (e.g. dedup) that should
    not inflate access statistics.

    Returns an empty list on transient errors (network, embedding failures)
    so callers can degrade gracefully.
    """
    if top_k is None:
        top_k = get_top_k()
    if threshold is None:
        threshold = get_threshold()
    try:
        return _backend.search(query, top_k, threshold, bump_access=bump_access)
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


def get_dreaming_config() -> dict:
    """Return the dreaming sub-config (public accessor for cross-module use)."""
    return _config.get("dreaming", {})


def get_top_accessed(limit: int) -> list[dict]:
    return _backend.get_top_accessed(limit)


def track_access(entry_ids: list[str]) -> None:
    """Bump access counts for specific entries."""
    if _backend is not None:
        _backend.track_access(entry_ids)


def get_by_source(source: str, limit: int = 50, bump_access: bool = True) -> list[dict]:
    """Return memories filtered by source tag (e.g. 'dreaming_prediction').

    When *bump_access* is True (default), access counts are bumped for
    returned entries to feed the promotion pipeline.  Pass False for
    background checks that should not inflate access counts.
    """
    return _backend.get_by_source(source, limit, bump_access=bump_access)


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


# ---------------------------------------------------------------------------
# Async dedup-aware add
# ---------------------------------------------------------------------------

_DEDUP_SIMILARITY_THRESHOLD = 0.80

# Source priority for merge promotion — lower index = higher protection.
_SOURCE_PRIORITY = ("user_explicit", "dreaming_schema", "harvest", "unknown")


def _source_rank(s: str) -> int:
    """Return priority rank for a source tag (lower = higher priority)."""
    try:
        return _SOURCE_PRIORITY.index(s)
    except ValueError:
        return len(_SOURCE_PRIORITY)


def _promote_source(entry_id: str, new_source: str) -> None:
    """Promote the source tag on *entry_id* if *new_source* has higher priority.

    If the active backend implements a ``promote_source(entry_id, new_source)``
    method, this function delegates to it so that locking and storage-specific
    logic are handled internally. Backends that do not implement this method
    simply skip source promotion.
    """
    if _backend is not None and hasattr(_backend, "promote_source"):
        _backend.promote_source(entry_id, new_source)


async def add_with_dedup(entry: str, source: str = "unknown", *, pool=None) -> str:
    """Add a memory entry, merging with an existing near-duplicate if found.

    1. Search for similar entries above ``_DEDUP_SIMILARITY_THRESHOLD``.
    2. If a match is found and *pool* is provided, merge via LLM.
    3. Upsert the merged text under the existing entry_id.  Both
       LocalVectorBackend and QdrantBackend preserve metadata
       (created_at, access_count, last_accessed) on upsert.
    4. Promote the source tag if the new entry's source has higher
       priority (e.g. ``user_explicit`` > ``harvest``).
    5. If no match, fall back to plain ``add()``.

    Returns the entry_id of the new/merged entry.
    """
    import asyncio
    from wintermute.infra import prompt_loader

    hits = await asyncio.to_thread(
        search, entry, top_k=1, threshold=_DEDUP_SIMILARITY_THRESHOLD,
        bump_access=False,
    )
    if hits and pool is not None and pool.enabled:
        existing = hits[0]
        existing_text = existing.get("text", "")
        existing_id = existing.get("id", "")

        try:
            merge_prompt = prompt_loader.load(
                "MEMORY_MERGE_PROMPT.txt",
                entry_1=existing_text,
                entry_2=entry,
            )
            response = await pool.call(
                messages=[{"role": "user", "content": merge_prompt}],
                max_tokens_override=512,
            )
            merged = (response.content or "").strip()
            if merged:
                # Upsert under the existing entry_id.  Both backends'
                # ON CONFLICT / upsert logic preserves created_at,
                # access_count, and last_accessed — no delete needed.
                new_id = await asyncio.to_thread(
                    add, merged, entry_id=existing_id, source=source
                )
                # Promote source if the new entry has higher priority
                # (e.g. user_explicit merged into a harvest entry).
                await asyncio.to_thread(_promote_source, existing_id, source)
                logger.info(
                    "Memory dedup: merged existing %s (%.0f%% sim) → %s",
                    existing_id, hits[0].get("score", 0) * 100, new_id,
                )
                return new_id
        except Exception as exc:
            logger.warning("Memory dedup merge failed, adding as new: %s", exc)

    # No duplicate or merge failed — plain add.
    return await asyncio.to_thread(add, entry, source=source)


def search_neighbors_batch(
    entry_ids: list[str],
    limit: int = 20,
    score_threshold: float = 0.0,
) -> dict[str, list[dict]]:
    """Find nearest neighbors for multiple entries via Qdrant search_batch."""
    if isinstance(_backend, QdrantBackend):
        return _backend.search_neighbors_batch(entry_ids, limit, score_threshold)
    return {}


def is_vector_enabled() -> bool:
    """True when an embedding-based vector backend is active.

    Returns False before ``init()`` or if initialization failed.
    After successful init, always True since all supported backends
    are embedding-based (FTS5 has been removed).
    """
    return _backend is not None


def is_memory_backend_initialized() -> bool:
    """True when a memory backend has been initialized (local_vector or qdrant)."""
    return _backend is not None


def get_top_k() -> int:
    return _config.get("top_k", 10)


def get_threshold() -> float:
    return _config.get("score_threshold", 0.3)
