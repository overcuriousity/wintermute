"""
Vector-indexed skill storage for Wintermute.

Mirrors the ``memory_store`` architecture with two embedding-based backends:
  - local_vector — SQLite + numpy cosine similarity (default)
  - qdrant       — Qdrant vector DB with embedding-based semantic search

An OpenAI-compatible embeddings endpoint is **required**.

Module-level singleton pattern (like memory_store.py).
Call ``init(config, embed_cfg)`` once at startup; all other functions use
the active backend.

On first run, performs a one-time migration from ``data/skills/*.md`` files
into the active backend.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from wintermute.infra.llm_utils import (
    embed as _embed,
    log_store_interaction as _log_interaction_impl,
    make_content_id as _make_id,
)
from wintermute.infra.paths import (
    SKILLS_DIR,
    SKILLS_VECTOR_DB_PATH,
)

logger = logging.getLogger(__name__)

# Module-level singleton state.
_backend: SkillBackend | None = None
_config: dict = {}
_embed_cfg: dict = {}


def _log_interaction(timestamp: float, action: str, input_text: str,
                     output_text: str, status: str = "ok",
                     llm: str = "") -> None:
    """Log a skill store interaction (delegates to llm_utils)."""
    _log_interaction_impl(timestamp, action, input_text, output_text,
                          status, llm=llm, session="system:skill_store")


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SkillBackend(Protocol):
    def init(self) -> None: ...
    def add(self, name: str, summary: str, documentation: str,
            changelog: str = "") -> str: ...
    def get(self, name: str) -> dict | None: ...
    def search(self, query: str, top_k: int, threshold: float) -> list[dict]: ...
    def get_all(self) -> list[dict]: ...
    def delete(self, name: str) -> bool: ...
    def update(self, name: str, summary: str | None = None,
               documentation: str | None = None,
               changelog: str | None = None) -> bool: ...
    def exists(self, name: str) -> bool: ...
    def count(self) -> int: ...
    def stats(self) -> dict: ...
    def get_stale(self, max_age_days: int, min_access: int) -> list[dict]: ...
    def get_all_with_vectors(self) -> list[dict]: ...
    def bulk_delete(self, names: list[str]) -> int: ...


def _build_full_text(summary: str, documentation: str) -> str:
    """Combine summary + documentation for embedding / FTS indexing."""
    return f"{summary.strip()}\n\n{documentation.strip()}" if summary else documentation.strip()


# ---------------------------------------------------------------------------
# Local vector backend (numpy cosine similarity, persisted in SQLite BLOBs)
# ---------------------------------------------------------------------------

class LocalVectorSkillBackend:
    """numpy cosine similarity over skill vectors stored in SQLite BLOBs.

    Requires an OpenAI-compatible embeddings endpoint.
    """

    def __init__(self, embed_cfg: dict) -> None:
        self._embed_cfg = embed_cfg
        self._db_path = SKILLS_VECTOR_DB_PATH
        self._lock = threading.Lock()

    def init(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS skill_vectors ("
                "  skill_id TEXT PRIMARY KEY,"
                "  name TEXT NOT NULL UNIQUE,"
                "  summary TEXT NOT NULL DEFAULT '',"
                "  documentation TEXT NOT NULL DEFAULT '',"
                "  full_text TEXT NOT NULL DEFAULT '',"
                "  vector BLOB NOT NULL,"
                "  created_at REAL NOT NULL,"
                "  last_accessed REAL NOT NULL DEFAULT 0,"
                "  access_count INTEGER NOT NULL DEFAULT 0,"
                "  version INTEGER NOT NULL DEFAULT 1,"
                "  changelog TEXT NOT NULL DEFAULT ''"
                ")"
            )
            conn.commit()
            conn.close()
        logger.info("Skill backend: local_vector (SQLite+numpy at %s)", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _vec_to_blob(vec: list[float]) -> bytes:
        import numpy as np
        return np.array(vec, dtype=np.float32).tobytes()

    def add(self, name: str, summary: str, documentation: str,
            changelog: str = "") -> str:
        sid = _make_id(name)
        full_text = _build_full_text(summary, documentation)
        vectors = _embed([full_text], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        blob = self._vec_to_blob(vectors[0])
        now = time.time()
        t0 = now
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT version, changelog FROM skill_vectors WHERE name = ?",
                    (name,),
                ).fetchone()
                if row:
                    version = row[0] + 1
                    old_changelog = row[1] or ""
                    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if changelog:
                        new_changelog = changelog
                    elif old_changelog:
                        new_changelog = f"{old_changelog}\n- {date_str}: updated"
                    else:
                        new_changelog = f"## Changelog\n- {date_str}: updated"
                    conn.execute(
                        "UPDATE skill_vectors SET summary=?, documentation=?, "
                        "full_text=?, vector=?, version=?, changelog=?, "
                        "last_accessed=? WHERE name=?",
                        (summary, documentation, full_text, blob, version,
                         new_changelog, now, name),
                    )
                else:
                    version = 1
                    conn.execute(
                        "INSERT INTO skill_vectors "
                        "(skill_id, name, summary, documentation, full_text, vector, "
                        "created_at, last_accessed, access_count, version, changelog) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                        (sid, name, summary, documentation, full_text, blob,
                         now, now, version, changelog),
                    )
                conn.commit()
            finally:
                conn.close()
        _log_interaction(t0, "skill_vector_add", f"{name}: {summary[:100]}",
                         f"v{version}", llm="local_vector")
        logger.info("Skill '%s' saved (v%d) via local_vector", name, version)
        return sid

    def exists(self, name: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT 1 FROM skill_vectors WHERE name = ?", (name,),
                ).fetchone()
            finally:
                conn.close()
        return row is not None

    def get(self, name: str) -> dict | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT skill_id, name, summary, documentation, "
                    "created_at, last_accessed, access_count, version, changelog "
                    "FROM skill_vectors WHERE name = ?",
                    (name,),
                ).fetchone()
                if not row:
                    return None
                now = time.time()
                conn.execute(
                    "UPDATE skill_vectors SET last_accessed = ?, "
                    "access_count = access_count + 1 WHERE name = ?",
                    (now, name),
                )
                conn.commit()
            finally:
                conn.close()
        return {
            "id": row[0], "name": row[1], "summary": row[2],
            "documentation": row[3], "created_at": row[4],
            "last_accessed": row[5], "access_count": row[6],
            "version": row[7], "changelog": row[8],
        }

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        import numpy as np

        t0 = time.time()
        vectors = _embed([query], self._embed_cfg, task="query")
        if not vectors:
            logger.warning("Skill vector search: embedding failed, returning all")
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
                    "SELECT skill_id, name, summary, documentation, vector, "
                    "created_at, last_accessed, access_count, version, changelog "
                    "FROM skill_vectors ORDER BY name"
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        results: list[dict] = []
        for r in rows:
            vec = np.frombuffer(r[4], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            score = float(np.dot(q_vec, vec / norm))
            if score >= threshold:
                results.append({
                    "id": r[0], "name": r[1], "summary": r[2],
                    "documentation": r[3], "created_at": r[5],
                    "last_accessed": r[6], "access_count": r[7],
                    "version": r[8], "changelog": r[9], "score": score,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        hits = results[:top_k]
        _log_interaction(
            t0, "skill_vector_search", query[:200],
            f"{len(hits)} hits (top_k={top_k})", llm="local_vector",
        )
        return hits

    def get_all(self) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT skill_id, name, summary, documentation, "
                    "created_at, last_accessed, access_count, version, changelog "
                    "FROM skill_vectors ORDER BY name"
                ).fetchall()
            finally:
                conn.close()
        return [
            {"id": r[0], "name": r[1], "summary": r[2], "documentation": r[3],
             "created_at": r[4], "last_accessed": r[5], "access_count": r[6],
             "version": r[7], "changelog": r[8], "score": 1.0}
            for r in rows
        ]

    def delete(self, name: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM skill_vectors WHERE name = ?", (name,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def update(self, name: str, summary: str | None = None,
               documentation: str | None = None,
               changelog: str | None = None) -> bool:
        # Phase 1: Read current row under lock.
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT summary, documentation, version, changelog "
                    "FROM skill_vectors WHERE name = ?",
                    (name,),
                ).fetchone()
            finally:
                conn.close()
        if not row:
            return False
        new_summary = summary if summary is not None else row[0]
        new_doc = documentation if documentation is not None else row[1]
        new_version = row[2] + 1
        if changelog is not None:
            new_changelog = changelog
        else:
            old_cl = row[3] or ""
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if old_cl:
                new_changelog = f"{old_cl}\n- {date_str}: updated"
            else:
                new_changelog = f"## Changelog\n- {date_str}: updated"
        full_text = _build_full_text(new_summary, new_doc)
        # Phase 2: Embed outside the lock (potentially slow network call).
        vectors = _embed([full_text], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        blob = self._vec_to_blob(vectors[0])
        # Phase 3: Write back under lock.
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE skill_vectors SET summary=?, documentation=?, "
                    "full_text=?, vector=?, version=?, changelog=?, "
                    "last_accessed=? WHERE name=?",
                    (new_summary, new_doc, full_text, blob, new_version,
                     new_changelog, time.time(), name),
                )
                conn.commit()
            finally:
                conn.close()
        logger.info("Skill '%s' updated (v%d) via local_vector", name, new_version)
        return True

    def count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM skill_vectors").fetchone()[0]
            finally:
                conn.close()

    def stats(self) -> dict:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT name, created_at, last_accessed, access_count, "
                    "version FROM skill_vectors ORDER BY name"
                ).fetchall()
            finally:
                conn.close()
        result: dict[str, dict] = {}
        for r in rows:
            result[r[0]] = {
                "created": r[1], "last_read": r[2], "read_count": r[3],
                "sessions_loaded": r[3],
                "version": r[4],
                "success_count": 0, "failure_count": 0,  # not currently persisted; always 0
            }
        return result

    def get_stale(self, max_age_days: int, min_access: int) -> list[dict]:
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT name, summary, last_accessed, access_count "
                    "FROM skill_vectors WHERE last_accessed < ? AND access_count < ?",
                    (cutoff, min_access),
                ).fetchall()
            finally:
                conn.close()
        return [
            {"name": r[0], "summary": r[1], "last_accessed": r[2],
             "access_count": r[3]}
            for r in rows
        ]

    def get_all_with_vectors(self) -> list[dict]:
        import numpy as np
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT skill_id, name, summary, documentation, vector, "
                    "created_at, last_accessed, access_count, version, changelog "
                    "FROM skill_vectors ORDER BY name"
                ).fetchall()
            finally:
                conn.close()
        results = []
        for r in rows:
            vec = np.frombuffer(r[4], dtype=np.float32).tolist()
            results.append({
                "id": r[0], "name": r[1], "summary": r[2],
                "documentation": r[3], "vector": vec, "created_at": r[5],
                "last_accessed": r[6], "access_count": r[7],
                "version": r[8], "changelog": r[9],
            })
        return results

    def bulk_delete(self, names: list[str]) -> int:
        if not names:
            return 0
        placeholders = ",".join("?" for _ in names)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"DELETE FROM skill_vectors WHERE name IN ({placeholders})",
                    names,
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Qdrant backend (vector semantic search)
# ---------------------------------------------------------------------------

class QdrantSkillBackend:
    """Qdrant vector DB with embedding-based semantic search for skills.

    Uses a separate collection from memories (derived from the memory
    collection name by default, e.g. ``wintermute_memories`` → ``wintermute_skills``).
    """

    def __init__(self, config: dict, embed_cfg: dict) -> None:
        self._qdrant_cfg = config.get("qdrant", {})
        self._embed_cfg = embed_cfg
        # Connection config inherited from memory.qdrant if not overridden.
        memory_qdrant = config.get("_memory_qdrant", {})
        self._url = self._qdrant_cfg.get("url",
                     memory_qdrant.get("url", "http://localhost:6333"))
        self._api_key = self._qdrant_cfg.get("api_key",
                        memory_qdrant.get("api_key", "")) or None
        # Derive default skill collection name from the memory collection name
        # so separate instances using different memory collections automatically
        # get isolated skill collections too.
        mem_collection = memory_qdrant.get("collection", "wintermute_memories")
        if mem_collection.endswith("_memories"):
            default_skill_collection = mem_collection[:-len("_memories")] + "_skills"
        else:
            default_skill_collection = mem_collection + "_skills"
        self._collection = self._qdrant_cfg.get("collection", default_skill_collection)
        self._dimensions = embed_cfg.get("dimensions", 1536)
        self._client: Any = None
        self._lock = threading.Lock()

    def init(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from urllib.parse import urlparse

        parsed = urlparse(self._url)
        use_https = parsed.scheme == "https"
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if use_https else 6333)

        self._client = QdrantClient(
            host=host, port=port, https=use_https,
            api_key=self._api_key, timeout=30,
            prefer_grpc=False,
        )
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Qdrant: created skill collection %r (%d dims)",
                        self._collection, self._dimensions)
        logger.info("Skill backend: qdrant (url=%s, collection=%s)",
                     self._url, self._collection)
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        try:
            from qdrant_client.models import PayloadSchemaType
            for field, schema in [
                ("name", PayloadSchemaType.KEYWORD),
                ("last_accessed", PayloadSchemaType.FLOAT),
                ("access_count", PayloadSchemaType.INTEGER),
                ("created_at", PayloadSchemaType.FLOAT),
            ]:
                try:
                    self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception:
                    pass
        except Exception:
            logger.debug("Qdrant: failed to create skill payload indexes", exc_info=True)

    def _name_to_id(self, name: str) -> str:
        """Deterministic point ID from skill name."""
        return _make_id(f"skill:{name}")

    def add(self, name: str, summary: str, documentation: str,
            changelog: str = "") -> str:
        from qdrant_client.models import PointStruct

        full_text = _build_full_text(summary, documentation)
        vectors = _embed([full_text], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        now = time.time()
        t0 = now
        pid = self._name_to_id(name)

        # Check existing for version tracking.
        version = 1
        new_changelog = changelog
        existing = []  # safe default if retrieve() raises
        try:
            existing = self._client.retrieve(
                collection_name=self._collection,
                ids=[pid], with_payload=True,
            )
            if existing:
                old_payload = existing[0].payload or {}
                version = old_payload.get("version", 0) + 1
                if not changelog:
                    old_cl = old_payload.get("changelog", "")
                    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if old_cl:
                        new_changelog = f"{old_cl}\n- {date_str}: updated"
                    else:
                        new_changelog = f"## Changelog\n- {date_str}: updated"
        except Exception:
            pass

        # Normalize old payload once for safe attribute access.
        old_payload = (existing[0].payload or {}) if existing else {}

        point = PointStruct(
            id=pid,
            vector=vectors[0],
            payload={
                "name": name,
                "summary": summary,
                "documentation": documentation,
                "full_text": full_text,
                "created_at": now if version == 1 else old_payload.get("created_at", now),
                "last_accessed": now,
                "access_count": old_payload.get("access_count", 0),
                "version": version,
                "changelog": new_changelog,
            },
        )
        self._client.upsert(
            collection_name=self._collection,
            points=[point],
        )
        _log_interaction(t0, "qdrant_skill_add", f"{name}: {summary[:100]}",
                         f"v{version}", llm="qdrant")
        logger.info("Skill '%s' saved (v%d) via qdrant", name, version)
        return pid

    def exists(self, name: str) -> bool:
        pid = self._name_to_id(name)
        try:
            results = self._client.retrieve(
                collection_name=self._collection,
                ids=[pid], with_payload=False,
            )
            return bool(results)
        except Exception:
            return False

    def get(self, name: str) -> dict | None:
        pid = self._name_to_id(name)
        try:
            results = self._client.retrieve(
                collection_name=self._collection,
                ids=[pid], with_payload=True,
            )
        except Exception:
            return None
        if not results:
            return None
        p = results[0].payload or {}
        # Track access.
        now = time.time()
        try:
            self._client.set_payload(
                collection_name=self._collection,
                payload={"last_accessed": now,
                         "access_count": p.get("access_count", 0) + 1},
                points=[pid],
            )
        except Exception:
            pass
        return {
            "id": pid, "name": p.get("name", name),
            "summary": p.get("summary", ""),
            "documentation": p.get("documentation", ""),
            "created_at": p.get("created_at", 0),
            "last_accessed": p.get("last_accessed", 0),
            "access_count": p.get("access_count", 0),
            "version": p.get("version", 1),
            "changelog": p.get("changelog", ""),
        }

    def search(self, query: str, top_k: int, threshold: float) -> list[dict]:
        t0 = time.time()
        vectors = _embed([query], self._embed_cfg, task="query")
        if not vectors:
            return self.get_all()[:top_k]
        try:
            results = self._client.query_points(
                collection_name=self._collection,
                query=vectors[0],
                limit=top_k,
                score_threshold=threshold,
                with_payload=True,
            ).points
        except Exception:
            logger.exception("Qdrant skill search failed")
            return self.get_all()[:top_k]
        hits = []
        for r in results:
            p = r.payload or {}
            hits.append({
                "id": str(r.id), "name": p.get("name", ""),
                "summary": p.get("summary", ""),
                "documentation": p.get("documentation", ""),
                "created_at": p.get("created_at", 0),
                "last_accessed": p.get("last_accessed", 0),
                "access_count": p.get("access_count", 0),
                "version": p.get("version", 1),
                "changelog": p.get("changelog", ""),
                "score": r.score,
            })
        _log_interaction(
            t0, "qdrant_skill_search", query[:200],
            f"{len(hits)} hits (top_k={top_k})", llm="qdrant",
        )
        return hits

    def get_all(self) -> list[dict]:
        try:
            all_points = []
            offset = None
            while True:
                points_page, offset = self._client.scroll(
                    collection_name=self._collection,
                    limit=10000,
                    with_payload=True,
                    offset=offset,
                )
                if not points_page:
                    break
                all_points.extend(points_page)
                if offset is None:
                    break
        except Exception:
            return []
        return [
            {"id": str(p.id),
             "name": (p.payload or {}).get("name", ""),
             "summary": (p.payload or {}).get("summary", ""),
             "documentation": (p.payload or {}).get("documentation", ""),
             "created_at": (p.payload or {}).get("created_at", 0),
             "last_accessed": (p.payload or {}).get("last_accessed", 0),
             "access_count": (p.payload or {}).get("access_count", 0),
             "version": (p.payload or {}).get("version", 1),
             "changelog": (p.payload or {}).get("changelog", ""),
             "score": 1.0}
            for p in sorted(all_points,
                            key=lambda x: (x.payload or {}).get("name", ""))
        ]

    def delete(self, name: str) -> bool:
        from qdrant_client.models import PointIdsList
        pid = self._name_to_id(name)
        # Check existence first — Qdrant delete is a no-op for missing IDs.
        try:
            existing = self._client.retrieve(
                collection_name=self._collection,
                ids=[pid], with_payload=False,
            )
            if not existing:
                return False
        except Exception:
            return False
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[pid]),
            )
            return True
        except Exception:
            return False

    def update(self, name: str, summary: str | None = None,
               documentation: str | None = None,
               changelog: str | None = None) -> bool:
        # Fetch existing payload directly from Qdrant without updating access stats
        pid = self._name_to_id(name)
        try:
            points = self._client.retrieve(
                collection_name=self._collection,
                ids=[pid],
                with_payload=True,
            )
        except Exception:
            return False
        if not points:
            return False
        payload = points[0].payload or {}
        existing_summary = payload.get("summary", "")
        existing_doc = payload.get("documentation", "")
        new_summary = summary if summary is not None else existing_summary
        new_doc = documentation if documentation is not None else existing_doc
        new_version = payload.get("version", 0) + 1
        if changelog is not None:
            new_changelog = changelog
        else:
            old_cl = payload.get("changelog", "")
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if old_cl:
                new_changelog = f"{old_cl}\n- {date_str}: updated"
            else:
                new_changelog = f"## Changelog\n- {date_str}: updated"
        full_text = _build_full_text(new_summary, new_doc)
        vectors = _embed([full_text], self._embed_cfg)
        if not vectors:
            raise RuntimeError("Embedding call returned no vectors")
        from qdrant_client.models import PointStruct
        now = time.time()
        point = PointStruct(
            id=pid,
            vector=vectors[0],
            payload={
                "name": name,
                "summary": new_summary,
                "documentation": new_doc,
                "full_text": full_text,
                "created_at": payload.get("created_at", now),
                "last_accessed": now,
                "access_count": payload.get("access_count", 0),
                "version": new_version,
                "changelog": new_changelog,
            },
        )
        self._client.upsert(
            collection_name=self._collection,
            points=[point],
        )
        logger.info("Skill '%s' updated (v%d) via qdrant", name, new_version)
        return True

    def count(self) -> int:
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0

    def stats(self) -> dict:
        all_skills = self.get_all()
        result: dict[str, dict] = {}
        for s in all_skills:
            result[s["name"]] = {
                "created": s["created_at"], "last_read": s["last_accessed"],
                "read_count": s["access_count"],
                "sessions_loaded": s["access_count"],
                "version": s["version"],
                "success_count": 0, "failure_count": 0,
            }
        return result

    def get_stale(self, max_age_days: int, min_access: int) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, Range
        cutoff = time.time() - (max_age_days * 86400)
        try:
            all_points = []
            offset = None
            while True:
                points_page, offset = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="last_accessed",
                                       range=Range(lt=cutoff)),
                        FieldCondition(key="access_count",
                                       range=Range(lt=min_access)),
                    ]),
                    limit=10000,
                    with_payload=True,
                    offset=offset,
                )
                if not points_page:
                    break
                all_points.extend(points_page)
                if offset is None:
                    break
        except Exception:
            return []
        return [
            {"name": (p.payload or {}).get("name", ""),
             "summary": (p.payload or {}).get("summary", ""),
             "last_accessed": (p.payload or {}).get("last_accessed", 0),
             "access_count": (p.payload or {}).get("access_count", 0)}
            for p in all_points
        ]

    def get_all_with_vectors(self) -> list[dict]:
        try:
            all_points = []
            offset = None
            while True:
                points_page, offset = self._client.scroll(
                    collection_name=self._collection,
                    limit=10000,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset,
                )
                if not points_page:
                    break
                all_points.extend(points_page)
                if offset is None:
                    break
        except Exception:
            return []
        return [
            {"id": str(p.id),
             "name": (p.payload or {}).get("name", ""),
             "summary": (p.payload or {}).get("summary", ""),
             "documentation": (p.payload or {}).get("documentation", ""),
             "vector": (
                 list(next(iter(p.vector.values())))
                 if isinstance(p.vector, dict) and p.vector
                 else (list(p.vector) if p.vector else [])
             ),
             "created_at": (p.payload or {}).get("created_at", 0),
             "last_accessed": (p.payload or {}).get("last_accessed", 0),
             "access_count": (p.payload or {}).get("access_count", 0),
             "version": (p.payload or {}).get("version", 1),
             "changelog": (p.payload or {}).get("changelog", "")}
            for p in all_points
        ]

    def bulk_delete(self, names: list[str]) -> int:
        from qdrant_client.models import PointIdsList
        pids = [self._name_to_id(n) for n in names]
        try:
            with self._lock:
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=PointIdsList(points=pids),  # type: ignore[arg-type]
                )
            return len(pids)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# One-time migration from data/skills/*.md flat files
# ---------------------------------------------------------------------------

def _migrate_from_flat_files() -> None:
    """Import existing markdown skill files into the active backend.

    Only runs when the store is empty and data/skills/*.md files exist.
    Legacy stats from data/skill_stats.yaml are **not** imported; migration
    starts with fresh stats in the new backend.  The stats file is loaded
    only to emit a debug log entry for each skill that had prior usage.
    """
    if not SKILLS_DIR.exists():
        return
    skill_files = sorted(SKILLS_DIR.glob("*.md"))
    if not skill_files:
        return

    # Load existing stats for metadata enrichment.
    stats: dict[str, dict] = {}
    try:
        import yaml
        stats_path = SKILLS_DIR.parent / "skill_stats.yaml"
        if stats_path.exists():
            raw = yaml.safe_load(stats_path.read_text(encoding="utf-8")) or {}
            stats = raw.get("skills", {})
    except Exception:
        logger.debug("Could not load skill_stats.yaml for migration", exc_info=True)

    imported = 0
    for f in skill_files:
        try:
            content = f.read_text(encoding="utf-8").strip()
            if not content:
                continue
            name = f.stem
            # Validate name against the same rules as skill_io.
            if not re.match(r'^[A-Za-z0-9][A-Za-z0-9_-]*$', name):
                logger.warning("Skipping migration of '%s': invalid skill name", name)
                continue
            # Parse: first line = summary, rest = documentation.
            lines = content.split("\n", 1)
            summary = lines[0].strip()
            documentation = lines[1].strip() if len(lines) > 1 else ""
            # Extract changelog if present.
            changelog = ""
            if "## Changelog" in documentation:
                idx = documentation.index("## Changelog")
                changelog = documentation[idx:].strip()
                documentation = documentation[:idx].strip()

            _b().add(name, summary, documentation, changelog=changelog)

            # Legacy stats from skill_stats.yaml (noted but not merged
            # into backend state to avoid mutating version history during
            # migration without proper support for setting access_count /
            # last_accessed directly).
            skill_stat = stats.get(name, {})
            if skill_stat:
                try:
                    read_count = skill_stat.get("read_count", 0)
                    last_read = skill_stat.get("last_read", 0)
                    if read_count or last_read:
                        logger.debug(
                            "Found legacy stats for skill '%s' (reads=%d, last_read=%s); "
                            "not applied by migration",
                            name, read_count, last_read,
                        )
                except Exception:
                    logger.debug(
                        "Error while inspecting legacy stats for skill '%s'",
                        name, exc_info=True,
                    )

            imported += 1
        except Exception:
            logger.warning("Failed to migrate skill file %s", f.name, exc_info=True)

    if imported:
        logger.info("Skill migration: imported %d skills from data/skills/*.md", imported)


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

def init(config: dict, embed_cfg: dict | None = None) -> None:
    """Select and initialize the skill backend.

    ``config`` is the ``skills`` section from config.yaml.
    ``embed_cfg`` is the shared ``memory.embeddings`` config (inherited).

    Requires an OpenAI-compatible embeddings endpoint.  The legacy ``fts5``
    backend has been removed.
    On first run with existing skill files, performs one-time migration.
    """
    global _backend, _config, _embed_cfg
    _config = dict(config)
    _embed_cfg = dict(embed_cfg or {})

    backend_name = config.get("backend", "local_vector")

    # ── Fail-fast for removed FTS5 backend ──────────────────────────────
    if backend_name == "fts5":
        raise ValueError(
            "The 'fts5' skill backend has been removed. "
            "An embedding-based backend (local_vector or qdrant) is now required.\n"
            "  Configure memory.embeddings.endpoint in config.yaml and set "
            "skills.backend to 'local_vector' (default) or 'qdrant'."
        )

    if backend_name not in ("local_vector", "qdrant"):
        raise ValueError(
            f"Unknown skill backend {backend_name!r}. "
            f"Supported backends: local_vector, qdrant"
        )

    if not _embed_cfg.get("endpoint"):
        raise ValueError(
            "memory.embeddings.endpoint is required for the skill store. "
            "Configure an OpenAI-compatible /v1/embeddings endpoint in config.yaml."
        )

    if backend_name == "local_vector":
        _backend = LocalVectorSkillBackend(_embed_cfg)
    elif backend_name == "qdrant":
        _backend = QdrantSkillBackend(config, _embed_cfg)
    _backend.init()

    # One-time migration from flat files.
    try:
        if _b().count() == 0:
            _migrate_from_flat_files()
    except Exception:
        logger.warning("Skill migration failed", exc_info=True)


def _b() -> "SkillBackend":
    """Return the active backend, raising if init() was not called."""
    if _backend is None:
        raise RuntimeError("skill_store.init() has not been called")
    return _backend


def add(name: str, summary: str, documentation: str,
        changelog: str = "") -> str:
    return _b().add(name, summary, documentation, changelog)


def exists(name: str) -> bool:
    """Check if a skill exists without updating access stats."""
    return _b().exists(name)


def get(name: str) -> dict | None:
    return _b().get(name)


def search(query: str, top_k: int | None = None,
           threshold: float | None = None) -> list[dict]:
    _top_k: int = top_k if top_k is not None else int(_config.get("top_k", 5))
    _threshold: float = threshold if threshold is not None else float(_config.get("score_threshold", 0.3))
    try:
        return _b().search(query, _top_k, _threshold)
    except Exception as exc:
        logger.warning("skill_store.search failed: %s", exc)
        return []


def get_all() -> list[dict]:
    return _b().get_all()


def delete(name: str) -> bool:
    return _b().delete(name)


def update(name: str, summary: str | None = None,
           documentation: str | None = None,
           changelog: str | None = None) -> bool:
    return _b().update(name, summary, documentation, changelog)


def count() -> int:
    return _b().count()


def stats() -> dict:
    return _b().stats()


def get_stale(max_age_days: int, min_access: int) -> list[dict]:
    return _b().get_stale(max_age_days, min_access)


def get_all_with_vectors() -> list[dict]:
    return _b().get_all_with_vectors()


def bulk_delete(names: list[str]) -> int:
    return _b().bulk_delete(names)


def is_vector_enabled() -> bool:
    """True when the active backend supports vector search."""
    if _backend is None:
        return False
    return isinstance(_backend, (LocalVectorSkillBackend, QdrantSkillBackend))


def is_qdrant_backend() -> bool:
    """True when the active backend is QdrantSkillBackend."""
    return isinstance(_backend, QdrantSkillBackend)


def get_top_k() -> int:
    return _config.get("top_k", 5)


def get_threshold() -> float:
    return _config.get("score_threshold", 0.3)
