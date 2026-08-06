"""Tests for the local ONNX embedding fallback.

No test here downloads a model or touches the network.  The one test that
would need real inference is skipped unless the model is already cached.
"""

import pytest

from wintermute.infra import local_embeddings


def test_constants_are_stable():
    """These values are persisted in the store's dimension guard; changing
    them invalidates existing local-vector databases."""
    assert local_embeddings.DIMENSIONS == 384
    assert local_embeddings.MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"
    assert local_embeddings.provider_name() == "local:all-MiniLM-L6-v2"


def test_is_available_returns_a_bool_without_raising():
    assert isinstance(local_embeddings.is_available(), bool)


def test_is_available_reports_false_when_extra_is_missing(monkeypatch):
    def _boom():
        raise ImportError("no onnxruntime")

    monkeypatch.setattr(local_embeddings, "_import_backend", _boom)
    local_embeddings.reset_cache()
    try:
        assert local_embeddings.is_available() is False
    finally:
        # is_available memoises; clear the poisoned False so later tests
        # (and the skipif below) see the real environment again.
        local_embeddings.reset_cache()


def test_embed_local_raises_a_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(local_embeddings, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as exc:
        local_embeddings.embed_local(["hello"])
    assert "local-embeddings" in str(exc.value)


def test_embed_local_returns_empty_for_empty_input():
    assert local_embeddings.embed_local([]) == []


@pytest.mark.skipif(
    not local_embeddings.is_model_cached(),
    reason="model not downloaded; run once online to enable this test",
)
def test_embed_local_produces_normalised_vectors_of_the_right_shape():
    vectors = local_embeddings.embed_local(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert all(len(v) == local_embeddings.DIMENSIONS for v in vectors)
    norms = [sum(x * x for x in v) ** 0.5 for v in vectors]
    assert all(abs(n - 1.0) < 1e-3 for n in norms)


def test_dimension_mismatch_is_refused(tmp_path, monkeypatch):
    """A store built at one dimension must refuse a provider with another."""
    import sqlite3

    from wintermute.infra import memory_store

    db_path = tmp_path / "local_vectors.db"
    monkeypatch.setattr(memory_store, "LOCAL_VECTOR_DB_PATH", db_path)

    backend = memory_store.LocalVectorBackend({"embeddings": {}})
    backend._db_path = db_path
    backend.init()

    # Rewrite the recorded dimension to simulate a store built elsewhere.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('embedding_dimension', '1536')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError) as exc:
        backend._check_provider_compatibility()
    assert "1536" in str(exc.value)
    assert "384" in str(exc.value)
