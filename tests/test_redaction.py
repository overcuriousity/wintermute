"""Tests for credential redaction in the Convergence Protocol.

redact_credentials reads a module-level global, so every test that touches it
must restore the previous value — otherwise state leaks between tests.
"""

import pytest

from wintermute.core import convergence_protocol as cp


@pytest.fixture()
def secrets():
    """Install redaction secrets for one test, then restore the originals."""
    original = cp._redaction_secrets

    def _install(*values: str) -> None:
        cp.set_redaction_secrets(frozenset(values))

    yield _install
    cp._redaction_secrets = original


def test_redact_replaces_known_secret(secrets):
    secrets("sk-abcdefghijklmnop")
    text, was_redacted = cp.redact_credentials("The key is sk-abcdefghijklmnop okay")
    assert was_redacted is True
    assert "sk-abcdefghijklmnop" not in text
    assert cp._SECRET_PLACEHOLDER in text


def test_redact_leaves_clean_text_untouched(secrets):
    secrets("sk-abcdefghijklmnop")
    text, was_redacted = cp.redact_credentials("Nothing sensitive here.")
    assert was_redacted is False
    assert text == "Nothing sensitive here."


def test_redact_handles_overlapping_secrets_longest_first(secrets):
    """A short secret that is a substring of a long one must not cause the
    long one to leak its tail."""
    secrets("sk-abcdefgh", "sk-abcdefghijklmnop")
    text, was_redacted = cp.redact_credentials("key sk-abcdefghijklmnop end")
    assert was_redacted is True
    assert "ijklmnop" not in text


def test_redact_noop_when_no_secrets_configured(secrets):
    secrets()
    text, was_redacted = cp.redact_credentials("sk-abcdefghijklmnop")
    assert was_redacted is False
    assert text == "sk-abcdefghijklmnop"


def test_redact_handles_empty_text(secrets):
    secrets("sk-abcdefghijklmnop")
    assert cp.redact_credentials("") == ("", False)


def test_extract_config_secrets_collects_every_known_path():
    cfg = {
        "inference_backends": [{"api_key": "backend-key-123456"}],
        "matrix": {"password": "matrix-pass-123456", "access_token": "matrix-token-123456"},
        "whisper": {"api_key": "whisper-key-123456"},
        "memory": {
            "embeddings": {"api_key": "embed-key-123456"},
            "qdrant": {"api_key": "qdrant-key-123456"},
        },
        "skills": {"qdrant": {"api_key": "skills-qdrant-key-123456"}},
    }
    found = cp.extract_config_secrets(cfg)
    assert "backend-key-123456" in found
    assert "matrix-pass-123456" in found
    assert "matrix-token-123456" in found
    assert "whisper-key-123456" in found
    assert "embed-key-123456" in found
    assert "qdrant-key-123456" in found
    assert "skills-qdrant-key-123456" in found


def test_extract_config_secrets_drops_short_and_placeholder_values():
    cfg = {
        "inference_backends": [{"api_key": "short"}, {"api_key": "none"}],
        "whisper": {"api_key": "whisper-1"},
        "memory": {"embeddings": {"api_key": "llama-server"}},
    }
    assert cp.extract_config_secrets(cfg) == frozenset()


def test_extract_config_secrets_tolerates_empty_config():
    assert cp.extract_config_secrets({}) == frozenset()
