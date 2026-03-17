# Credential Redaction

Wintermute includes a dual-layer credential redaction system that prevents the LLM from leaking API keys, passwords, and access tokens from the system configuration in its responses.

---

## What it protects against

LLM models can inadvertently expose secrets when:
- A user asks about system configuration or setup
- The model hallucinates tool output containing config values
- The model includes raw config snippets in its response

The redaction system catches all of these cases before the response reaches the user.

---

## Dual-Layer Architecture

### Layer 1: Pre-dispatch text filter

A synchronous filter in `llm_thread.py` that runs **before** the response is delivered to any output channel (future-based user replies, broadcast system events, convergence corrections). It performs simple substring replacement of known secret values with `[API-KEY-REDACTED]`.

This layer is **always active** and cannot be bypassed — it does not depend on the Convergence Protocol being enabled.

### Layer 2: Convergence Protocol hook

The `credential_redaction` CP hook fires in the `post_inference` phase (main thread scope). It detects `[API-KEY-REDACTED]` markers left by Layer 1 and:

1. Logs the incident at WARNING level
2. Fires a Gotify alert (currently a stub — see below)
3. Injects a correction telling the LLM not to include credentials in future responses

This provides observability and behavioral correction on top of the safety guarantee from Layer 1.

---

## Monitored config fields

Secrets are extracted at startup from these config paths:

| Config path | Description |
|-------------|-------------|
| `inference_backends[*].api_key` | LLM provider API keys |
| `matrix.password` | Matrix bot password |
| `matrix.access_token` | Matrix access token |
| `matrix.device_id` | Matrix device identifier |
| `whisper.api_key` | Whisper transcription API key |
| `memory.embeddings.api_key` | Embeddings endpoint API key |
| `memory.qdrant.api_key` | Qdrant vector DB API key |
| `skills.qdrant.api_key` | Skills Qdrant API key |

Values that are filtered out (not treated as secrets):
- Empty strings
- `"none"`, `"llama-server"` (common placeholders for local unauthenticated endpoints)
- Any value shorter than 8 characters

---

## Gotify alert stub

The `send_gotify_alert()` function in `wintermute/infra/alerts.py` currently logs at WARNING level. Future versions will forward alerts to a Gotify server when `gotify.url` and `gotify.token` are added to `config.yaml`.

---

## Default behavior

Credential redaction is **enabled by default** — no configuration needed. The pre-dispatch filter activates automatically whenever secrets are found in the config. The CP hook is registered as a built-in hook and enabled by default like other programmatic validators.

To disable the CP hook (Layer 2 only — Layer 1 always runs):

```yaml
convergence_protocol:
  validators:
    credential_redaction: false
```

---

## Scope

The redaction filter and CP hook apply only to the **main thread**. Sub-sessions are not affected — they operate in isolated contexts and do not have direct user-facing output channels.
