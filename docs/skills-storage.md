# Skills Storage

Skills are learned, reusable procedures that the AI assistant can create, retrieve, and refine over time. They are stored in a **vector-indexed backend** — the same architecture used for memories.

## Backend Selection

| Backend | Description | When to use |
| --- | --- | --- |
| `fts5` | SQLite FTS5 full-text search | Default / low-resource setups |
| `local_vector` | SQLite + numpy cosine similarity | Local embeddings without external services |
| `qdrant` | Qdrant vector database | Production / large skill libraries |

The backend is **auto-selected** based on the available embeddings configuration unless explicitly overridden in the `skills` section. When no `skills.backend` is set:

- If an embeddings endpoint is configured (`memory.embeddings.endpoint`), defaults to `local_vector`
- Otherwise defaults to `fts5` (no embedding endpoint required)

> **Note:** The skills backend does **not** automatically inherit `memory.backend`. It is configured independently via `skills.backend` or auto-selected based on whether `memory.embeddings.endpoint` is present. For example, setting `memory.backend: qdrant` does not make skills use Qdrant — you must set `skills.backend: qdrant` explicitly.

```yaml
# config.yaml

# Explicit override (optional):
skills:
  backend: "local_vector"

# Or rely on auto-detection from embeddings config:
# (skills backend does NOT inherit memory.backend)
memory:
  backend: "local_vector"
  embeddings:
    endpoint: "http://localhost:11434/v1"   # presence triggers local_vector default for skills
    model: "nomic-embed-text"
```

> **Note:** `flat_file` is **not** supported for skills. If your memory backend is `flat_file`, skills automatically fall back to `fts5`.

### Embedding Configuration

Skills inherit embedding settings from `memory.embeddings`. No separate embedding config is needed. The shared settings include:

- `endpoint` — Embedding API base URL (the code appends `/embeddings`)
- `model` — Embedding model name
- `dimensions` — Vector dimensions (auto-detected)
- `api_key` — Optional API key

### Qdrant Configuration

When using the Qdrant backend, skills are stored in a **separate collection** (default: `wintermute_skills`) to keep them isolated from memories:

```yaml
skills:
  backend: "qdrant"
  qdrant:
    collection: "wintermute_skills"   # default
    url: "http://localhost:6333"      # inherited from memory.qdrant if omitted
    api_key: null
```

## The `skill` Tool

The `skill` tool replaces the former `add_skill` tool and supports three actions:

### `add` — Create or Update a Skill

```json
{
  "action": "add",
  "skill_name": "deploy-docker",
  "summary": "Docker Compose deployment workflow",
  "documentation": "1. cd to project directory\n2. `docker compose up -d`\n3. Verify: `docker ps`"
}
```

### `read` — Retrieve a Skill by Name

```json
{
  "action": "read",
  "skill_name": "deploy-docker"
}
```

### `search` — Semantic Search for Skills

```json
{
  "action": "search",
  "query": "container deployment",
  "top_k": 5
}
```

The search action uses vector similarity (with `local_vector` / `qdrant`) or full-text search (with `fts5`).

## System Prompt Integration (Query-Ranked TOC)

The skills TOC injected into the system prompt mirrors the memory ranking pattern:

- **Vector backend active + user query available**: Skills are ranked by cosine similarity to the query. Only the top-k most relevant skills appear in the TOC, with relevance scores.
- **Non-vector backend or no query**: All skills are listed alphabetically.

This ensures the LLM sees the most relevant skills for the current conversation without consuming context window on unrelated entries.

## Migration from Flat Files

On first startup after upgrading, `skill_store` automatically migrates existing `data/skills/*.md` files into the configured backend:

1. Each `.md` file is parsed:
   - **First line** → `summary`
   - **Remaining content** → `documentation`
   - **`## Changelog` section** → extracted and preserved in metadata
2. Stats from `data/skill_stats.yaml` (if present) are **not** imported; migration starts with fresh stats in the new backend
3. Original `.md` files are **not deleted** — they remain as a backup
4. A log entry is written for each migrated skill
5. The `skill_stats.py` module and `skill_stats.yaml` file are no longer used after migration — historical stats are not carried over; new stats are tracked natively by the skill store backends

The migration runs **only when the skill store is empty** (i.e., `count() == 0` on first startup after upgrading). It is not per-skill idempotent: if migration is interrupted, you may need to clear the skill store and restart to re-trigger it. It does not re-import individual missing skills into a partially populated store.

## Dreaming (Skill Consolidation)

During dreaming cycles, skills are automatically maintained:

- **Auto-retirement**: Skills not accessed in 90+ days are deleted
- **Deduplication**: Overlapping skills are merged (LLM-assisted)
- **Condensation**: Skills with documentation >600 chars are condensed (less aggressive than the former per-skill-always approach)

## Reflection (Skill Analysis)

The reflection cycle monitors skill health:

- **Failure correlation**: Skills loaded in 3+ failed sessions are flagged
- **LLM analysis**: Enriched with per-skill stats from `skill_store.stats()` (access counts, versions, failure rate)
- **Mutation sub-sessions**: The `skill` tool (with actions `add` and `read`) replaces the former `add_skill` + `read_file` pattern
- **Synthesis**: Recurring successful patterns are proposed as new skills

## Web Interface

The `/api/skills` endpoints are backend-agnostic:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/skills` | List all skills with stats |
| `GET` | `/api/skills/{name}` | Read a specific skill |
| `POST` | `/api/skills` | Create a new skill |
| `PUT` | `/api/skills/{name}` | Update an existing skill |
| `DELETE` | `/api/skills/{name}` | Delete a skill |

## NL Translation

When NL translation is enabled, the `skill` tool accepts a plain-English description:

```yaml
nl_translation:
  enabled: true
  tools:
    - task
    - worker_delegation
    - skill                # formerly "add_skill"
```

The translator LLM determines the action (add/read/search) and extracts structured arguments from the description.
