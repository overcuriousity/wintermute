# Skills Storage

Skills are learned, reusable procedures that the AI assistant can create, retrieve, and refine over time. They are stored in a **vector-indexed backend** — the same architecture used for memories.

## Backend Selection

| Backend | Description | When to use |
| --- | --- | --- |
| `local_vector` | SQLite + numpy cosine similarity | Default — local embeddings without external services |
| `qdrant` | Qdrant vector database | Production / large skill libraries |

An OpenAI-compatible embeddings endpoint is **required** (configured via `memory.embeddings.endpoint`).

When no `skills.backend` is set, the value is inherited from `memory.backend` (defaults to `local_vector`).

> **Migration note:** The legacy `fts5` backend has been removed. If your config still has `skills.backend: "fts5"`, remove it or change it to `"local_vector"`.

```yaml
# config.yaml

# Explicit override (optional):
skills:
  backend: "local_vector"

# Or rely on auto-detection from memory config:
memory:
  backend: "local_vector"
  embeddings:
    endpoint: "http://localhost:11434/v1"
    model: "nomic-embed-text"
```


### Embedding Configuration

Skills inherit embedding settings from `memory.embeddings`. No separate embedding config is needed. The shared settings include:

- `endpoint` — Embedding API base URL (the code appends `/embeddings`)
- `model` — Embedding model name
- `dimensions` — Vector dimensions (auto-detected)
- `api_key` — Optional API key

### Qdrant Configuration

When using the Qdrant backend, skills are stored in a **separate collection** to keep them isolated from memories. The collection name is derived from the memory collection by default (e.g., `wintermute_memories` → `wintermute_skills`), or can be set explicitly:

```yaml
skills:
  backend: "qdrant"
  qdrant:
    collection: "wintermute_skills"   # derived from memory.qdrant.collection if omitted
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

The search action uses vector similarity (cosine similarity with `local_vector`, or Qdrant's native search with `qdrant`).

## System Prompt Integration (Query-Ranked TOC)

The skills TOC injected into the system prompt mirrors the memory ranking pattern:

- **Vector backend active + user query available**: Skills are ranked by cosine similarity to the query. Only the top-k most relevant skills appear in the TOC, with relevance scores.
- **No query available**: All skills are listed alphabetically.

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
- **LLM analysis**: Enriched with per-skill stats from `skill_store.stats()` (access counts, versions, staleness/recency)
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
