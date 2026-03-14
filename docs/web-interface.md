# Web Interface

The web interface is an aiohttp HTTP server, enabled by default at `http://127.0.0.1:8080`. It provides a debug/administration panel only — chat interaction happens through Matrix (or any future chat interface). The `/debug` URL is the main entry point.

## Debug Panel

Available at `/debug`. Provides a live inspection and administration view with auto-refresh (every 5 seconds) via a server-sent events stream.

### Tabs

| Tab | Content |
|-----|---------|
| **Sessions** | All active conversation threads (web, Matrix, system) with context utilisation bars, message counts, and token breakdown (system prompt + tools + history). Click a session to inspect its messages, view the assembled system prompt, inject messages, compact context, or delete the session. |
| **Sub-sessions** | All background workers with ID, workflow, dependencies, parent thread, status, objective, system prompt mode, creation time, duration, and result/error preview. Click a sub-session to inspect its full message history. |
| **Workflows** | Workflow DAGs with per-node status, objectives, dependencies, and result previews. Collapsible sections per workflow. |
| **Jobs** | APScheduler jobs with trigger type, next run time, and arguments. |
| **Tasks** | Full task management: create, edit, pause, resume, and delete tasks. Sections for active, paused, completed, and scheduled tasks. Supports all schedule types (once, daily, weekly, monthly, interval). |
| **Outcomes** | Historical sub-session outcomes with status, duration, tool call counts, Convergence Protocol verdicts, and objectives. Aggregate stats (success rate, avg duration, timeout rate) displayed at the top. Filterable by status. |

## Debug REST API

The debug panel is backed by a REST API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sessions` | List all sessions with token budgets |
| GET | `/api/sessions/{thread_id}/messages` | Get messages for a session |
| POST | `/api/sessions/{thread_id}/send` | Inject a message into a session |
| POST | `/api/sessions/{thread_id}/delete` | Archive and clear a session |
| POST | `/api/sessions/{thread_id}/compact` | Force context compaction |
| GET | `/api/subsessions` | List all sub-sessions |
| GET | `/api/subsessions/{id}/messages` | Get full message history for a sub-session |
| GET | `/api/workflows` | List all workflows |
| GET | `/api/jobs` | List all scheduler jobs |
| GET | `/api/config` | Dump the current (sanitised) runtime config |
| GET | `/api/system-prompt` | Get the assembled system prompt + tool schemas with token counts |
| GET | `/api/tasks` | List all tasks |
| POST | `/api/tasks` | Create a new task |
| PUT | `/api/tasks/{task_id}` | Update a task |
| DELETE | `/api/tasks/{task_id}` | Delete a task |
| GET | `/api/interaction-log` | List interaction log entries (Convergence Protocol, dreaming, embedding calls, Qdrant operations) |
| GET | `/api/interaction-log/{id}` | Get a specific interaction log entry |
| GET | `/api/memory` | Get memory store stats and search results (query: `q`, `k`) |
| GET | `/api/outcomes` | List sub-session outcomes with aggregate stats (supports `status`, `limit`, `offset` query params) |
| GET | `/api/stream` | Server-sent events stream for live panel updates |
| GET | `/api/thread-config` | List all per-thread config overrides and available backends |
| GET | `/api/thread-config/{thread_id}` | Get resolved config for a thread (values + sources) |
| POST | `/api/thread-config/{thread_id}` | Set per-thread config overrides (JSON body with keys to set; `null` removes an override) |
| DELETE | `/api/thread-config/{thread_id}` | Remove all per-thread config overrides |
