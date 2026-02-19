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
| **Routines** | Full routine management: create, edit, and delete routines. Sections for active, completed, failed, and cancelled routines. Supports all schedule types (once, daily, weekly, monthly, interval). |

## Debug REST API

The debug panel is backed by a REST API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/debug/sessions` | List all sessions with token budgets |
| GET | `/api/debug/sessions/{thread_id}/messages` | Get messages for a session |
| POST | `/api/debug/sessions/{thread_id}/send` | Inject a message into a session |
| POST | `/api/debug/sessions/{thread_id}/delete` | Archive and clear a session |
| POST | `/api/debug/sessions/{thread_id}/compact` | Force context compaction |
| GET | `/api/debug/subsessions` | List all sub-sessions |
| GET | `/api/debug/subsessions/{id}/messages` | Get full message history for a sub-session |
| GET | `/api/debug/workflows` | List all workflows |
| GET | `/api/debug/jobs` | List all scheduler jobs |
| GET | `/api/debug/config` | Dump the current (sanitised) runtime config |
| GET | `/api/debug/system-prompt` | Get the assembled system prompt + tool schemas with token counts |
| GET | `/api/debug/routines` | List all routines |
| POST | `/api/debug/routines` | Create a new routine |
| DELETE | `/api/debug/routines/{job_id}` | Delete a routine |
| GET | `/api/debug/agenda` | List current agenda items |
| GET | `/api/debug/interaction-log` | List Turing Protocol interaction log entries |
| GET | `/api/debug/interaction-log/{id}` | Get a specific interaction log entry |
| GET | `/api/debug/stream` | Server-sent events stream for live panel updates |
