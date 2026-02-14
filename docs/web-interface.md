# Web Interface

The web interface runs as an aiohttp HTTP + WebSocket server, enabled by default at `http://127.0.0.1:8080`.

## Chat UI

The main page (`/`) provides a WebSocket-based chat interface:

- Each browser tab gets its own independent `thread_id` (prefixed `web_`)
- Supports all [slash commands](commands.md)
- Real-time message delivery via WebSocket
- Auto-reconnect on disconnect
- Link to the debug panel in the header

## Debug Panel

Available at `/debug`. Provides a live inspection view with auto-refresh (every 5 seconds).

### Tabs

| Tab | Content |
|-----|---------|
| **Sessions** | All active conversation threads (web, Matrix, system) with context utilisation bars, message counts, and token breakdown (system prompt + tools + history). Click a session to inspect its messages, view the assembled system prompt, inject messages, compact context, or delete the session. |
| **Sub-sessions** | All background workers with ID, workflow, dependencies, parent thread, status, objective, system prompt mode, creation time, duration, and result/error preview. |
| **Workflows** | Workflow DAGs with per-node status, objectives, dependencies, and result previews. Collapsible sections per workflow. |
| **Jobs** | APScheduler jobs with trigger type, next run time, and arguments. |
| **Reminders** | Full reminder management: create, edit, and delete reminders. Sections for active, completed, failed, and cancelled reminders. Supports all schedule types (once, daily, weekly, monthly, interval). |

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
| GET | `/api/debug/workflows` | List all workflows |
| GET | `/api/debug/jobs` | List all scheduler jobs |
| GET | `/api/debug/system-prompt` | Get the assembled system prompt + tool schemas with token counts |
| GET | `/api/debug/reminders` | List all reminders |
| POST | `/api/debug/reminders` | Create a new reminder |
| DELETE | `/api/debug/reminders/{job_id}` | Delete a reminder |
