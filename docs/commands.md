# Slash Commands

All commands are available in both Matrix and the web UI.

| Command | Description |
|---------|-------------|
| `/new` | Reset conversation history for the current thread. Archives all messages and clears the compaction summary. Also cancels any running sub-sessions bound to this thread. |
| `/compact` | Force context compaction. Summarises older messages and archives them, keeping the last 10 messages intact. Shows before/after token counts. |
| `/tasks` | List all tasks (active, paused, completed) with their schedules. |
| `/status` | Show detailed system status: running asyncio tasks, active sub-sessions, workflow state, dreaming loop status, and task count. |
| `/dream` | Manually trigger the nightly dreaming consolidation of MEMORIES.txt and tasks. Shows before/after counts. |
| `/memory-stats` | Show memory store status: backend type, entry count, and backend-specific details (Qdrant URL, collection, dimensions, status). |
| `/rebuild-index` | Rebuild the vector memory index from MEMORIES.txt. Only available when a vector backend (`fts5` or `qdrant`) is active. Reports stats after completion. |
| `/kimi-auth` | Start Kimi-Code OAuth device-code flow. Broadcasts the verification URL to the current chat. Only relevant when a `kimi-code` backend is configured. |
| `/verify-session` | Send an E2EE SAS verification request to all `allowed_users`. Matrix-only. Useful after key changes or to establish cross-signing trust. Has no effect if E2EE is not available. |
| `/commands` | List all available slash commands with short descriptions. |
