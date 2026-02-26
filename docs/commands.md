# Slash Commands

All commands are available in both Matrix and the web UI.

| Command | Description |
|---------|-------------|
| `/new` | Reset conversation history for the current thread. Archives all messages and clears the compaction summary. Also cancels any running sub-sessions bound to this thread. |
| `/compact` | Force context compaction. Summarises older messages and archives them, keeping the last 10 messages intact. Shows before/after token counts. |
| `/tasks` | List all tasks (active, paused, completed) with their schedules. |
| `/status` | Show detailed system status: LLM backends, context budget, memory, tasks, background loops (dreaming, harvest, reflection), self-model summary, active sub-sessions. |
| `/dream` | Manually trigger the dreaming cycle. Runs all phases (housekeeping + creative) and shows per-phase results with ✓/✗ status, before/after counts for memories, tasks, and skills. |
| `/reflect` | Manually trigger the reflection cycle (rule engine + LLM analysis + self-model update). Reports findings and any auto-tuning changes inline. |
| `/memory-stats` | Show memory store status: backend type, entry count, and backend-specific details (Qdrant URL, collection, dimensions, status). |
| `/rebuild-index` | Rebuild the vector memory index from MEMORIES.txt. Only available when a vector backend (`fts5` or `qdrant`) is active. Reports stats after completion. |
| `/config` | Show the resolved configuration for the current thread, including which values are overridden vs. defaults. |
| `/config <key> <value>` | Set a per-thread configuration override. Keys: `backend_name`, `session_timeout_minutes`, `sub_sessions_enabled`, `system_prompt_mode`. |
| `/config reset` | Remove all per-thread configuration overrides for the current thread. |
| `/config reset <key>` | Remove a single per-thread override, reverting that key to the default. |
| `/kimi-auth` | Start Kimi-Code OAuth device-code flow. Broadcasts the verification URL to the current chat. Only relevant when a `kimi-code` backend is configured. |
| `/verify-session` | Send an E2EE SAS verification request to all `allowed_users`. Matrix-only. Useful after key changes or to establish cross-signing trust. Has no effect if E2EE is not available. |
| `/commands` | List all available slash commands with short descriptions. |
