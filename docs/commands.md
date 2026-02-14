# Slash Commands

All commands are available in both Matrix and the web UI.

| Command | Description |
|---------|-------------|
| `/new` | Reset conversation history for the current thread. Archives all messages and clears the compaction summary. Also cancels any running sub-sessions bound to this thread. |
| `/compact` | Force context compaction. Summarises older messages and archives them, keeping the last 10 messages intact. Shows before/after token counts. |
| `/reminders` | List all scheduled reminders (active, completed, and failed). |
| `/pulse` | Manually trigger a pulse review. Enqueues a system event asking the AI to review PULSE.txt and report any actions taken. |
| `/status` | Show detailed system status: running asyncio tasks, active sub-sessions, workflow state, pulse/dreaming loop status, and reminder count. |
| `/dream` | Manually trigger the nightly dreaming consolidation of MEMORIES.txt and PULSE.txt. Shows before/after character counts. |
| `/commands` | List all available slash commands with short descriptions. |
