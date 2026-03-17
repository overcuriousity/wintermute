# Documentation Index

This index organizes Wintermute's documentation by operator intent so guides are easier to find and maintain.

## Quick Paths

- New deployment: [Installation](installation.md) -> [Configuration](configuration.md) -> [Best Practices](best-practices.md)
- Chat and interfaces: [Web Interface](web-interface.md) -> [Matrix Setup](matrix-setup.md) or [Signal Setup](signal-setup.md)
- Runtime internals: [Architecture](architecture.md) -> [Autonomy](autonomy.md) -> [Convergence Protocol](convergence-protocol.md)
- Prompt and tool behavior: [System Prompts](system-prompts.md) -> [Tools](tools.md) -> [Commands](commands.md)

## Guides By Area

### 1. Setup & Operations

| Guide | Purpose |
|-------|---------|
| [Installation](installation.md) | Onboarding flow, manual install, systemd deployment |
| [Configuration](configuration.md) | Full `config.yaml` reference and provider examples |
| [Best Practices](best-practices.md) | Isolation, trust model, model tiering, credential handling |
| [Lite Mode](lite-mode.md) | Disable worker delegation globally to reduce cost |

### 2. Interfaces & Channels

| Guide | Purpose |
|-------|---------|
| [Web Interface](web-interface.md) | Debug panel and REST endpoints |
| [Matrix Setup](matrix-setup.md) | Matrix account setup, E2EE behavior, SAS verification |
| [Signal Setup](signal-setup.md) | signal-cli installation, registration, allowlists, troubleshooting |

### 3. Runtime Internals

| Guide | Purpose |
|-------|---------|
| [Architecture](architecture.md) | Components, startup flow, data flow, workflow DAG lifecycle |
| [Autonomy](autonomy.md) | Dreaming, reflection, harvesting, sub-session orchestration |
| [Convergence Protocol](convergence-protocol.md) | Detect/validate/correct pipeline, hooks, configuration |
| [System Prompts](system-prompts.md) | Prompt assembly pipeline and component sizing |

### 4. Tooling & Control Surface

| Guide | Purpose |
|-------|---------|
| [Tools](tools.md) | Tool categories, filtering, profiles, and parameter reference |
| [Commands](commands.md) | Slash commands across Matrix and web |
| [Outcome Tracking](outcome-tracking.md) | Sub-session outcome telemetry and historical feedback injection |
| [Skills Storage](skills-storage.md) | Skill backend architecture, migration, and API/tool behavior |

## Maintenance Notes

- If a new file is added under `docs/`, add it here and in the README documentation table.
- Keep category labels stable so external links and onboarding references stay predictable.
- Prefer adding guides over expanding unrelated pages when introducing major features.
