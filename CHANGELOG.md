# Changelog

## [0.15.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.14.0-alpha...v0.15.0-alpha) (2026-03-27)


### Features

* add send_message tool, fix send_file sub-session routing ([c550a2c](https://github.com/overcuriousity/wintermute/commit/c550a2c6c4567c7e4c69efda2dac23fa8fe4e1e1))
* add send_message tool, fix send_file sub-session routing, strengthen NL task translation ([c034f39](https://github.com/overcuriousity/wintermute/commit/c034f3965934909c35dacb7f728049a610058e04))


### Bug Fixes

* address PR review — store Matrix event_bus sub IDs, fix restart_self category in docs ([9f4d00a](https://github.com/overcuriousity/wintermute/commit/9f4d00a99c5fa500f900d13837ab8b8fc462d3cf))
* address PR review round 2 — keep send_file categorized, use exclude_names ([04a48e7](https://github.com/overcuriousity/wintermute/commit/04a48e783e47c8d8a70058791956f3c1b967473e))
* protect skill merge targets from contradictory delete instructions ([4e7e044](https://github.com/overcuriousity/wintermute/commit/4e7e04453a446e7659648757d617ac3615a78e7f))
* protect skill merge targets from contradictory deletes ([798232e](https://github.com/overcuriousity/wintermute/commit/798232ece206018a1ed1bbfbeb87bf26c91785c5))
* reject sub_xxx thread IDs as non-routable in delivery thread resolution ([3743006](https://github.com/overcuriousity/wintermute/commit/37430061dbb53e825fe6c4dd97aaad9df7ae82d0))
* update Signal event_bus subscription comment to reflect both events ([6ab3d4c](https://github.com/overcuriousity/wintermute/commit/6ab3d4cb3654bb1b2908eb140a9e9610ac3d405a))
* web interface bugs — HTTP status codes, system prompt fallback, frontend error handling ([faa6e86](https://github.com/overcuriousity/wintermute/commit/faa6e8655eec88af91483a14f334fb82084c3a53))
* web interface bugs — proper HTTP status codes, system prompt fallback, frontend error handling ([10c47fa](https://github.com/overcuriousity/wintermute/commit/10c47faa22f37c6a81d6930115c33c1a5ea882f7))

## [0.14.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.13.1-alpha...v0.14.0-alpha) (2026-03-25)


### Features

* expand per-thread config overrides and wire skill success tracking ([ac8cd39](https://github.com/overcuriousity/wintermute/commit/ac8cd39e95839b9d4c1eb1d8c6bbcc763fc159cb))
* expand per-thread config overrides and wire skill success tracking ([89c679b](https://github.com/overcuriousity/wintermute/commit/89c679b881d9523a2dbde6343fe5f03b935e8846))
* per-session backend overrides for all role-based pools ([29048a6](https://github.com/overcuriousity/wintermute/commit/29048a629c54a2bea23f81a1838e426d2011235a))
* per-session backend overrides for all role-based pools ([05726d5](https://github.com/overcuriousity/wintermute/commit/05726d5e193ab6c5f82e025c29248aafe1cf183e))
* weak-model resilience — programmatic validation for dreaming and memory writes ([b1c8c15](https://github.com/overcuriousity/wintermute/commit/b1c8c1501c0892159013b7e5c2e03dd343d7e84e))
* weak-model resilience for dreaming and memory writes ([d075faa](https://github.com/overcuriousity/wintermute/commit/d075faa23c95e77143c26c3b29496b29927e546f))


### Bug Fixes

* address additional copilot reviews on weak-model resilience PR ([6608ad6](https://github.com/overcuriousity/wintermute/commit/6608ad67e6dfcce50164f4c5d9ea0f767f94028a))
* address copilot review comments on context compaction PR ([36e6cd0](https://github.com/overcuriousity/wintermute/commit/36e6cd0dd1758771cbbc7aebfae023c9435e1d17))
* address Copilot review feedback on thread config and skill store ([d412ad2](https://github.com/overcuriousity/wintermute/commit/d412ad2f9c457d0aba84f615c164a15d3f22b141))
* address copilot review issues — case-insensitive repeat check, dead pred_type, force_enable normalization ([c0f64ec](https://github.com/overcuriousity/wintermute/commit/c0f64ec967bc12ede1a03c2103ef817df3482378))
* address copilot review issues — unbounded query and stale plan docs ([5839b48](https://github.com/overcuriousity/wintermute/commit/5839b48d25e378b1a34cc6df361a439501ad2f43))
* address copilot review issues on weak-model resilience PR ([b3cc1b3](https://github.com/overcuriousity/wintermute/commit/b3cc1b31f6bd509421973bc8ae32243616236f4f))
* address fifth round of copilot review comments on context compaction ([96f54fa](https://github.com/overcuriousity/wintermute/commit/96f54fa9d72a3b2921fadeb7dee11248fec92e16))
* address final copilot review issues — source_indices validation and batch survival updates ([6a0d1c1](https://github.com/overcuriousity/wintermute/commit/6a0d1c1276d3022864f78eb332fb0ff00be9be8a))
* address fourth round of copilot review comments on context compaction ([34d4c30](https://github.com/overcuriousity/wintermute/commit/34d4c3017d23cf8773f4d1adc84623ec5f9b71fc))
* address PR review feedback for per-session backend overrides ([df6959c](https://github.com/overcuriousity/wintermute/commit/df6959c90a34a735336346fe1488aef971a10a7d))
* address remaining copilot review comments on context compaction ([b2cd43d](https://github.com/overcuriousity/wintermute/commit/b2cd43d98a78dc774ebf86664eede232a8f03f6f))
* address remaining copilot review issues on weak-model resilience PR ([20b72f3](https://github.com/overcuriousity/wintermute/commit/20b72f39a067150ceab6534cf1f1e2c414d7cbee))
* address second round of copilot review comments on shrink pre-pass ([2a110da](https://github.com/overcuriousity/wintermute/commit/2a110da96fa8c5f2e921590e368b1f167774ea10))
* address seventh round of copilot review comments on context compaction ([d9febd3](https://github.com/overcuriousity/wintermute/commit/d9febd3d61a7c59af2ddd6003886cbf71cc6cbdf))
* address sixth round of copilot review comments on context compaction ([1606e2e](https://github.com/overcuriousity/wintermute/commit/1606e2edda5bd7598c7f3c1b1c46f0c2b60ea7dd))
* address third round of copilot review comments on context compaction ([999fbfd](https://github.com/overcuriousity/wintermute/commit/999fbfdec01e739d360669dbf81a38d552b698c8))
* cap kept-message shrink ops at _MAX_KEPT_SHRINK_OPS=50 ([fe83575](https://github.com/overcuriousity/wintermute/commit/fe83575fbbcdf17d11800458f5df1e7fb80db2de))
* correct shrink_input_limit computation in context compactor ([66b6fae](https://github.com/overcuriousity/wintermute/commit/66b6fae3d7daee8e8c932a846cba8bd61f593508))
* debug panel always shows last actual LLM system prompt ([3bb9ff9](https://github.com/overcuriousity/wintermute/commit/3bb9ff9f0ee97f82769e5445f69df75426ac6d07))
* debug panel always shows last actual LLM system prompt ([90267d2](https://github.com/overcuriousity/wintermute/commit/90267d2095d062c160c279513d2c80b7ec67f351))
* guard backend_overrides type in resolve/resolve_as_dict ([40b2c3d](https://github.com/overcuriousity/wintermute/commit/40b2c3df15355a3bda030495e99acc24fc4133cc))
* make thread_id required (default='default') in update_message_content ([28ccb92](https://github.com/overcuriousity/wintermute/commit/28ccb928112db795918db2380ea5391da131272f))
* restore per-thread config overrides for CP validator and pool ([dcd3757](https://github.com/overcuriousity/wintermute/commit/dcd3757eb23207c15a3f1163a549350d6f92d672))
* restore per-thread config overrides for CP validator and pool ([444f24f](https://github.com/overcuriousity/wintermute/commit/444f24f1af7cf13194b15897ed5dbf48a50673e4))
* **scheduler:** address additional copilot review comments ([07c28fa](https://github.com/overcuriousity/wintermute/commit/07c28fae2c7152822085f4522b25fa1ad2a26bcb))
* **scheduler:** address copilot review - startup blocking and relative past-due check ([fe97584](https://github.com/overcuriousity/wintermute/commit/fe975842a800899630cf2e9e2dcec16d26e9f449))
* **scheduler:** always check task status in _fire_task regardless of schedule_type ([9116b6a](https://github.com/overcuriousity/wintermute/commit/9116b6aeb68e5e7f515b78fdc10749bc2787c510))
* **scheduler:** guard non-active task firing and fix dreaming job removal order ([d5ce0c4](https://github.com/overcuriousity/wintermute/commit/d5ce0c430fdb098132f4a32e9830694ebf2a9e9d))
* **scheduler:** handle None misfire_grace_time in _should_complete_stale_once ([19c53a1](https://github.com/overcuriousity/wintermute/commit/19c53a108d417dc4bda741352b7b3b4a024a6575))
* **scheduler:** respect misfire_grace_time before auto-completing stale once-tasks ([911fcb6](https://github.com/overcuriousity/wintermute/commit/911fcb697be6292ccd7ec082dc8580843da0f6c6))
* **scheduler:** thread-safe async maintenance and remove dreaming.py retention duplicate ([e73ae4e](https://github.com/overcuriousity/wintermute/commit/e73ae4e8f9d48192d871f5c0d82203a4d6d4f4e3))
* **sub-session:** address copilot review comments on PR [#221](https://github.com/overcuriousity/wintermute/issues/221) ([a4bdcb0](https://github.com/overcuriousity/wintermute/commit/a4bdcb0b619f1509375bc57f194d03f131a8347c))
* **sub-session:** address second round of copilot review comments on PR [#221](https://github.com/overcuriousity/wintermute/issues/221) ([f8c04cb](https://github.com/overcuriousity/wintermute/commit/f8c04cbc758d442054c2f5319531e5087bd20b37))
* **sub-session:** improve timing accuracy, context efficiency, and result relay ([c84681a](https://github.com/overcuriousity/wintermute/commit/c84681a7101354258ea416f7a0e948771d3a3bb8))
* **sub-session:** improve timing, context efficiency, and result relay ([832a276](https://github.com/overcuriousity/wintermute/commit/832a2766ed37300667d166241ec7022f590e1fa7))
* **sub-session:** skip [NO_ACTION] sentinel when back-filling task result summary ([b2a2a8b](https://github.com/overcuriousity/wintermute/commit/b2a2a8b021ec64c2ace036cf7b5689c7d5fc733a))
* **task:** align execution-mode runtime/logging semantics ([4eb82ce](https://github.com/overcuriousity/wintermute/commit/4eb82ce291e749d14d06ad30d2a94ee3b31746ce))
* **task:** keep scheduled reminders passive unless ai_prompt is explicit ([674fcf0](https://github.com/overcuriousity/wintermute/commit/674fcf00ec64584d6f52a583747d3b85bf938b7c))
* **task:** keep scheduled reminders passive unless ai_prompt is explicit ([e29fc2a](https://github.com/overcuriousity/wintermute/commit/e29fc2a50e8680ec004f53f719c425a858d2d9d1))
* **task:** preserve legacy default notify + record attempted failures ([2a72ffd](https://github.com/overcuriousity/wintermute/commit/2a72ffd7028584d1eb0c524c8e34e4f8e1255f06))
* **task:** preserve notify semantics and accurate delivery labels ([c623741](https://github.com/overcuriousity/wintermute/commit/c623741440c946bf8236ff91d0dc9e46cb28cc71))
* **tasks:** address copilot review comments ([89b10d1](https://github.com/overcuriousity/wintermute/commit/89b10d1c2e84aa590722ecfbaf988acf8f168dc6))
* **tasks:** make DB authoritative for ai_prompt/execution_mode; support updating them ([1d2125b](https://github.com/overcuriousity/wintermute/commit/1d2125bf5b6cfe6869298f984b9ab68df168fc5a))
* **tasks:** support updating ai_prompt/execution_mode; make DB authoritative at fire time ([f7e36d3](https://github.com/overcuriousity/wintermute/commit/f7e36d3212f23681489f0e0d00f5c6019dd6fd5d))
* two-phase context compaction for few very large messages ([f31ae80](https://github.com/overcuriousity/wintermute/commit/f31ae80a38063acd46d3d32db16585cb884df7d5))
* two-phase context compaction handles few very large messages ([5987ae1](https://github.com/overcuriousity/wintermute/commit/5987ae18c851d9f4deb6a5d5219c06fcd879bba8))


### Dependencies

* **python:** bump anthropic from 0.84.0 to 0.86.0 ([f7bf808](https://github.com/overcuriousity/wintermute/commit/f7bf808432d6f4fba4ed7a8480640aa1ccb92134))
* **python:** bump json-repair from 0.58.5 to 0.58.6 ([9894b25](https://github.com/overcuriousity/wintermute/commit/9894b25efdeef94cd4febc154155ba82ea20bf52))
* **python:** bump openai from 2.28.0 to 2.29.0 ([1cc61ad](https://github.com/overcuriousity/wintermute/commit/1cc61adba0602e4cca842effb2bc1c98de4c7553))
* **python:** bump ruff from 0.15.6 to 0.15.7 ([74f65ef](https://github.com/overcuriousity/wintermute/commit/74f65ef002b52f6af1087ea1d5914d22b01148bd))

## [0.13.1-alpha](https://github.com/overcuriousity/wintermute/compare/v0.13.0-alpha...v0.13.1-alpha) (2026-03-18)


### Bug Fixes

* accurate docstring for log_level; add context to fire-and-forget logs ([0e26e6d](https://github.com/overcuriousity/wintermute/commit/0e26e6d05e4b4d1313042cf0dad638af26971f47))
* address Copilot review feedback on scheduled task auto-promotion ([a390e4b](https://github.com/overcuriousity/wintermute/commit/a390e4b52aec6732931b5a45cd7488da6cadd9ee))
* address PR review — normalize ai_prompt and avoid content duplication ([c189c3d](https://github.com/overcuriousity/wintermute/commit/c189c3d0d6aee94be97b5a0744d6403914bc6fa5))
* align setup.sh and onboarding.sh config output with config.yaml.example ([383d521](https://github.com/overcuriousity/wintermute/commit/383d5218fce812bfe8e3c299e830a21ca9d22571))
* auto-generate ai_prompt for scheduled tasks missing one ([4d1cf47](https://github.com/overcuriousity/wintermute/commit/4d1cf47d290761648c9cb4354d08eef1cca70d30))
* auto-generate ai_prompt for scheduled tasks missing one ([5d98ba6](https://github.com/overcuriousity/wintermute/commit/5d98ba6de30fe852c7aeb161fa4c8ef9297250e8))
* clarify skill tool returns docs, not executes procedures ([5d05202](https://github.com/overcuriousity/wintermute/commit/5d0520212fbc37139259c6a78fa0c8b8beb2baf0))
* debug panel accuracy + remove dead code ([d6f1d66](https://github.com/overcuriousity/wintermute/commit/d6f1d66d038049ffd7140ff37c8c5f98f15f8d00))
* debug panel shows accurate tool schemas; remove dead code ([031c935](https://github.com/overcuriousity/wintermute/commit/031c935c874a525ac163e955ceeb857c04186ac6))
* derive skill collection name from memory collection ([18d8300](https://github.com/overcuriousity/wintermute/commit/18d83007b989b3fb321f0f7e8255736fb41a3b54))
* derive skill collection name from memory collection for instance isolation ([f5e773f](https://github.com/overcuriousity/wintermute/commit/f5e773f90a32d2991c3fbe6ded786225aac61890))
* disable inline_tool_limit CP hook when worker_delegation excluded ([25a4503](https://github.com/overcuriousity/wintermute/commit/25a45039459c8c933f5da9a2ac8a34f6f7d469bd))
* prefer UUID over phone number for Signal read receipts ([1aa057b](https://github.com/overcuriousity/wintermute/commit/1aa057b28779eab0a243450ba30be9eb6a088555))
* prefer UUID over phone number for Signal read receipts ([52653f5](https://github.com/overcuriousity/wintermute/commit/52653f5c22474838521617f7175e4f6a393cd2c7))
* re-enable inline tool limit in lite mode ([edc271a](https://github.com/overcuriousity/wintermute/commit/edc271ace493e05c0da885f9d03563d97b86b67b))
* re-enable inline tool limit in lite mode with adapted correction ([4297936](https://github.com/overcuriousity/wintermute/commit/4297936d5d8794903394ca0db47217544e24dce2))
* restore _update_config_yaml and clear caches on session reset ([ad1e0bb](https://github.com/overcuriousity/wintermute/commit/ad1e0bb5e62925517cf58ffcce7d7a884e8688cc))
* Signal voice transcription and read receipt visibility ([3db0a3f](https://github.com/overcuriousity/wintermute/commit/3db0a3f2d4512529ad21ad7ef784eece3c4dfb60))
* Signal voice transcription and read receipt visibility ([cdc0df1](https://github.com/overcuriousity/wintermute/commit/cdc0df1ef094d6af04b67d3dfd6a021a2d4a3754))

## [0.13.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.12.0-alpha...v0.13.0-alpha) (2026-03-17)


### Features

* add credential redaction CP hook to prevent API key leaks ([f05c117](https://github.com/overcuriousity/wintermute/commit/f05c1179aef99d731b5a9d72a3ec72428dfee45c))
* add credential redaction CP hook to prevent API key leaks ([661b7de](https://github.com/overcuriousity/wintermute/commit/661b7de3cefd4aa5b24808d7a2a00d01854dce24))
* add group chat mode for Matrix rooms ([cfb4575](https://github.com/overcuriousity/wintermute/commit/cfb45756dd3782f89517f682ab4f526041ad2fa9))
* add group chat mode for Matrix rooms ([29910db](https://github.com/overcuriousity/wintermute/commit/29910dbef1d00be6b04103649fac98464482880f))
* add Signal messenger interface via signal-cli ([7b2f70c](https://github.com/overcuriousity/wintermute/commit/7b2f70cedf85098c0306d6447d5470fe8b5e7c01))
* add Signal messenger interface via signal-cli ([670bd0a](https://github.com/overcuriousity/wintermute/commit/670bd0a81bf91aa9be4213e1ed5fd00398ac5148))
* credential redaction CP hook ([a2c1c71](https://github.com/overcuriousity/wintermute/commit/a2c1c717615e168911fe96abd24f30292020e8f9))
* enforce sub_sessions_enabled as global config (lite mode) ([1ed624d](https://github.com/overcuriousity/wintermute/commit/1ed624de12c99915339518a4d3309d1a66e38b11))
* enforce sub_sessions_enabled as global config (lite mode) ([3a5da64](https://github.com/overcuriousity/wintermute/commit/3a5da6450fc3328dde556576920a0e19eebc245f))


### Bug Fixes

* address Copilot review — stale comment, mx-reply pill false positive, docs ([91a90e6](https://github.com/overcuriousity/wintermute/commit/91a90e61bd049486d880ce52406e97aaf51fb1f1))
* address Copilot review — token model, activity tracking, image mention strip ([9ccc0c8](https://github.com/overcuriousity/wintermute/commit/9ccc0c8730dfdd692587032c3ff2f1e61f0927c9))
* address Copilot review feedback on credential redaction ([35b60e1](https://github.com/overcuriousity/wintermute/commit/35b60e1630dc89a10282e76ece5e61690d00b72d))
* address PR review — add concurrency guard, cooldown sync, and emit throttle ([82981f9](https://github.com/overcuriousity/wintermute/commit/82981f91428d1bb1b687cd26e4794925b5cc19ce))
* address PR review feedback for Signal interface ([61dfc45](https://github.com/overcuriousity/wintermute/commit/61dfc45c833ff8029cc74e9175139b8627e53915))
* address remaining PR review — race-free guard, cooldown logic, fallback ([bbd10db](https://github.com/overcuriousity/wintermute/commit/bbd10db31ad0549eea2c34054d4b7a02e222de42))
* address second Copilot review round ([2c01da2](https://github.com/overcuriousity/wintermute/commit/2c01da2868b8de11d64ad0d3eb979805de8f68d6))
* address second round of PR review feedback ([9e2a188](https://github.com/overcuriousity/wintermute/commit/9e2a188e5b77b31d50823e2bb0b637a08ab49210))
* address third round of PR review feedback ([e041ee4](https://github.com/overcuriousity/wintermute/commit/e041ee483138e02c2447fd8b07982ebb5dbe5603))
* check _firing before reset/log in event handlers ([54860b4](https://github.com/overcuriousity/wintermute/commit/54860b457e77f51a6e311a792cb9b60246a30380))
* drop inline summarisation fallback for skills — missing event bus is an error ([421f05a](https://github.com/overcuriousity/wintermute/commit/421f05a071081b844c2c259de5926480ec0528e4))
* gate compaction summaries on ephemeral, skip reply fetch in group mode ([9997b9a](https://github.com/overcuriousity/wintermute/commit/9997b9ace3d44ab44f6704736fceaed74e839e7e))
* gate group-mode mentions on allowed_users (match Matrix behavior) ([e9b79df](https://github.com/overcuriousity/wintermute/commit/e9b79df46ba3fad6459a7d0289376e57931064b5))
* log full UUID in Signal ACL rejection message ([4ffe815](https://github.com/overcuriousity/wintermute/commit/4ffe81512e669b148d4720ec37e7026e8ffae1db))
* pass available_tools in pre-compaction prompt reassembly ([d930b6f](https://github.com/overcuriousity/wintermute/commit/d930b6f0b022eebe09701fdd03c48a3007990e53))
* rename cooldown field, only set on success, throttle warning log ([a3bc66f](https://github.com/overcuriousity/wintermute/commit/a3bc66f351897e308c713b6848b4c5bed9f6979f))
* resolve per-thread model for silent store, require allowed_rooms in group mode ([0449070](https://github.com/overcuriousity/wintermute/commit/044907005e3f5f2769b3910ecd8db6ce25ea5aa1))
* restrict group mode to single-turn ephemeral conversations ([b0a52b6](https://github.com/overcuriousity/wintermute/commit/b0a52b6f63631efe426cc180d1e8569f1963a9b6))
* robust mention detection, suppress events for silent stores ([e552ac5](https://github.com/overcuriousity/wintermute/commit/e552ac5f92b36199c0ff5e848f26c0d0e312a948))
* route skills oversize to dreaming cycle instead of inline LLM ([8bac8a0](https://github.com/overcuriousity/wintermute/commit/8bac8a067dd01b2074aceb2e200ab3b161dc689b))
* route skills oversize to dreaming cycle instead of inline LLM ([78d229e](https://github.com/overcuriousity/wintermute/commit/78d229eea9a11f85b64871e2f82eb2456cc27c0c)), closes [#190](https://github.com/overcuriousity/wintermute/issues/190)
* strip plaintext reply fallback, consolidate group-mode gating, skip seed ([c85d759](https://github.com/overcuriousity/wintermute/commit/c85d7597d967942e95a67ff893cae09a23e11c8e))
* tighten mention detection to pill/structured only, guard get() ([39a6a34](https://github.com/overcuriousity/wintermute/commit/39a6a342d2eae6b3e1d636271afe88107c72d423))
* use daemon mode without --json for newer signal-cli versions ([c8764af](https://github.com/overcuriousity/wintermute/commit/c8764af0e5f04f3d7f596baf4daeebb2eae90b6b))
* use jsonRpc subcommand and elevate stderr logging to WARNING ([64fb716](https://github.com/overcuriousity/wintermute/commit/64fb716c1d426d449edbcaa17678605f650febee))
* use jsonRpc subcommand for newer signal-cli ([1217c3e](https://github.com/overcuriousity/wintermute/commit/1217c3eb8539fb256777f74b0192fa490c4cf83d))
* use jsonRpc subcommand for newer signal-cli versions ([ddbc584](https://github.com/overcuriousity/wintermute/commit/ddbc58460999eb79642240e32f0be3aade117692))


### Documentation

* add instructions for finding Signal group IDs ([fe4c203](https://github.com/overcuriousity/wintermute/commit/fe4c203840678e4c4015963a5cc76fb99f4adb05))
* add instructions for finding Signal UUIDs ([6b0dba3](https://github.com/overcuriousity/wintermute/commit/6b0dba3ce919cbd8f4fdd43dc19ebbc036975704))
* clarify UUID requirement for users with hidden phone numbers ([463da45](https://github.com/overcuriousity/wintermute/commit/463da4509836ad97eaab84b07b3dae914a7861d7))
* update signal-setup.md for HTTP daemon mode and UUID support ([3671679](https://github.com/overcuriousity/wintermute/commit/367167943ae5570bf12eba6b0a86b27d9eecf653))

## [0.12.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.11.2-alpha...v0.12.0-alpha) (2026-03-15)


### Features

* add response scaffolding for sub-sessions ([0a18a16](https://github.com/overcuriousity/wintermute/commit/0a18a1660135a308c17663d20af419d394f569e1))
* add response scaffolding for sub-sessions ([2824d2c](https://github.com/overcuriousity/wintermute/commit/2824d2c864fd672c0c921af07789218e78f324f9)), closes [#179](https://github.com/overcuriousity/wintermute/issues/179)


### Bug Fixes

* add json-repair fallback for malformed tool call arguments ([32c1884](https://github.com/overcuriousity/wintermute/commit/32c18847aacbe7968f10edc2803cc35335bbe443)), closes [#177](https://github.com/overcuriousity/wintermute/issues/177)
* add json-repair fallback for malformed tool calls ([de1a700](https://github.com/overcuriousity/wintermute/commit/de1a700f0861ebb75e2522cefaaba2bf2b6cb447))


### Dependencies

* **python:** Bump numpy from 2.4.2 to 2.4.3 ([8da0408](https://github.com/overcuriousity/wintermute/commit/8da04088afa3fd1edb111454ad72e1c1f69dac7d))
* **python:** Bump openai from 2.26.0 to 2.28.0 ([b49101b](https://github.com/overcuriousity/wintermute/commit/b49101bc86e5cc4cb546acb51273eb4946c20b0a))
* **python:** Bump qdrant-client from 1.17.0 to 1.17.1 ([1a75e35](https://github.com/overcuriousity/wintermute/commit/1a75e356a5406340042f91cb9f737fdfd0f0a11a))
* **python:** Bump ruff from 0.15.5 to 0.15.6 ([0a1ea0b](https://github.com/overcuriousity/wintermute/commit/0a1ea0be544111f76205bddcd8af83484f24dd23))

## [0.11.2-alpha](https://github.com/overcuriousity/wintermute/compare/v0.11.1-alpha...v0.11.2-alpha) (2026-03-14)


### Bug Fixes

* address race conditions, missing timeouts, and resource leaks ([4a77daf](https://github.com/overcuriousity/wintermute/commit/4a77dafd07def9e1f8ffa370da032ae4df696030))

## [0.11.1-alpha](https://github.com/overcuriousity/wintermute/compare/v0.11.0-alpha...v0.11.1-alpha) (2026-03-14)


### Bug Fixes

* address multiple bugs found during code review ([75836c8](https://github.com/overcuriousity/wintermute/commit/75836c80a866640d9dad1a7b0f9392ab42f8d4d4))

## [0.11.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.10.0-alpha...v0.11.0-alpha) (2026-03-14)


### Features

* add restart_self tool for self-initiated process restart ([df7d2cb](https://github.com/overcuriousity/wintermute/commit/df7d2cbf59ce1316ff6960f36602495dbcbb7a26))
* add restart_self tool for self-initiated process restart ([7375484](https://github.com/overcuriousity/wintermute/commit/737548472a50a04909addc127d316a2446246efb))
* add send_file tool, replace [send_file:] text parsing ([35524dc](https://github.com/overcuriousity/wintermute/commit/35524dc13287202d6246caeca365e2e249fdeea5))
* add send_file tool, replace text-marker parsing ([e6b8f61](https://github.com/overcuriousity/wintermute/commit/e6b8f61f24822f8b26e95198418f00cca7344f84))
* formalize execute_tool() into a syscall-like interface ([2ee5487](https://github.com/overcuriousity/wintermute/commit/2ee5487a7c4338df59ce00e637f99fb523bef58c))


### Bug Fixes

* address Copilot review findings ([44b31a4](https://github.com/overcuriousity/wintermute/commit/44b31a499525f835e96a412c198dcd5b20a03638))
* address Copilot review findings on memory store PR ([b206d5c](https://github.com/overcuriousity/wintermute/commit/b206d5cf8cc8e5a97d0a84a1758fe66d385878ac))
* address eighth round of Copilot review findings ([c252f4f](https://github.com/overcuriousity/wintermute/commit/c252f4fba24059ac60279d5065989bfde1126f37))
* address fifth round of Copilot review findings ([cc8d45b](https://github.com/overcuriousity/wintermute/commit/cc8d45bbbea8d438513052a10e6b7659336006ce))
* address fifth round of Copilot review findings ([594faf2](https://github.com/overcuriousity/wintermute/commit/594faf233658ddef3708ede44267895be6187710))
* address fourth round of Copilot review findings ([3e59081](https://github.com/overcuriousity/wintermute/commit/3e590814cf0b4f3c25419631910f9880207e6d88))
* address fourth round of Copilot review findings ([a0d6b3c](https://github.com/overcuriousity/wintermute/commit/a0d6b3caa625f837c4ff5ef1e9e41ed93c58646c))
* address PR review — thread safety, execv args, named constant ([1040bd2](https://github.com/overcuriousity/wintermute/commit/1040bd226bd4913f5dc62e360fe9b9ab96c9e3dd))
* address second round of Copilot review findings ([934838b](https://github.com/overcuriousity/wintermute/commit/934838b146764e6a372f9d3629f6f34d9136d715))
* address second round of Copilot review findings ([af70cbb](https://github.com/overcuriousity/wintermute/commit/af70cbb44fa1e1c5ebbbd4727eb699b37ed54c16))
* address second round of Copilot review findings ([9cd0961](https://github.com/overcuriousity/wintermute/commit/9cd0961dae52306b3512af74a3a361fe63030963))
* address seventh round of Copilot review findings ([844fd2a](https://github.com/overcuriousity/wintermute/commit/844fd2a500224df9a82b181140ebce3d45c9bafb))
* address sixth round of Copilot review findings ([e564d87](https://github.com/overcuriousity/wintermute/commit/e564d876759513c3ae0500d04b44cff3ab5ba5df))
* address third round of Copilot review findings ([0a77b95](https://github.com/overcuriousity/wintermute/commit/0a77b95150f683d91866ec9626fb2c2a84cbef18))
* address third round of Copilot review findings ([d848c51](https://github.com/overcuriousity/wintermute/commit/d848c51fccd9eb635a677be8a13314122a397b19))
* ensure legacy turing_* actions get convergence color coding in d… ([44e2c7a](https://github.com/overcuriousity/wintermute/commit/44e2c7a9129150c64ce3f2eb0f4ab1e2c0e004a7))
* ensure legacy turing_* actions get convergence color coding in debug UI ([c66bfd1](https://github.com/overcuriousity/wintermute/commit/c66bfd1f6774c23bb450ff0e3889db0f370b7d05))
* make embeddings endpoint mandatory in setup and onboarding scripts ([4033f2b](https://github.com/overcuriousity/wintermute/commit/4033f2bc1ef9c2a9d0db0cb30174081d4a9e2ed6))
* memory dedup overhaul — remove MEMORIES.txt, add add-time dedup ([868e963](https://github.com/overcuriousity/wintermute/commit/868e96365040829fd65f4fb44ceb7424f320e8fb))
* recategorize skill tool from orchestration to research ([b86055d](https://github.com/overcuriousity/wintermute/commit/b86055de9809239e2dc0acacf3baa76875da898b))
* remove MEMORIES.txt, add similarity-based dedup to memory store ([0e8f6e2](https://github.com/overcuriousity/wintermute/commit/0e8f6e2ec98515e52155dee7a476f9dd1dc620d1))
* remove redundant RestartForceExitStatus, clarify fallback naming ([cbfd464](https://github.com/overcuriousity/wintermute/commit/cbfd4645c2576c4a7a25297880e7f35af21e917d))
* restore bump_access forwarding in module-level search() ([4f8e286](https://github.com/overcuriousity/wintermute/commit/4f8e2866a86b9aad27278f7f95e3a2a6876a73f0))
* single-source event emission to prevent double-emit ([cccac6a](https://github.com/overcuriousity/wintermute/commit/cccac6aa381421e98154672b094ceff1d46ae24b))


### Reverts

* remove syscall interface (not requested) ([752529c](https://github.com/overcuriousity/wintermute/commit/752529c1be0a84cf2d839a02527b84d103307ddf))


### Documentation

* update tool category references after skill recategorization ([6a346c5](https://github.com/overcuriousity/wintermute/commit/6a346c5b4c0ebc22da88e1d073600ea300be93af))

## [Unreleased]

### Breaking Changes

* **remove: flat-file memory backend** — The `flat_file` memory backend has been removed. All memory storage now uses ranked/searchable backends (`fts5`, `local_vector`, `qdrant`). If your `config.yaml` has `backend: "flat_file"`, change it to `backend: "fts5"` for a zero-config replacement. Startup with an unknown backend name now raises a `ValueError` instead of silently falling back (init-time failures for valid backends still fall back to `fts5`). The startup fallback on init failure has changed from `flat_file` to `fts5`. MEMORIES.txt remains as a git-versioned export target (written by `append_memory` dual-write and dreaming `working_set_export` phase).

## [0.10.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.9.0-alpha...v0.10.0-alpha) (2026-03-12)


### Features

* wire up dead autonomy systems — self-model injection, prediction persistence, session timeouts ([aeb42f6](https://github.com/overcuriousity/wintermute/commit/aeb42f6483a533d34e9800dd107a048b4179cbb2))
* wire up dead autonomy systems — self-model injection, prediction persistence, session timeouts ([54078a7](https://github.com/overcuriousity/wintermute/commit/54078a70d63167fb20efdc095209ef01c26c08cf))


### Bug Fixes

* 6 bugs — query param validation, SSE broadcast, time-gate cleanup, silent logging ([ed0f1be](https://github.com/overcuriousity/wintermute/commit/ed0f1bed4a4e28f23f8eebe3f99c92042e3f2031))
* address 3 more Copilot review findings ([d3df642](https://github.com/overcuriousity/wintermute/commit/d3df6429d875c101775329388436d0337d7f2b32))
* address 3 more Copilot review findings ([074630c](https://github.com/overcuriousity/wintermute/commit/074630c7b247182542042f3c35ac379b3bf10747))
* address 4 more Copilot review findings ([b721f32](https://github.com/overcuriousity/wintermute/commit/b721f32d9e21203dca471346e2374e83ead260b1))
* address 4 more Copilot review findings ([f27fcd0](https://github.com/overcuriousity/wintermute/commit/f27fcd0a5ec89bd778cc107aeb97bf53e578d191))
* address 5 more Copilot review findings ([8c3dff1](https://github.com/overcuriousity/wintermute/commit/8c3dff12493234909d9f3a9cb61a8549b4a57f24))
* address 6 additional Copilot review findings ([3dea2bc](https://github.com/overcuriousity/wintermute/commit/3dea2bcdf7c4570de1d700ddd0da83f6c30270bb))
* address 7 Copilot review findings on autonomy wiring PR ([7551788](https://github.com/overcuriousity/wintermute/commit/7551788fe03d898f2794df0ac1fcfb096b71b905))
* address latest Copilot review findings ([e429159](https://github.com/overcuriousity/wintermute/commit/e429159113f6fdf910d4ed41d0ae256d71e09a7d))
* address remaining Copilot review findings on autonomy PR ([5d383b4](https://github.com/overcuriousity/wintermute/commit/5d383b4d29124ab5714d4ee6aebcfc86a5082769))
* gate self-model injection on explicit enablement and normalize reflection task scheduling ([53aaae2](https://github.com/overcuriousity/wintermute/commit/53aaae2f8611dc02ff823fd2fe9616d545c26990))
* session timeout re-check logic and docs numbering ([f4bd616](https://github.com/overcuriousity/wintermute/commit/f4bd61625c5d71fb093d8b974d2303c614ffe71c))
* sync TaskNode status before workflow done-count in _report() ([d9064d8](https://github.com/overcuriousity/wintermute/commit/d9064d8a4558035554c98030868dae403469b41b))
* use public resolve_config API and module-level json import ([5a16ba3](https://github.com/overcuriousity/wintermute/commit/5a16ba385f42b81f0ec244115a044a0ba778c195))

## [0.9.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.8.0-alpha...v0.9.0-alpha) (2026-03-11)


### Features

* preserve scratchpad dirs post-workflow; enrich /status output ([95fc00b](https://github.com/overcuriousity/wintermute/commit/95fc00b6ad2918efc3b8a3d4c989c988a9cd9fe6))


### Bug Fixes

* address 4 bugs in sub_session — empty response loop, workflow flood, visibility, cleanup ([0a6c0a8](https://github.com/overcuriousity/wintermute/commit/0a6c0a854b458a423ea0e4c9816d8af1e17d44fb))
* address Copilot review — harden per-thread queue concurrency ([469b002](https://github.com/overcuriousity/wintermute/commit/469b002f0efd15da2fb4885f6a7eff20e4c42da5))
* address PR review feedback from Copilot ([897d292](https://github.com/overcuriousity/wintermute/commit/897d2920b265c7654e53ca1d2c1b5efd91c9defc))
* harden dispatch/cleanup race, gate workers on startup, await bg tasks ([5d3913e](https://github.com/overcuriousity/wintermute/commit/5d3913ef5543526cdd7d771581c482007ba00b58))
* import Callable in session_manager; drop stale line counts from docs ([9ee089d](https://github.com/overcuriousity/wintermute/commit/9ee089d8afcb002396c2bce585df2f370541d6b3))
* prevent orphaned queue on idle exit with pending items ([e1d84c4](https://github.com/overcuriousity/wintermute/commit/e1d84c432ee42137de866f530d4a382a674ec957))
* reset empty_retries on success; guard workflow report double-delivery ([3045714](https://github.com/overcuriousity/wintermute/commit/3045714dc5f7d9fc45f80db161c732a023b1d214))
* soften profile field description per review ([4449607](https://github.com/overcuriousity/wintermute/commit/44496074bdc1ec1c8a54548558b4055bad4d94a4))

## [0.8.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.7.0-alpha...v0.8.0-alpha) (2026-03-09)


### Features

* prediction consumption pipeline ([db0b678](https://github.com/overcuriousity/wintermute/commit/db0b6787c7c34ceb0f6df0b3ff24aee203e05b2f))
* prediction consumption pipeline — close the generate→act loop ([ee298cd](https://github.com/overcuriousity/wintermute/commit/ee298cd9462a3fd4e7263ecbfad35555c58bf208))
* **web:** upgrade skills panel for vector skill store ([8dea2ba](https://github.com/overcuriousity/wintermute/commit/8dea2baed4e2c9bec3cd7ed193d094355161b304))


### Bug Fixes

* add track_access to MemoryBackend protocol and all backends ([16d8a88](https://github.com/overcuriousity/wintermute/commit/16d8a88f9f78c027f5693303e21ee53ef990c30a))
* address Copilot review — payload safety, pred_type accuracy, docs ([ee38f38](https://github.com/overcuriousity/wintermute/commit/ee38f38f5b88368371f8c9280f3ed97cd62f775f))
* address Copilot review feedback (AM/PM parsing, imports, docs) ([a47966d](https://github.com/overcuriousity/wintermute/commit/a47966dfb7894fb32336a00056f30c450feb8462))
* address eighth round of PR review feedback ([3c16e0c](https://github.com/overcuriousity/wintermute/commit/3c16e0cab189c621f0f325e91977ff39753b2d50))
* address fifth round of PR review feedback ([3bb98ea](https://github.com/overcuriousity/wintermute/commit/3bb98eaf2c4023566c910bf60d2b1205d48f96a4))
* address fourth round of PR review feedback ([161f2d4](https://github.com/overcuriousity/wintermute/commit/161f2d4bc6e59207c9c6a6b294d8e6bbf07e82c0))
* address PR review comments ([20b5aba](https://github.com/overcuriousity/wintermute/commit/20b5aba80fff1d243ea7fb2ae46660bc9391e39c))
* address PR review feedback on prediction consumption ([162cf0e](https://github.com/overcuriousity/wintermute/commit/162cf0eba1da79f83f6ba2fee3d3299d05f60d2d))
* address second round of PR review feedback ([08c5173](https://github.com/overcuriousity/wintermute/commit/08c51738e4ef233782083a75fd6ba9414cdd1e4c))
* address seventh round of PR review feedback ([7a1dc03](https://github.com/overcuriousity/wintermute/commit/7a1dc03ea09242954676b7f0adee037d04b85a8e))
* address sixth round of PR review feedback ([76f8e2d](https://github.com/overcuriousity/wintermute/commit/76f8e2dfc15b7f3342359500b650a4cb1494c849))
* address third round of PR review feedback ([32f6894](https://github.com/overcuriousity/wintermute/commit/32f6894f7d8db546dfc2f3ad04a9d20733f7d802))
* capture AM/PM per-endpoint and clear retired_at on prediction re-insert ([2628c4a](https://github.com/overcuriousity/wintermute/commit/2628c4a5c0dea40541643555bf3e31b581288119))
* eliminate two-phase construction in main.py ([#82](https://github.com/overcuriousity/wintermute/issues/82)) ([1953262](https://github.com/overcuriousity/wintermute/commit/19532628f14f005ee51bacbc685c441b4235a136))
* eliminate two-phase construction in main.py ([#82](https://github.com/overcuriousity/wintermute/issues/82)) ([d46d292](https://github.com/overcuriousity/wintermute/commit/d46d292aab4c7e0ce488f67510a5f79b058981a9))
* preserve prediction accuracy counters on re-insert; per-point Qdrant access bump ([a5ba271](https://github.com/overcuriousity/wintermute/commit/a5ba271a8f7a4656227c7bfb8a6c498c56e4e138))
* route proactive prediction sub-sessions to target thread ([c640089](https://github.com/overcuriousity/wintermute/commit/c640089951de85c2ff91d6cc0aa14a0dd87129a8))
* use module-level function for prediction check job ([96ea458](https://github.com/overcuriousity/wintermute/commit/96ea45830c52fe653f3e63d08b153780e9609fc3))

## [0.7.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.6.0-alpha...v0.7.0-alpha) (2026-03-06)


### Features

* add inline_tool_limit Turing Protocol hook ([43656fd](https://github.com/overcuriousity/wintermute/commit/43656fda2e44d010901532e0cd046be7bab3e09a))
* add scratchpad view to web interface ([ce00210](https://github.com/overcuriousity/wintermute/commit/ce00210005b76a297d8cd5a0af1ed3a4b2f418f4))
* autonomy-first overhaul for scheduled tasks ([453a8cf](https://github.com/overcuriousity/wintermute/commit/453a8cff313003eb6c2847a8b0d85d8c6242c506))
* migrate skills from flat files to vector-indexed storage ([a2d2ac0](https://github.com/overcuriousity/wintermute/commit/a2d2ac0f61cd3ddf955ba3f55d09c372ffe1bb05))
* migrate skills from flat files to vector-indexed storage ([7d504d4](https://github.com/overcuriousity/wintermute/commit/7d504d4c0ac11b22c0ce2ea3754071b2f9e0fc70))
* **rescue:** add [TOOL_CALL] bracket pattern with hash-rocket and CLI-arg parsing ([7b53cd6](https://github.com/overcuriousity/wintermute/commit/7b53cd66f73650a72698c154746f7815588b74f9))
* track LLM backend in outcome stats and TP violation statistics ([d847de5](https://github.com/overcuriousity/wintermute/commit/d847de54e5f04cca795e48766793c852efc739c1))


### Bug Fixes

* abort delete on archive failure, fix merge upsert, align NL prompt ([5786a65](https://github.com/overcuriousity/wintermute/commit/5786a6523f83b775b4b98c93a70b4aacd5c29f3e))
* address Copilot review comments on PR [#157](https://github.com/overcuriousity/wintermute/issues/157) ([eb8f250](https://github.com/overcuriousity/wintermute/commit/eb8f2509864af124efb9fa35e1da5b221bfea160))
* address Copilot review comments on PR [#157](https://github.com/overcuriousity/wintermute/issues/157) ([081d7e6](https://github.com/overcuriousity/wintermute/commit/081d7e63c1b545a09e6815c174a088c78121cb63))
* address remaining Copilot review comments on PR [#157](https://github.com/overcuriousity/wintermute/issues/157) ([bddf9b4](https://github.com/overcuriousity/wintermute/commit/bddf9b4fa08dc10d201d50c7cc55ca1d0c969e35))
* address review findings — bugs, stale docs, missing exists() ([981c399](https://github.com/overcuriousity/wintermute/commit/981c3997e03397b46e3783c56b6df85169e75783))
* address second round of Copilot review comments ([0f6fc17](https://github.com/overcuriousity/wintermute/commit/0f6fc172e862d287bc02a791ca005f267fe05a3b))
* pass explicit max_tokens=context_size when limit is 0 to prevent server-side truncation ([8a985f4](https://github.com/overcuriousity/wintermute/commit/8a985f4c48f6bab906e7a71a5638dd59ec4a1159))
* remove oneOf from skill schema, upgrade dreaming log level ([f431dc6](https://github.com/overcuriousity/wintermute/commit/f431dc6d106df4eda717f69d2957c4dc0772f5ec))
* replace warning with neutral mode field in task response ([cecd7ac](https://github.com/overcuriousity/wintermute/commit/cecd7acf6713d91963a7feec8356ffab874cdb16))
* resolve Copilot review comments on PR [#157](https://github.com/overcuriousity/wintermute/issues/157) ([960f74a](https://github.com/overcuriousity/wintermute/commit/960f74a60cc90b3ba63a9adbd6aae66f31ee81a6))
* run Turing Protocol on system event responses ([9b59e4a](https://github.com/overcuriousity/wintermute/commit/9b59e4a3b7249bebc95876d491c26f8783bc04d2))
* skill name validation in web endpoints + SIM201 in skill_io ([6fd13c9](https://github.com/overcuriousity/wintermute/commit/6fd13c9c8335ce70976b0d7e5d504c10c243d816))
* suppress cross-turn false positives in Turing Protocol Stage 2 validators ([74a47dc](https://github.com/overcuriousity/wintermute/commit/74a47dc69432b5e2906aff988aa44ad82472683b))


### Documentation

* update stale add_skill references to unified skill tool ([74ed30c](https://github.com/overcuriousity/wintermute/commit/74ed30c98cc8995983b0357ee6c8274031fe911a))

## [0.6.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.5.1-alpha...v0.6.0-alpha) (2026-03-01)


### Features

* add status/cancel actions to worker_delegation + audit tool descriptions ([0a1ca38](https://github.com/overcuriousity/wintermute/commit/0a1ca38bdaa97e4262e88af9452407a1506c0432))
* add status/cancel to worker_delegation + audit descriptions ([d8d6834](https://github.com/overcuriousity/wintermute/commit/d8d6834b224aab4daccfae9761f361f1a522c69f))


### Bug Fixes

* add configurable batch_size for embeddings to prevent LiteLLM 500 errors ([384ec8b](https://github.com/overcuriousity/wintermute/commit/384ec8b6943c0b6a27e4b4214af1dd243e66e9de))
* add input validation for tuning config values ([609fbb6](https://github.com/overcuriousity/wintermute/commit/609fbb621b0c036d430e04c0ea9abae21efee96a))
* address PR [#143](https://github.com/overcuriousity/wintermute/issues/143) review - unused import and reasoning_content round-trip ([2c05be3](https://github.com/overcuriousity/wintermute/commit/2c05be3eeb7e412291d6d66c51d5d6204a233fc9))
* address PR review — thread safety, scoping, validation ([e4840a8](https://github.com/overcuriousity/wintermute/commit/e4840a8330d750482c75403464ac178b9e3d2e87))
* address PR review — truncation notice cap, multi-item coverage, token→char conversion ([008e379](https://github.com/overcuriousity/wintermute/commit/008e3795e3f1c52e7dd8441ab836119cd24299f9))
* address PR review — type annotations and retry path consistency ([cbfc68a](https://github.com/overcuriousity/wintermute/commit/cbfc68a018b161148c1f8edeb48f58950a35a9f4))
* address PR review comments ([594e6b2](https://github.com/overcuriousity/wintermute/commit/594e6b2daf7517cf599bcccb152649461446ea69))
* address review comments on TuringProtocolRunner ([0a760d3](https://github.com/overcuriousity/wintermute/commit/0a760d38711a46f844f3f7513794c54e13c6d6f5))
* address review comments on TuringProtocolRunner ([bd690d0](https://github.com/overcuriousity/wintermute/commit/bd690d0b7d3933d7af09ebd125995ffc72703cbf))
* cap combined multi-item output to prevent context overflow ([b462a30](https://github.com/overcuriousity/wintermute/commit/b462a30d55582872bd8f59daf603d4a4718d5958))
* catch concurrent.futures.TimeoutError, fix status callable signature ([27a8874](https://github.com/overcuriousity/wintermute/commit/27a88742cc9529ab2be27147fe4791696d1f8ea6))
* configurable embeddings batch_size to prevent LiteLLM 500 errors ([a244df4](https://github.com/overcuriousity/wintermute/commit/a244df4d031e7779f58eaa6af74cf55023628ec5))
* pass tool_deps as parameter to static _build_system_prompt ([f78101d](https://github.com/overcuriousity/wintermute/commit/f78101dddfdc54c351e9016cb8334a9ceffb885b))
* remove redundant _adopt_orphan_deps call from _find_or_create_workflow ([b393be0](https://github.com/overcuriousity/wintermute/commit/b393be0fae87f83f8a72bb15607a282697373e0a))
* replace lambda with named function to satisfy E731 ([aa4e767](https://github.com/overcuriousity/wintermute/commit/aa4e7679f61212d53ed93cac488ba20ad5acb085))
* truncate oversized tool outputs to prevent context overflow in sub-sessions ([6e2ce5c](https://github.com/overcuriousity/wintermute/commit/6e2ce5ce873f0fe61d8077bba499d4d8314f7a68))
* truncate oversized tool outputs to prevent sub-session context overflow ([f3f0e60](https://github.com/overcuriousity/wintermute/commit/f3f0e60324c176fd93880a3a91fcef7b8f7ce75e))


### Documentation

* add refactor strategy for remaining issues [#79](https://github.com/overcuriousity/wintermute/issues/79)-[#108](https://github.com/overcuriousity/wintermute/issues/108) ([a246424](https://github.com/overcuriousity/wintermute/commit/a246424749f6539b333cf1e18c83e5083ed88cf4))
* refactor strategy for remaining issues [#79](https://github.com/overcuriousity/wintermute/issues/79)–[#108](https://github.com/overcuriousity/wintermute/issues/108) ([e855b3e](https://github.com/overcuriousity/wintermute/commit/e855b3e657494bc9b257cc95eb82680560a96c99))

## [0.5.1-alpha](https://github.com/overcuriousity/wintermute/compare/v0.5.0-alpha...v0.5.1-alpha) (2026-02-28)


### Bug Fixes

* address all PR [#134](https://github.com/overcuriousity/wintermute/issues/134) review comments on tool_call_rescue ([e914f6d](https://github.com/overcuriousity/wintermute/commit/e914f6d2a2faa1c513daaf697b8c7ea858d007a4))
* address Copilot review comments ([9388fcc](https://github.com/overcuriousity/wintermute/commit/9388fcc0f7e3caaaac7fe97a57d5320b459eb08c))
* address Copilot review comments on PR [#127](https://github.com/overcuriousity/wintermute/issues/127) ([9c9581b](https://github.com/overcuriousity/wintermute/commit/9c9581b71a68beae9135f2acbb86f274f5328bf3))
* address Copilot review comments on PR [#128](https://github.com/overcuriousity/wintermute/issues/128) ([394d66a](https://github.com/overcuriousity/wintermute/commit/394d66ae8f6de6355cab0c0f006c328d85cb847d))
* address Copilot review comments on PR [#129](https://github.com/overcuriousity/wintermute/issues/129) ([d777388](https://github.com/overcuriousity/wintermute/commit/d77738883d9ff84456a42c88e098b9c9947c0616))
* address latest PR [#134](https://github.com/overcuriousity/wintermute/issues/134) review comments on tool_call_rescue ([caf70de](https://github.com/overcuriousity/wintermute/commit/caf70de91bd95e5b2ec9d43cb0cbd1a540b7397b))
* address new PR [#134](https://github.com/overcuriousity/wintermute/issues/134) review comments on tool_call_rescue ([f35ded3](https://github.com/overcuriousity/wintermute/commit/f35ded36e67fb0588ae3b1593355fd895a7a971e))
* address PR [#129](https://github.com/overcuriousity/wintermute/issues/129) review comments in _update_config_yaml ([99cac82](https://github.com/overcuriousity/wintermute/commit/99cac82093859da623a7b80ac92171ddc56e553d))
* address PR [#137](https://github.com/overcuriousity/wintermute/issues/137) review comments ([23479ad](https://github.com/overcuriousity/wintermute/commit/23479ad6197e5827cc8cc055a9c361491c17bcbf))
* address remaining PR [#129](https://github.com/overcuriousity/wintermute/issues/129) review comments in matrix_thread ([56a370f](https://github.com/overcuriousity/wintermute/commit/56a370f31b334731621fc28f7656faf545db8885))
* address remaining PR [#134](https://github.com/overcuriousity/wintermute/issues/134) review comments ([b8a718e](https://github.com/overcuriousity/wintermute/commit/b8a718edf1754cc3d48611dd6010f8ce1fc6bfdf))
* **backend_pool:** address Copilot review issues in raise-none guard ([a739b5e](https://github.com/overcuriousity/wintermute/commit/a739b5ef5ea8b1013284b3a9230e421701ffee49))
* **backend_pool:** guard against raise None when all backends skipped ([283fdde](https://github.com/overcuriousity/wintermute/commit/283fddec4bfe90647b266f7371f60709a2ed0d9b))
* **backend_pool:** guard against raise None when all backends skipped ([2ff00a0](https://github.com/overcuriousity/wintermute/commit/2ff00a0d244cb5727eb435d9f762314ff8cbb142)), closes [#69](https://github.com/overcuriousity/wintermute/issues/69)
* **backend_pool:** guard against raise None when all backends skipped ([a499d63](https://github.com/overcuriousity/wintermute/commit/a499d6371c523148a1c80cebd82d22a74e621302)), closes [#69](https://github.com/overcuriousity/wintermute/issues/69)
* complete DATA_DIR centralization; keep SEARXNG env var fallback ([a3f0b85](https://github.com/overcuriousity/wintermute/commit/a3f0b858df8a7b44cf36a8c261c4cc7a57207fe4))
* **data_versioning:** address Copilot review issues in commit_async/drain ([2238545](https://github.com/overcuriousity/wintermute/commit/2238545b01524fcbdd4572f95a81cd3c37028c0f))
* **data_versioning:** close drain/commit_async race condition ([ac9bb2c](https://github.com/overcuriousity/wintermute/commit/ac9bb2cb388459290a8a1f06b97e5c112c5c6ff6))
* **data_versioning:** drain commit threads on shutdown ([8d6e23c](https://github.com/overcuriousity/wintermute/commit/8d6e23c035468424cf2984b464bbc929cddf3a14))
* **data_versioning:** drain commit threads on shutdown instead of daemon=True ([a5acd4e](https://github.com/overcuriousity/wintermute/commit/a5acd4e180b05c06e31c75df2fbec4eb2a34ef9f)), closes [#75](https://github.com/overcuriousity/wintermute/issues/75)
* **data_versioning:** remove misleading timeout from drain(); run via asyncio.to_thread() ([9dc2790](https://github.com/overcuriousity/wintermute/commit/9dc2790276b50262cbbbcedeb71718c671d631dc))
* defer kimi-code client creation for unused backends ([031610e](https://github.com/overcuriousity/wintermute/commit/031610ee03b8fe2c8d3c162d5e57bb4aa9d5f5a6))
* **event_bus:** address Copilot review issues in thread-safe emit ([8a7812c](https://github.com/overcuriousity/wintermute/commit/8a7812c74f48037ac92b341994db60755628b7e7))
* **event_bus:** extend lock scope to cover debounce state ([7915498](https://github.com/overcuriousity/wintermute/commit/7915498af60492c18b11399454a7017c345b0da2))
* **event_bus:** extend lock scope to cover debounce state mutations ([67f4c8b](https://github.com/overcuriousity/wintermute/commit/67f4c8bb2bdf88c4ca6571f9a536af14add3febc)), closes [#78](https://github.com/overcuriousity/wintermute/issues/78)
* **event_bus:** use call_soon_threadsafe for thread-pool emit() ([40dec9c](https://github.com/overcuriousity/wintermute/commit/40dec9cd9f65d005d49c3b674dd380c4d910ce94))
* **event_bus:** use call_soon_threadsafe for thread-pool emit() ([d52ece8](https://github.com/overcuriousity/wintermute/commit/d52ece89377fab64898367b835dd8004a0e60334)), closes [#66](https://github.com/overcuriousity/wintermute/issues/66)
* guard against search: null in config.yaml ([746ab32](https://github.com/overcuriousity/wintermute/commit/746ab328fbcdf69c6327bf20364265363d3f185b))
* handle list-type content safely in normalized messages ([42af5d8](https://github.com/overcuriousity/wintermute/commit/42af5d8106e07b84e430bddeb853bdea6c4c447a))
* **llm_thread:** add retry limit for empty-choices loop ([5510924](https://github.com/overcuriousity/wintermute/commit/5510924f67f6326623c91e142c33bba5b87083b2))
* **llm_thread:** add retry limit for empty-choices loop ([27b6843](https://github.com/overcuriousity/wintermute/commit/27b6843ecae4c74dd36d316deb39dd50f559646f)), closes [#73](https://github.com/overcuriousity/wintermute/issues/73)
* load debug HTML lazily in handler, not at import time (PR [#132](https://github.com/overcuriousity/wintermute/issues/132)) ([4a7a10e](https://github.com/overcuriousity/wintermute/commit/4a7a10e900371da52e84032baa6066b549f7f6ac))
* **matrix_thread:** atomic config.yaml writes with lock ([f541043](https://github.com/overcuriousity/wintermute/commit/f541043528a07e23f5ba46fa3b6272fdd3ce48a1))
* **matrix_thread:** atomic config.yaml writes with lock ([7c278a5](https://github.com/overcuriousity/wintermute/commit/7c278a55168e8596e2efb3125db2575dbff96ef0)), closes [#74](https://github.com/overcuriousity/wintermute/issues/74)
* **memory_harvest:** address Copilot review issues in background task tracking ([f87957a](https://github.com/overcuriousity/wintermute/commit/f87957ae5462d7151304c68059e65b45394dd884))
* **memory_harvest:** handle CancelledError explicitly in _await_harvest ([09c9933](https://github.com/overcuriousity/wintermute/commit/09c9933e78ca8cc107e4d73fac236199e56b7891))
* **memory_harvest:** re-raise CancelledError in inner interaction_log block ([821b7d5](https://github.com/overcuriousity/wintermute/commit/821b7d552393b26fac0ba47509eefc62d8619eb2))
* **memory_harvest:** replace deprecated ensure_future with create_task ([8705342](https://github.com/overcuriousity/wintermute/commit/87053422724eab57fc993b7ae51d7302b2e02deb))
* **memory_harvest:** replace deprecated ensure_future with create_task ([034274c](https://github.com/overcuriousity/wintermute/commit/034274c76d46deb3a6552501acbed839c816e750)), closes [#77](https://github.com/overcuriousity/wintermute/issues/77)
* **memory_store:** address Copilot review issues in QdrantBackend ([1d3e371](https://github.com/overcuriousity/wintermute/commit/1d3e37133a73946e96aa12b79870c8566332e7d4))
* **memory_store:** chunked SQLite deletes; raise on metadata-read failures ([bad06b1](https://github.com/overcuriousity/wintermute/commit/bad06b1b6f82505382905c9552a81f0e2dff0f31))
* **memory_store:** hold lock across retrieve+upsert in QdrantBackend.add() ([cd4e401](https://github.com/overcuriousity/wintermute/commit/cd4e401f2ea9cedff4dc5f406f78263106d2adb9))
* **memory_store:** move Filter/HasIdCondition import to method top ([479c7cc](https://github.com/overcuriousity/wintermute/commit/479c7ccd9e2a3c98551dcd1dcfe0f8eae6ec6e4b))
* **memory_store:** paginate all Qdrant scroll calls; fix offset sentinel check ([e86e795](https://github.com/overcuriousity/wintermute/commit/e86e795f3d48c7d8481ed9ec741a9c77f143c0be))
* **memory_store:** preserve metadata on add() and replace_all() ([01063ed](https://github.com/overcuriousity/wintermute/commit/01063ed3742d02ad3687284a05dff87eef0428b6))
* **memory_store:** preserve metadata on add() and replace_all() ([30eb16c](https://github.com/overcuriousity/wintermute/commit/30eb16cc0721c628df8ef44a7d34fe6eedbea27a))
* **memory_store:** use batched retrieve() in replace_all(); fix add() exception handler; single-lock critical section ([a74b5d6](https://github.com/overcuriousity/wintermute/commit/a74b5d610e4a22eea5828d7dd0c8e9f147b29a18))
* **prompt_assembler:** parse Unix timestamp correctly in _get_reflection_observations ([2c27174](https://github.com/overcuriousity/wintermute/commit/2c2717406dc954dd7d09c5858fd7ff3a1047e1ec))
* **prompt_assembler:** parse Unix timestamp correctly in _get_reflection_observations ([a049817](https://github.com/overcuriousity/wintermute/commit/a049817ef96f340db609262c249f29ce10dc528e)), closes [#70](https://github.com/overcuriousity/wintermute/issues/70)
* reorder imports in main.py (third-party before first-party) ([2edc3cb](https://github.com/overcuriousity/wintermute/commit/2edc3cbb64e09b970a9d0aa81d9baea01ffb2e1d))
* replace deprecated asyncio.get_event_loop() with get_running_loop() ([6cab735](https://github.com/overcuriousity/wintermute/commit/6cab7354e1e56f961af89bbe49ad458fc45d66b3))
* replace deprecated asyncio.get_event_loop() with get_running_loop() ([da3b463](https://github.com/overcuriousity/wintermute/commit/da3b46305d01253f10d4141fdd434de3a679a2a1)), closes [#76](https://github.com/overcuriousity/wintermute/issues/76)
* rescue XML/text-encoded tool calls from weak models ([3417779](https://github.com/overcuriousity/wintermute/commit/34177797c649452e9d9a09c17db27583e5e4b964))
* rescue XML/text-encoded tool calls from weak models ([2a73e1f](https://github.com/overcuriousity/wintermute/commit/2a73e1fc582b4808ac1d15679d979ec1129f4269))
* restore interaction_log for multi-item calls; drop unused Path import ([a29e2af](https://github.com/overcuriousity/wintermute/commit/a29e2afd188391c47339a97cd561462205a9b6f4))
* store references to fire-and-forget asyncio tasks ([50c50d7](https://github.com/overcuriousity/wintermute/commit/50c50d740f1e66cc28742e42436d7336adb49692))
* store references to fire-and-forget asyncio tasks ([c21ab50](https://github.com/overcuriousity/wintermute/commit/c21ab50919e42a89e56b50e8795f9bdfc3d84473)), closes [#68](https://github.com/overcuriousity/wintermute/issues/68)
* **sub_session:** address Copilot review issues in context trim sanitizer ([3e1603a](https://github.com/overcuriousity/wintermute/commit/3e1603a5f7e7c6caa540d1368edc3ef431359bee))
* **sub_session:** sanitize tool-call boundaries after context trim ([c954b82](https://github.com/overcuriousity/wintermute/commit/c954b827312e3173a9efa659821d5954362a645d))
* **sub_session:** sanitize tool-call boundaries after context trim ([164dabb](https://github.com/overcuriousity/wintermute/commit/164dabb7c0663696b5c63093052e0e9cc438328c)), closes [#67](https://github.com/overcuriousity/wintermute/issues/67)
* use MutableMapping for config write isinstance checks ([1f5b06f](https://github.com/overcuriousity/wintermute/commit/1f5b06f7da6ceaf00947af194c4f36838d5c79a2))
* use MutableMapping instead of Mapping for config write checks ([3a8dbba](https://github.com/overcuriousity/wintermute/commit/3a8dbbaba70d12eb82985f74c1c29c7208bcc107))

## [0.5.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.4.0-alpha...v0.5.0-alpha) (2026-02-27)


### Features

* add /reflect command and self-model/reflection to /status ([0357a26](https://github.com/overcuriousity/wintermute/commit/0357a26139d611b4cd5bfba33cb7d5f3f5fbec05))
* add async event bus infrastructure wired into all components ([4f5e5e4](https://github.com/overcuriousity/wintermute/commit/4f5e5e4d2c57655c0da76cb5dde0676c96c50fff))
* add skill merge action to dreaming dedup ([b58de0e](https://github.com/overcuriousity/wintermute/commit/b58de0e4384a44813dfe8c5328cc03426cec1183))
* apply PR [#60](https://github.com/overcuriousity/wintermute/issues/60) changes (introspection tool, reflection, skill synthesis) ([f33bc00](https://github.com/overcuriousity/wintermute/commit/f33bc002ce7c286a957c2eb0299ec63ec704bfac))
* biologically-inspired multi-phase dreaming system ([baa651f](https://github.com/overcuriousity/wintermute/commit/baa651f346be5b04944e664ae1ec740489febd62))
* enhance startup logging with visual indicator ([6a71b1e](https://github.com/overcuriousity/wintermute/commit/6a71b1e7a849861731a54c79d5d695a1c1a5e985))
* implement Phase 1 reflection cycle (feedback loop) ([6720f56](https://github.com/overcuriousity/wintermute/commit/6720f56bcb96b0f8c9c78c9e9c8d0fdeb6e6a770))
* implement Phase 1 reflection cycle (feedback loop) ([a253438](https://github.com/overcuriousity/wintermute/commit/a253438bec34707599affa514207474a3e0ea147))
* improve delegation heuristics and surface reflection observations ([52a6b88](https://github.com/overcuriousity/wintermute/commit/52a6b88676ab8c46836c42c75e69a3e73983e85c))
* per-thread configuration overrides ([#7](https://github.com/overcuriousity/wintermute/issues/7)) ([88c5052](https://github.com/overcuriousity/wintermute/commit/88c505290a35e52f870094b05d2605fcf85cf4b6))
* Phase 3 — Skill Evolution ([a8f160a](https://github.com/overcuriousity/wintermute/commit/a8f160af509e58a429664d1c255f19c8cc4fa2c9))
* Phase 3 — Skill Evolution (usage tracking, outcome correlation, auto-retirement) ([f72bb38](https://github.com/overcuriousity/wintermute/commit/f72bb386babae6f0634cb76f95263d4b2735e27d))
* self-model profiler with operational metrics and auto-tuning ([0b0c605](https://github.com/overcuriousity/wintermute/commit/0b0c6059cc06271fd19324e190b05a8f4d805894))
* vector-native dreaming with access tracking and source tagging ([2154c19](https://github.com/overcuriousity/wintermute/commit/2154c1903d3d4936118f1cb4a17931467794e8a1))


### Bug Fixes

* add retry with exponential backoff for embedding endpoint ([c73c679](https://github.com/overcuriousity/wintermute/commit/c73c6790de9bf4c9d5242a0ab33f9bdf1bf66b4b))
* address Copilot review — 5 bugs in skill evolution ([3554271](https://github.com/overcuriousity/wintermute/commit/35542718c11c8dd7eada6382e2222dc92512709f))
* address Copilot review comments on outcome tracking ([88f1c1b](https://github.com/overcuriousity/wintermute/commit/88f1c1b7167e88ad12a59505c62e622167d4692e))
* address Copilot review feedback in reflection.py ([50084cc](https://github.com/overcuriousity/wintermute/commit/50084ccad7908375d867798a9a774333a6a6fa16))
* address valid Copilot review items from PR [#60](https://github.com/overcuriousity/wintermute/issues/60) (synthesis, encapsulation, path checks) ([b2da51d](https://github.com/overcuriousity/wintermute/commit/b2da51d1f7767f49895ee99656e112488e4b4653))
* auto-inject parent context into sub-sessions ([d4b8fb9](https://github.com/overcuriousity/wintermute/commit/d4b8fb956f85475fcb8b277d7f967ca48c63d973))
* don't wipe crypto store on transient network errors ([e98cf66](https://github.com/overcuriousity/wintermute/commit/e98cf66eb92c36eeb011b28ff6069d9097b39aeb))
* fall back to system service when user D-Bus bus is unavailable ([12e77f9](https://github.com/overcuriousity/wintermute/commit/12e77f98f60557305909048dc6acae367f90456a))
* gracefully handle missing systemd user bus in LXC containers ([ff03e83](https://github.com/overcuriousity/wintermute/commit/ff03e835c0ed090edac2377005d2bcc976368bc8))
* guard self-model init failure to prevent blocking harvest task startup ([893c555](https://github.com/overcuriousity/wintermute/commit/893c5554b97165c370024ff4c604eec9b596c338))
* implement valid Copilot review topics from PR [#60](https://github.com/overcuriousity/wintermute/issues/60) ([e50a6ca](https://github.com/overcuriousity/wintermute/commit/e50a6cafcc25e67b21ddb9aa529ee61c31dac266))
* make Tier 3 trigger language-neutral; fix action_filter bug ([dca4506](https://github.com/overcuriousity/wintermute/commit/dca4506ea1b4a85542253bf91d17f0931ab78259))
* resolve outcome tracking bugs and implement Outcomes debug panel ([0c79ba9](https://github.com/overcuriousity/wintermute/commit/0c79ba9f348d214c37a0bb5db6caa53707a86e09))
* robust JSON extraction in dreaming — tolerate prose-wrapped responses ([06c1bb3](https://github.com/overcuriousity/wintermute/commit/06c1bb3570b9e7f2c9cafe09a2708354c339fd43))
* SAS verification MAC info string and MUnknownToken handling ([73467ce](https://github.com/overcuriousity/wintermute/commit/73467ce213e73f5f1a3abfa53c9fb8e854736ebb))
* systemd user→system service fallback in onboarding.py and docs ([6974d2b](https://github.com/overcuriousity/wintermute/commit/6974d2b3a53154a756b0300d409d5a089033452a))


### Dependencies

* **python:** bump ruff from 0.15.1 to 0.15.2 ([1a42def](https://github.com/overcuriousity/wintermute/commit/1a42def8a17159d1d2c10d61fbcb70c5445dc8f6))


### Documentation

* add reflection cycle documentation, web UI filters, roadmap update ([1567eda](https://github.com/overcuriousity/wintermute/commit/1567eda1ebb2703dbe3d6820493a52243937be5b))
* document self-model, /reflect command, and /status updates ([6a2d63f](https://github.com/overcuriousity/wintermute/commit/6a2d63f3d7c4b7256f09e35142474746910cef8f))
* mark Phase 2 (Self-Model) as done in roadmap ([4a14cb2](https://github.com/overcuriousity/wintermute/commit/4a14cb2556ea837ce2bc21bb8b90fa7121e26193))
* updated commands.md, configuration.md, web-interface.md ([88c5052](https://github.com/overcuriousity/wintermute/commit/88c505290a35e52f870094b05d2605fcf85cf4b6))

## [0.4.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.3.0-alpha...v0.4.0-alpha) (2026-02-23)


### Features

* add dedicated memory_harvest LLM role for cheaper backend routing ([c25ec18](https://github.com/overcuriousity/wintermute/commit/c25ec18bca500fa6b0d1c6417b0c8b5b2dd98211))
* add interaction logging for embedding/Qdrant ops, update docs ([0552d25](https://github.com/overcuriousity/wintermute/commit/0552d250ffb6a43e91ad2f19b9b9c55584b2072a))
* add LocalVectorBackend (numpy+SQLite) to close [#39](https://github.com/overcuriousity/wintermute/issues/39) ([13ec0d3](https://github.com/overcuriousity/wintermute/commit/13ec0d34c86f1f369fdc24a0cfca780ba631513a))
* add max_rounds and skip_tp_on_exit to sub-sessions ([2ad0360](https://github.com/overcuriousity/wintermute/commit/2ad0360e8bf2aa47b9b723df26231c375264f904))
* add sectioned BASE_PROMPT and named tool profiles ([1948a3a](https://github.com/overcuriousity/wintermute/commit/1948a3a77c1598912c21662fe48fb23778ba2265))
* add vector-indexed memory retrieval with flat_file/fts5/qdrant backends ([dd77c1f](https://github.com/overcuriousity/wintermute/commit/dd77c1fd970cfba066443789a85f1413652acc5d))
* complete vector memory storage gaps (issue [#39](https://github.com/overcuriousity/wintermute/issues/39)) ([20c82ac](https://github.com/overcuriousity/wintermute/commit/20c82acdf03fc451f16f380180bdf60d1bf74b1d))


### Bug Fixes

* add api_key support to embeddings endpoint ([b5b6a17](https://github.com/overcuriousity/wintermute/commit/b5b6a1778121d0a1948f860ed280e2474ae645cd))
* add fast-path for agenda list in NL translator ([43a11b3](https://github.com/overcuriousity/wintermute/commit/43a11b3f288d91e788ff20d442fcbb13d0caa846))
* address PR [#44](https://github.com/overcuriousity/wintermute/issues/44) review comments (async I/O, race conditions, FTS5 ranking) ([e86efef](https://github.com/overcuriousity/wintermute/commit/e86efefdc697b2c42f8d3b19706332c607789195))
* cache memory count in SSE snapshot to avoid Qdrant polling spam ([81f52dc](https://github.com/overcuriousity/wintermute/commit/81f52dc1da1557dba32ec2ae9e052f84a6d91559))
* exclude pool_override from sub-session JSON serialization ([6688f51](https://github.com/overcuriousity/wintermute/commit/6688f51fba67b9a15a5c2515f4276112aa1b2418))
* make dreaming memory prompt backend-agnostic ([7d1cc3c](https://github.com/overcuriousity/wintermute/commit/7d1cc3cc668534ff2203f64fdd5a95ecbde5f176))
* merge orphan metadata in NL-translated worker_delegation lists ([27bc3aa](https://github.com/overcuriousity/wintermute/commit/27bc3aa7a16aa125cb37804c4d24d4b5bf6c0808))
* missing database import in slash commands, don't send dimensions in embed request ([3f90579](https://github.com/overcuriousity/wintermute/commit/3f90579466d7077507c4521b30fceb2aa5767f8d))
* parse Qdrant URL into host/port/https for reliable HTTPS connections ([344f72b](https://github.com/overcuriousity/wintermute/commit/344f72b8bdea8317a6d19b1e78f36d8f30aca45f))
* use REST API instead of gRPC for Qdrant client ([85f05e5](https://github.com/overcuriousity/wintermute/commit/85f05e5eb48edc0e1b3ba7d4306c989531fd5e26))
* use UUID format for memory entry IDs (Qdrant requires UUID or int) ([9f46d38](https://github.com/overcuriousity/wintermute/commit/9f46d38689b22c5f1465bfd4939b5cfd31337929))

## [0.3.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.2.1-alpha...v0.3.0-alpha) (2026-02-21)


### Features

* fix MEMORIES.txt race condition and add git auto-versioning for data/ ([308150a](https://github.com/overcuriousity/wintermute/commit/308150a1e54ea87be343017dca33f2cbc5f164f5))
* unified ContextTooLargeError detection and per-caller recovery ([e4fcee7](https://github.com/overcuriousity/wintermute/commit/e4fcee757a67bfdbbda57b432e56eef3687034af))


### Bug Fixes

* add exponential backoff retry for rate-limited LLM requests ([9fd4053](https://github.com/overcuriousity/wintermute/commit/9fd4053dc59323170a21e720d8b0ce9254be1688))
* add write_file to phantom_tool_result detection and document all TP hooks in config ([2940bcc](https://github.com/overcuriousity/wintermute/commit/2940bcc8cd35083dae656d14da911415e6b76da9))
* await dependency resolution in cancel_for_thread to prevent deadlocks ([b32adb0](https://github.com/overcuriousity/wintermute/commit/b32adb07e90455a096b270725e563c9e34385700))
* collect recent_assistant_messages before saving reply to DB ([d9f81ac](https://github.com/overcuriousity/wintermute/commit/d9f81ac9939f58d91b12dc34889c01db85f7b4b4))
* defer memory harvest state update until sub-session completes ([59461d0](https://github.com/overcuriousity/wintermute/commit/59461d05489a9a423575c204df660e1673839626))
* handle ChatCompletionMessage objects in sub-session TP context ([9f22e62](https://github.com/overcuriousity/wintermute/commit/9f22e620f27cb594663a9cfcc75c0ef16c7a1bf1))
* harden Turing Protocol pipeline against weak-model failure modes ([7eb75bb](https://github.com/overcuriousity/wintermute/commit/7eb75bbd7045a9009d3ce8054ba50626503f2f07))
* improve Turing Protocol correction effectiveness for weak models ([d8c3093](https://github.com/overcuriousity/wintermute/commit/d8c309360109be98a58a29ba664eb32d89ca0e4c))
* log and report malformed tool args instead of silently defaulting to {} ([1e00036](https://github.com/overcuriousity/wintermute/commit/1e00036bf5af26e21235a6c18346188c9934fc87))
* reduce phantom_tool_result false positives when tools were called ([8415a58](https://github.com/overcuriousity/wintermute/commit/8415a58e956d54466c6da63d6fac3043989de70a))
* replace get_event_loop().run_in_executor with daemon threads in prompt_assembler ([67060ed](https://github.com/overcuriousity/wintermute/commit/67060ed0198d0f12882ccfc589163582e6e8674a))
* resolve dependents on CancelledError to prevent workflow deadlock ([7b7fdb6](https://github.com/overcuriousity/wintermute/commit/7b7fdb6609534f995f7dfa4a01b86d8ccd74cf77))
* three minor correctness issues ([7a8ffc0](https://github.com/overcuriousity/wintermute/commit/7a8ffc09e335dc9d3bb7741472b4bd362804f814))

## [0.2.1-alpha](https://github.com/overcuriousity/wintermute/compare/v0.2.0-alpha...v0.2.1-alpha) (2026-02-20)


### Bug Fixes

* always broadcast Turing correction responses to Matrix ([f2e2763](https://github.com/overcuriousity/wintermute/commit/f2e2763c899ad0e2322a99fe71063129f0cc8a57))
* convert nl_tools set to list before JSON serialization in Turing Protocol ([0ab74d4](https://github.com/overcuriousity/wintermute/commit/0ab74d4661cb6cbe32731f90b7e9207cf50e691d))
* guard against LLM returning empty/null choices across all modules ([e62f9b5](https://github.com/overcuriousity/wintermute/commit/e62f9b5eb3b61c65187a86c2413ec6a48a090a21))
* make Turing Protocol corrections self-healing without user interaction ([1107778](https://github.com/overcuriousity/wintermute/commit/1107778651b2b1cea143d770e25230ba8e9f1e84))
* make Turing Protocol NL-translation aware to prevent validation loops ([fc35a66](https://github.com/overcuriousity/wintermute/commit/fc35a66ffae76b6dabd1b336878d7ee3157a44f5))
* re-check correction responses to catch models ignoring Turing Protocol ([7c7baaf](https://github.com/overcuriousity/wintermute/commit/7c7baafeb53517adcb202bdf0e1415f42a0f89ea))
* use graceful fallback when Turing correction is ignored ([e3bf6de](https://github.com/overcuriousity/wintermute/commit/e3bf6de5a15d583cd2febad0266aa31a4e0667d5))
* use thread-safe event loop scheduling in sub-session spawning ([ecbc0d6](https://github.com/overcuriousity/wintermute/commit/ecbc0d65b12ca31ddac96b0442917dc61910c9fb))
* use thread-safe event loop scheduling in sub-session spawning ([de749b1](https://github.com/overcuriousity/wintermute/commit/de749b198eca5d46ba24833383050b3d622c63d7))

## [0.2.0-alpha](https://github.com/overcuriousity/wintermute/compare/v0.1.0-alpha...v0.2.0-alpha) (2026-02-20)


### Features

* add periodic update checker with Matrix notifications ([#32](https://github.com/overcuriousity/wintermute/issues/32)) ([cf091e5](https://github.com/overcuriousity/wintermute/commit/cf091e504ad4f99395810ef19e95d77c43c012df))
* add time-gated workflows (not_before) and current time injection ([a2b3c64](https://github.com/overcuriousity/wintermute/commit/a2b3c64cc38ff21711cc491388e111ad82595130))
* expand /status command to show full configuration details ([2075fa9](https://github.com/overcuriousity/wintermute/commit/2075fa99869a052af8e020d10c92d9601ee3f73e))
* implement SAS key verification for E2EE device trust ([8964055](https://github.com/overcuriousity/wintermute/commit/896405566b2d91771d3e4a456349f27ef32fdd74))
* make dreaming and compaction prompts configurable via data/ files ([3902524](https://github.com/overcuriousity/wintermute/commit/3902524c76a4855bdfd681f02a4f2d1458585182))
* replace memory system section with structured knowledge routing guidelines ([3bf0ed8](https://github.com/overcuriousity/wintermute/commit/3bf0ed8bfe6e1ffcf2d20cb84f2ecb928f49782e))
* show typing indicator while LLM is processing a Matrix message ([e420685](https://github.com/overcuriousity/wintermute/commit/e42068563178d8d08af2a1e349a32a92616db999))
* sub-agent delegation, tool filtering, research tools, and timeout continuation ([b421c83](https://github.com/overcuriousity/wintermute/commit/b421c83672a0309370b40790610e598edbf48c94))
* two-stage NL translation pipe for complex tool calls ([608a2b6](https://github.com/overcuriousity/wintermute/commit/608a2b686f195b9d63f494fe5f375c8b1c1270e4))
* two-stage NL translation pipe for complex tool calls ([dbc6932](https://github.com/overcuriousity/wintermute/commit/dbc6932f03bf91081a9da869047872a1e588aee8))


### Bug Fixes

* _connect() self-recursion in database.py ([fb90b6f](https://github.com/overcuriousity/wintermute/commit/fb90b6f575216bef198e860a1523b61952f0911d))
* 10 bugs from codebase audit ([cc13ed4](https://github.com/overcuriousity/wintermute/commit/cc13ed42de55a0d6da939986515e0023162c772d))
* also trust bot's own device when sharing Megolm session keys ([d283401](https://github.com/overcuriousity/wintermute/commit/d283401ce2512fd7ca2c724d7d14482ad4c20e13))
* auto-trust allowed users' devices for E2EE sending ([12136b3](https://github.com/overcuriousity/wintermute/commit/12136b3604b8f83261237dac8e130e148e842d4a))
* guard keys_query() with users_for_key_query check ([1682e2d](https://github.com/overcuriousity/wintermute/commit/1682e2dfa72d5faea1b535667afc2d42a334953b))
* implement full modern SAS verification flow (request/ready/done) ([2d84f9b](https://github.com/overcuriousity/wintermute/commit/2d84f9b1d7698139e5f1a21cd4d51c81705c7928))
* keep typing indicator alive during tool calls and send read receipts ([9d15ed9](https://github.com/overcuriousity/wintermute/commit/9d15ed9fcc80363ea48ce7915f130c61323c5223))
* load Olm store after setting access_token, fix unclosed sessions ([953b5da](https://github.com/overcuriousity/wintermute/commit/953b5da3fdfc678f9dafe8c4dbfc7c891e35f299))
* load_store() is synchronous in matrix-nio, remove await ([1cf709e](https://github.com/overcuriousity/wintermute/commit/1cf709e5a87b3df59401069eb39fdea350bce753))
* remove invalid 'store' kwarg from AsyncClient constructor ([461ee18](https://github.com/overcuriousity/wintermute/commit/461ee18ca5deab1bdeff16f3e5c0bbc2adb0f01b))
* UnboundLocalError for nl_enabled when tool_names whitelist is used ([c9c3a5c](https://github.com/overcuriousity/wintermute/commit/c9c3a5cb63e36aab8670983bd04b7772e89411c2))


### Documentation

* audit and update all documentation for accuracy and small-LLM USP ([0599c36](https://github.com/overcuriousity/wintermute/commit/0599c3629c63742d573df87607bc15359d94efbc))
* audit and update all documentation for accuracy and small-LLM USP ([320d71d](https://github.com/overcuriousity/wintermute/commit/320d71d5f016b757453f9623d92e65f2cedc7b13))
* expand Matrix setup as primary interface documentation ([a1e8a8a](https://github.com/overcuriousity/wintermute/commit/a1e8a8a7a9316a25a003b247bfb09d2ca7643978))
* fix README intro — add Kimi, reframe debug as audit trail ([db8d298](https://github.com/overcuriousity/wintermute/commit/db8d298e747ed2b95eb09b41cc3db4de92273da1))
* regenerate README for Wintermute rebranding ([a42463c](https://github.com/overcuriousity/wintermute/commit/a42463c6b1f33dfa22a5cb081b2614441b131d01))
* split README into lean overview + dedicated docs/ pages ([02bb028](https://github.com/overcuriousity/wintermute/commit/02bb028d148cdc6683c9e5111dbac749bfd2abcc))
* update docs for append_memory, compaction chaining, SearXNG recommendation ([1239f33](https://github.com/overcuriousity/wintermute/commit/1239f33a2ba7dfc1a03f20c1e02e3b1892e29f3a))
* update README and .gitignore for refactored architecture ([85e4d29](https://github.com/overcuriousity/wintermute/commit/85e4d296525d5abc8db4fb07d8eddb89473ccb02))
