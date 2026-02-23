# Changelog

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
* merge orphan metadata in NL-translated spawn_sub_session lists ([27bc3aa](https://github.com/overcuriousity/wintermute/commit/27bc3aa7a16aa125cb37804c4d24d4b5bf6c0808))
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
