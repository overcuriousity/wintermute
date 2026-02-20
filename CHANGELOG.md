# Changelog

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
