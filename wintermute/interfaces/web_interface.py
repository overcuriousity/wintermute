"""
Web Interface

Debug panel available at /debug.  Provides a read/write inspection view of all
running sessions, sub-sessions, scheduled jobs, and tasks.
REST API under /api/debug/* is consumed by the embedded SPA.
"""

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

from wintermute import tools as tool_module

logger = logging.getLogger(__name__)

_DEBUG_HTML_PATH = Path(__file__).parent / "static" / "debug.html"


# ---------------------------------------------------------------------------
# WebInterface class
# ---------------------------------------------------------------------------


class WebInterface:
    """
    Runs an aiohttp web server serving the debug panel at /debug.

    Optional debug dependencies (injected post-construction in main.py):
      _sub_sessions  – SubSessionManager
      _scheduler     – TaskScheduler
      _matrix        – MatrixThread
      _main_pool     – BackendPool (for context_size / max_tokens)
    """

    def __init__(self, host: str, port: int, llm_thread) -> None:
        self._host = host
        self._port = port
        self._llm = llm_thread
        # Map thread_id -> set of WebSocket connections
        self._threads: dict[str, set[web.WebSocketResponse]] = {}
        # Optional debug dependencies (set after construction)
        self._sub_sessions = None
        self._scheduler = None
        self._matrix = None
        self._main_pool = None   # BackendPool for main role
        self._multi_cfg = None
        self._background_tasks: set[asyncio.Task] = set()
        # Shared slash-command handler (injected from main.py).
        self._slash_handler = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def connected_thread_ids(self) -> set[str]:
        """Return thread IDs with active WebSocket connections."""
        return {tid for tid, clients in self._threads.items() if clients}

    async def broadcast(self, text: str, thread_id: str = None, *,
                        reasoning: str = None) -> None:
        """Push a message to all connected clients in a specific thread."""
        if thread_id is None:
            return
        clients = self._threads.get(thread_id, set())
        if not clients:
            return
        msg = {"role": "assistant", "text": text, "thread_id": thread_id}
        if reasoning:
            msg["reasoning"] = reasoning
        payload = json.dumps(msg)
        dead = set()
        for ws in clients:
            try:
                await ws.send_str(payload)
            except Exception:  # noqa: BLE001
                dead.add(ws)
        clients -= dead

    async def run(self) -> None:
        app = web.Application()
        # Debug panel
        app.router.add_get("/debug", self._handle_debug)
        # Debug REST API
        app.router.add_get("/api/debug/sessions",                        self._api_sessions)
        app.router.add_get("/api/debug/sessions/{thread_id}/messages",  self._api_session_messages)
        app.router.add_post("/api/debug/sessions/{thread_id}/send",     self._api_session_send)
        app.router.add_post("/api/debug/sessions/{thread_id}/delete",   self._api_session_delete)
        app.router.add_post("/api/debug/sessions/{thread_id}/compact",  self._api_session_compact)
        app.router.add_get("/api/debug/subsessions",                    self._api_subsessions)
        app.router.add_get("/api/debug/subsessions/{id}/messages",      self._api_subsession_messages)
        app.router.add_get("/api/debug/workflows",                     self._api_workflows)
        app.router.add_get("/api/debug/jobs",                           self._api_jobs)
        app.router.add_get("/api/debug/config",                          self._api_config)
        app.router.add_get("/api/debug/system-prompt",                  self._api_system_prompt)
        app.router.add_get("/api/debug/tasks",                          self._api_tasks)
        app.router.add_get("/api/debug/memory",                           self._api_memory)
        app.router.add_get("/api/debug/interaction-log",                 self._api_interaction_log)
        app.router.add_get("/api/debug/interaction-log/{id}",            self._api_interaction_log_entry)
        app.router.add_get("/api/debug/outcomes",                        self._api_outcomes)
        app.router.add_get("/api/debug/stream",                          self._api_stream)
        # Per-thread config API
        app.router.add_get("/api/debug/thread-config",                   self._api_thread_configs)
        app.router.add_get("/api/debug/thread-config/{thread_id}",       self._api_thread_config_get)
        app.router.add_post("/api/debug/thread-config/{thread_id}",      self._api_thread_config_set)
        app.router.add_delete("/api/debug/thread-config/{thread_id}",    self._api_thread_config_reset)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info("Web interface listening on http://%s:%d", self._host, self._port)
        logger.info("Debug panel at http://%s:%d/debug", self._host, self._port)

        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            for clients in list(self._threads.values()):
                for ws in list(clients):
                    await ws.close()
            self._threads.clear()
            await runner.cleanup()

    # ------------------------------------------------------------------
    # Debug panel handler
    # ------------------------------------------------------------------

    async def _handle_debug(self, _request: web.Request) -> web.Response:
        try:
            html = _DEBUG_HTML_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Could not read debug panel HTML from %s: %s", _DEBUG_HTML_PATH, exc)
            return web.Response(status=500, text="Debug panel unavailable")
        return web.Response(text=html, content_type="text/html")

    # ------------------------------------------------------------------
    # Debug REST API — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json(data) -> web.Response:
        return web.Response(text=json.dumps(data), content_type="application/json")

    def _token_budget(self, thread_id: str = "default") -> dict:
        """Delegate to LLMThread.get_token_budget() for accurate accounting."""
        if self._llm:
            return self._llm.get_token_budget(thread_id)
        return {"total_limit": 4096, "sp_tokens": 0, "tools_tokens": 0,
                "hist_tokens": 0, "total_used": 0, "pct": 0.0, "msg_count": 0}

    # ------------------------------------------------------------------
    # Debug REST API — sessions
    # ------------------------------------------------------------------

    async def _api_sessions(self, _request: web.Request) -> web.Response:
        from wintermute.infra import database

        db_threads = set(await database.async_call(database.get_active_thread_ids))
        web_live = set(self._threads.keys())

        matrix_rooms: set[str] = set()
        if self._matrix is not None:
            try:
                matrix_rooms = self._matrix.joined_room_ids
            except Exception:  # noqa: BLE001
                pass

        all_ids = db_threads | web_live | matrix_rooms

        sessions = []
        for tid in sorted(all_ids):
            budget = self._token_budget(tid)
            ttype = (
                "web" if tid.startswith("web_")
                else "matrix" if (tid.startswith("!") and ":" in tid)
                else "system"
            )
            sessions.append({
                "id": tid,
                "type": ttype,
                "live": tid in web_live or tid in matrix_rooms,
                "msg_count": budget["msg_count"],
                "sp_tokens": budget["sp_tokens"],
                "tools_tokens": budget["tools_tokens"],
                "hist_tokens": budget["hist_tokens"],
                "total_used": budget["total_used"],
                "total_limit": budget["total_limit"],
                "context_pct": budget["pct"],
            })

        return self._json({"sessions": sessions})

    async def _api_session_messages(self, request: web.Request) -> web.Response:
        from wintermute.infra import database

        thread_id = request.match_info["thread_id"]
        msgs = await database.async_call(database.load_active_messages, thread_id)
        budget = self._token_budget(thread_id)

        result = {
            "thread_id": thread_id,
            "messages": [
                {
                    "id": m["id"],
                    "ts": m["timestamp"],
                    "role": m["role"],
                    "content": m["content"],
                    "tokens": m["token_count"],
                }
                for m in msgs
            ],
            **budget,
        }
        # JS expects context_pct, budget has pct
        result["context_pct"] = result.pop("pct", 0)
        return self._json(result)

    async def _api_session_send(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        try:
            data = await request.json()
            text = data.get("text", "").strip()
            if not text:
                return web.Response(
                    text=json.dumps({"error": "No text provided"}),
                    content_type="application/json",
                    status=400,
                )
            # Try slash commands first (shared handler).
            if self._slash_handler and text.startswith("/"):
                async def _send(msg: str) -> None:
                    await self.broadcast(msg, thread_id)
                if await self._slash_handler.dispatch(text, thread_id, _send):
                    return self._json({"ok": True, "thread_id": thread_id, "command": True})

            _task = asyncio.create_task(self._llm.enqueue_user_message(text, thread_id))
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
            return self._json({"ok": True, "thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)})

    async def _api_session_delete(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        try:
            await self._llm.reset_session(thread_id)
            from wintermute.infra import prompt_loader
            try:
                seed_prompt = prompt_loader.load_seed(self._llm.seed_language)
                await self._llm.enqueue_system_event(seed_prompt, thread_id)
            except Exception:  # noqa: BLE001
                pass
            return self._json({"ok": True, "thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)})

    async def _api_session_compact(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        try:
            await self._llm.force_compact(thread_id)
            return self._json({"ok": True, "thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)})

    async def _api_config(self, _request: web.Request) -> web.Response:
        def _backend_list(configs):
            """Serialize a list of ProviderConfig into dicts."""
            return [
                {
                    "name": cfg.name,
                    "provider": cfg.provider,
                    "base_url": cfg.base_url,
                    "model": cfg.model,
                    "context_size": cfg.context_size,
                    "max_tokens": cfg.max_tokens,
                    "reasoning": cfg.reasoning,
                }
                for cfg in configs
            ]
        mc = self._multi_cfg
        if mc:
            return self._json({
                "main": _backend_list(mc.main),
                "compaction": _backend_list(mc.compaction),
                "sub_sessions": _backend_list(mc.sub_sessions),
                "dreaming": _backend_list(mc.dreaming),
                "turing_protocol": _backend_list(mc.turing_protocol),
            })
        return self._json({})

    async def _api_system_prompt(self, _request: web.Request) -> web.Response:
        from wintermute.infra import prompt_assembler
        # Prefer the exact system prompt last sent to the LLM (includes
        # vector-ranked memories, compaction summary, etc.).
        thread_id = _request.query.get("thread", "default")
        prompt = None
        if self._llm:
            prompt = self._llm.get_last_system_prompt(thread_id)
        if prompt is None:
            try:
                prompt = prompt_assembler.assemble()
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": str(exc)})
        from wintermute.core.llm_thread import _count_tokens
        _cfg = self._main_pool.primary if (self._main_pool and self._main_pool.enabled) else None
        model = _cfg.model if _cfg else "gpt-4"
        sp_tokens = _count_tokens(prompt, model)
        # Use NL-aware schemas if the LLM thread has NL translation enabled.
        nl_tools = None
        if self._llm:
            nl_cfg = getattr(self._llm, '_nl_translation_config', {})
            if nl_cfg.get("enabled", False):
                nl_tools = nl_cfg.get("tools", set())
        active_schemas = tool_module.get_tool_schemas(nl_tools=nl_tools)
        tools_tokens = _count_tokens(json.dumps(active_schemas), model)
        total_limit = max(_cfg.context_size - _cfg.max_tokens, 1) if _cfg else 4096
        combined_tokens = sp_tokens + tools_tokens
        return self._json({
            "prompt": prompt,
            "sp_tokens": sp_tokens,
            "tools_tokens": tools_tokens,
            "tokens": combined_tokens,
            "total_limit": total_limit,
            "pct": round(min(combined_tokens / total_limit * 100, 100), 1),
            "tool_schemas": active_schemas,
        })

    # ------------------------------------------------------------------
    # Debug REST API — sub-sessions
    # ------------------------------------------------------------------

    async def _api_subsessions(self, _request: web.Request) -> web.Response:
        if self._sub_sessions is None:
            return self._json({"sessions": []})
        return self._json({"sessions": self._sub_sessions.list_all()})

    async def _api_subsession_messages(self, request: web.Request) -> web.Response:
        sid = request.match_info["id"]
        if self._sub_sessions is None:
            return self._json({"session_id": sid, "messages": []})
        msgs = self._sub_sessions.get_messages(sid)
        # Serialize messages for the frontend
        serialized = []
        for m in msgs:
            if isinstance(m, dict):
                entry = {
                    "role": m.get("role", "unknown"),
                    "content": m.get("content", ""),
                }
                if m.get("tool_call_id"):
                    entry["tool_call_id"] = m["tool_call_id"]
                serialized.append(entry)
            else:
                # OpenAI message object
                entry = {"role": getattr(m, "role", "assistant")}
                entry["content"] = getattr(m, "content", None) or ""
                if hasattr(m, "tool_calls") and m.tool_calls:
                    entry["tool_calls"] = [
                        {"id": tc.id, "name": tc.function.name,
                         "arguments": tc.function.arguments}
                        for tc in m.tool_calls
                    ]
                serialized.append(entry)
        return self._json({"session_id": sid, "messages": serialized})

    async def _api_workflows(self, _request: web.Request) -> web.Response:
        if self._sub_sessions is None:
            return self._json({"workflows": []})
        return self._json({"workflows": self._sub_sessions.list_workflows()})

    # ------------------------------------------------------------------
    # Debug REST API — scheduler jobs
    # ------------------------------------------------------------------

    async def _api_jobs(self, _request: web.Request) -> web.Response:
        if self._scheduler is None:
            return self._json({"jobs": []})
        return self._json({"jobs": self._scheduler.list_jobs()})

    # ------------------------------------------------------------------
    # Debug REST API — tasks
    # ------------------------------------------------------------------

    async def _api_tasks(self, _request: web.Request) -> web.Response:
        from wintermute.infra import database
        items = await database.async_call(database.list_tasks, "all")
        return self._json({"items": items, "count": len(items)})

    # ------------------------------------------------------------------
    # Per-thread config API
    # ------------------------------------------------------------------

    async def _api_thread_configs(self, _request: web.Request) -> web.Response:
        """GET /api/debug/thread-config — list all overrides + available backends."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            return self._json({"error": "Thread config not available"})
        overrides = mgr.get_all_overrides()
        return self._json({
            "overrides": overrides,
            "available_backends": sorted(mgr.get_available_backends()),
            "keys": ["backend_name", "session_timeout_minutes",
                     "sub_sessions_enabled", "system_prompt_mode"],
        })

    async def _api_thread_config_get(self, request: web.Request) -> web.Response:
        """GET /api/debug/thread-config/{thread_id} — resolved config with sources."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            return self._json({"error": "Thread config not available"})
        thread_id = request.match_info["thread_id"]
        resolved_dict = mgr.resolve_as_dict(thread_id)
        raw = mgr.get(thread_id)
        return self._json({
            "thread_id": thread_id,
            "resolved": resolved_dict,
            "overrides": {k: v for k, v in (raw.__dict__ if raw else {}).items()
                          if v is not None} if raw else {},
        })

    async def _api_thread_config_set(self, request: web.Request) -> web.Response:
        """POST /api/debug/thread-config/{thread_id} — set overrides (JSON body)."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            return self._json({"error": "Thread config not available"})
        thread_id = request.match_info["thread_id"]
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON body")
        try:
            for key, value in body.items():
                mgr.set(thread_id, key, value)
        except (ValueError, TypeError) as exc:
            return self._json({"error": str(exc)})
        resolved_dict = mgr.resolve_as_dict(thread_id)
        return self._json({"ok": True, "resolved": resolved_dict})

    async def _api_thread_config_reset(self, request: web.Request) -> web.Response:
        """DELETE /api/debug/thread-config/{thread_id} — remove all overrides."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            return self._json({"error": "Thread config not available"})
        thread_id = request.match_info["thread_id"]
        mgr.reset(thread_id)
        return self._json({"ok": True})

    # ------------------------------------------------------------------
    # Interaction Log
    # ------------------------------------------------------------------

    async def _api_interaction_log(self, request: web.Request) -> web.Response:
        from wintermute.infra import database
        limit = int(request.query.get("limit", "200"))
        offset = int(request.query.get("offset", "0"))
        session = request.query.get("session") or None
        action = request.query.get("action") or None
        before_id_s = request.query.get("before_id")
        after_id_s = request.query.get("after_id")
        before_id = int(before_id_s) if before_id_s else None
        after_id = int(after_id_s) if after_id_s else None
        entries = await database.async_call(
            database.get_interaction_log, limit=limit, offset=offset,
            session_filter=session, action_filter=action,
            before_id=before_id, after_id=after_id)
        total = await database.async_call(
            database.count_interaction_log, session_filter=session,
            action_filter=action)
        return self._json({"entries": entries, "total": total})

    async def _api_interaction_log_entry(self, request: web.Request) -> web.Response:
        from wintermute.infra import database
        entry_id = int(request.match_info["id"])
        entry = await database.async_call(database.get_interaction_log_entry, entry_id)
        if not entry:
            return web.json_response({"error": "not found"}, status=404)
        return self._json(entry)

    # ------------------------------------------------------------------
    # Outcomes API
    # ------------------------------------------------------------------

    async def _api_outcomes(self, request: web.Request) -> web.Response:
        from wintermute.infra import database
        limit = int(request.query.get("limit", "200"))
        offset = int(request.query.get("offset", "0"))
        status_filter = request.query.get("status") or None
        rows, total, stats = await database.async_call(
            database.get_outcomes_page, limit=limit, offset=offset, status_filter=status_filter
        )
        return self._json({"entries": rows, "total": total, "stats": stats})

    # ------------------------------------------------------------------
    # Memory debug API
    # ------------------------------------------------------------------

    _memory_count_cache: tuple[float, int] = (0.0, 0)

    def _get_memory_count(self) -> int:
        import time
        now = time.monotonic()
        if now - self._memory_count_cache[0] < 30:
            return self._memory_count_cache[1]
        try:
            from wintermute.infra import memory_store
            count = memory_store.count()
        except Exception:  # noqa: BLE001
            count = 0
        self._memory_count_cache = (now, count)
        return count

    async def _api_memory(self, request: web.Request) -> web.Response:
        from wintermute.infra import memory_store

        query = request.query.get("q", "")
        top_k = int(request.query.get("k", "5"))

        def _build_result() -> dict:
            r: dict = {
                "backend": memory_store.stats().get("backend", "unknown"),
                "count": memory_store.count(),
                "vector_enabled": memory_store.is_vector_enabled(),
                "top_k": memory_store.get_top_k(),
                "threshold": memory_store.get_threshold(),
                "stats": memory_store.stats(),
            }
            if query:
                r["results"] = memory_store.search(query, top_k=top_k)
            return r

        result = await asyncio.get_running_loop().run_in_executor(None, _build_result)
        return self._json(result)

    # ------------------------------------------------------------------
    # SSE stream
    # ------------------------------------------------------------------

    async def _build_stream_snapshot(self) -> dict:
        from wintermute.infra import database

        # Sessions
        db_threads = set(await database.async_call(database.get_active_thread_ids))
        web_live = set(self._threads.keys())
        matrix_rooms: set[str] = set()
        if self._matrix is not None:
            try:
                matrix_rooms = self._matrix.joined_room_ids
            except Exception:  # noqa: BLE001
                pass
        all_ids = db_threads | web_live | matrix_rooms
        sessions = []
        for tid in sorted(all_ids):
            budget = self._token_budget(tid)
            ttype = (
                "web" if tid.startswith("web_")
                else "matrix" if (tid.startswith("!") and ":" in tid)
                else "system"
            )
            sessions.append({
                "id": tid,
                "type": ttype,
                "live": tid in web_live or tid in matrix_rooms,
                "msg_count": budget["msg_count"],
                "sp_tokens": budget["sp_tokens"],
                "tools_tokens": budget["tools_tokens"],
                "hist_tokens": budget["hist_tokens"],
                "total_used": budget["total_used"],
                "total_limit": budget["total_limit"],
                "context_pct": budget["pct"],
            })

        # Sub-sessions and workflows
        subsessions = self._sub_sessions.list_all() if self._sub_sessions else []
        workflows = self._sub_sessions.list_workflows() if self._sub_sessions else []

        # Scheduled jobs
        jobs = self._scheduler.list_jobs() if self._scheduler else []

        # Tasks (all statuses for the debug view)
        task_items = await database.async_call(database.list_tasks, "all")

        # Interaction log counts
        interactions_total = await database.async_call(database.count_interaction_log)
        interactions_max_id = await database.async_call(database.get_interaction_log_max_id)

        # Per-thread config overrides
        mgr = getattr(self, "_thread_config_manager", None)
        thread_config_overrides = mgr.get_all_overrides() if mgr else {}

        return {
            "sessions": sessions,
            "subsessions": subsessions,
            "workflows": workflows,
            "jobs": jobs,
            "tasks": task_items,
            "interactions_total": interactions_total,
            "interactions_max_id": interactions_max_id,
            "memory_count": self._get_memory_count(),
            "thread_config_overrides": thread_config_overrides,
        }

    async def _api_stream(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        response.headers.update({
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)
        try:
            while True:
                try:
                    payload = await self._build_stream_snapshot()
                    await response.write(
                        ("data: " + json.dumps(payload) + "\n\n").encode()
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("SSE snapshot error: %s", exc)
                await asyncio.sleep(3)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return response
