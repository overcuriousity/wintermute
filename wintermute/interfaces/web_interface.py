"""
Web Interface

Dashboard SPA served at webroot (/).  Provides a read/write inspection view of
all running sessions, sub-sessions, scheduled jobs, tasks, and skills.
REST API under /api/* is consumed by the embedded SPA.
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
    Runs an aiohttp web server serving the dashboard SPA at /.

    All dependencies are passed via the constructor.
    """

    def __init__(self, host: str, port: int, llm_thread,
                 *,
                 sub_sessions=None,
                 scheduler=None,
                 matrix=None,
                 main_pool=None,
                 multi_cfg=None,
                 thread_config_manager=None,
                 slash_handler=None) -> None:
        self._host = host
        self._port = port
        self._llm = llm_thread
        # Map thread_id -> set of WebSocket connections
        self._threads: dict[str, set[web.WebSocketResponse]] = {}
        self._sub_sessions = sub_sessions
        self._scheduler = scheduler
        self._matrix = matrix
        self._main_pool = main_pool   # BackendPool for main role
        self._multi_cfg = multi_cfg
        self._thread_config_manager = thread_config_manager
        self._background_tasks: set[asyncio.Task] = set()
        # SSE broadcast queue: pending messages for connected SSE clients.
        self._sse_queues: set[asyncio.Queue] = set()
        # Shared slash-command handler.
        self._slash_handler = slash_handler

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def connected_thread_ids(self) -> set[str]:
        """Return thread IDs with active WebSocket connections."""
        return {tid for tid, clients in self._threads.items() if clients}

    async def broadcast(self, text: str, thread_id: str = None, *,
                        reasoning: str = None) -> None:
        """Push a message to all connected SSE and WebSocket clients."""
        if thread_id is None:
            return
        # SSE push: notify all connected SSE clients.
        msg = {"type": "broadcast", "role": "assistant", "text": text,
               "thread_id": thread_id}
        if reasoning:
            msg["reasoning"] = reasoning
        dead_queues: set[asyncio.Queue] = set()
        for q in self._sse_queues:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead_queues.add(q)
        self._sse_queues -= dead_queues
        # Legacy WebSocket push (if any WS clients are connected).
        clients = self._threads.get(thread_id, set())
        if not clients:
            return
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
        # Dashboard SPA
        app.router.add_get("/", self._handle_dashboard)
        app.router.add_get("/debug", self._handle_debug_redirect)
        # REST API
        app.router.add_get("/api/sessions",                        self._api_sessions)
        app.router.add_get("/api/sessions/{thread_id}/messages",  self._api_session_messages)
        app.router.add_post("/api/sessions/{thread_id}/send",     self._api_session_send)
        app.router.add_post("/api/sessions/{thread_id}/delete",   self._api_session_delete)
        app.router.add_post("/api/sessions/{thread_id}/compact",  self._api_session_compact)
        app.router.add_get("/api/subsessions",                    self._api_subsessions)
        app.router.add_get("/api/subsessions/{id}/messages",      self._api_subsession_messages)
        app.router.add_get("/api/workflows",                     self._api_workflows)
        app.router.add_get("/api/workflows/{workflow_id}/scratchpad", self._api_workflow_scratchpad)
        app.router.add_get("/api/jobs",                           self._api_jobs)
        app.router.add_get("/api/config",                          self._api_config)
        app.router.add_get("/api/system-prompt",                  self._api_system_prompt)
        app.router.add_get("/api/tasks",                          self._api_tasks)
        app.router.add_post("/api/tasks",                         self._api_task_create)
        app.router.add_post("/api/tasks/purge",                   self._api_tasks_purge)
        app.router.add_put("/api/tasks/{task_id}",                self._api_task_update)
        app.router.add_delete("/api/tasks/{task_id}",             self._api_task_delete)
        app.router.add_post("/api/tasks/{task_id}/{action}",      self._api_task_action)
        app.router.add_get("/api/memory",                           self._api_memory)
        app.router.add_get("/api/memory/all",                      self._api_memory_list)
        app.router.add_post("/api/memory/bulk-delete",             self._api_memory_bulk_delete)
        app.router.add_put("/api/memory/{entry_id}",               self._api_memory_update)
        app.router.add_delete("/api/memory/{entry_id}",            self._api_memory_delete)
        app.router.add_get("/api/interaction-log",                 self._api_interaction_log)
        app.router.add_get("/api/interaction-log/{id}",            self._api_interaction_log_entry)
        app.router.add_get("/api/outcomes",                        self._api_outcomes)
        app.router.add_get("/api/cp-violations",                    self._api_cp_violations)
        app.router.add_get("/api/stream",                          self._api_stream)
        # Per-thread config API
        app.router.add_get("/api/thread-config",                   self._api_thread_configs)
        app.router.add_get("/api/thread-config/{thread_id}",       self._api_thread_config_get)
        app.router.add_post("/api/thread-config/{thread_id}",      self._api_thread_config_set)
        app.router.add_delete("/api/thread-config/{thread_id}",    self._api_thread_config_reset)
        # Skills API
        app.router.add_get("/api/skills",                          self._api_skills)
        app.router.add_get("/api/skills/info",                     self._api_skill_info)
        app.router.add_get("/api/skills/search",                   self._api_skill_search)
        app.router.add_get("/api/skills/{name}",                   self._api_skill_get)
        app.router.add_post("/api/skills",                         self._api_skill_create)
        app.router.add_put("/api/skills/{name}",                   self._api_skill_update)
        app.router.add_delete("/api/skills/{name}",                self._api_skill_delete)
        # Prediction accuracy API
        app.router.add_get("/api/prediction-accuracy",             self._api_prediction_accuracy)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info("Web interface listening on http://%s:%d", self._host, self._port)

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
    # Dashboard handler
    # ------------------------------------------------------------------

    async def _handle_dashboard(self, _request: web.Request) -> web.Response:
        try:
            html = _DEBUG_HTML_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Could not read dashboard HTML from %s: %s", _DEBUG_HTML_PATH, exc)
            return web.Response(status=500, text="Dashboard unavailable")
        return web.Response(text=html, content_type="text/html")

    async def _handle_debug_redirect(self, _request: web.Request) -> web.Response:
        raise web.HTTPFound("/")

    # ------------------------------------------------------------------
    # REST API — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json(data, *, status: int = 200) -> web.Response:
        return web.Response(text=json.dumps(data), content_type="application/json", status=status)

    def _error_response(
        self,
        message: str = "Internal server error",
        *,
        status: int = 500,
        exc: Exception | None = None,
    ) -> web.Response:
        """
        Build a generic JSON error response and log the underlying exception.

        This avoids sending potentially sensitive exception details to clients.
        """
        if exc is not None:
            logger.exception("Unhandled exception in API: %s", message)
        return self._json({"error": message}, status=status)

    def _token_budget(self, thread_id: str = "default") -> dict:
        """Delegate to LLMThread.get_token_budget() for accurate accounting."""
        if self._llm:
            return self._llm.get_token_budget(thread_id)
        return {"total_limit": 4096, "sp_tokens": 0, "tools_tokens": 0,
                "hist_tokens": 0, "total_used": 0, "pct": 0.0, "msg_count": 0}

    # ------------------------------------------------------------------
    # REST API — sessions
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
                else "signal" if tid.startswith("sig_")
                else "matrix" if (tid.startswith("!") and ":" in tid)
                else "system"
            )
            sessions.append({
                "id": tid,
                "type": ttype,
                "live": tid in web_live or tid in matrix_rooms,
                "group_mode": ttype == "matrix" and self._matrix is not None
                              and self._matrix.group_mode,
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
                async def _handle_slash() -> None:
                    async def _send(msg: str) -> None:
                        await self.broadcast(msg, thread_id)
                    try:
                        handled = await self._slash_handler.dispatch(text, thread_id, _send)
                        if not handled:
                            await self._llm.enqueue_user_message(text, thread_id)
                    except Exception:  # noqa: BLE001
                        logger.exception("Slash command error for thread %s", thread_id)
                        try:
                            await self.broadcast("Error handling slash command.", thread_id)
                        except Exception:  # noqa: BLE001
                            pass

                _cmd_task = asyncio.create_task(_handle_slash())
                self._background_tasks.add(_cmd_task)
                _cmd_task.add_done_callback(self._background_tasks.discard)
                return self._json({"ok": True, "thread_id": thread_id, "command": True})

            _task = asyncio.create_task(self._llm.enqueue_user_message(text, thread_id))
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
            return self._json({"ok": True, "thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001
            return self._error_response("Failed to send session request", status=500, exc=exc)

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
            return self._error_response("Failed to delete session", status=500, exc=exc)

    async def _api_session_compact(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        try:
            await self._llm.force_compact(thread_id)
            return self._json({"ok": True, "thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001
            return self._error_response("Failed to compact session", status=500, exc=exc)

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
                "convergence_protocol": _backend_list(mc.convergence_protocol),
            })
        return self._json({})

    async def _api_system_prompt(self, _request: web.Request) -> web.Response:
        # Show the exact system prompt last sent to the LLM (includes
        # vector-ranked memories, compaction summary, etc.).
        # Falls back to a freshly assembled prompt (without ranked memories)
        # when no cached version exists yet (e.g. after a restart).
        thread_id = _request.query.get("thread", "default")
        prompt = None
        is_fallback = False
        if self._llm:
            prompt = self._llm.get_last_system_prompt(thread_id)
        if prompt is None:
            # Assemble an approximate prompt so the debug panel is useful
            # even before the first inference call (e.g. after restart).
            try:
                from wintermute.infra import prompt_assembler
                summary = None
                if self._llm and hasattr(self._llm, '_store'):
                    summary = self._llm._store.compaction_summaries.get(thread_id)
                prompt = prompt_assembler.assemble(extra_summary=summary)
                is_fallback = True
            except Exception:  # noqa: BLE001
                return self._json(
                    {"error": f"Could not assemble system prompt for thread '{thread_id}'."},
                    status=503,
                )
        from wintermute.core.llm_thread import _count_tokens
        _cfg = self._main_pool.primary if (self._main_pool and self._main_pool.enabled) else None
        model = _cfg.model if _cfg else "gpt-4"
        sp_tokens = _count_tokens(prompt, model)
        # Prefer the exact tool schemas last sent to the LLM (respects
        # lite-mode exclusions, tool profiles, NL translation, etc.).
        active_schemas = None
        if self._llm:
            active_schemas = self._llm.get_last_tool_schemas(thread_id)
        # Fallback when there is no cached schema from a prior LLM call.
        if active_schemas is None:
            # Prefer a helper on the LLM thread that mirrors inference-time
            # tool selection (lite-mode exclusions, tool profiles, etc.),
            # if such a helper is available.
            if self._llm and hasattr(self._llm, "get_active_tool_schemas"):
                try:
                    active_schemas = self._llm.get_active_tool_schemas(thread_id)
                except Exception:  # noqa: BLE001
                    # If anything goes wrong, fall back to the generic schema builder.
                    active_schemas = None
        if active_schemas is None:
            nl_tools = None
            if self._llm:
                nl_cfg = getattr(self._llm, '_nl_translation_config', {})
                if nl_cfg.get("enabled", False):
                    nl_tools = nl_cfg.get("tools", set())
            active_schemas = tool_module.get_tool_schemas(nl_tools=nl_tools)
        tools_tokens = _count_tokens(json.dumps(active_schemas), model)
        total_limit = max(_cfg.context_size - _cfg.max_tokens, 1) if _cfg else 4096
        combined_tokens = sp_tokens + tools_tokens
        result = {
            "prompt": prompt,
            "sp_tokens": sp_tokens,
            "tools_tokens": tools_tokens,
            "tokens": combined_tokens,
            "total_limit": total_limit,
            "pct": round(min(combined_tokens / total_limit * 100, 100), 1),
            "tool_schemas": active_schemas,
        }
        if is_fallback:
            result["fallback"] = True
        return self._json(result)

    # ------------------------------------------------------------------
    # REST API — sub-sessions
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
    # REST API — workflow scratchpad
    # ------------------------------------------------------------------

    async def _api_workflow_scratchpad(self, request: web.Request) -> web.Response:
        """GET /api/workflows/{workflow_id}/scratchpad — list scratchpad files with contents."""
        workflow_id = request.match_info["workflow_id"]
        # Sanitise: only allow alphanumeric, dash, underscore
        if not all(c.isalnum() or c in "-_" for c in workflow_id):
            return self._json({"error": "invalid workflow_id"}, status=400)
        scratchpad_dir = Path("data/scratchpad") / workflow_id
        if not scratchpad_dir.is_dir():
            return self._json({"workflow_id": workflow_id, "files": [], "exists": False})
        files = []
        for p in sorted(scratchpad_dir.iterdir()):
            if p.is_file():
                try:
                    content = p.read_text(errors="replace")
                except Exception as exc:
                    logger.exception("Failed to read scratchpad file %s", p)
                    content = "[read error]"
                files.append({"name": p.name, "size": p.stat().st_size, "content": content})
        return self._json({"workflow_id": workflow_id, "files": files, "exists": True})

    # ------------------------------------------------------------------
    # REST API — scheduler jobs
    # ------------------------------------------------------------------

    async def _api_jobs(self, _request: web.Request) -> web.Response:
        if self._scheduler is None:
            return self._json({"jobs": []})
        return self._json({"jobs": self._scheduler.list_jobs()})

    # ------------------------------------------------------------------
    # REST API — tasks
    # ------------------------------------------------------------------

    async def _api_tasks(self, _request: web.Request) -> web.Response:
        from wintermute.infra import database
        items = await database.async_call(database.list_tasks, "all")
        return self._json({"items": items, "count": len(items)})

    async def _api_task_create(self, request: web.Request) -> web.Response:
        """POST /api/tasks — create a new task."""
        from wintermute.infra import database
        from wintermute.tools.task_tools import _describe_schedule, _resolve_execution_mode
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        content = (data.get("content") or "").strip()
        if not content:
            return web.json_response({"error": "content is required"}, status=400)

        schedule_type = (data.get("schedule_type") or "").strip() or None
        ai_prompt = (data.get("ai_prompt") or "").strip() or None
        execution_mode = (data.get("execution_mode") or "").strip() or None
        background_provided = "background" in data
        background = bool(data.get("background", False))

        try:
            execution_mode, background = _resolve_execution_mode(
                schedule_type=schedule_type,
                ai_prompt=ai_prompt,
                execution_mode=execution_mode,
                background=background,
                background_provided=background_provided,
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        schedule_config = None
        schedule_desc = None
        if schedule_type:
            _sched_keys = ("schedule_type", "at", "day_of_week", "day_of_month",
                           "interval_seconds", "window_start", "window_end")
            sched_inputs = {k: data[k] for k in _sched_keys if k in data}
            schedule_config = json.dumps(sched_inputs)
            schedule_desc = _describe_schedule(sched_inputs)

        task_id = await database.async_call(
            database.add_task,
            content,
            data.get("thread_id"),
            schedule_type,
            schedule_desc,
            schedule_config,
            ai_prompt,
            background,
            execution_mode,
        )

        if schedule_type and self._scheduler is not None:
            try:
                self._scheduler.ensure_job(
                    task_id, json.loads(schedule_config),
                    ai_prompt, data.get("thread_id"), background, execution_mode,
                )
                await database.async_call(database.update_task, task_id, None,
                                          apscheduler_job_id=task_id)
            except Exception:
                logger.exception("Failed to schedule APScheduler job for task %s", task_id)

        result: dict = {"ok": True, "task_id": task_id}
        if schedule_desc:
            result["schedule"] = schedule_desc
        return self._json(result)

    async def _api_task_update(self, request: web.Request) -> web.Response:
        """PUT /api/tasks/{task_id} — update task fields."""
        from wintermute.infra import database
        from wintermute.tools.task_tools import _resolve_execution_mode
        task_id = request.match_info["task_id"]
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        allowed = {"content", "status", "ai_prompt", "execution_mode"}
        kwargs = {k: v for k, v in data.items() if k in allowed and v is not None}
        if not kwargs:
            return web.json_response({"error": "No valid fields to update"}, status=400)

        # Validate execution_mode/ai_prompt consistency when either is changing.
        if "execution_mode" in kwargs or "ai_prompt" in kwargs:
            task = await database.async_call(database.get_task, task_id)
            if not task:
                return web.json_response({"error": "not found"}, status=404)
            merged_execution_mode = kwargs.get("execution_mode", task.get("execution_mode"))
            merged_ai_prompt = (kwargs.get("ai_prompt", task.get("ai_prompt")) or "").strip() or None
            try:
                _resolve_execution_mode(
                    schedule_type=task.get("schedule_type"),
                    ai_prompt=merged_ai_prompt,
                    execution_mode=(merged_execution_mode or "").strip() or None,
                    background=bool(task.get("background", False)),
                    background_provided=True,
                )
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=400)

        ok = await database.async_call(database.update_task, task_id, None, **kwargs)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return self._json({"ok": True})

    async def _api_task_delete(self, request: web.Request) -> web.Response:
        """DELETE /api/tasks/{task_id} — soft-delete a task."""
        from wintermute.infra import database
        task_id = request.match_info["task_id"]
        task = await database.async_call(database.get_task, task_id)
        if not task:
            return web.json_response({"error": "not found"}, status=404)
        if task.get("apscheduler_job_id") and self._scheduler:
            self._scheduler.remove_job(task_id)
        ok = await database.async_call(database.delete_task, task_id)
        return self._json({"ok": ok})

    async def _api_task_action(self, request: web.Request) -> web.Response:
        """POST /api/tasks/{task_id}/{action} — pause/resume/complete a task."""
        from wintermute.infra import database
        task_id = request.match_info["task_id"]
        action = request.match_info["action"]
        if action == "pause":
            task = await database.async_call(database.get_task, task_id)
            if task and task.get("apscheduler_job_id") and self._scheduler:
                try:
                    self._scheduler.remove_job(task_id)
                except Exception:
                    logger.warning("Could not remove APScheduler job for paused task %s", task_id)
            ok = await database.async_call(database.pause_task, task_id)
        elif action == "resume":
            task = await database.async_call(database.get_task, task_id)
            ok = await database.async_call(database.resume_task, task_id)
            if ok and task and task.get("schedule_config") and self._scheduler:
                try:
                    sched_cfg = json.loads(task["schedule_config"])
                    self._scheduler.ensure_job(
                        task_id, sched_cfg,
                        task.get("ai_prompt"), task.get("thread_id"),
                        bool(task.get("background", False)), task.get("execution_mode"),
                    )
                except Exception:
                    logger.warning("Could not re-schedule APScheduler job for resumed task %s", task_id)
        elif action == "complete":
            try:
                data = await request.json()
            except Exception:
                data = {}
            reason = (data.get("reason") or "Completed via web interface").strip()
            task = await database.async_call(database.get_task, task_id)
            if task and task.get("apscheduler_job_id") and self._scheduler:
                self._scheduler.remove_job(task_id)
            ok = await database.async_call(database.complete_task, task_id, reason)
        else:
            return web.json_response({"error": f"Unknown action: {action}"}, status=400)
        if not ok:
            return web.json_response({"error": "not found or no change"}, status=404)
        return self._json({"ok": True})

    async def _api_tasks_purge(self, _request: web.Request) -> web.Response:
        """POST /api/tasks/purge — delete all completed tasks."""
        from wintermute.infra import database
        count = await database.async_call(database.delete_old_completed_tasks, 0)
        return self._json({"ok": True, "deleted": count})

    # ------------------------------------------------------------------
    # Per-thread config API
    # ------------------------------------------------------------------

    async def _api_thread_configs(self, _request: web.Request) -> web.Response:
        """GET /api/thread-config — list all overrides + available backends."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            return self._json({"error": "Thread config not available"}, status=503)
        overrides = mgr.get_all_overrides()
        backends = sorted(mgr.get_available_backends())
        return self._json({
            "overrides": overrides,
            "available_backends": backends,
            "keys": [
                {"name": "backend_name", "type": "select", "options": backends},
                {"name": "backend_overrides.compaction", "type": "select", "options": backends, "nullable": True},
                {"name": "backend_overrides.sub_sessions", "type": "select", "options": backends, "nullable": True},
                {"name": "backend_overrides.convergence_protocol", "type": "select", "options": backends, "nullable": True},
                {"name": "backend_overrides.nl_translation", "type": "select", "options": backends, "nullable": True},
                {"name": "session_timeout_minutes", "type": "int", "min": 1, "nullable": True},
                {"name": "sub_sessions_enabled", "type": "bool"},
                {"name": "system_prompt_mode", "type": "select", "options": ["full", "minimal"]},
                {"name": "seed_language", "type": "str", "placeholder": "en"},
                {"name": "nl_translation_enabled", "type": "bool"},
                {"name": "memory_top_k", "type": "int", "min": 1},
                {"name": "memory_score_threshold", "type": "float", "min": 0, "max": 1, "step": 0.05},
                {"name": "compaction_keep_recent", "type": "int", "min": 1},
                {"name": "max_inline_tool_rounds", "type": "int", "min": 0},
            ],
        })

    async def _api_thread_config_get(self, request: web.Request) -> web.Response:
        """GET /api/thread-config/{thread_id} — resolved config with sources."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            return self._json({"error": "Thread config not available"}, status=503)
        thread_id = request.match_info["thread_id"]
        resolved_dict = mgr.resolve_as_dict(thread_id)
        raw = mgr.get(thread_id)
        # Flatten backend_overrides dict into dotted keys for UI compatibility.
        flat_overrides: dict = {}
        if raw:
            for k, v in raw.__dict__.items():
                if v is None:
                    continue
                if k == "backend_overrides" and isinstance(v, dict):
                    for role, bname in v.items():
                        flat_overrides[f"backend_overrides.{role}"] = bname
                else:
                    flat_overrides[k] = v
        return self._json({
            "thread_id": thread_id,
            "resolved": resolved_dict,
            "overrides": flat_overrides,
        })

    async def _api_thread_config_set(self, request: web.Request) -> web.Response:
        """POST /api/thread-config/{thread_id} — set overrides (JSON body)."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            return self._json({"error": "Thread config not available"}, status=503)
        thread_id = request.match_info["thread_id"]
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON body")
        try:
            for key, value in body.items():
                mgr.set(thread_id, key, value)
        except (ValueError, TypeError) as exc:
            logger.exception("Invalid thread configuration for %s", thread_id)
            return self._json({"error": "Invalid thread configuration"}, status=400)
        resolved_dict = mgr.resolve_as_dict(thread_id)
        return self._json({"ok": True, "resolved": resolved_dict})

    async def _api_thread_config_reset(self, request: web.Request) -> web.Response:
        """DELETE /api/thread-config/{thread_id} — remove all overrides."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            return self._json({"error": "Thread config not available"}, status=503)
        thread_id = request.match_info["thread_id"]
        mgr.reset(thread_id)
        return self._json({"ok": True})

    # ------------------------------------------------------------------
    # Interaction Log
    # ------------------------------------------------------------------

    @staticmethod
    def _int_param(request: web.Request, name: str, default: int) -> int:
        """Parse an integer query parameter, returning *default* on bad input."""
        raw = request.query.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except (ValueError, TypeError):
            return default

    async def _api_interaction_log(self, request: web.Request) -> web.Response:
        from wintermute.infra import database
        limit = self._int_param(request, "limit", 200)
        offset = self._int_param(request, "offset", 0)
        session = request.query.get("session") or None
        action = request.query.get("action") or None
        before_id_s = request.query.get("before_id")
        after_id_s = request.query.get("after_id")
        before_id = self._int_param(request, "before_id", 0) or None if before_id_s else None
        after_id = self._int_param(request, "after_id", 0) or None if after_id_s else None
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
        try:
            entry_id = int(request.match_info["id"])
        except (ValueError, TypeError):
            return web.json_response({"error": "invalid id"}, status=400)
        entry = await database.async_call(database.get_interaction_log_entry, entry_id)
        if not entry:
            return web.json_response({"error": "not found"}, status=404)
        return self._json(entry)

    # ------------------------------------------------------------------
    # Outcomes API
    # ------------------------------------------------------------------

    async def _api_outcomes(self, request: web.Request) -> web.Response:
        from wintermute.infra import database
        limit = self._int_param(request, "limit", 200)
        offset = self._int_param(request, "offset", 0)
        status_filter = request.query.get("status") or None
        source_filter = request.query.get("source") or None
        rows, total, stats = await database.async_call(
            database.get_outcomes_page, limit=limit, offset=offset,
            status_filter=status_filter, source_filter=source_filter,
        )
        return self._json({"entries": rows, "total": total, "stats": stats})

    async def _api_cp_violations(self, _request: web.Request) -> web.Response:
        from wintermute.infra import database
        stats = await database.async_call(database.get_cp_violation_stats)
        return self._json(stats)

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
        top_k = self._int_param(request, "k", 5)

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

    async def _api_memory_list(self, _request: web.Request) -> web.Response:
        """GET /api/memory/all — return all memory entries."""
        from wintermute.infra import memory_store
        loop = asyncio.get_running_loop()
        items = await loop.run_in_executor(None, memory_store.get_all)
        return self._json({"items": items, "count": len(items)})

    async def _api_memory_delete(self, request: web.Request) -> web.Response:
        """DELETE /api/memory/{entry_id} — delete a single memory entry."""
        from wintermute.infra import memory_store
        entry_id = request.match_info["entry_id"]
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, memory_store.delete, entry_id)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return self._json({"ok": True})

    async def _api_memory_update(self, request: web.Request) -> web.Response:
        """PUT /api/memory/{entry_id} — update a memory entry (delete + re-add)."""
        from wintermute.infra import memory_store
        entry_id = request.match_info["entry_id"]
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        text = (data.get("text") or "").strip()
        if not text:
            return web.json_response({"error": "text is required"}, status=400)
        loop = asyncio.get_running_loop()
        # Verify the entry exists before updating.
        existing = await loop.run_in_executor(None, memory_store.exists_batch, [entry_id])
        if not existing:
            return web.json_response({"error": "not found"}, status=404)
        source = data.get("source", "user_explicit")
        # memory_store.add uses ON CONFLICT DO UPDATE, so this is a safe upsert
        # that preserves metadata (created_at, access stats) rather than
        # deleting and re-inserting.
        new_id = await loop.run_in_executor(None, memory_store.add, text, entry_id, source)
        return self._json({"ok": True, "id": new_id})

    async def _api_memory_bulk_delete(self, request: web.Request) -> web.Response:
        """POST /api/memory/bulk-delete — delete multiple memory entries."""
        from wintermute.infra import memory_store
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)
        ids = data.get("ids", [])
        if not ids:
            return web.json_response({"error": "ids list required"}, status=400)
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(None, memory_store.bulk_delete, ids)
        return self._json({"ok": True, "deleted": count})

    # ------------------------------------------------------------------
    # Prediction accuracy API
    # ------------------------------------------------------------------

    async def _api_prediction_accuracy(self, _request: web.Request) -> web.Response:
        """GET /api/prediction-accuracy — return active prediction accuracy data."""
        from wintermute.infra import database
        records = await database.async_call(database.get_active_predictions_accuracy)
        return self._json(records)

    # ------------------------------------------------------------------
    # Skills API
    # ------------------------------------------------------------------

    async def _api_skills(self, _request: web.Request) -> web.Response:
        """GET /api/skills — list all skills with metadata and stats."""
        from wintermute.infra import skill_store

        skills = []
        try:
            all_skills = skill_store.get_all()
            store_stats = skill_store.stats()
        except Exception as exc:
            logger.exception("Failed to list skills")
            return self._json({"error": "Failed to list skills", "skills": [], "count": 0}, status=500)

        for rec in all_skills:
            name = rec["name"]
            sstat = store_stats.get(name, {})
            skills.append({
                "name": name,
                "summary": rec.get("summary", ""),
                "doc_chars": len(rec.get("documentation", "")),
                "last_accessed": rec.get("last_accessed", 0),
                "read_count": sstat.get("read_count", 0),
                "sessions_loaded": sstat.get("sessions_loaded", 0),
                "success_count": sstat.get("success_count", 0),
                "failure_count": sstat.get("failure_count", 0),
                "version": sstat.get("version", 1),
                "last_read": sstat.get("last_read"),
            })
        return self._json({"skills": skills, "count": len(skills)})

    async def _api_skill_info(self, _request: web.Request) -> web.Response:
        """GET /api/skills/info — skill store backend metadata."""
        from wintermute.infra import skill_store

        backend = "unknown"
        if skill_store.is_qdrant_backend():
            backend = "qdrant"
        elif skill_store.is_vector_enabled():
            backend = "local_vector"
        return self._json({
            "backend": backend,
            "vector_enabled": skill_store.is_vector_enabled(),
            "count": skill_store.count(),
            "top_k": skill_store.get_top_k(),
            "threshold": skill_store.get_threshold(),
        })

    async def _api_skill_search(self, request: web.Request) -> web.Response:
        """GET /api/skills/search?q=...&k=5 — semantic/keyword skill search."""
        from wintermute.infra import skill_store

        query = request.query.get("q", "").strip()
        if not query:
            return web.json_response(
                {"error": "query parameter 'q' is required"}, status=400)
        try:
            top_k = int(request.query.get("k", str(skill_store.get_top_k())))
        except (ValueError, TypeError):
            top_k = skill_store.get_top_k()
        try:
            threshold = float(request.query.get("t", str(skill_store.get_threshold())))
        except (ValueError, TypeError):
            threshold = skill_store.get_threshold()

        def _search() -> list:
            return skill_store.search(query, top_k=top_k, threshold=threshold)

        results = await asyncio.get_running_loop().run_in_executor(None, _search)
        return self._json({"query": query, "results": results, "count": len(results)})

    async def _api_skill_get(self, request: web.Request) -> web.Response:
        """GET /api/skills/{name} — read full skill content."""
        from wintermute.infra import skill_store
        from wintermute.infra.skill_io import _validate_skill_name

        name = request.match_info["name"]
        try:
            name = _validate_skill_name(name)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        rec = skill_store.get(name)
        if rec is None:
            return web.json_response({"error": "not found"}, status=404)
        return self._json({
            "name": name,
            "summary": rec.get("summary", ""),
            "documentation": rec.get("documentation", ""),
            "content": f"{rec.get('summary', '')}\n\n{rec.get('documentation', '')}".strip(),
            "version": rec.get("version", 1),
            "changelog": rec.get("changelog", ""),
            "access_count": rec.get("access_count", 0),
            "last_accessed": rec.get("last_accessed", 0),
            "created_at": rec.get("created_at", 0),
        })

    async def _api_skill_create(self, request: web.Request) -> web.Response:
        """POST /api/skills — create a new skill."""
        from wintermute.infra.skill_io import add_skill, _validate_skill_name
        from wintermute.infra import skill_store

        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON body")
        name = (data.get("skill_name") or "").strip()
        summary = (data.get("summary") or "").strip()
        documentation = (data.get("documentation") or "").strip()
        if not name or not documentation:
            return web.json_response(
                {"error": "skill_name and documentation are required"}, status=400)
        try:
            name = _validate_skill_name(name)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        # Reject if already exists (non-tracking check).
        if skill_store.exists(name):
            return web.json_response(
                {"error": f"Skill '{name}' already exists. Use PUT to update."}, status=409)
        try:
            add_skill(name, documentation, summary=summary or None)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return self._json({"ok": True, "name": name})

    async def _api_skill_update(self, request: web.Request) -> web.Response:
        """PUT /api/skills/{name} — update an existing skill."""
        from wintermute.infra import skill_store
        from wintermute.infra.skill_io import _validate_skill_name

        name = request.match_info["name"]
        try:
            name = _validate_skill_name(name)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if not skill_store.exists(name):
            return web.json_response({"error": "not found"}, status=404)
        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON body")
        content = (data.get("content") or "").strip()
        summary = (data.get("summary") or "").strip()
        documentation = (data.get("documentation") or "").strip()
        # Backward compatibility for legacy debug UI:
        # If only "content" is provided, treat first line as summary and the
        # remainder as documentation (matching migration logic).
        if not documentation and content:
            lines = content.splitlines()
            if lines:
                if not summary:
                    summary = lines[0].strip()
                documentation = "\n".join(lines[1:]).strip()
        if not documentation:
            return web.json_response(
                {"error": "content or documentation is required"}, status=400)
        skill_store.update(name, summary=summary or None, documentation=documentation)
        from wintermute.infra import data_versioning
        data_versioning.commit_async(f"skill: {name}")
        return self._json({"ok": True, "name": name})

    async def _api_skill_delete(self, request: web.Request) -> web.Response:
        """DELETE /api/skills/{name} — soft-delete (archive) a skill.

        Writes a backup of the skill to data/skills/.archive/ before
        removing it from the store, preserving the ability to restore.
        """
        from wintermute.infra import skill_store
        from wintermute.infra import data_versioning
        from wintermute.infra.skill_io import _validate_skill_name
        from wintermute.infra.paths import SKILLS_DIR

        name = request.match_info["name"]
        try:
            name = _validate_skill_name(name)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if not skill_store.exists(name):
            return web.json_response({"error": "not found"}, status=404)
        # Archive: write a backup .md before removing from store.
        # Use get() here to fetch full content (stat bump is irrelevant before delete).
        rec = skill_store.get(name) or {}
        try:
            archive_dir = SKILLS_DIR / ".archive"
            archive_dir_resolved = archive_dir.resolve()
            archive_dir_resolved.mkdir(parents=True, exist_ok=True)
            archive_path = (archive_dir_resolved / f"{name}.md").resolve()
            # Guard: ensure resolved path stays strictly within the archive directory.
            if not archive_path.is_relative_to(archive_dir_resolved):
                raise ValueError(f"Archive path escapes directory: {name!r}")
            content = f"{rec.get('summary', '')}\n\n{rec.get('documentation', '')}".strip()
            changelog = rec.get("changelog", "")
            if changelog:
                content = f"{content}\n\n{changelog}"
            archive_path.write_text(content, encoding="utf-8")
        except Exception:
            logger.error("Failed to archive skill '%s' to .archive/; aborting delete", name, exc_info=True)
            return web.json_response(
                {"error": "failed to archive skill before deletion"},
                status=500,
            )
        skill_store.delete(name)
        data_versioning.commit_async(f"skill: archive {name}")
        logger.info("Skill '%s' archived and deleted via web API", name)
        return self._json({"ok": True, "name": name, "archived": True})

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
                else "signal" if tid.startswith("sig_")
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
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._sse_queues.add(q)
        try:
            while True:
                try:
                    payload = await self._build_stream_snapshot()
                    await response.write(
                        ("data: " + json.dumps(payload) + "\n\n").encode()
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("SSE snapshot error: %s", exc)
                # Drain any broadcast messages that arrived since last snapshot.
                while not q.empty():
                    try:
                        msg = q.get_nowait()
                        await response.write(
                            ("event: broadcast\ndata: " + json.dumps(msg) + "\n\n").encode()
                        )
                    except asyncio.QueueEmpty:
                        break
                await asyncio.sleep(3)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._sse_queues.discard(q)
        return response
