"""
Web Interface

A minimal aiohttp HTTP + WebSocket server that mirrors the Matrix interface.
Every connected browser tab is an independent chat session routed through the
same LLM thread.  Supports all special commands (/new, /compact, /reminders,
/heartbeat).

Works standalone when no Matrix account is configured.
"""

import logging
from typing import Optional

from aiohttp import web

import tools as tool_module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded single-page UI
# ---------------------------------------------------------------------------

_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ganglion</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    display: flex; flex-direction: column; height: 100dvh;
  }
  header {
    padding: .6rem 1rem;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
    display: flex; align-items: center; gap: .8rem;
  }
  header h1 { font-size: 1.1rem; font-weight: 600; color: #a8d8ea; }
  #status {
    font-size: .75rem; padding: .2rem .5rem;
    border-radius: 99px;
    background: #333; color: #aaa;
  }
  #status.connected { background: #0f3460; color: #a8d8ea; }
  #log {
    flex: 1; overflow-y: auto;
    padding: 1rem;
    display: flex; flex-direction: column; gap: .6rem;
  }
  .msg {
    max-width: 80%; padding: .5rem .8rem;
    border-radius: .6rem; line-height: 1.45;
    white-space: pre-wrap; word-break: break-word;
    font-size: .9rem;
  }
  .msg.user {
    align-self: flex-end;
    background: #0f3460; color: #e0e0e0;
    border-bottom-right-radius: .1rem;
  }
  .msg.assistant {
    align-self: flex-start;
    background: #16213e; color: #e0e0e0;
    border-bottom-left-radius: .1rem;
  }
  .msg.system {
    align-self: center;
    background: transparent; color: #888;
    font-size: .78rem; font-style: italic;
  }
  .ts { font-size: .65rem; color: #666; margin-top: .25rem; }
  form {
    display: flex; gap: .5rem;
    padding: .75rem 1rem;
    background: #16213e;
    border-top: 1px solid #0f3460;
  }
  input {
    flex: 1; padding: .55rem .8rem;
    background: #1a1a2e; color: #e0e0e0;
    border: 1px solid #0f3460; border-radius: .4rem;
    font-size: .9rem; outline: none;
  }
  input:focus { border-color: #a8d8ea; }
  button {
    padding: .55rem 1.1rem;
    background: #0f3460; color: #a8d8ea;
    border: none; border-radius: .4rem;
    cursor: pointer; font-size: .9rem;
  }
  button:hover { background: #1a4a80; }
  button:disabled { opacity: .4; cursor: default; }
</style>
</head>
<body>
<header>
  <h1>Ganglion</h1>
  <span id="status">connecting…</span>
</header>
<div id="log"></div>
<form id="form">
  <input id="input" type="text" placeholder="Message or /command…" autocomplete="off" disabled>
  <button id="send" type="submit" disabled>Send</button>
</form>
<script>
const log    = document.getElementById('log');
const form   = document.getElementById('form');
const input  = document.getElementById('input');
const send   = document.getElementById('send');
const status = document.getElementById('status');

let ws;

function ts() {
  return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

function addMsg(role, text) {
  const wrap = document.createElement('div');
  const msg  = document.createElement('div');
  msg.className = 'msg ' + role;
  msg.textContent = text;
  const t = document.createElement('div');
  t.className = 'ts';
  t.textContent = ts();
  wrap.style.display = 'flex';
  wrap.style.flexDirection = 'column';
  wrap.style.alignItems = role === 'user' ? 'flex-end'
                        : role === 'system' ? 'center' : 'flex-start';
  wrap.appendChild(msg);
  wrap.appendChild(t);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws');

  ws.onopen = () => {
    status.textContent = 'connected';
    status.className = 'connected';
    input.disabled = false;
    send.disabled  = false;
    input.focus();
  };

  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    addMsg(d.role, d.text);
  };

  ws.onclose = () => {
    status.textContent = 'disconnected – retrying…';
    status.className = '';
    input.disabled = true;
    send.disabled  = true;
    setTimeout(connect, 3000);
  };

  ws.onerror = () => ws.close();
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || ws.readyState !== WebSocket.OPEN) return;
  addMsg('user', text);
  ws.send(JSON.stringify({text}));
  input.value = '';
});

connect();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# WebInterface class
# ---------------------------------------------------------------------------


class WebInterface:
    """
    Runs an aiohttp web server.
    ``broadcast`` can be called from any task to push a message to all
    currently connected browser tabs.
    """

    def __init__(self, host: str, port: int, llm_thread) -> None:
        self._host = host
        self._port = port
        self._llm = llm_thread
        self._clients: set[web.WebSocketResponse] = set()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def broadcast(self, text: str) -> None:
        """Push a message to every connected browser tab."""
        import json
        payload = json.dumps({"role": "assistant", "text": text})
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_str(payload)
            except Exception:  # noqa: BLE001
                dead.add(ws)
        self._clients -= dead

    async def run(self) -> None:
        app = web.Application()
        app.router.add_get("/",   self._handle_index)
        app.router.add_get("/ws", self._handle_ws)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info("Web interface listening on http://%s:%d", self._host, self._port)

        # Keep running until cancelled.
        try:
            import asyncio
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_index(self, _request: web.Request) -> web.Response:
        return web.Response(text=_HTML, content_type="text/html")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        import json

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        logger.info("Web client connected (%d total)", len(self._clients))

        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                    text = data.get("text", "").strip()
                except (json.JSONDecodeError, AttributeError):
                    continue
                if not text:
                    continue

                reply = await self._dispatch(text, ws)
                if reply:
                    await ws.send_str(json.dumps({"role": "assistant", "text": reply}))
        finally:
            self._clients.discard(ws)
            logger.info("Web client disconnected (%d remaining)", len(self._clients))

        return ws

    # ------------------------------------------------------------------
    # Command dispatch (mirrors matrix_thread.py)
    # ------------------------------------------------------------------

    async def _dispatch(self, text: str, ws: web.WebSocketResponse) -> Optional[str]:
        import json

        async def system(msg: str) -> None:
            await ws.send_str(json.dumps({"role": "system", "text": msg}))

        if text == "/new":
            await self._llm.reset_session()
            await system("Session reset.")
            return None

        if text == "/compact":
            await self._llm.force_compact()
            await system("Context compaction complete.")
            return None

        if text == "/reminders":
            result = tool_module.execute_tool("list_reminders", {})
            return result

        if text == "/heartbeat":
            await system("Heartbeat review triggered.")
            await self._llm.enqueue_system_event(
                "The user manually triggered a heartbeat review. "
                "Review your HEARTBEATS.txt and report what actions, if any, you take."
            )
            return None

        return await self._llm.enqueue_user_message(text)
