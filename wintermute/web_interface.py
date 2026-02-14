"""
Web Interface

A minimal aiohttp HTTP + WebSocket server that mirrors the Matrix interface.
Each WebSocket connection gets its own thread_id for independent conversation
context.  Supports all special commands (/new, /compact, /reminders, /pulse).

Debug panel
-----------
Available at /debug (same host/port).  Provides a read/write inspection view
of all running sessions, sub-sessions, scheduled jobs, and reminders.
REST API under /api/debug/* is consumed by the embedded SPA.

Works standalone when no Matrix account is configured.
"""

import asyncio
import json
import logging
import secrets
from typing import Optional

from aiohttp import web

from wintermute import tools as tool_module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat UI (unchanged from original)
# ---------------------------------------------------------------------------

_CHAT_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wintermute</title>
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
  header a { font-size: .75rem; color: #a8d8ea; text-decoration: none; margin-left: auto; }
  header a:hover { text-decoration: underline; }
  #status {
    font-size: .75rem; padding: .2rem .5rem;
    border-radius: 99px;
    background: #333; color: #aaa;
  }
  #status.connected { background: #0f3460; color: #a8d8ea; }
  #thread-id {
    font-size: .65rem; color: #666;
  }
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
  <h1>Wintermute</h1>
  <span id="status">connecting\u2026</span>
  <span id="thread-id"></span>
  <a href="/debug">Debug \u2197</a>
</header>
<div id="log"></div>
<form id="form">
  <input id="input" type="text" placeholder="Message or /command\u2026" autocomplete="off" disabled>
  <button id="send" type="submit" disabled>Send</button>
</form>
<script>
const log    = document.getElementById('log');
const form   = document.getElementById('form');
const input  = document.getElementById('input');
const send   = document.getElementById('send');
const status = document.getElementById('status');
const threadEl = document.getElementById('thread-id');

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
    if (d.thread_id) threadEl.textContent = d.thread_id;
    addMsg(d.role, d.text);
  };

  ws.onclose = () => {
    status.textContent = 'disconnected \u2013 retrying\u2026';
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
# Debug panel SPA
# ---------------------------------------------------------------------------

_DEBUG_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wintermute Debug</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    display: flex; flex-direction: column; height: 100dvh;
    font-size: 13px;
  }
  header {
    padding: .5rem 1rem;
    background: #0d1526;
    border-bottom: 1px solid #0f3460;
    display: flex; align-items: center; gap: .8rem;
    flex-shrink: 0;
  }
  header h1 { font-size: 1rem; font-weight: 600; color: #a8d8ea; }
  header a { color: #a8d8ea; text-decoration: none; font-size: .82rem; }
  header a:hover { text-decoration: underline; }
  .header-right { margin-left: auto; display: flex; align-items: center; gap: .8rem; }
  #refresh-status { font-size: .72rem; color: #555; }
  #refresh-toggle {
    font-size: .75rem; padding: .2rem .6rem;
    background: #16213e; color: #a8d8ea;
    border: 1px solid #0f3460; border-radius: 99px; cursor: pointer;
  }
  #refresh-toggle:hover { background: #1a4a80; }
  #layout { display: flex; flex: 1; overflow: hidden; }
  /* Sidebar */
  #sidebar {
    width: 150px; min-width: 150px;
    background: #0d1526;
    border-right: 1px solid #0f3460;
    display: flex; flex-direction: column;
    padding: .4rem 0;
    flex-shrink: 0;
  }
  .tab-btn {
    padding: .55rem .9rem;
    background: none; border: none;
    color: #777; cursor: pointer;
    text-align: left; font-size: .82rem;
    border-left: 2px solid transparent;
    transition: color .12s, background .12s;
    white-space: nowrap;
  }
  .tab-btn:hover { color: #ccc; background: #16213e; }
  .tab-btn.active { color: #a8d8ea; border-left-color: #a8d8ea; background: #16213e; }
  .tab-count {
    float: right; font-size: .68rem;
    background: #1a3060; color: #7aa; padding: .05rem .35rem;
    border-radius: 99px;
  }
  /* Main content */
  #main { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
  .tab-panel { display: none; flex: 1; overflow: hidden; flex-direction: column; }
  .tab-panel.active { display: flex; }
  /* Sessions split layout */
  .sessions-split { display: flex; flex: 1; overflow: hidden; }
  .session-list {
    width: 270px; min-width: 270px;
    overflow-y: auto;
    border-right: 1px solid #0f3460;
    padding: .4rem;
    flex-shrink: 0;
  }
  .session-item {
    padding: .45rem .6rem;
    border-radius: .35rem;
    cursor: pointer;
    margin-bottom: .25rem;
    border: 1px solid transparent;
    transition: background .1s;
  }
  .session-item:hover { background: #16213e; }
  .session-item.selected { border-color: #2a5a8a; background: #16213e; }
  .session-id {
    font-family: monospace; font-size: .72rem;
    word-break: break-all; color: #ccc;
  }
  .session-meta { display: flex; gap: .3rem; align-items: center; margin-top: .3rem; flex-wrap: wrap; }
  .ctx-bar-wrap {
    height: 4px; background: #252535; border-radius: 2px;
    margin-top: .4rem; overflow: hidden;
  }
  .ctx-bar { height: 4px; border-radius: 2px; transition: width .4s; }
  .ctx-label { font-size: .65rem; color: #666; margin-top: .15rem; }
  /* Message view */
  .msg-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .msg-panel-header {
    padding: .5rem .9rem;
    background: #0d1526;
    border-bottom: 1px solid #0f3460;
    display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
    flex-shrink: 0;
  }
  .msg-panel-header .ctx-info { font-size: .72rem; color: #888; margin-left: auto; }
  .msg-view {
    flex: 1; overflow-y: auto;
    padding: .8rem 1rem;
    display: flex; flex-direction: column; gap: .4rem;
  }
  .msg-wrap { display: flex; flex-direction: column; }
  .msg-wrap.user { align-items: flex-end; }
  .msg-wrap.assistant { align-items: flex-start; }
  .msg-wrap.system { align-items: center; }
  .msg-bubble {
    max-width: 85%; padding: .4rem .7rem;
    border-radius: .5rem;
    white-space: pre-wrap; word-break: break-word;
    line-height: 1.45; font-size: .84rem;
  }
  .msg-bubble.user { background: #0f3460; }
  .msg-bubble.assistant { background: #16213e; }
  .msg-bubble.system { background: transparent; color: #777; font-style: italic; font-size: .78rem; }
  .msg-meta { font-size: .62rem; color: #555; margin-top: .1rem; }
  .msg-expand { font-size: .7rem; color: #4a7ab0; cursor: pointer; }
  .msg-expand:hover { color: #a8d8ea; }
  .msg-inject {
    display: flex; gap: .5rem; padding: .5rem .8rem;
    background: #0d1526; border-top: 1px solid #0f3460;
    flex-shrink: 0;
  }
  .msg-inject input {
    flex: 1; padding: .35rem .6rem;
    background: #1a1a2e; color: #e0e0e0;
    border: 1px solid #0f3460; border-radius: .35rem;
    font-size: .82rem; outline: none;
  }
  .msg-inject input:focus { border-color: #a8d8ea; }
  /* Badges */
  .badge {
    font-size: .62rem; padding: .1rem .38rem;
    border-radius: 99px; font-weight: 600;
    text-transform: uppercase; white-space: nowrap;
  }
  .badge-web      { background: #0f3460; color: #a8d8ea; }
  .badge-matrix   { background: #2d1b4e; color: #c9b0ff; }
  .badge-system   { background: #1a3a1a; color: #90ee90; }
  .badge-live     { background: #0a3a0a; color: #6fe06f; }
  .badge-running  { background: #0f3460; color: #a8d8ea; }
  .badge-completed{ background: #1a3a1a; color: #90ee90; }
  .badge-failed   { background: #3a1a1a; color: #ff9090; }
  .badge-timeout  { background: #3a2a1a; color: #ffcc90; }
  .badge-pending  { background: #2a2a1a; color: #cccc80; }
  .badge-cancelled{ background: #2a2a2a; color: #999; }
  /* Scrollable area */
  .scroll-area { flex: 1; overflow-y: auto; padding: .6rem 1rem; }
  /* Tables */
  .data-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
  .data-table th {
    text-align: left; padding: .4rem .6rem;
    background: #0d1526; color: #777;
    border-bottom: 1px solid #0f3460;
    font-weight: 600; white-space: nowrap;
    position: sticky; top: 0; z-index: 1;
  }
  .data-table td {
    padding: .38rem .6rem;
    border-bottom: 1px solid #141428;
    vertical-align: top;
  }
  .data-table tr:hover td { background: #16213e; }
  .mono { font-family: monospace; font-size: .75rem; }
  .trunc {
    max-width: 280px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }
  .dim { color: #666; }
  /* Buttons */
  .btn {
    padding: .3rem .7rem;
    border: 1px solid #0f3460; border-radius: .35rem;
    cursor: pointer; font-size: .78rem;
    background: #16213e; color: #a8d8ea;
    transition: background .1s;
  }
  .btn:hover { background: #1a4a80; }
  .btn-danger { background: #2a1010; border-color: #5a2020; color: #ff9090; }
  .btn-danger:hover { background: #3a1515; }
  .btn-sm { padding: .15rem .45rem; font-size: .72rem; }
  .btn-primary { background: #0f3460; border-color: #1a5490; color: #a8d8ea; }
  .btn-primary:hover { background: #1a4a80; }
  /* Forms */
  .form-bar {
    display: flex; gap: .5rem; flex-wrap: wrap; align-items: flex-end;
    padding: .6rem .8rem;
    background: #0d1526;
    border-bottom: 1px solid #0f3460;
    flex-shrink: 0;
  }
  .form-group { display: flex; flex-direction: column; gap: .2rem; }
  .form-group label { font-size: .7rem; color: #777; }
  .form-group input, .form-group select, .form-group textarea {
    padding: .3rem .5rem;
    background: #1a1a2e; color: #e0e0e0;
    border: 1px solid #0f3460; border-radius: .35rem;
    font-size: .82rem; outline: none;
  }
  .form-group input:focus,
  .form-group select:focus,
  .form-group textarea:focus { border-color: #a8d8ea; }
  .form-group textarea { resize: vertical; min-height: 48px; }
  /* Section headers for reminders */
  .section-hdr {
    padding: .4rem .6rem;
    background: #0d1526;
    border: 1px solid #0f3460;
    border-radius: .3rem;
    cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: .3rem; margin-top: .6rem;
    font-size: .82rem; color: #888;
    user-select: none;
  }
  .section-hdr:hover { color: #ccc; }
  .section-hdr.open { color: #a8d8ea; }
  .section-body { margin-bottom: .4rem; }
  /* Empty state */
  .empty { padding: 1.5rem; text-align: center; color: #555; font-size: .82rem; }
  /* Context bar in header */
  .ctx-bar-header {
    display: inline-flex; align-items: center; gap: .4rem;
    font-size: .72rem;
  }
  .ctx-bar-header .bar-wrap {
    width: 80px; height: 5px; background: #252535; border-radius: 3px; overflow: hidden;
  }
  .ctx-bar-header .bar-fill { height: 5px; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>Wintermute Debug</h1>
  <a href="/">&#8592; Chat</a>
  <div class="header-right">
    <span id="refresh-status"></span>
    <button id="refresh-toggle" onclick="toggleRefresh()">Auto-refresh: ON</button>
  </div>
</header>
<div id="layout">
  <nav id="sidebar">
    <button class="tab-btn active" data-tab="sessions" onclick="showTab('sessions')">
      Sessions <span class="tab-count" id="cnt-sessions">0</span>
    </button>
    <button class="tab-btn" data-tab="subsessions" onclick="showTab('subsessions')">
      Sub-sessions <span class="tab-count" id="cnt-subsessions">0</span>
    </button>
    <button class="tab-btn" data-tab="workflows" onclick="showTab('workflows')">
      Workflows <span class="tab-count" id="cnt-workflows">0</span>
    </button>
    <button class="tab-btn" data-tab="jobs" onclick="showTab('jobs')">
      Jobs <span class="tab-count" id="cnt-jobs">0</span>
    </button>
    <button class="tab-btn" data-tab="reminders" onclick="showTab('reminders')">
      Reminders <span class="tab-count" id="cnt-reminders">0</span>
    </button>
  </nav>

  <div id="main">
    <!-- ── Sessions ── -->
    <div class="tab-panel active" id="panel-sessions">
      <div class="sessions-split">
        <div class="session-list" id="session-list">
          <div class="empty">Loading sessions\u2026</div>
        </div>
        <div class="msg-panel">
          <div class="msg-panel-header" id="msg-panel-header">
            <span style="color:#555">Select a session to inspect it</span>
          </div>
          <div class="msg-view" id="msg-view">
          </div>
          <div class="msg-inject">
            <input id="inject-input" type="text"
              placeholder="Inject message into selected session\u2026"
              onkeydown="if(event.key==='Enter')sendToSession()">
            <button class="btn btn-primary" onclick="sendToSession()">Send</button>
          </div>
        </div>
      </div>
    </div>

    <!-- ── Sub-sessions ── -->
    <div class="tab-panel" id="panel-subsessions">
      <div class="scroll-area">
        <table class="data-table">
          <thead>
            <tr>
              <th>ID</th><th>Workflow</th><th>Deps</th><th>Parent</th><th>Status</th>
              <th>Objective</th><th>Mode</th>
              <th>Created</th><th>Duration</th><th>Result / Error</th>
            </tr>
          </thead>
          <tbody id="subsessions-body">
            <tr><td colspan="10" class="empty">Loading\u2026</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ── Workflows ── -->
    <div class="tab-panel" id="panel-workflows">
      <div class="scroll-area" id="workflows-area">
        <div class="empty">Loading\u2026</div>
      </div>
    </div>

    <!-- ── Jobs ── -->
    <div class="tab-panel" id="panel-jobs">
      <div class="scroll-area">
        <table class="data-table">
          <thead>
            <tr>
              <th>Job ID</th><th>Trigger</th><th>Next Run</th><th>Args</th>
            </tr>
          </thead>
          <tbody id="jobs-body">
            <tr><td colspan="4" class="empty">Loading\u2026</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ── Reminders ── -->
    <div class="tab-panel" id="panel-reminders">
      <form id="reminder-form" class="form-bar" onsubmit="createReminder(event)">
        <div class="form-group">
          <label>Type</label>
          <select name="schedule_type" onchange="onScheduleTypeChange(this.value)" required>
            <option value="once">Once</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
            <option value="interval">Interval</option>
          </select>
        </div>
        <div class="form-group" id="fg-at">
          <label id="lbl-at">At (HH:MM or &ldquo;in 2 hours&rdquo;)</label>
          <input name="at" placeholder="e.g. 09:00 / tomorrow 09:00" style="width:190px">
        </div>
        <div class="form-group" id="fg-dow" style="display:none">
          <label>Day of week</label>
          <select name="day_of_week">
            <option value="mon">Mon</option><option value="tue">Tue</option>
            <option value="wed">Wed</option><option value="thu">Thu</option>
            <option value="fri">Fri</option><option value="sat">Sat</option>
            <option value="sun">Sun</option>
          </select>
        </div>
        <div class="form-group" id="fg-dom" style="display:none">
          <label>Day of month</label>
          <input name="day_of_month" type="number" min="1" max="31" style="width:70px" placeholder="1">
        </div>
        <div class="form-group" id="fg-interval" style="display:none">
          <label>Interval (seconds)</label>
          <input name="interval_seconds" type="number" min="1" style="width:100px" placeholder="3600">
        </div>
        <div class="form-group" id="fg-window" style="display:none">
          <label>Window start (HH:MM)</label>
          <input name="window_start" placeholder="08:00" style="width:70px">
        </div>
        <div class="form-group" id="fg-window-end" style="display:none">
          <label>Window end (HH:MM)</label>
          <input name="window_end" placeholder="20:00" style="width:70px">
        </div>
        <div class="form-group">
          <label>Message</label>
          <input name="message" placeholder="Reminder text" style="width:180px" required>
        </div>
        <div class="form-group">
          <label>AI prompt (optional)</label>
          <input name="ai_prompt" placeholder="Prompt for AI when it fires" style="width:200px">
        </div>
        <div class="form-group" style="justify-content:flex-end">
          <label>&nbsp;</label>
          <label style="display:flex;align-items:center;gap:.3rem;cursor:pointer">
            <input type="checkbox" name="system_reminder" style="width:auto"> System
          </label>
        </div>
        <div class="form-group" style="justify-content:flex-end">
          <label>&nbsp;</label>
          <button type="submit" class="btn btn-primary">+ Create</button>
        </div>
      </form>
      <div class="scroll-area" id="reminders-area">
        <div class="empty">Loading\u2026</div>
      </div>
    </div>
  </div>
</div>

<script>
// ── State ──
let currentTab = 'sessions';
let selectedSession = null;
let autoRefresh = true;
let refreshTimer = null;
const REFRESH_MS = 5000;

// ── Helpers ──
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmtTokens(n) {
  if (n >= 1000) return (n/1000).toFixed(1) + 'k';
  return String(n);
}
function fmtTime(iso) {
  if (!iso) return '\u2014';
  return new Date(iso).toLocaleString();
}
function fmtDuration(startIso, endIso) {
  const start = new Date(startIso);
  const end = endIso ? new Date(endIso) : new Date();
  const secs = ((end - start) / 1000);
  if (endIso) return secs.toFixed(1) + 's';
  return secs.toFixed(0) + 's\u2026';
}
function ctxColor(pct) {
  if (pct > 80) return '#e74c3c';
  if (pct > 50) return '#f39c12';
  return '#27ae60';
}
function badge(cls, text) {
  return `<span class="badge badge-${esc(cls)}">${esc(text)}</span>`;
}
function sessionType(id) {
  if (id.startsWith('web_')) return 'web';
  if (id.startsWith('!') && id.includes(':')) return 'matrix';
  return 'system';
}

// ── Tab switching ──
function showTab(name) {
  currentTab = name;
  document.querySelectorAll('.tab-btn')
    .forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel')
    .forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
  loadTab(name);
}

async function loadTab(name) {
  try {
    switch (name) {
      case 'sessions':    await loadSessions(); break;
      case 'subsessions': await loadSubSessions(); break;
      case 'workflows':   await loadWorkflows(); break;
      case 'jobs':        await loadJobs(); break;
      case 'reminders':   await loadReminders(); break;
    }
  } catch (e) {
    console.error('loadTab error:', e);
  }
}

// ── Auto-refresh ──
function scheduleRefresh() {
  clearTimeout(refreshTimer);
  if (!autoRefresh) return;
  refreshTimer = setTimeout(async () => {
    await loadTab(currentTab);
    document.getElementById('refresh-status').textContent =
      'Updated ' + new Date().toLocaleTimeString();
    scheduleRefresh();
  }, REFRESH_MS);
}

function toggleRefresh() {
  autoRefresh = !autoRefresh;
  document.getElementById('refresh-toggle').textContent =
    'Auto-refresh: ' + (autoRefresh ? 'ON' : 'OFF');
  if (autoRefresh) scheduleRefresh();
  else clearTimeout(refreshTimer);
}

// ── Sessions ──
async function loadSessions() {
  const r = await fetch('/api/debug/sessions');
  const data = await r.json();
  const sessions = data.sessions || [];
  document.getElementById('cnt-sessions').textContent = sessions.length;
  renderSessionList(sessions);
  if (selectedSession) {
    // Re-load messages for current selection (may have new entries)
    await loadSessionMessages(selectedSession);
  }
}

function renderSessionList(sessions) {
  const el = document.getElementById('session-list');
  if (!sessions.length) {
    el.innerHTML = '<div class="empty">No active sessions</div>';
    return;
  }
  el.innerHTML = sessions.map(s => {
    const type = sessionType(s.id);
    const isSelected = s.id === selectedSession;
    const shortId = s.id.length > 32 ? s.id.slice(0, 28) + '\u2026' : s.id;
    // stacked bar: sp (gray) + tools (purple) + hist (colored)
    const spPct    = Math.min(s.sp_tokens    / s.total_limit * 100, 100);
    const toolsPct = Math.min((s.tools_tokens || 0) / s.total_limit * 100, 100 - spPct);
    const hiPct    = Math.min(s.hist_tokens   / s.total_limit * 100, 100 - spPct - toolsPct);
    const histColor = ctxColor(s.context_pct);
    const tip = `SP: ${fmtTokens(s.sp_tokens)} + Tools: ${fmtTokens(s.tools_tokens||0)} + Hist: ${fmtTokens(s.hist_tokens)} = ${fmtTokens(s.total_used)} / ${fmtTokens(s.total_limit)} tok (${s.context_pct}%)`;
    return `<div class="session-item${isSelected ? ' selected' : ''}"
                 data-id="${esc(s.id)}" onclick="selectSession(this.dataset.id)">
      <div class="session-id">${esc(shortId)}</div>
      <div class="session-meta">
        ${badge(type, type)}
        ${s.live ? badge('live', 'live') : ''}
        <span class="dim">${s.msg_count} msg${s.msg_count !== 1 ? 's' : ''}</span>
      </div>
      <div class="ctx-bar-wrap" title="${tip}">
        <div style="display:flex;height:4px">
          <div style="width:${spPct}%;background:#445;flex-shrink:0"></div>
          <div style="width:${toolsPct}%;background:#5a3a8a;flex-shrink:0"></div>
          <div style="width:${hiPct}%;background:${histColor};flex-shrink:0"></div>
        </div>
      </div>
      <div class="ctx-label">${s.context_pct}% &middot; SP&nbsp;${fmtTokens(s.sp_tokens)}&nbsp;+&nbsp;tools&nbsp;${fmtTokens(s.tools_tokens||0)}&nbsp;+&nbsp;hist&nbsp;${fmtTokens(s.hist_tokens)}&nbsp;/&nbsp;${fmtTokens(s.total_limit)}</div>
    </div>`;
  }).join('');
}

function selectSession(id) {
  selectedSession = id;
  document.querySelectorAll('.session-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.id === id);
  });
  loadSessionMessages(id);
}

async function loadSessionMessages(threadId) {
  const hdr = document.getElementById('msg-panel-header');
  const view = document.getElementById('msg-view');
  const type = sessionType(threadId);

  const r = await fetch('/api/debug/sessions/' + encodeURIComponent(threadId) + '/messages');
  const data = await r.json();
  const msgs = data.messages || [];

  // Context bar (stacked: sp + tools + hist)
  let ctxHtml = '';
  if (data.total_limit !== undefined) {
    const spPct    = Math.min(data.sp_tokens    / data.total_limit * 100, 100);
    const toolsPct = Math.min((data.tools_tokens||0) / data.total_limit * 100, 100 - spPct);
    const hiPct    = Math.min(data.hist_tokens   / data.total_limit * 100, 100 - spPct - toolsPct);
    const histColor = ctxColor(data.context_pct || 0);
    const tip2 = `SP: ${fmtTokens(data.sp_tokens)} + Tools: ${fmtTokens(data.tools_tokens||0)} + Hist: ${fmtTokens(data.hist_tokens)} = ${fmtTokens(data.total_used)} / ${fmtTokens(data.total_limit)} tokens`;
    ctxHtml = `<span class="ctx-info ctx-bar-header" title="${tip2}">
      <span class="bar-wrap" style="display:flex">
        <span style="display:block;height:5px;width:${spPct}%;background:#445;flex-shrink:0"></span>
        <span style="display:block;height:5px;width:${toolsPct}%;background:#5a3a8a;flex-shrink:0"></span>
        <span style="display:block;height:5px;width:${hiPct}%;background:${histColor};flex-shrink:0"></span>
      </span>
      ${data.context_pct}% &middot; SP&nbsp;${fmtTokens(data.sp_tokens)}&nbsp;+&nbsp;tools&nbsp;${fmtTokens(data.tools_tokens||0)}&nbsp;+&nbsp;hist&nbsp;${fmtTokens(data.hist_tokens)}&nbsp;/&nbsp;${fmtTokens(data.total_limit)}
    </span>`;
  }

  hdr.innerHTML = `
    ${badge(type, type)}
    <span class="mono" style="font-size:.75rem">${esc(threadId)}</span>
    <span class="dim">${msgs.length} msg${msgs.length !== 1 ? 's' : ''}</span>
    ${ctxHtml}
    <span style="margin-left:auto;display:flex;gap:.4rem">
      <button class="btn btn-sm" onclick="showSystemPrompt()">Sys&nbsp;Prompt</button>
      <button class="btn btn-sm" onclick="compactSession()">Compact</button>
      <button class="btn btn-danger btn-sm" onclick="deleteSession()">Delete</button>
    </span>`;

  const scrolledToBottom = view.scrollHeight - view.scrollTop <= view.clientHeight + 40;

  view.innerHTML = msgs.length ? msgs.map(m => {
    const ts = new Date(m.ts * 1000).toLocaleString();
    const tokInfo = m.tokens ? ` &middot; ${m.tokens} tok` : '';
    const MAX = 800;
    let content = m.content || '';
    const truncated = content.length > MAX;
    const displayContent = truncated ? content.slice(0, MAX) : content;
    const expandLink = truncated
      ? `<span class="msg-expand" data-full="${esc(content)}" onclick="expandMsg(this)">[show more +${content.length-MAX} chars]</span>`
      : '';
    return `<div class="msg-wrap ${esc(m.role)}">
      <div class="msg-bubble ${esc(m.role)}">${esc(displayContent)}${truncated ? '\u2026' : ''}</div>
      ${expandLink}
      <div class="msg-meta">${ts}${tokInfo}</div>
    </div>`;
  }).join('') : '<div class="empty">No messages in this thread</div>';

  if (scrolledToBottom) view.scrollTop = view.scrollHeight;
}

function expandMsg(el) {
  const bubble = el.previousElementSibling;
  bubble.textContent = el.dataset.full;
  el.remove();
}

async function sendToSession() {
  const input = document.getElementById('inject-input');
  const text = input.value.trim();
  if (!text) return;
  if (!selectedSession) { alert('Select a session first'); return; }
  input.value = '';
  input.disabled = true;
  try {
    const r = await fetch('/api/debug/sessions/' + encodeURIComponent(selectedSession) + '/send', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text}),
    });
    const data = await r.json();
    if (!data.ok) alert('Error: ' + (data.error || JSON.stringify(data)));
    else setTimeout(() => loadSessionMessages(selectedSession), 800);
  } finally {
    input.disabled = false;
    input.focus();
  }
}

async function deleteSession() {
  if (!selectedSession) return;
  if (!confirm('Archive all messages for session ' + selectedSession + '? (Same as /new in chat)')) return;
  const r = await fetch('/api/debug/sessions/' + encodeURIComponent(selectedSession) + '/delete', {method: 'POST'});
  const data = await r.json();
  if (data.ok) { selectedSession = null; await loadSessions(); }
  else alert('Error: ' + (data.error || JSON.stringify(data)));
}

async function compactSession() {
  if (!selectedSession) return;
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = 'Compacting\u2026';
  try {
    const r = await fetch('/api/debug/sessions/' + encodeURIComponent(selectedSession) + '/compact', {method: 'POST'});
    const data = await r.json();
    if (data.ok) await loadSessionMessages(selectedSession);
    else alert('Error: ' + (data.error || JSON.stringify(data)));
  } finally {
    btn.disabled = false;
    btn.textContent = 'Compact';
  }
}

// ── System prompt modal ──
async function showSystemPrompt() {
  // Always recreate so we get fresh data.
  const existing = document.getElementById('sp-modal');
  if (existing) existing.remove();
  const modal = document.createElement('div');
  modal.id = 'sp-modal';
  modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;display:flex;align-items:center;justify-content:center';
  modal.innerHTML = `
    <div style="background:#16213e;border:1px solid #0f3460;border-radius:.5rem;width:82vw;max-height:88vh;display:flex;flex-direction:column;overflow:hidden">
      <div style="padding:.6rem 1rem;background:#0d1526;border-bottom:1px solid #0f3460;display:flex;align-items:center;gap:.7rem;flex-shrink:0;flex-wrap:wrap">
        <span style="font-weight:600;color:#a8d8ea">System Prompt + Tool Schemas</span>
        <span id="sp-stats" style="font-size:.75rem;color:#888">Loading\u2026</span>
        <button class="btn btn-sm" style="margin-left:auto" onclick="document.getElementById('sp-modal').remove()">Close</button>
      </div>
      <div style="flex:1;overflow:auto;display:flex;flex-direction:column">
        <div style="padding:.4rem .7rem;background:#0a1020;border-bottom:1px solid #0f3460;font-size:.72rem;color:#888;display:flex;gap:1.5rem" id="sp-token-bar"></div>
        <div style="padding:.4rem .7rem;background:#0d1526;border-bottom:1px solid #0f3460;font-size:.78rem;color:#a8d8ea;cursor:pointer;user-select:none" onclick="toggleSpSection('sp-section')">&#9660; System Prompt</div>
        <pre id="sp-section" style="padding:1rem;font-size:.78rem;line-height:1.5;white-space:pre-wrap;word-break:break-word;color:#ccc;margin:0">Loading\u2026</pre>
        <div style="padding:.4rem .7rem;background:#0d1526;border-bottom:1px solid #0f3460;font-size:.78rem;color:#c9b0ff;cursor:pointer;user-select:none" onclick="toggleSpSection('tools-section')">&#9660; Tool Schemas</div>
        <pre id="tools-section" style="padding:1rem;font-size:.75rem;line-height:1.5;white-space:pre-wrap;word-break:break-word;color:#b0a0cc;margin:0">Loading\u2026</pre>
      </div>
    </div>`;
  modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
  document.body.appendChild(modal);
  try {
    const r = await fetch('/api/debug/system-prompt');
    const data = await r.json();
    if (data.error) {
      document.getElementById('sp-section').textContent = 'Error: ' + data.error;
      document.getElementById('tools-section').textContent = '';
      return;
    }
    const spTok    = data.sp_tokens || 0;
    const toolsTok = data.tools_tokens || 0;
    const total    = data.tokens || (spTok + toolsTok);
    document.getElementById('sp-stats').textContent =
      `${fmtTokens(total)} tok total \u2022 ${data.pct}% of context`;
    document.getElementById('sp-token-bar').innerHTML =
      `<span style="color:#8899aa">SP: <strong>${fmtTokens(spTok)}</strong> tok</span>` +
      `<span style="color:#c9b0ff">Tools: <strong>${fmtTokens(toolsTok)}</strong> tok</span>` +
      `<span style="color:#666">Limit: ${fmtTokens(data.total_limit)} tok</span>`;
    document.getElementById('sp-section').textContent = data.prompt;
    document.getElementById('tools-section').textContent =
      JSON.stringify(data.tool_schemas, null, 2);
  } catch (e) {
    document.getElementById('sp-section').textContent = 'Failed to load: ' + e;
    document.getElementById('tools-section').textContent = '';
  }
}

function toggleSpSection(id) {
  const el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? '' : 'none';
}

// ── Sub-sessions ──
async function loadSubSessions() {
  const r = await fetch('/api/debug/subsessions');
  const data = await r.json();
  const sessions = data.sessions || [];
  document.getElementById('cnt-subsessions').textContent = sessions.length;
  const tbody = document.getElementById('subsessions-body');
  if (!sessions.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty">No sub-sessions recorded</td></tr>';
    return;
  }
  tbody.innerHTML = sessions.map(s => {
    const duration = fmtDuration(s.created_at, s.completed_at);
    const resultText = s.result
      ? s.result.slice(0, 120) + (s.result.length > 120 ? '\u2026' : '')
      : (s.error ? '\u26a0 ' + s.error.slice(0, 80) : '');
    const parentShort = s.parent_thread_id
      ? (s.parent_thread_id.length > 20 ? s.parent_thread_id.slice(0, 18) + '\u2026' : s.parent_thread_id)
      : '\u2014';
    const wfShort = s.workflow_id || '\u2014';
    const deps = (s.depends_on || []);
    const depsStr = deps.length ? deps.join(', ') : '\u2014';
    return `<tr>
      <td class="mono">${esc(s.session_id)}</td>
      <td class="mono dim">${esc(wfShort)}</td>
      <td class="mono dim" title="${esc(depsStr)}">${deps.length ? esc(deps.map(d => d.slice(0,12)).join(', ')) : '\u2014'}</td>
      <td class="mono dim" title="${esc(s.parent_thread_id || '')}">${esc(parentShort)}</td>
      <td>${badge(s.status, s.status)}</td>
      <td class="trunc" title="${esc(s.objective)}">${esc(s.objective.slice(0, 60))}${s.objective.length > 60 ? '\u2026' : ''}</td>
      <td class="dim">${esc(s.system_prompt_mode)}</td>
      <td class="dim" style="white-space:nowrap">${fmtTime(s.created_at)}</td>
      <td class="dim" style="white-space:nowrap">${duration}</td>
      <td class="trunc" title="${esc(resultText)}">${esc(resultText)}</td>
    </tr>`;
  }).join('');
}

// ── Workflows ──
async function loadWorkflows() {
  const r = await fetch('/api/debug/workflows');
  const data = await r.json();
  const workflows = data.workflows || [];
  document.getElementById('cnt-workflows').textContent = workflows.length;
  const area = document.getElementById('workflows-area');
  if (!workflows.length) {
    area.innerHTML = '<div class="empty">No workflows recorded</div>';
    return;
  }
  area.innerHTML = workflows.map(wf => {
    const nodesHtml = wf.nodes.map(n => {
      const depsStr = n.depends_on.length ? ' \u2190 ' + n.depends_on.join(', ') : '';
      const detail = n.error ? '\u26a0 ' + esc(n.error)
        : (n.result_preview ? esc(n.result_preview) : '');
      return `<tr>
        <td class="mono">${esc(n.node_id)}</td>
        <td>${badge(n.status, n.status)}</td>
        <td class="trunc" title="${esc(n.objective)}">${esc(n.objective.slice(0, 80))}${n.objective.length > 80 ? '\u2026' : ''}</td>
        <td class="mono dim">${esc(depsStr || 'none')}</td>
        <td class="trunc dim" title="${esc(detail)}">${detail.slice(0, 100)}${detail.length > 100 ? '\u2026' : ''}</td>
      </tr>`;
    }).join('');
    return `
      <div class="section-hdr open" onclick="toggleSection(this)">
        <span>
          <span class="mono" style="font-size:.78rem">${esc(wf.workflow_id)}</span>
          ${badge(wf.status, wf.status)}
          <span class="dim">${wf.node_count} node${wf.node_count !== 1 ? 's' : ''}</span>
          <span class="dim">${esc(wf.parent_thread_id || 'no parent')}</span>
        </span>
        <span>\u25bc</span>
      </div>
      <div class="section-body">
        <table class="data-table">
          <thead><tr>
            <th>Node ID</th><th>Status</th><th>Objective</th><th>Depends On</th><th>Result / Error</th>
          </tr></thead>
          <tbody>${nodesHtml}</tbody>
        </table>
      </div>`;
  }).join('');
}

// ── Jobs ──
async function loadJobs() {
  const r = await fetch('/api/debug/jobs');
  const data = await r.json();
  const jobs = data.jobs || [];
  document.getElementById('cnt-jobs').textContent = jobs.length;
  const tbody = document.getElementById('jobs-body');
  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty">No scheduled jobs</td></tr>';
    return;
  }
  tbody.innerHTML = jobs.map(j => {
    const nextRun = j.next_run_time ? fmtTime(j.next_run_time) : '\u2014';
    const args = Object.entries(j.kwargs || {})
      .map(([k, v]) => `${k}=${v}`)
      .join(', ');
    return `<tr>
      <td class="mono">${esc(j.id)}</td>
      <td class="dim">${esc(j.trigger)}</td>
      <td style="white-space:nowrap">${esc(nextRun)}</td>
      <td class="trunc dim" title="${esc(args)}">${esc(args.slice(0, 100))}${args.length > 100 ? '\u2026' : ''}</td>
    </tr>`;
  }).join('');
}

// ── Reminders ──
async function loadReminders() {
  const r = await fetch('/api/debug/reminders');
  const data = await r.json();
  const active = data.active || [];
  const completed = data.completed || [];
  const failed = data.failed || [];
  const cancelled = data.cancelled || [];
  document.getElementById('cnt-reminders').textContent = active.length;

  const area = document.getElementById('reminders-area');
  area.innerHTML =
    renderReminderSection('active-body',    'Active',    active,    true)  +
    renderReminderSection('completed-body', 'Completed', completed, false) +
    renderReminderSection('failed-body',    'Failed',    failed,    false) +
    (cancelled.length ? renderReminderSection('cancelled-body', 'Cancelled', cancelled, false) : '');
}

function renderReminderSection(id, label, reminders, showActions) {
  const cols = showActions ? 8 : 6;
  return `
    <div class="section-hdr open" onclick="toggleSection(this)">
      <span>${label} <small>(${reminders.length})</small></span>
      <span>&#9660;</span>
    </div>
    <div class="section-body">
      <table class="data-table">
        <thead><tr>
          <th>ID</th><th>Schedule</th><th>Message</th>
          <th>Next run</th><th>Thread</th><th>AI prompt</th>
          ${showActions ? '<th colspan="2"></th>' : ''}
        </tr></thead>
        <tbody id="${id}">
          ${reminderRows(reminders, showActions)}
        </tbody>
      </table>
    </div>`;
}

function reminderRows(reminders, showActions) {
  if (!reminders.length) return `<tr><td colspan="${showActions ? 8 : 6}" class="empty">None</td></tr>`;
  return reminders.map(rem => {
    // Store serialised reminder on a data attribute to avoid inline JS quoting issues.
    const dataRem = esc(JSON.stringify(rem));
    return `<tr>
    <td class="mono">${esc(rem.id)}</td>
    <td class="dim">${esc(rem.schedule)}</td>
    <td class="trunc" title="${esc(rem.message)}">${esc(rem.message.slice(0, 60))}${rem.message.length > 60 ? '\u2026' : ''}</td>
    <td class="dim" style="white-space:nowrap">${rem.next_run ? fmtTime(rem.next_run) : '\u2014'}</td>
    <td class="mono dim">${esc(rem.thread_id || 'system')}</td>
    <td class="trunc dim" title="${esc(rem.ai_prompt || '')}">${esc((rem.ai_prompt || '\u2014').slice(0, 40))}</td>
    ${showActions ? `
    <td><button class="btn btn-sm" data-rem="${dataRem}" onclick="editReminder(JSON.parse(this.dataset.rem))">Edit</button></td>
    <td><button class="btn btn-danger btn-sm" data-id="${esc(rem.id)}" onclick="deleteReminder(this.dataset.id)">Delete</button></td>
    ` : ''}
  </tr>`;
  }).join('');
}

function toggleSection(hdr) {
  hdr.classList.toggle('open');
  const body = hdr.nextElementSibling;
  body.style.display = body.style.display === 'none' ? '' : 'none';
}

async function deleteReminder(jobId) {
  if (!confirm('Delete reminder ' + jobId + '?')) return;
  const r = await fetch('/api/debug/reminders/' + encodeURIComponent(jobId), {method: 'DELETE'});
  const data = await r.json();
  if (data.ok) await loadReminders();
  else alert('Error: ' + (data.error || JSON.stringify(data)));
}

// Show/hide fields based on schedule type.
function onScheduleTypeChange(val) {
  document.getElementById('fg-at').style.display         = val === 'interval' ? 'none' : '';
  document.getElementById('fg-dow').style.display        = val === 'weekly'   ? '' : 'none';
  document.getElementById('fg-dom').style.display        = val === 'monthly'  ? '' : 'none';
  document.getElementById('fg-interval').style.display   = val === 'interval' ? '' : 'none';
  document.getElementById('fg-window').style.display     = val === 'interval' ? '' : 'none';
  document.getElementById('fg-window-end').style.display = val === 'interval' ? '' : 'none';
  const lbl = document.getElementById('lbl-at');
  if (lbl) lbl.textContent = val === 'once'
    ? 'At (HH:MM or \u201cin 2 hours\u201d)'
    : 'Time (HH:MM)';
}

// Fill the create form with existing values so the user can edit and re-submit.
// Stores the old job_id; on submit the old reminder is deleted after creating the new one.
let _editReplaceId = null;

function editReminder(rem) {
  const form = document.getElementById('reminder-form');
  const stype = rem.type || 'once';
  form.schedule_type.value    = stype;
  form.message.value          = rem.message   || '';
  form.ai_prompt.value        = rem.ai_prompt || '';
  form.system_reminder.checked = !rem.thread_id || rem.thread_id === 'system';
  onScheduleTypeChange(stype);

  // Populate type-specific fields from the registry schedule string where possible.
  // The schedule field is a human-readable string built by _describe_schedule().
  if (stype === 'interval') {
    // "every 3600s from 08:00 to 20:00"
    const mSec = (rem.schedule || '').match(/every\s+(\d+)s/);
    if (mSec) form.interval_seconds.value = mSec[1];
    const mWin = (rem.schedule || '').match(/from\s+(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})/);
    if (mWin) { form.window_start.value = mWin[1]; form.window_end.value = mWin[2]; }
  } else {
    // "daily at 09:00" / "weekly on mon at 09:00" / "monthly on day 1 at 09:00" / "once at ..."
    const mAt = (rem.schedule || '').match(/at\s+(.+)$/);
    if (mAt) form.at.value = mAt[1].trim();
    if (stype === 'weekly') {
      const mDow = (rem.schedule || '').match(/on\s+(mon|tue|wed|thu|fri|sat|sun)/);
      if (mDow) form.day_of_week.value = mDow[1];
    }
    if (stype === 'monthly') {
      const mDom = (rem.schedule || '').match(/on day\s+(\d+)/);
      if (mDom) form.day_of_month.value = mDom[1];
    }
  }

  _editReplaceId = rem.id;
  form.querySelector('[type=submit]').textContent = 'Save (replaces ' + rem.id + ')';
  form.scrollIntoView({behavior: 'smooth'});
}

async function createReminder(e) {
  e.preventDefault();
  const form = e.target;
  const stype = form.schedule_type.value;
  const payload = {
    schedule_type: stype,
    message:       form.message.value.trim(),
  };
  const at = form.at.value.trim();
  if (at && stype !== 'interval') payload.at = at;
  if (stype === 'weekly')   payload.day_of_week   = form.day_of_week.value;
  if (stype === 'monthly')  payload.day_of_month  = parseInt(form.day_of_month.value, 10) || 1;
  if (stype === 'interval') {
    payload.interval_seconds = parseInt(form.interval_seconds.value, 10);
    const ws = form.window_start.value.trim();
    const we = form.window_end.value.trim();
    if (ws) payload.window_start = ws;
    if (we) payload.window_end   = we;
  }
  const aiPrompt = form.ai_prompt.value.trim();
  if (aiPrompt) payload.ai_prompt = aiPrompt;
  if (form.system_reminder.checked) payload.system = true;

  const r = await fetch('/api/debug/reminders', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const data = await r.json();
  if (data.error) { alert('Error: ' + data.error); return; }

  // If editing an existing reminder, delete the old one now.
  if (_editReplaceId) {
    await fetch('/api/debug/reminders/' + encodeURIComponent(_editReplaceId), {method: 'DELETE'});
    _editReplaceId = null;
    form.querySelector('[type=submit]').textContent = '+ Create';
  }
  form.reset();
  onScheduleTypeChange('once');
  await loadReminders();
}

// ── Init ──
loadTab('sessions');
scheduleRefresh();
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
    Each WebSocket connection gets its own thread_id.
    ``broadcast`` can be called from any task to push a message to clients
    in a specific thread.

    Optional debug dependencies (injected post-construction in main.py):
      _sub_sessions  – SubSessionManager
      _scheduler     – ReminderScheduler
      _matrix        – MatrixThread
      _llm_cfg       – LLMConfig (for context_size / max_tokens)
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
        self._llm_cfg = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def broadcast(self, text: str, thread_id: str = None) -> None:
        """Push a message to all connected clients in a specific thread."""
        if thread_id is None:
            return
        clients = self._threads.get(thread_id, set())
        if not clients:
            return
        payload = json.dumps({"role": "assistant", "text": text, "thread_id": thread_id})
        dead = set()
        for ws in clients:
            try:
                await ws.send_str(payload)
            except Exception:  # noqa: BLE001
                dead.add(ws)
        clients -= dead

    async def run(self) -> None:
        app = web.Application()
        # Chat
        app.router.add_get("/",   self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        # Debug panel
        app.router.add_get("/debug", self._handle_debug)
        # Debug REST API
        app.router.add_get("/api/debug/sessions",                        self._api_sessions)
        app.router.add_get("/api/debug/sessions/{thread_id}/messages",  self._api_session_messages)
        app.router.add_post("/api/debug/sessions/{thread_id}/send",     self._api_session_send)
        app.router.add_post("/api/debug/sessions/{thread_id}/delete",   self._api_session_delete)
        app.router.add_post("/api/debug/sessions/{thread_id}/compact",  self._api_session_compact)
        app.router.add_get("/api/debug/subsessions",                    self._api_subsessions)
        app.router.add_get("/api/debug/workflows",                     self._api_workflows)
        app.router.add_get("/api/debug/jobs",                           self._api_jobs)
        app.router.add_get("/api/debug/system-prompt",                  self._api_system_prompt)
        app.router.add_get("/api/debug/reminders",                      self._api_reminders)
        app.router.add_post("/api/debug/reminders",                     self._api_reminder_create)
        app.router.add_delete("/api/debug/reminders/{job_id}",          self._api_reminder_delete)

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
    # Chat handlers
    # ------------------------------------------------------------------

    async def _handle_index(self, _request: web.Request) -> web.Response:
        return web.Response(text=_CHAT_HTML, content_type="text/html")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        thread_id = f"web_{secrets.token_hex(8)}"
        self._threads.setdefault(thread_id, set()).add(ws)
        logger.info("Web client connected as thread %s (%d threads total)",
                     thread_id, len(self._threads))

        await ws.send_str(json.dumps({
            "role": "system", "text": f"Connected as thread {thread_id}",
            "thread_id": thread_id,
        }))

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

                reply = await self._dispatch(text, ws, thread_id)
                if reply:
                    await ws.send_str(json.dumps({
                        "role": "assistant", "text": reply, "thread_id": thread_id,
                    }))
        finally:
            clients = self._threads.get(thread_id, set())
            clients.discard(ws)
            if not clients:
                self._threads.pop(thread_id, None)
            logger.info("Web client disconnected from thread %s", thread_id)

        return ws

    async def _dispatch(self, text: str, ws: web.WebSocketResponse,
                        thread_id: str) -> Optional[str]:
        async def system(msg: str) -> None:
            await ws.send_str(json.dumps({
                "role": "system", "text": msg, "thread_id": thread_id,
            }))

        if text == "/new":
            await self._llm.reset_session(thread_id)
            await system("Session reset.")
            return None

        if text == "/compact":
            before = self._llm.get_token_budget(thread_id)
            await self._llm.force_compact(thread_id)
            after = self._llm.get_token_budget(thread_id)
            await system(
                f"Context compacted.\n"
                f"Before: {before['total_used']} tokens ({before['msg_count']} msgs, {before['pct']}%)\n"
                f"After: {after['total_used']} tokens ({after['msg_count']} msgs, {after['pct']}%)"
            )
            return None

        if text == "/reminders":
            result = tool_module.execute_tool("list_reminders", {})
            return result

        if text == "/pulse":
            await system("Pulse review triggered.")
            await self._llm.enqueue_system_event(
                "The user manually triggered a pulse review. "
                "Review your PULSE.txt and report what actions, if any, you take.",
                thread_id,
            )
            return None

        if text == "/status":
            await self._handle_status_command(system)
            return None

        if text == "/dream":
            await self._handle_dream_command(system)
            return None

        if text == "/commands":
            await system(
                "**Available commands:**\n"
                "- `/new` – Reset conversation history\n"
                "- `/compact` – Compact context (summarise old messages)\n"
                "- `/reminders` – List active reminders\n"
                "- `/pulse` – Trigger a pulse review\n"
                "- `/status` – Show system status\n"
                "- `/dream` – Trigger a dream cycle\n"
                "- `/commands` – Show this list"
            )
            return None

        return await self._llm.enqueue_user_message(text, thread_id)

    # ------------------------------------------------------------------
    # /status and /dream command helpers
    # ------------------------------------------------------------------

    async def _handle_status_command(self, system) -> None:
        import asyncio as _asyncio
        lines = ["**Wintermute Status**\n"]

        # Asyncio tasks
        tasks = sorted(_asyncio.all_tasks(), key=lambda t: t.get_name())
        running_names = [t.get_name() for t in tasks if not t.done()]
        lines.append(f"**Core tasks:** {', '.join(running_names)}\n")

        # Sub-sessions
        if self._sub_sessions:
            active = self._sub_sessions.list_active()
            if active:
                lines.append(f"**Active sub-sessions ({len(active)}):**")
                for s in active:
                    lines.append(f"- `{s['session_id']}` [{s['status']}] {s['objective'][:80]}")
            else:
                lines.append("**Sub-sessions:** none active")
            workflows = self._sub_sessions.list_workflows()
            running_wfs = [w for w in workflows if w["status"] == "running"]
            if running_wfs:
                lines.append(f"\n**Active workflows ({len(running_wfs)}):**")
                for w in running_wfs:
                    nodes_summary = ", ".join(
                        f"{n['node_id']}[{n['status']}]" for n in w["nodes"]
                    )
                    lines.append(f"- `{w['workflow_id']}`: {nodes_summary}")
        else:
            lines.append("**Sub-sessions:** not available")

        # Pulse loop
        if hasattr(self, "_pulse_loop") and self._pulse_loop:
            state = "running" if self._pulse_loop._running else "stopped"
            lines.append(f"\n**Pulse loop:** {state} (interval: {self._pulse_loop._interval // 60}m)")

        # Dreaming loop
        if hasattr(self, "_dreaming_loop") and self._dreaming_loop:
            state = "running" if self._dreaming_loop._running else "stopped"
            lines.append(f"**Dreaming loop:** {state} (target: {self._dreaming_loop._cfg.hour:02d}:{self._dreaming_loop._cfg.minute:02d} UTC, model: {self._dreaming_loop._model})")

        # Scheduler
        if hasattr(self, "_scheduler") and self._scheduler:
            reminders = self._scheduler.list_reminders()
            lines.append(f"**Reminders:** {len(reminders.get('active', []))} active")

        await system("\n".join(lines))

    async def _handle_dream_command(self, system) -> None:
        from wintermute import dreaming, prompt_assembler

        if not hasattr(self, "_dreaming_loop") or not self._dreaming_loop:
            await system("Dreaming loop not available.")
            return

        dl = self._dreaming_loop
        mem_before = len(prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or "")
        pulse_before = len(prompt_assembler._read(prompt_assembler.PULSE_FILE) or "")

        await system("Starting dream cycle...")
        try:
            await dreaming.run_dream_cycle(client=dl._client, model=dl._model)
        except Exception as exc:
            await system(f"Dream cycle failed: {exc}")
            return

        mem_after = len(prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or "")
        pulse_after = len(prompt_assembler._read(prompt_assembler.PULSE_FILE) or "")

        await system(
            f"Dream cycle complete.\n"
            f"MEMORIES.txt: {mem_before} -> {mem_after} chars\n"
            f"PULSE.txt: {pulse_before} -> {pulse_after} chars"
        )

    # ------------------------------------------------------------------
    # Debug panel handler
    # ------------------------------------------------------------------

    async def _handle_debug(self, _request: web.Request) -> web.Response:
        return web.Response(text=_DEBUG_HTML, content_type="text/html")

    # ------------------------------------------------------------------
    # Debug REST API — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json(data) -> web.Response:
        return web.Response(text=json.dumps(data), content_type="application/json")

    @staticmethod
    def _count_tokens(text: str, model: str = "gpt-4") -> int:
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:  # noqa: BLE001
            return len(text) // 4

    def _token_budget(self, thread_id: str = "default") -> dict:
        """
        Return a dict with accurate token accounting for a thread.

        total_limit   = context_size - max_tokens
        sp_tokens     = actual assembled system prompt tokens (incl. compaction summary)
        tools_tokens  = tokens consumed by the tool schema JSON sent with every request
        hist_tokens   = sum of stored message token counts for thread
        total_used    = sp_tokens + tools_tokens + hist_tokens
        pct           = total_used / total_limit * 100
        """
        if self._llm_cfg is None:
            return {"total_limit": 4096, "sp_tokens": 0, "tools_tokens": 0,
                    "hist_tokens": 0, "total_used": 0, "pct": 0.0}

        from wintermute import database, prompt_assembler

        total_limit = max(self._llm_cfg.context_size - self._llm_cfg.max_tokens, 1)
        model = self._llm_cfg.model

        # System prompt (with per-thread compaction summary if available)
        summary = self._llm.get_compaction_summary(thread_id) if self._llm else None
        try:
            sp_text = prompt_assembler.assemble(extra_summary=summary)
        except Exception:  # noqa: BLE001
            sp_text = ""
        sp_tokens = self._count_tokens(sp_text, model)

        # Tool schemas are sent with every API request and count toward context.
        tools_tokens = self._count_tokens(json.dumps(tool_module.TOOL_SCHEMAS), model)

        stats = database.get_thread_stats(thread_id)
        hist_tokens = stats["token_used"]

        total_used = sp_tokens + tools_tokens + hist_tokens
        pct = round(min(total_used / total_limit * 100, 100), 1)

        return {
            "total_limit": total_limit,
            "sp_tokens": sp_tokens,
            "tools_tokens": tools_tokens,
            "hist_tokens": hist_tokens,
            "total_used": total_used,
            "pct": pct,
            "msg_count": stats["msg_count"],
        }

    # ------------------------------------------------------------------
    # Debug REST API — sessions
    # ------------------------------------------------------------------

    async def _api_sessions(self, _request: web.Request) -> web.Response:
        from wintermute import database

        db_threads = set(database.get_active_thread_ids())
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
        from wintermute import database

        thread_id = request.match_info["thread_id"]
        msgs = database.load_active_messages(thread_id)
        budget = self._token_budget(thread_id)

        return self._json({
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
        })

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
            asyncio.create_task(self._llm.enqueue_user_message(text, thread_id))
            return self._json({"ok": True, "thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)})

    async def _api_session_delete(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        try:
            await self._llm.reset_session(thread_id)
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

    async def _api_system_prompt(self, _request: web.Request) -> web.Response:
        from wintermute import prompt_assembler
        try:
            prompt = prompt_assembler.assemble()
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)})
        model = self._llm_cfg.model if self._llm_cfg else "gpt-4"
        sp_tokens = self._count_tokens(prompt, model)
        tools_tokens = self._count_tokens(json.dumps(tool_module.TOOL_SCHEMAS), model)
        total_limit = (
            max(self._llm_cfg.context_size - self._llm_cfg.max_tokens, 1)
            if self._llm_cfg else 4096
        )
        combined_tokens = sp_tokens + tools_tokens
        return self._json({
            "prompt": prompt,
            "sp_tokens": sp_tokens,
            "tools_tokens": tools_tokens,
            "tokens": combined_tokens,
            "total_limit": total_limit,
            "pct": round(min(combined_tokens / total_limit * 100, 100), 1),
            "tool_schemas": tool_module.TOOL_SCHEMAS,
        })

    # ------------------------------------------------------------------
    # Debug REST API — sub-sessions
    # ------------------------------------------------------------------

    async def _api_subsessions(self, _request: web.Request) -> web.Response:
        if self._sub_sessions is None:
            return self._json({"sessions": []})
        return self._json({"sessions": self._sub_sessions.list_all()})

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
    # Debug REST API — reminders
    # ------------------------------------------------------------------

    async def _api_reminders(self, _request: web.Request) -> web.Response:
        raw = tool_module.execute_tool("list_reminders", {})
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"active": [], "completed": [], "failed": []}
        return self._json(data)

    async def _api_reminder_create(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": f"Invalid JSON: {exc}"})
        raw = tool_module.execute_tool("set_reminder", payload)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"raw": raw}
        return self._json(result)

    async def _api_reminder_delete(self, request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        if self._scheduler is None:
            return self._json({"error": "Scheduler not available"})
        ok = self._scheduler.delete_reminder(job_id)
        return self._json({"ok": ok, "job_id": job_id})
