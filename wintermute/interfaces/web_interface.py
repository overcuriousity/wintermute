"""
Web Interface

Debug panel available at /debug.  Provides a read/write inspection view of all
running sessions, sub-sessions, scheduled jobs, and tasks.
REST API under /api/debug/* is consumed by the embedded SPA.
"""

import asyncio
import json
import logging

from aiohttp import web

from wintermute import tools as tool_module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug panel SPA
# ---------------------------------------------------------------------------

_DEBUG_HTML = r"""\
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
  .stream-connected { color: #6fe06f; }
  .stream-error { color: #f39c12; }
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
  .badge-turing   { background: #410F5F; color: #e0b3ff; }
  .badge-tool     { background: #1a1a2e; color: #8888cc; }
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
  /* Section headers */
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
  <span id="cfg-info" style="font-size:.72rem;color:#888"></span>
  <div class="header-right">
    <span id="stream-status" style="font-size:.72rem;color:#555">\u25cf connecting</span>
  </div>
</header>
<div id="layout">
  <nav id="sidebar">
    <button class="tab-btn active" data-tab="sessions" onclick="showTab('sessions')">
      Sessions <span class="tab-count" id="cnt-sessions">0</span>
    </button>
    <button class="tab-btn" data-tab="workers" onclick="showTab('workers')">
      Workers <span class="tab-count" id="cnt-workers">0</span>
    </button>
    <button class="tab-btn" data-tab="jobs" onclick="showTab('jobs')">
      Jobs <span class="tab-count" id="cnt-jobs">0</span>
    </button>
    <button class="tab-btn" data-tab="tasks" onclick="showTab('tasks')">
      Tasks <span class="tab-count" id="cnt-tasks">0</span>
    </button>
    <button class="tab-btn" data-tab="memory" onclick="showTab('memory')">
      Memory <span class="tab-count" id="cnt-memory">0</span>
    </button>
    <button class="tab-btn" data-tab="interactions" onclick="showTab('interactions')">
      Interactions <span class="tab-count" id="cnt-interactions">0</span>
    </button>
    <button class="tab-btn" data-tab="outcomes" onclick="showTab('outcomes')">
      Outcomes <span class="tab-count" id="cnt-outcomes">0</span>
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

    <!-- ── Workers ── -->
    <div class="tab-panel" id="panel-workers">
      <div class="scroll-area" id="workers-area">
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

    <!-- ── Tasks ── -->
    <div class="tab-panel" id="panel-tasks">
      <div class="scroll-area">
        <table class="data-table">
          <thead>
            <tr>
              <th>ID</th><th>Priority</th><th>Status</th><th>Content</th><th>Schedule</th><th>Thread</th><th>Created</th><th>Last Run</th><th>Runs</th>
            </tr>
          </thead>
          <tbody id="tasks-body">
            <tr><td colspan="9" class="empty">Loading\u2026</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ── Interaction Log ── -->
    <!-- ── Memory ── -->
    <div class="tab-panel" id="panel-memory">
      <div class="scroll-area" style="padding:1rem">
        <div id="memory-stats" style="margin-bottom:1rem">
          <div class="empty">Loading memory stats\u2026</div>
        </div>
        <div class="form-bar" style="gap:.6rem;margin-bottom:1rem">
          <div class="form-group" style="flex:1">
            <label>Test search query</label>
            <input type="text" id="memory-search-q" placeholder="Enter query\u2026" style="width:100%">
          </div>
          <div class="form-group">
            <label>top_k</label>
            <input type="number" id="memory-search-k" value="5" style="width:60px" min="1" max="50">
          </div>
          <div class="form-group" style="align-self:flex-end">
            <button class="action-btn" onclick="doMemorySearch()">Search</button>
          </div>
        </div>
        <div id="memory-results"></div>
      </div>
    </div>

    <!-- ── Outcomes ── -->
    <div class="tab-panel" id="panel-outcomes">
      <div class="form-bar" style="gap:.8rem">
        <div class="form-group">
          <label>Filter by status</label>
          <select id="oc-filter-status" onchange="loadOutcomes()" style="min-width:160px">
            <option value="">All statuses</option>
            <option value="completed">completed</option>
            <option value="timeout">timeout</option>
            <option value="failed">failed</option>
          </select>
        </div>
      </div>
      <div id="outcomes-stats" style="padding:.5rem 1rem;font-size:.8rem;color:#888;display:flex;gap:2rem;border-bottom:1px solid #0f3460"></div>
      <div class="scroll-area">
        <table class="data-table">
          <thead>
            <tr>
              <th>Time</th><th>Status</th><th>Session</th><th>Mode</th>
              <th>Duration</th><th>Tools used</th><th>TP verdict</th><th>Objective</th>
            </tr>
          </thead>
          <tbody id="outcomes-body">
            <tr><td colspan="8" class="empty">Loading\u2026</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="tab-panel" id="panel-interactions">
      <div class="form-bar" style="gap:.8rem">
        <div class="form-group">
          <label>Filter by session</label>
          <select id="il-filter-session" onchange="loadInteractionLog()" style="min-width:180px">
            <option value="">All sessions</option>
          </select>
        </div>
        <div class="form-group">
          <label>Filter by action</label>
          <select id="il-filter-action" onchange="loadInteractionLog()" style="min-width:160px">
            <option value="">All actions</option>
            <option value="chat">chat</option>
            <option value="inference_round">inference_round</option>
            <option value="tool_call">tool_call</option>
            <option value="nl_translation">nl_translation</option>
            <option value="sub_session">sub_session</option>
            <option value="turing_detection">turing_detection</option>
            <option value="turing_validation">turing_validation</option>
            <option value="turing_correction">turing_correction</option>
            <option value="turing_objective">turing_objective</option>
            <option value="turing_response">turing_response</option>
            <option value="compaction">compaction</option>
            <option value="dreaming">dreaming</option>
            <option value="embedding">embedding</option>
            <option value="local_vector_add">local_vector_add</option>
            <option value="local_vector_search">local_vector_search</option>
            <option value="local_vector_replace_all">local_vector_replace_all</option>
            <option value="qdrant_add">qdrant_add</option>
            <option value="qdrant_search">qdrant_search</option>
            <option value="qdrant_replace_all">qdrant_replace_all</option>
            <option value="system_event">system_event</option>
          </select>
        </div>
      </div>
      <div class="scroll-area">
        <table class="data-table">
          <thead>
            <tr>
              <th>Time</th><th>Action</th><th>Session</th><th>LLM</th>
              <th>Input</th><th>Output</th><th>Status</th>
            </tr>
          </thead>
          <tbody id="interactions-body">
            <tr><td colspan="7" class="empty">Loading\u2026</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
// ── State ──
let currentTab = 'sessions';
let selectedSession = null;
const _closedWorkerGroups = new Set();
let ilMaxId = 0;
let ilMinId = null;
let ilAllLoaded = false;
let ilLoading = false;
let ilObserver;

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
      case 'workers':     await loadWorkers(); break;
      case 'jobs':        await loadJobs(); break;
      case 'tasks':      await loadTasks(); break;
      case 'memory':       await loadMemory(); break;
      case 'interactions': await loadInteractionLog(); break;
      case 'outcomes':     await loadOutcomes(); break;
    }
  } catch (e) {
    console.error('loadTab error:', e);
  }
}

// ── SSE stream ──
function connectStream() {
  const es = new EventSource('/api/debug/stream');
  const statusEl = document.getElementById('stream-status');
  es.onopen = () => {
    statusEl.textContent = '\u25cf live';
    statusEl.className = 'stream-connected';
  };
  es.onerror = () => {
    statusEl.textContent = '\u25cf reconnecting';
    statusEl.className = 'stream-error';
  };
  es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    renderSessions(data.sessions || []);
    renderWorkers(data.subsessions || [], data.workflows || []);
    renderJobs(data.jobs || []);
    renderTasks(data.tasks || []);
    updateInteractionsCount(data.interactions_total, data.interactions_max_id);
    updateMemoryCount(data.memory_count);
  };
}

// ── Sessions ──
function renderSessions(sessions) {
  document.getElementById('cnt-sessions').textContent = sessions.length;
  renderSessionList(sessions);
  if (selectedSession) loadSessionMessages(selectedSession);
}

async function loadSessions() {
  const r = await fetch('/api/debug/sessions');
  const d = await r.json();
  renderSessions(d.sessions || []);
}

function renderSessionList(sessions) {
  const el = document.getElementById('session-list');
  const scrollTop = el.scrollTop;
  if (!sessions.length) {
    el.innerHTML = '<div class="empty">No active sessions</div>';
    el.scrollTop = scrollTop;
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
  el.scrollTop = scrollTop;
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
    return `<div class="msg-wrap ${esc(m.role)}">
      <div class="msg-bubble ${esc(m.role)}">${esc(m.content || '')}</div>
      <div class="msg-meta">${ts}${tokInfo}</div>
    </div>`;
  }).join('') : '<div class="empty">No messages in this thread</div>';

  if (scrolledToBottom) view.scrollTop = view.scrollHeight;
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

// ── Workers (unified sub-sessions + workflows) ──
let _expandedWorkers = new Set();

function _workerSessionRow(s, indent) {
  const duration = fmtDuration(s.created_at, s.completed_at);
  const resultText = s.result
    ? s.result.slice(0, 120) + (s.result.length > 120 ? '\u2026' : '')
    : (s.error ? '\u26a0 ' + s.error.slice(0, 80) : '');
  const deps = (s.depends_on || []);
  const depsStr = deps.length ? deps.map(d => d.slice(0, 12)).join(', ') : '\u2014';
  const tcCount = s.tool_call_count || 0;
  const isExpanded = _expandedWorkers.has(s.session_id);
  const arrow = isExpanded ? '\u25bc' : '\u25b6';
  const activityIndicator = s.status === 'running' && tcCount > 0
    ? ' <span style="color:#6fe06f" title="Active tool calls">\u25cf</span>' : '';
  const indentStyle = indent ? 'padding-left:1.6rem' : '';
  return `<tr style="cursor:pointer" onclick="toggleWorker('${esc(s.session_id)}')">
    <td style="width:1.5em;text-align:center;color:#666;${indentStyle}">${arrow}</td>
    <td class="mono">${esc(s.session_id)}${activityIndicator}</td>
    <td class="mono dim" title="${esc(deps.join(', '))}">${depsStr}</td>
    <td>${badge(s.status, s.status)}</td>
    <td class="dim">${tcCount}</td>
    <td class="trunc" title="${esc(s.objective)}">${esc(s.objective.slice(0, 60))}${s.objective.length > 60 ? '\u2026' : ''}</td>
    <td class="dim">${esc(s.system_prompt_mode)}</td>
    <td class="dim" style="white-space:nowrap">${fmtTime(s.created_at)}</td>
    <td class="dim" style="white-space:nowrap">${duration}</td>
    <td class="trunc" title="${esc(resultText)}">${esc(resultText)}</td>
  </tr>
  <tr id="wkdetail-${esc(s.session_id)}" style="display:${isExpanded ? '' : 'none'}">
    <td colspan="10" style="background:#0d1526;padding:.6rem 1rem">
      <div id="wkcontent-${esc(s.session_id)}" style="font-size:.8rem;color:#999">Loading\u2026</div>
    </td>
  </tr>`;
}

function renderWorkers(sessions, workflows) {
  document.getElementById('cnt-workers').textContent = sessions.length;
  const area = document.getElementById('workers-area');
  const scrollTop = area.scrollTop;

  // Save expanded detail content + inner scroll so SSE updates don't wipe them
  const savedDetails = {};
  for (const sid of _expandedWorkers) {
    const contentEl = document.getElementById('wkcontent-' + sid);
    if (contentEl) {
      const innerDiv = contentEl.querySelector('div[style*="overflow-y"]');
      savedDetails[sid] = {
        html: contentEl.innerHTML,
        innerScroll: innerDiv ? innerDiv.scrollTop : 0,
      };
    }
  }

  if (!sessions.length) {
    area.innerHTML = '<div class="empty">No workers recorded</div>';
    area.scrollTop = scrollTop;
    return;
  }

  // Build a map of workflow_id -> workflow metadata, and session_id -> session
  const wfMap = {};
  for (const wf of (workflows || [])) wfMap[wf.workflow_id] = wf;
  const sessionMap = {};
  for (const s of sessions) sessionMap[s.session_id] = s;

  // Group sessions: workflow groups first (ordered by workflow created_at desc),
  // then standalone sessions (no workflow_id), also newest-first
  const wfSessions = {};   // workflow_id -> [session, ...]
  const standalone = [];
  for (const s of sessions) {
    if (s.workflow_id) {
      if (!wfSessions[s.workflow_id]) wfSessions[s.workflow_id] = [];
      wfSessions[s.workflow_id].push(s);
    } else {
      standalone.push(s);
    }
  }

  // Collect workflow IDs present in sessions (some may not be in wfMap yet)
  const wfIds = Object.keys(wfSessions);
  // Sort by earliest created_at of member sessions, newest first
  wfIds.sort((a, b) => {
    const ta = Math.max(...wfSessions[a].map(s => new Date(s.created_at)));
    const tb = Math.max(...wfSessions[b].map(s => new Date(s.created_at)));
    return tb - ta;
  });

  let html = '';

  // Table header helper
  const tableHeader = `<table class="data-table" style="width:100%">
    <thead><tr>
      <th></th><th>ID</th><th>Deps</th><th>Status</th>
      <th>Tools</th><th>Objective</th><th>Mode</th><th>Created</th><th>Duration</th><th>Result / Error</th>
    </tr></thead><tbody>`;
  const tableFooter = `</tbody></table>`;

  // Render workflow groups
  for (const wfId of wfIds) {
    const wf = wfMap[wfId];
    const members = wfSessions[wfId];
    const runningCount = members.filter(s => s.status === 'running').length;
    const activityNote = runningCount > 0
      ? ' <span style="color:#6fe06f">\u25cf ' + runningCount + ' active</span>' : '';
    const wfStatus = wf ? wf.status : (runningCount > 0 ? 'running' : 'done');
    const isClosed = _closedWorkerGroups.has(wfId);
    html += `
      <div class="section-hdr${isClosed ? '' : ' open'}"
           data-section-id="${esc(wfId)}" data-section-type="workergroup"
           onclick="toggleSection(this)">
        <span>
          <span class="mono" style="font-size:.78rem">${esc(wfId)}</span>
          ${badge(wfStatus, wfStatus)}${activityNote}
          <span class="dim">${members.length} worker${members.length !== 1 ? 's' : ''}</span>
          ${wf ? '<span class="dim">' + esc(wf.parent_thread_id || '') + '</span>' : ''}
        </span>
        <span>\u25bc</span>
      </div>
      <div class="section-body" style="${isClosed ? 'display:none' : ''}">
        ${tableHeader}
        ${members.map(s => _workerSessionRow(s, true)).join('')}
        ${tableFooter}
      </div>`;
  }

  // Render standalone sessions
  if (standalone.length) {
    html += tableHeader;
    html += standalone.map(s => _workerSessionRow(s, false)).join('');
    html += tableFooter;
  }

  area.innerHTML = html;

  // Restore expanded detail content + inner scroll; only fetch if not yet loaded
  for (const sid of _expandedWorkers) {
    const contentEl = document.getElementById('wkcontent-' + sid);
    if (!contentEl) continue;
    if (savedDetails[sid]) {
      contentEl.innerHTML = savedDetails[sid].html;
      const innerDiv = contentEl.querySelector('div[style*="overflow-y"]');
      if (innerDiv) innerDiv.scrollTop = savedDetails[sid].innerScroll;
    } else {
      fetchWorkerDetail(sid);
    }
  }

  area.scrollTop = scrollTop;
}

async function loadWorkers() {
  const [r1, r2] = await Promise.all([
    fetch('/api/debug/subsessions'),
    fetch('/api/debug/workflows'),
  ]);
  const [d1, d2] = await Promise.all([r1.json(), r2.json()]);
  renderWorkers(d1.sessions || [], d2.workflows || []);
}

async function toggleWorker(sid) {
  const detailRow = document.getElementById('wkdetail-' + sid);
  if (!detailRow) return;
  if (_expandedWorkers.has(sid)) {
    _expandedWorkers.delete(sid);
    detailRow.style.display = 'none';
  } else {
    _expandedWorkers.add(sid);
    detailRow.style.display = '';
    fetchWorkerDetail(sid);
  }
  // Update arrow in the trigger row
  const triggerRow = detailRow.previousElementSibling;
  if (triggerRow) {
    const arrowCell = triggerRow.querySelector('td');
    if (arrowCell) arrowCell.textContent = _expandedWorkers.has(sid) ? '\u25bc' : '\u25b6';
  }
}

async function fetchWorkerDetail(sid) {
  const el = document.getElementById('wkcontent-' + sid);
  if (!el) return;
  try {
    const r = await fetch('/api/debug/subsessions/' + encodeURIComponent(sid) + '/messages');
    const data = await r.json();
    const msgs = data.messages || [];
    if (!msgs.length) {
      el.innerHTML = '<span class="dim">No messages recorded (session may still be initializing)</span>';
      return;
    }
    let html = '<div style="margin-bottom:.5rem"><strong style="color:#a8d8ea">Messages (' + msgs.length + ')</strong></div>';
    html += '<div style="max-height:400px;overflow-y:auto;border:1px solid #0f3460;border-radius:.3rem;padding:.4rem">';
    msgs.forEach(m => {
      const roleColor = m.role === 'assistant' ? '#a8d8ea' : m.role === 'user' ? '#90ee90' : m.role === 'tool' ? '#c9b0ff' : '#888';
      const content = (m.content || '').slice(0, 500) + ((m.content || '').length > 500 ? '\u2026' : '');
      html += '<div style="margin-bottom:.4rem;padding:.3rem .5rem;border-left:2px solid ' + roleColor + '">';
      html += '<span style="color:' + roleColor + ';font-weight:600;font-size:.72rem;text-transform:uppercase">' + esc(m.role) + '</span>';
      if (m.tool_call_id) html += ' <span class="mono dim" style="font-size:.68rem">' + esc(m.tool_call_id) + '</span>';
      if (m.tool_calls) {
        m.tool_calls.forEach(tc => {
          html += ' <span class="badge badge-system" style="font-size:.62rem">' + esc(tc.name) + '</span>';
        });
      }
      html += '<div style="white-space:pre-wrap;word-break:break-word;margin-top:.2rem;font-size:.78rem;color:#ccc">' + esc(content) + '</div>';
      html += '</div>';
    });
    html += '</div>';
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = '<span style="color:#ff9090">Failed to load: ' + esc(String(e)) + '</span>';
  }
}

// ── Jobs ──
function renderJobs(jobs) {
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

async function loadJobs() {
  const r = await fetch('/api/debug/jobs');
  const d = await r.json();
  renderJobs(d.jobs || []);
}

// ── Tasks ──
function renderTasks(items) {
  document.getElementById('cnt-tasks').textContent = items.filter(i => i.status === 'active').length;
  const tbody = document.getElementById('tasks-body');
  if (!items.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty">No tasks</td></tr>';
    return;
  }
  const priorityLabel = p => p <= 1 ? 'P1 \u2605\u2605\u2605' : p <= 2 ? 'P2 \u2605\u2605' : p <= 3 ? 'P3 \u2605' : 'P' + p;
  tbody.innerHTML = items.map(it => {
    const statusColor = it.status === 'active' ? '#4caf8a' : it.status === 'paused' ? '#f0ad4e' : '#888';
    return `<tr>
      <td class="mono">${esc(String(it.id))}</td>
      <td class="dim">${esc(priorityLabel(it.priority))}</td>
      <td style="color:${statusColor}">${esc(it.status)}</td>
      <td class="trunc" title="${esc(it.content)}">${esc(it.content.slice(0, 100))}${it.content.length > 100 ? '\u2026' : ''}</td>
      <td class="dim">${esc(it.schedule_desc || '\u2014')}</td>
      <td class="mono dim">${esc(it.thread_id || '\u2014')}</td>
      <td class="dim" style="white-space:nowrap">${it.created ? new Date(it.created * 1000).toLocaleString() : '\u2014'}</td>
      <td class="dim" style="white-space:nowrap">${it.last_run_at ? new Date(it.last_run_at * 1000).toLocaleString() : '\u2014'}</td>
      <td class="dim">${it.run_count || 0}</td>
    </tr>`;
  }).join('');
}

async function loadTasks() {
  const r = await fetch('/api/debug/tasks');
  const d = await r.json();
  renderTasks(d.items || []);
}

function toggleSection(hdr) {
  hdr.classList.toggle('open');
  const body = hdr.nextElementSibling;
  body.style.display = hdr.classList.contains('open') ? '' : 'none';
  const id   = hdr.dataset.sectionId;
  const type = hdr.dataset.sectionType;
  if (!id) return;
  const set = _closedWorkerGroups;
  hdr.classList.contains('open') ? set.delete(id) : set.add(id);
}
// ── Interaction Log ──
function _makeILRow(e) {
  const ts = new Date(e.timestamp * 1000).toLocaleString();
  const inputPreview = (e.input || '').length > 120 ? e.input.slice(0, 117) + '\u2026' : (e.input || '');
  const outputPreview = (e.output || '').length > 120 ? e.output.slice(0, 117) + '\u2026' : (e.output || '');
  const statusCls = e.status === 'ok' ? 'completed' : (e.status === 'error' ? 'failed' : 'pending');
  const sessionShort = (e.session || '').length > 24 ? e.session.slice(0, 22) + '\u2026' : (e.session || '');
  const actionStyle = e.action.startsWith('turing') ? 'turing' : (e.action === 'tool_call' ? 'tool' : (e.action === 'chat' ? 'web' : (e.action === 'inference_round' ? 'pending' : (e.action === 'error' ? 'failed' : 'system'))));
  const _rowBg = e.action === 'tool_call' ? ';background:#0a0f1a' : (e.action === 'inference_round' ? ';background:#0a1020' : '');
  return (
    '<tr style="cursor:pointer' + _rowBg + '" onclick="toggleILDetail(this, ' + e.id + ')">' +
    '<td class="dim" style="white-space:nowrap">' + esc(ts) + '</td>' +
    '<td>' + badge(actionStyle, e.action) + '</td>' +
    '<td class="mono" style="font-size:.72rem" title="' + esc(e.session || '') + '">' + esc(sessionShort) + '</td>' +
    '<td class="mono" style="font-size:.72rem">' + esc(e.llm || '') + '</td>' +
    '<td class="trunc" title="' + esc(e.input || '') + '">' + esc(inputPreview) + '</td>' +
    '<td class="trunc" title="' + esc(e.output || '') + '">' + esc(outputPreview) + '</td>' +
    '<td>' + badge(statusCls, e.status) + '</td>' +
    '</tr>'
  );
}

function _fetchIL({limit = 200, before_id, after_id} = {}) {
  const session = document.getElementById('il-filter-session').value;
  const action  = document.getElementById('il-filter-action').value;
  const params = new URLSearchParams({limit});
  if (session) params.set('session', session);
  if (action)  params.set('action', action);
  if (before_id != null) params.set('before_id', before_id);
  if (after_id != null)  params.set('after_id', after_id);
  return fetch('/api/debug/interaction-log?' + params)
    .then(r => r.json()).then(d => d.entries || []);
}

async function _refreshILSessions() {
  const filterSel = document.getElementById('il-filter-session');
  const currentFilter = filterSel.value;
  const r = await fetch('/api/debug/interaction-log?limit=1000&offset=0');
  const d = await r.json();
  const sessions = [...new Set((d.entries || []).map(e => e.session).filter(Boolean))];
  const existing = new Set([...filterSel.options].map(o => o.value));
  sessions.forEach(s => {
    if (!existing.has(s)) {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      filterSel.appendChild(opt);
    }
  });
  filterSel.value = currentFilter;
}

function renderILEntries(entries, append, prepend = false) {
  if (!entries.length) return;
  const ids = entries.map(e => e.id);
  if (!prepend) ilMinId = Math.min(ilMinId ?? Infinity, ...ids);
  ilMaxId = Math.max(ilMaxId, ...ids);

  const tbody = document.getElementById('interactions-body');
  const sentinel = document.getElementById('il-sentinel');
  if (sentinel) sentinel.remove();

  // after_id returns ASC order — reverse so newest renders at top when prepending
  const ordered = prepend ? [...entries].reverse() : entries;
  const html = ordered.map(e => _makeILRow(e)).join('');

  if (prepend)       tbody.insertAdjacentHTML('afterbegin', html);
  else if (append)   tbody.insertAdjacentHTML('beforeend', html);
  else               tbody.innerHTML = html;

  const sentinelRow = document.createElement('tr');
  sentinelRow.id = 'il-sentinel';
  sentinelRow.innerHTML = '<td colspan="7"></td>';
  tbody.appendChild(sentinelRow);
  if (ilObserver) ilObserver.observe(sentinelRow);
}

async function loadInteractionLog() {
  ilMaxId = 0; ilMinId = null; ilAllLoaded = false;
  const tbody = document.getElementById('interactions-body');
  tbody.innerHTML = '<tr><td colspan="7" class="empty">Loading\u2026</td></tr>';
  await _refreshILSessions();
  const entries = await _fetchIL({limit: 200});
  if (!entries.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No interactions recorded yet</td></tr>';
    return;
  }
  renderILEntries(entries, false);
}

async function loadOlderInteractions() {
  if (ilLoading || ilAllLoaded || ilMinId === null) return;
  ilLoading = true;
  try {
    const entries = await _fetchIL({limit: 200, before_id: ilMinId});
    if (!entries.length) { ilAllLoaded = true; }
    else renderILEntries(entries, true);
  } finally {
    ilLoading = false;
  }
}

async function loadNewInteractions() {
  if (!ilMaxId) return;
  const entries = await _fetchIL({limit: 100, after_id: ilMaxId});
  if (entries.length) renderILEntries(entries, false, true);
}

function updateInteractionsCount(total, maxId) {
  if (total != null) document.getElementById('cnt-interactions').textContent = total;
  if (maxId > ilMaxId && ilMaxId > 0) loadNewInteractions();
}

function setupILObserver() {
  ilObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && !ilLoading && !ilAllLoaded) {
      loadOlderInteractions();
    }
  }, {rootMargin: '200px'});
}

async function toggleILDetail(row, id) {
  const next = row.nextElementSibling;
  if (next && next.classList.contains('il-detail')) {
    next.remove();
    return;
  }
  const r = await fetch('/api/debug/interaction-log/' + id);
  const e = await r.json();
  const detail = document.createElement('tr');
  detail.className = 'il-detail';
  let rawSection = '';
  if (e.raw_output) {
    try {
      const raw = JSON.parse(e.raw_output);
      const parts = [];
      if (raw.tool_calls && raw.tool_calls.length) {
        parts.push('<div style="margin:.6rem 0"><strong style="color:#f0c040">Tool Calls (' + raw.tool_calls.length + '):</strong></div>');
        raw.tool_calls.forEach((tc, i) => {
          parts.push('<div style="margin-left:1rem;margin-bottom:.4rem"><span style="color:#a8d8ea">' + esc(tc.name) + '</span>');
          parts.push('<pre style="white-space:pre-wrap;word-break:break-word;color:#888;font-size:.75rem;margin:.2rem 0">' + esc(tc.arguments || '') + '</pre>');
          parts.push('<pre style="white-space:pre-wrap;word-break:break-word;color:#6a6;font-size:.75rem;margin:.2rem 0">' + esc((tc.result || '').slice(0, 300)) + '</pre></div>');
        });
      }
      if (raw.reasoning) {
        parts.push('<div style="margin:.6rem 0"><strong style="color:#c0a0f0">Reasoning:</strong></div>');
        parts.push('<pre style="white-space:pre-wrap;word-break:break-word;color:#999;font-size:.75rem">' + esc(raw.reasoning) + '</pre>');
      }
      rawSection = parts.join('');
    } catch(ex) {
      rawSection = '<div style="margin:.6rem 0"><strong style="color:#a8d8ea">Raw Output:</strong></div><pre style="white-space:pre-wrap;word-break:break-word;color:#888;font-size:.75rem">' + esc(e.raw_output) + '</pre>';
    }
  }
  detail.innerHTML = '<td colspan="7" style="background:#0d1526;padding:1rem">' +
    '<div style="margin-bottom:.6rem"><strong style="color:#a8d8ea">Input:</strong></div>' +
    '<pre style="white-space:pre-wrap;word-break:break-word;color:#ccc;max-height:none;overflow:visible;font-size:.8rem">' + esc(e.input || '') + '</pre>' +
    '<div style="margin:.6rem 0"><strong style="color:#a8d8ea">Output:</strong></div>' +
    '<pre style="white-space:pre-wrap;word-break:break-word;color:#ccc;max-height:none;overflow:visible;font-size:.8rem">' + esc(e.output || '') + '</pre>' +
    rawSection +
    '</td>';
  row.after(detail);
}

// ── Memory ──
async function loadMemory() {
  try {
    const r = await fetch('/api/debug/memory');
    const d = await r.json();
    document.getElementById('cnt-memory').textContent = d.count || 0;
    const el = document.getElementById('memory-stats');
    const rows = [
      '<table class="data-table" style="max-width:500px">',
      '<tr><td class="dim">Backend</td><td>' + esc(d.backend || 'unknown') + '</td></tr>',
      '<tr><td class="dim">Entries</td><td>' + (d.count || 0) + '</td></tr>',
      '<tr><td class="dim">Vector enabled</td><td>' + (d.vector_enabled ? 'yes' : 'no') + '</td></tr>',
      '<tr><td class="dim">top_k</td><td>' + (d.top_k || '-') + '</td></tr>',
      '<tr><td class="dim">threshold</td><td>' + (d.threshold || '-') + '</td></tr>',
    ];
    if (d.stats) {
      for (const [k, v] of Object.entries(d.stats)) {
        if (k !== 'backend' && k !== 'count')
          rows.push('<tr><td class="dim">' + esc(k) + '</td><td>' + esc(String(v)) + '</td></tr>');
      }
    }
    rows.push('</table>');
    el.innerHTML = rows.join('');
  } catch(e) { console.error('loadMemory:', e); }
}

async function doMemorySearch() {
  const q = document.getElementById('memory-search-q').value.trim();
  if (!q) return;
  const k = document.getElementById('memory-search-k').value || 5;
  const el = document.getElementById('memory-results');
  el.innerHTML = '<div class="empty">Searching\u2026</div>';
  try {
    const r = await fetch('/api/debug/memory?q=' + encodeURIComponent(q) + '&k=' + k);
    const d = await r.json();
    const results = d.results || [];
    if (!results.length) {
      el.innerHTML = '<div class="empty">No results</div>';
      return;
    }
    const html = '<table class="data-table"><thead><tr><th>#</th><th>Score</th><th>Text</th></tr></thead><tbody>' +
      results.map((r, i) =>
        '<tr><td>' + (i+1) + '</td><td style="color:#6fe06f">' + (r.score != null ? r.score.toFixed(4) : '-') + '</td>' +
        '<td style="white-space:pre-wrap;word-break:break-word">' + esc(r.text || '') + '</td></tr>'
      ).join('') + '</tbody></table>';
    el.innerHTML = html;
  } catch(e) {
    el.innerHTML = '<div class="empty">Error: ' + esc(e.message) + '</div>';
  }
}

function updateMemoryCount(count) {
  if (count != null) document.getElementById('cnt-memory').textContent = count;
}

// ── Outcomes ──
async function loadOutcomes() {
  const status = document.getElementById('oc-filter-status').value;
  const params = new URLSearchParams({limit: 200, offset: 0});
  if (status) params.set('status', status);
  try {
    const r = await fetch('/api/debug/outcomes?' + params);
    const d = await r.json();
    const entries = d.entries || [];
    document.getElementById('cnt-outcomes').textContent = d.total || 0;
    const st = d.stats || {};
    const byStatus = st.by_status || {};
    const statsEl = document.getElementById('outcomes-stats');
    statsEl.innerHTML = [
      '<span>Total: <strong>' + (st.total || 0) + '</strong></span>',
      Object.entries(byStatus).map(([k,v]) => '<span>' + esc(k) + ': <strong>' + v + '</strong></span>').join(''),
      st.avg_duration_seconds != null ? '<span>Avg duration: <strong>' + st.avg_duration_seconds + 's</strong></span>' : '',
      st.avg_tool_calls != null ? '<span>Avg tool calls: <strong>' + st.avg_tool_calls + '</strong></span>' : '',
      st.timeout_rate_pct != null ? '<span>Timeout rate: <strong>' + st.timeout_rate_pct + '%</strong></span>' : '',
    ].flat().filter(Boolean).join(' \u2502 ');
    const tbody = document.getElementById('outcomes-body');
    if (!entries.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty">No outcomes recorded yet</td></tr>';
      return;
    }
    tbody.innerHTML = entries.map(o => {
      const ts = o.timestamp ? new Date(o.timestamp * 1000).toLocaleString() : '-';
      const dur = o.duration_seconds != null ? o.duration_seconds.toFixed(1) + 's' : '-';
      const tc = o.tool_call_count != null ? o.tool_call_count : '-';
      let toolsUsed = '-';
      try { toolsUsed = o.tools_used ? JSON.parse(o.tools_used).join(', ') || '-' : '-'; } catch(e) {}
      const verdict = o.turing_verdict || '-';
      const verdictColor = verdict === 'pass' ? '#6fe06f' : verdict === 'fail' ? '#e06f6f' : '#888';
      const statusColor = o.status === 'completed' ? '#6fe06f' : o.status === 'timeout' ? '#f0c040' : '#e06f6f';
      const obj = (o.objective || '').slice(0, 80);
      return '<tr>' +
        '<td style="white-space:nowrap;color:#888">' + esc(ts) + '</td>' +
        '<td style="color:' + statusColor + '">' + esc(o.status || '') + '</td>' +
        '<td style="font-family:monospace;font-size:.75rem;color:#888">' + esc((o.session_id || '').slice(0,12)) + '</td>' +
        '<td>' + esc(o.system_prompt_mode || '') + '</td>' +
        '<td style="color:#a8d8ea">' + esc(dur) + '</td>' +
        '<td style="font-size:.75rem;color:#999">' + esc(tc) + (toolsUsed !== '-' ? ' (' + esc(toolsUsed) + ')' : '') + '</td>' +
        '<td style="color:' + verdictColor + '">' + esc(verdict) + '</td>' +
        '<td style="white-space:pre-wrap;word-break:break-word;font-size:.8rem">' + esc(obj) + (o.objective && o.objective.length > 80 ? '\u2026' : '') + '</td>' +
        '</tr>';
    }).join('');
  } catch(e) { console.error('loadOutcomes:', e); }
}

// ── Config info ──
async function loadConfig() {
  try {
    const r = await fetch('/api/debug/config');
    const d = await r.json();
    const el = document.getElementById('cfg-info');
    const parts = [];
    // Each role is a list of backends (ordered failover); show the primary (first)
    const first = arr => Array.isArray(arr) ? arr[0] : arr;
    const main  = first(d.main);
    const comp  = first(d.compaction);
    const sub   = first(d.sub_sessions);
    const dream = first(d.dreaming);
    if (main) parts.push('<span style="color:#a8d8ea">' + esc(main.model) + '</span> <span style="color:#555">' + fmtTokens(main.context_size) + ' ctx</span>');
    if (comp && comp.model !== (main && main.model))
      parts.push('<span style="color:#666">compact: ' + esc(comp.model) + '</span>');
    if (sub && (sub.model !== (main && main.model) || sub.base_url !== (main && main.base_url)))
      parts.push('<span style="color:#666">sub: ' + esc(sub.model) + '</span>');
    if (dream && dream.model !== (main && main.model))
      parts.push('<span style="color:#666">dream: ' + esc(dream.model) + '</span>');
    el.innerHTML = parts.join(' <span style="color:#333">\u2502</span> ');
  } catch(e) { console.error('loadConfig:', e); }
}

// ── Init ──
loadConfig();
loadTab('sessions');
connectStream();
setupILObserver();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# WebInterface class
# ---------------------------------------------------------------------------


class WebInterface:
    """
    Runs an aiohttp web server serving the debug panel at /debug.

    Optional debug dependencies (injected post-construction in main.py):
      _sub_sessions  – SubSessionManager
      _scheduler     – RoutineScheduler
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
        return web.Response(text=_DEBUG_HTML, content_type="text/html")

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
            asyncio.create_task(self._llm.enqueue_user_message(text, thread_id))
            return self._json({"ok": True, "thread_id": thread_id})
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)})

    async def _api_session_delete(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        try:
            await self._llm.reset_session(thread_id)
            from wintermute.infra import prompt_loader
            try:
                seed_prompt = prompt_loader.load_seed(self._llm._seed_language)
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

        result: dict = {
            "backend": memory_store.stats().get("backend", "unknown"),
            "count": memory_store.count(),
            "vector_enabled": memory_store.is_vector_enabled(),
            "top_k": memory_store.get_top_k(),
            "threshold": memory_store.get_threshold(),
            "stats": memory_store.stats(),
        }

        if query:
            results = memory_store.search(query, top_k=top_k)
            result["results"] = results

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

        return {
            "sessions": sessions,
            "subsessions": subsessions,
            "workflows": workflows,
            "jobs": jobs,
            "tasks": task_items,
            "interactions_total": interactions_total,
            "interactions_max_id": interactions_max_id,
            "memory_count": self._get_memory_count(),
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
