"""
Web Interface

Debug panel available at /debug.  Provides a read/write inspection view of all
running sessions, sub-sessions, scheduled jobs, and reminders.
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
    <button class="tab-btn" data-tab="interactions" onclick="showTab('interactions')">
      Interactions <span class="tab-count" id="cnt-interactions">0</span>
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
              <th></th><th>ID</th><th>Workflow</th><th>Deps</th><th>Parent</th><th>Status</th>
              <th>Tools</th><th>Objective</th><th>Mode</th>
              <th>Created</th><th>Duration</th><th>Result / Error</th>
            </tr>
          </thead>
          <tbody id="subsessions-body">
            <tr><td colspan="12" class="empty">Loading\u2026</td></tr>
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

    <!-- ── Interaction Log ── -->
    <div class="tab-panel" id="panel-interactions">
      <div class="form-bar" style="gap:.8rem">
        <div class="form-group">
          <label>Filter by session</label>
          <select id="il-filter-session" onchange="loadInteractionLog()" style="min-width:180px">
            <option value="">All sessions</option>
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
const _closedWorkflows = new Set();
const _closedReminders = new Set();
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
      case 'subsessions': await loadSubSessions(); break;
      case 'workflows':   await loadWorkflows(); break;
      case 'jobs':        await loadJobs(); break;
      case 'reminders':   await loadReminders(); break;
      case 'interactions': await loadInteractionLog(); break;
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
    renderSubSessions(data.subsessions || []);
    renderWorkflows(data.workflows || []);
    renderJobs(data.jobs || []);
    renderReminders(data.reminders || {});
    updateInteractionsCount(data.interactions_total, data.interactions_max_id);
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

// ── Sub-sessions ──
let _expandedSubSessions = new Set();

function renderSubSessions(sessions) {
  document.getElementById('cnt-subsessions').textContent = sessions.length;
  const tbody = document.getElementById('subsessions-body');
  const scrollArea = tbody.closest('.scroll-area');
  const outerScroll = scrollArea ? scrollArea.scrollTop : 0;

  // Save expanded detail content + inner scroll so SSE updates don't wipe them
  const savedDetails = {};
  for (const sid of _expandedSubSessions) {
    const contentEl = document.getElementById('sscontent-' + sid);
    if (contentEl) {
      const innerDiv = contentEl.querySelector('div[style*="overflow-y"]');
      savedDetails[sid] = {
        html: contentEl.innerHTML,
        innerScroll: innerDiv ? innerDiv.scrollTop : 0,
      };
    }
  }

  if (!sessions.length) {
    tbody.innerHTML = '<tr><td colspan="12" class="empty">No sub-sessions recorded</td></tr>';
    if (scrollArea) scrollArea.scrollTop = outerScroll;
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
    const tcCount = s.tool_call_count || 0;
    const isExpanded = _expandedSubSessions.has(s.session_id);
    const arrow = isExpanded ? '\u25bc' : '\u25b6';
    const activityIndicator = s.status === 'running' && tcCount > 0
      ? ' <span style="color:#6fe06f" title="Active tool calls">\u25cf</span>' : '';
    return `<tr style="cursor:pointer" onclick="toggleSubSession('${esc(s.session_id)}')">
      <td style="width:1.5em;text-align:center;color:#666">${arrow}</td>
      <td class="mono">${esc(s.session_id)}${activityIndicator}</td>
      <td class="mono dim">${esc(wfShort)}</td>
      <td class="mono dim" title="${esc(depsStr)}">${deps.length ? esc(deps.map(d => d.slice(0,12)).join(', ')) : '\u2014'}</td>
      <td class="mono dim" title="${esc(s.parent_thread_id || '')}">${esc(parentShort)}</td>
      <td>${badge(s.status, s.status)}</td>
      <td class="dim">${tcCount}</td>
      <td class="trunc" title="${esc(s.objective)}">${esc(s.objective.slice(0, 60))}${s.objective.length > 60 ? '\u2026' : ''}</td>
      <td class="dim">${esc(s.system_prompt_mode)}</td>
      <td class="dim" style="white-space:nowrap">${fmtTime(s.created_at)}</td>
      <td class="dim" style="white-space:nowrap">${duration}</td>
      <td class="trunc" title="${esc(resultText)}">${esc(resultText)}</td>
    </tr>
    <tr id="ssdetail-${esc(s.session_id)}" style="display:${isExpanded ? '' : 'none'}">
      <td colspan="12" style="background:#0d1526;padding:.6rem 1rem">
        <div id="sscontent-${esc(s.session_id)}" style="font-size:.8rem;color:#999">Loading\u2026</div>
      </td>
    </tr>`;
  }).join('');

  // Restore expanded detail content + inner scroll; only fetch if not yet loaded
  for (const sid of _expandedSubSessions) {
    const contentEl = document.getElementById('sscontent-' + sid);
    if (!contentEl) continue;
    if (savedDetails[sid]) {
      contentEl.innerHTML = savedDetails[sid].html;
      const innerDiv = contentEl.querySelector('div[style*="overflow-y"]');
      if (innerDiv) innerDiv.scrollTop = savedDetails[sid].innerScroll;
    } else {
      fetchSubSessionDetail(sid);
    }
  }

  if (scrollArea) scrollArea.scrollTop = outerScroll;
}

async function loadSubSessions() {
  const r = await fetch('/api/debug/subsessions');
  const d = await r.json();
  renderSubSessions(d.sessions || []);
}

async function toggleSubSession(sid) {
  const detailRow = document.getElementById('ssdetail-' + sid);
  if (!detailRow) return;
  if (_expandedSubSessions.has(sid)) {
    _expandedSubSessions.delete(sid);
    detailRow.style.display = 'none';
  } else {
    _expandedSubSessions.add(sid);
    detailRow.style.display = '';
    fetchSubSessionDetail(sid);
  }
  // Update arrow
  const tbody = document.getElementById('subsessions-body');
  const rows = tbody.querySelectorAll('tr[onclick]');
  rows.forEach(r => {
    const m = r.getAttribute('onclick').match(/toggleSubSession\('([^']+)'\)/);
    if (m) {
      const arrow = _expandedSubSessions.has(m[1]) ? '\u25bc' : '\u25b6';
      r.querySelector('td').textContent = arrow;
    }
  });
}

async function fetchSubSessionDetail(sid) {
  const el = document.getElementById('sscontent-' + sid);
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
    msgs.forEach((m, i) => {
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

// ── Workflows ──
function renderWorkflows(workflows) {
  document.getElementById('cnt-workflows').textContent = workflows.length;
  const area = document.getElementById('workflows-area');
  const scrollTop = area.scrollTop;
  if (!workflows.length) {
    area.innerHTML = '<div class="empty">No workflows recorded</div>';
    area.scrollTop = scrollTop;
    return;
  }
  area.innerHTML = workflows.map(wf => {
    const runningNodes = wf.nodes.filter(n => n.status === 'running');
    const activityNote = runningNodes.length > 0
      ? ' <span style="color:#6fe06f">\u25cf ' + runningNodes.length + ' active</span>'
      : '';
    const isClosed = _closedWorkflows.has(wf.workflow_id);
    const nodesHtml = wf.nodes.map(n => {
      const depsStr = n.depends_on.length ? ' \u2190 ' + n.depends_on.join(', ') : '';
      const detail = n.error ? '\u26a0 ' + esc(n.error)
        : (n.result_preview ? esc(n.result_preview) : '');
      const nodeActivity = n.status === 'running'
        ? '<span style="color:#6fe06f" title="Running">\u25cf</span> ' : '';
      return `<tr>
        <td class="mono">${nodeActivity}${esc(n.node_id)}</td>
        <td>${badge(n.status, n.status)}</td>
        <td class="trunc" title="${esc(n.objective)}">${esc(n.objective.slice(0, 80))}${n.objective.length > 80 ? '\u2026' : ''}</td>
        <td class="mono dim">${esc(depsStr || 'none')}</td>
        <td class="trunc dim" title="${esc(detail)}">${detail.slice(0, 100)}${detail.length > 100 ? '\u2026' : ''}</td>
      </tr>`;
    }).join('');
    return `
      <div class="section-hdr${isClosed ? '' : ' open'}"
           data-section-id="${esc(wf.workflow_id)}" data-section-type="workflow"
           onclick="toggleSection(this)">
        <span>
          <span class="mono" style="font-size:.78rem">${esc(wf.workflow_id)}</span>
          ${badge(wf.status, wf.status)}${activityNote}
          <span class="dim">${wf.node_count} node${wf.node_count !== 1 ? 's' : ''}</span>
          <span class="dim">${esc(wf.parent_thread_id || 'no parent')}</span>
        </span>
        <span>\u25bc</span>
      </div>
      <div class="section-body" style="${isClosed ? 'display:none' : ''}">
        <table class="data-table">
          <thead><tr>
            <th>Node ID</th><th>Status</th><th>Objective</th><th>Depends On</th><th>Result / Error</th>
          </tr></thead>
          <tbody>${nodesHtml}</tbody>
        </table>
      </div>`;
  }).join('');
  area.scrollTop = scrollTop;
}

async function loadWorkflows() {
  const r = await fetch('/api/debug/workflows');
  const d = await r.json();
  renderWorkflows(d.workflows || []);
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

// ── Reminders ──
function renderReminders(data) {
  const active = data.active || [];
  const completed = data.completed || [];
  const failed = data.failed || [];
  const cancelled = data.cancelled || [];
  document.getElementById('cnt-reminders').textContent = active.length;

  const area = document.getElementById('reminders-area');
  const scrollTop = area.scrollTop;
  area.innerHTML =
    renderReminderSection('active-body',    'active',    'Active',    active,    true)  +
    renderReminderSection('completed-body', 'completed', 'Completed', completed, false) +
    renderReminderSection('failed-body',    'failed',    'Failed',    failed,    false) +
    (cancelled.length ? renderReminderSection('cancelled-body', 'cancelled', 'Cancelled', cancelled, false) : '');
  area.scrollTop = scrollTop;
}

async function loadReminders() {
  const r = await fetch('/api/debug/reminders');
  const d = await r.json();
  renderReminders(d);
}

function renderReminderSection(id, sectionKey, label, reminders, showActions) {
  const isClosed = _closedReminders.has(sectionKey);
  return `
    <div class="section-hdr${isClosed ? '' : ' open'}"
         data-section-id="${esc(sectionKey)}" data-section-type="reminder"
         onclick="toggleSection(this)">
      <span>${label} <small>(${reminders.length})</small></span>
      <span>&#9660;</span>
    </div>
    <div class="section-body" style="${isClosed ? 'display:none' : ''}">
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
  body.style.display = hdr.classList.contains('open') ? '' : 'none';
  const id   = hdr.dataset.sectionId;
  const type = hdr.dataset.sectionType;
  if (!id) return;
  const set = type === 'workflow' ? _closedWorkflows : _closedReminders;
  hdr.classList.contains('open') ? set.delete(id) : set.add(id);
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
// ── Interaction Log ──
function _makeILRow(e) {
  const ts = new Date(e.timestamp * 1000).toLocaleString();
  const inputPreview = (e.input || '').length > 120 ? e.input.slice(0, 117) + '\u2026' : (e.input || '');
  const outputPreview = (e.output || '').length > 120 ? e.output.slice(0, 117) + '\u2026' : (e.output || '');
  const statusCls = e.status === 'ok' ? 'completed' : (e.status === 'error' ? 'failed' : 'pending');
  const sessionShort = (e.session || '').length > 24 ? e.session.slice(0, 22) + '\u2026' : (e.session || '');
  const actionStyle = e.action.startsWith('turing') ? 'turing' : (e.action === 'tool_call' ? 'tool' : (e.action === 'chat' ? 'web' : (e.action === 'error' ? 'failed' : 'system')));
  return `<tr style="cursor:pointer${e.action === 'tool_call' ? ';background:#0a0f1a' : ''}" onclick="toggleILDetail(this, ${e.id})">
    <td class="dim" style="white-space:nowrap">${esc(ts)}</td>
    <td>${badge(actionStyle, e.action)}</td>
    <td class="mono" style="font-size:.72rem" title="${esc(e.session || '')}">${esc(sessionShort)}</td>
    <td class="mono" style="font-size:.72rem">${esc(e.llm || '')}</td>
    <td class="trunc" title="${esc(e.input || '')}">${esc(inputPreview)}</td>
    <td class="trunc" title="${esc(e.output || '')}">${esc(outputPreview)}</td>
    <td>${badge(statusCls, e.status)}</td>
  </tr>`;
}

function _fetchIL({limit = 200, before_id, after_id} = {}) {
  const session = document.getElementById('il-filter-session').value;
  const params = new URLSearchParams({limit});
  if (session) params.set('session', session);
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

// ── Config info ──
async function loadConfig() {
  try {
    const r = await fetch('/api/debug/config');
    const d = await r.json();
    const el = document.getElementById('cfg-info');
    const parts = [];
    if (d.main) parts.push('<span style="color:#a8d8ea">' + esc(d.main.model) + '</span> <span style="color:#555">' + fmtTokens(d.main.context_size) + ' ctx</span>');
    if (d.compaction && d.compaction.model !== d.main.model)
      parts.push('<span style="color:#666">compact: ' + esc(d.compaction.model) + '</span>');
    if (d.sub_sessions && (d.sub_sessions.model !== d.main.model || d.sub_sessions.base_url !== d.main.base_url))
      parts.push('<span style="color:#666">sub: ' + esc(d.sub_sessions.model) + '</span>');
    if (d.dreaming && d.dreaming.model !== d.main.model)
      parts.push('<span style="color:#666">dream: ' + esc(d.dreaming.model) + '</span>');
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
      _scheduler     – ReminderScheduler
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
        app.router.add_get("/api/debug/reminders",                      self._api_reminders)
        app.router.add_post("/api/debug/reminders",                     self._api_reminder_create)
        app.router.add_delete("/api/debug/reminders/{job_id}",          self._api_reminder_delete)
        app.router.add_get("/api/debug/interaction-log",                 self._api_interaction_log)
        app.router.add_get("/api/debug/interaction-log/{id}",            self._api_interaction_log_entry)
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
        from wintermute import prompt_assembler
        try:
            prompt = prompt_assembler.assemble()
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc)})
        from wintermute.llm_thread import _count_tokens
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


    # ------------------------------------------------------------------
    # Interaction Log
    # ------------------------------------------------------------------

    async def _api_interaction_log(self, request: web.Request) -> web.Response:
        from wintermute import database
        limit = int(request.query.get("limit", "200"))
        offset = int(request.query.get("offset", "0"))
        session = request.query.get("session") or None
        before_id_s = request.query.get("before_id")
        after_id_s = request.query.get("after_id")
        before_id = int(before_id_s) if before_id_s else None
        after_id = int(after_id_s) if after_id_s else None
        entries = database.get_interaction_log(limit=limit, offset=offset,
                                               session_filter=session,
                                               before_id=before_id,
                                               after_id=after_id)
        total = database.count_interaction_log(session_filter=session)
        return self._json({"entries": entries, "total": total})

    async def _api_interaction_log_entry(self, request: web.Request) -> web.Response:
        from wintermute import database
        entry_id = int(request.match_info["id"])
        entry = database.get_interaction_log_entry(entry_id)
        if not entry:
            return web.json_response({"error": "not found"}, status=404)
        return self._json(entry)

    # ------------------------------------------------------------------
    # SSE stream
    # ------------------------------------------------------------------

    async def _build_stream_snapshot(self) -> dict:
        from wintermute import database

        # Sessions
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

        # Sub-sessions and workflows
        subsessions = self._sub_sessions.list_all() if self._sub_sessions else []
        workflows = self._sub_sessions.list_workflows() if self._sub_sessions else []

        # Scheduled jobs
        jobs = self._scheduler.list_jobs() if self._scheduler else []

        # Reminders
        raw = tool_module.execute_tool("list_reminders", {})
        try:
            reminders = json.loads(raw)
        except json.JSONDecodeError:
            reminders = {"active": [], "completed": [], "failed": []}

        # Interaction log counts
        interactions_total = database.count_interaction_log()
        interactions_max_id = database.get_interaction_log_max_id()

        return {
            "sessions": sessions,
            "subsessions": subsessions,
            "workflows": workflows,
            "jobs": jobs,
            "reminders": reminders,
            "interactions_total": interactions_total,
            "interactions_max_id": interactions_max_id,
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
