/* ════════════════════════════════════════════════════════════
   BOT-RL — SHARED SCRIPT
   Loaded on every page. Provides:
     • $ / post() / fetch helpers
     • Nav clock + live RUNNING/STOPPED indicator
     • Generic anti-"snapback" form sync (THE FIX for settings reverting
       before you can hit Save — see safeSet()/markDirty() below)
     • Shared formatters used across pages (fmt, esc, formatDur)
   Each page's own template still has its own small <script> block for
   page-specific rendering (open trades table, prediction cards, etc) —
   those call into the helpers here rather than re-implementing them.
   ════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

/* ── Anti-snapback form sync ──────────────────────────────────
   Root cause of the "confidence slider reverts before I can click
   Save" bug: every page polls /api/state every few seconds and
   blindly does `field.value = serverValue` for every input, including
   ones you're actively editing. If a poll lands mid-edit, your change
   gets overwritten before you ever reach the Save button.

   Fix: track which fields have unsaved local edits ("dirty") and skip
   overwriting those — and skip whatever field currently has focus —
   until the edit is actually saved (or the page is reloaded).
   Use safeSet() everywhere a poll would otherwise do `field.value = x`.
─────────────────────────────────────────────────────────────── */
const _dirtyFields = new Set();

function markDirty(id) { _dirtyFields.add(id); }
function clearDirty(...ids) { ids.forEach(id => _dirtyFields.delete(id)); }
function isProtected(el, id) {
  return _dirtyFields.has(id) || document.activeElement === el;
}

/** Safely set a form field's value from server/poll data — never
 *  stomps a field the user is actively editing or has unsaved changes in. */
function safeSet(id, value) {
  const el = $(id);
  if (!el) return;
  if (isProtected(el, id)) return;
  el.value = value;
}

/** Same idea for checkboxes/toggles. */
function safeSetChecked(id, checked) {
  const el = $(id);
  if (!el) return;
  if (isProtected(el, id)) return;
  el.checked = !!checked;
}

// Any input/select/checkbox on the page automatically marks itself
// dirty the moment the user touches it — no per-field wiring needed.
document.addEventListener('input', e => {
  if (e.target && e.target.id) markDirty(e.target.id);
});
document.addEventListener('change', e => {
  if (e.target && e.target.id) markDirty(e.target.id);
});

/* ── Nav: live clock + RUNNING/STOPPED indicator ──────────────── */
function updateNavClock() {
  const el = $('nav-time');
  if (el) el.textContent = new Date().toUTCString().split(' ')[4] + ' UTC';
}
setInterval(updateNavClock, 1000);
updateNavClock();

function updateNavStatus(botRunning) {
  const dot = $('nav-dot'), status = $('nav-status');
  if (dot) dot.className = 'dot' + (botRunning ? '' : ' off');
  if (status) status.textContent = botRunning ? 'RUNNING' : 'STOPPED';
}

/* ── Common API helpers ───────────────────────────────────────── */
async function post(url, data) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data || {})
    });
    return await r.json();
  } catch (e) { return { ok: false, error: String(e) }; }
}

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

/** Force-close a trade by id — used on Home, Dashboard, and Settings'
 *  open-trades panel. `onDone` is called (e.g. to re-fetch state) after. */
async function closeTrade(id, onDone) {
  if (!confirm(`Force-close trade ${id}?`)) return;
  try {
    const r = await fetch('/api/trade/close/' + id, { method: 'POST' });
    const d = await r.json();
    if (!d.ok) alert(d.error || 'Failed to close trade');
    if (onDone) await onDone();
  } catch (e) { alert('Request failed: ' + e); }
}

async function botAction(action, onDone) {
  const r = await post('/api/bot/' + action, {});
  if (!r.ok) alert(r.error || 'Action failed');
  if (onDone) await onDone();
}

/* ── Shared formatters ────────────────────────────────────────── */
function fmt(n) {
  return Number(n || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function formatDur(ms) {
  if (isNaN(ms) || ms < 0) return '—';
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
}

/* ── Flash / alert helpers (Settings-style panels) ────────────── */
function flash(id) {
  const el = $(id);
  if (!el) return;
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 2500);
}
function showOk(prefix) {
  const el = $(prefix + '-ok');
  if (el) { el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 3000); }
}
function showErr(prefix, msg) {
  if (msg) { const sp = $(prefix + '-err-msg'); if (sp) sp.textContent = msg; }
  const el = $(prefix + '-err');
  if (el) { el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 4000); }
}
function hideAlerts(prefix) {
  [$(prefix + '-ok'), $(prefix + '-err')].forEach(e => { if (e) e.classList.remove('show'); });
}
