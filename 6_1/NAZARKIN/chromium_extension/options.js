'use strict';

const DEFAULTS = {
  enabled: true,
  apiBase: 'http://127.0.0.1:8000',
  arcticBase: 'https://arctic-shift.photon-reddit.com',
  subreddits: ['wallstreetbets'],
  lowConfidence: 0.45
};

const $ = (id) => document.getElementById(id);

function say(text, cls) {
  const el = $('status');
  el.textContent = text;
  el.className = cls || '';
}

function readForm() {
  return {
    enabled: $('enabled').checked,
    apiBase: $('apiBase').value.trim().replace(/\/+$/, ''),
    arcticBase: $('arcticBase').value.trim().replace(/\/+$/, ''),
    subreddits: $('subreddits').value.split(',').map((s) => s.trim().replace(/^\/r\//i, ''))
                                         .filter(Boolean),
    lowConfidence: Math.max(0, Math.min(1, parseFloat($('lowConfidence').value) || 0))
  };
}

chrome.storage.local.get(DEFAULTS, (d) => {
  $('apiBase').value = d.apiBase;
  $('arcticBase').value = d.arcticBase;
  $('subreddits').value = d.subreddits.join(', ');
  $('lowConfidence').value = d.lowConfidence;
  $('enabled').checked = d.enabled;
});

// Same long-lived port the content script uses: one-shot sendMessage to the MV3 worker
// never completes on Chrome 131 ("message port closed before a response was received").
let port = null;
let seq = 0;
const waiters = new Map();

function openPort() {
  if (port) return port;
  port = chrome.runtime.connect({ name: 'wsb-classifier' });
  port.onMessage.addListener((reply) => {
    const w = waiters.get(reply && reply.seq);
    if (!w) return;
    waiters.delete(reply.seq);
    w.resolve(reply);
  });
  port.onDisconnect.addListener(() => {
    port = null;
    for (const [s, w] of waiters) { waiters.delete(s); w.reject(new Error('the background worker went away')); }
  });
  return port;
}

function call(msg) {
  return new Promise((resolve, reject) => {
    const s = ++seq;
    waiters.set(s, { resolve, reject });
    openPort().postMessage(Object.assign({ seq: s }, msg));
  });
}

$('save').addEventListener('click', () => {
  const v = readForm();
  if (!v.apiBase) return say('Base URL cannot be empty', 'bad');
  if (!v.arcticBase) return say('Arctic Shift URL cannot be empty', 'bad');
  chrome.storage.local.set(v, () => say('Saved', 'ok'));
});

$('test').addEventListener('click', async () => {
  say('Testing…');
  try {
    const reply = await call({ type: 'WSB_HEALTH' });
    if (!reply || !reply.ok) return say((reply && reply.error) || 'no reply', 'bad');
    const h = reply.health || {};
    say(`OK — ${h.model} v${h.version}, classes: ${(h.classes || []).join(', ')}`, 'ok');
  } catch (e) {
    say(String(e.message), 'bad');
  }
});

$('clear').addEventListener('click', async () => {
  try {
    const reply = await call({ type: 'WSB_CLEAR_CACHE' });
    say(`Cleared ${(reply && reply.removed) || 0} cached labels`, 'ok');
  } catch (e) {
    say(String(e.message), 'bad');
  }
});

(async () => {
  try {
    const reply = await call({ type: 'WSB_CACHE_STATS' });
    if (reply && reply.entries) say(`${reply.entries} labels cached`);
  } catch (e) { /* the cache count is informational only */ }
})();
