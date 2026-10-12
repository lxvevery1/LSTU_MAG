'use strict';

/*
 * Background worker / event page.
 *
 * Runs as an MV3 service worker in Chromium and as a non-persistent event page in
 * Firefox, so it holds no state in memory across events: everything durable lives
 * in chrome.storage.local. Both `background.service_worker` and `background.scripts`
 * are declared in the manifest because each browser honours exactly one of them.
 *
 * All cross-origin traffic lives here on purpose. In MV3 a content script does not
 * inherit the extension's host permissions, so a fetch issued from content.js to
 * arctic-shift would be blocked by CORS. From the background context it is allowed,
 * which is also why api_model/server.py needs no CORS headers.
 */

const DEFAULTS = {
  enabled: true,
  apiBase: 'http://127.0.0.1:8000',
  arcticBase: 'https://arctic-shift.photon-reddit.com',
  subreddits: ['wallstreetbets'],
  // below this probability the badge is drawn as "uncertain" instead of asserted
  lowConfidence: 0.45,
  // how long a cached label stays valid
  cacheTtlMs: 7 * 24 * 60 * 60 * 1000,
  requestTimeoutMs: 20000,
  /*
   * The archive is on the critical path for every label, but it is unreliable about
   * latency: measured from a real browser, requests for the same id count ranged from
   * 0.6s to over 20s for no reproducible reason. So it gets its own, much shorter
   * budget than the model server. Whatever the archive has not delivered in time is
   * classified from the text already in the page instead, which is a worse label but
   * an immediate one - a slow archive should never mean no answer at all.
   */
  arcticTimeoutMs: 6000
};

const cacheKey = (id) => 'c:' + id;
const trimSlash = (s) => String(s || '').replace(/\/+$/, '');

/*
 * Must match the training-time normalisation in wsb_classification/build_sample_8k.py:
 *   norm = lambda s: " ".join((s or "").split())
 * i.e. every whitespace run collapses to a single space and the result is stripped.
 * Nothing else is removed - the training data still contained markdown links, bare
 * URLs and quote markers, so stripping them here would shift the input distribution.
 */
function norm(s) {
  return String(s == null ? '' : s).split(/\s+/).filter(Boolean).join(' ');
}

// "t3_1wvjeke" and "1wvjeke" both mean the same thing: arctic-shift returns bare ids.
const bareId = (id) => String(id || '').replace(/^t\d_/, '');

// ---------------------------------------------------------------- chrome api

function getOptions(cb) {
  chrome.storage.local.get(DEFAULTS, cb);
}

function storageGet(keys, cb) {
  chrome.storage.local.get(keys, cb);
}

function storageSet(obj, cb) {
  chrome.storage.local.set(obj, cb);
}

const getOptionsP = () => new Promise((res) => getOptions(res));
const getStorageP = (keys) => new Promise((res) => storageGet(keys, res));
const removeStorageP = (keys) => new Promise((res) => chrome.storage.local.remove(keys, res));

// ------------------------------------------------------------------- fetching

class HttpError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.body = body;
  }
}

async function fetchJson(url, init, timeoutMs, attempt = 0) {
  const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
  const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
  let res;
  let text;
  try {
    res = await fetch(url, ctrl ? { ...init, signal: ctrl.signal } : init);
    // The abort has to stay armed across the body read. The archive has been seen
    // returning 200 and then stalling part way through the body, which resolves fetch()
    // as soon as the headers arrive - clearing the timer at that point would leave
    // res.text() waiting forever on a response that is never going to finish.
    text = await res.text();
  } catch (e) {
    // network-level failure: server down, DNS, blocked by the browser, or aborted
    throw new HttpError(e && e.name === 'AbortError' ? 'request timed out'
                                                        : 'cannot reach host', 0, '');
  } finally {
    if (timer) clearTimeout(timer);
  }
  if (!res.ok) {
    throw new HttpError(`HTTP ${res.status}`, res.status, text.slice(0, 300));
  }
  try {
    return JSON.parse(text);
  } catch (e) {
    // the archive has been observed returning a body that is cut off mid-object;
    // one retry is enough to make that a non-event
    if (attempt === 0) {
      return new Promise((r) => setTimeout(r, 400))
        .then(() => fetchJson(url, init, timeoutMs, 1));
    }
    throw new HttpError('response was not JSON (is the API base URL correct?)',
                        res.status, text.slice(0, 300));
  }
}

/*
 * One request for the whole viewport.
 *
 * There is no documented id limit and none could be measured: the same id count takes
 * 0.6s on one run and over 20s on the next, so this is latency variance, not a size
 * threshold. The id list is split only to bound the URL length, and each part is given
 * a short budget of its own so a stall degrades to the page text rather than to nothing.
 */
const ARCTIC_MAX_IDS = 20;

async function fetchArcticChunk(opts, ids) {
  const qs = ids.map(encodeURIComponent).join(',');
  const url = `${trimSlash(opts.arcticBase)}/api/posts/ids?ids=${qs}`;
  const json = await fetchJson(url, { headers: { Accept: 'application/json' } },
                               opts.arcticTimeoutMs);
  const byId = {};
  const rows = (json && json.data) || [];
  for (const r of rows) {
    if (r && r.id) byId[bareId(r.id)] = r;
  }
  return byId;
}

async function fetchArctic(opts, ids, errors) {
  const parts = [];
  for (let i = 0; i < ids.length; i += ARCTIC_MAX_IDS) {
    parts.push(ids.slice(i, i + ARCTIC_MAX_IDS));
  }
  if (!parts.length) return {};

  const chunks = await Promise.all(parts.map((c) =>
    fetchArcticChunk(opts, c).catch((e) => {
      // one slow part must not lose the posts the other parts resolved
      errors._arctic = (errors._arctic ? errors._arctic + '; ' : '') + e.message;
      return {};
    })));

  const byId = {};
  for (const c of chunks) Object.assign(byId, c);
  return byId;
}

// One request for the whole viewport. The API returns an `index` per prediction,
// which is the position in the submitted array, so results are re-keyed by index.
async function fetchPredictions(opts, jobs) {
  const url = `${trimSlash(opts.apiBase)}/predict`;
  const payload = { posts: jobs.map((j) => ({ title: j.title, body: j.body })) };
  const json = await fetchJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(payload)
  }, opts.requestTimeoutMs);

  const preds = (json && json.predictions) || [];
  if (preds.length !== jobs.length) {
    throw new HttpError(`expected ${jobs.length} predictions, got ${preds.length}`,
                        0, '');
  }
  const out = {};
  preds.forEach((p, i) => {
    const idx = Number.isInteger(p.index) ? p.index : i;
    out[jobs[idx].id] = p;
  });
  return out;
}

// -------------------------------------------------------------------- classify

async function classify(posts, opts) {
  const now = Date.now();
  const results = {};
  const errors = {};
  const toWrite = {};

  if (!Array.isArray(posts) || !posts.length) return { results, errors };

  // 1. serve whatever is already cached
  const keys = posts.map((p) => cacheKey(p.id));
  const stored = await new Promise((res) => storageGet(keys, res));

  const todo = [];
  for (const p of posts) {
    if (!p || !p.id) continue;
    const hit = stored[cacheKey(p.id)];
    if (!hit) { todo.push(p); continue; }
    if (now - hit.ts > opts.cacheTtlMs) { todo.push(p); continue; }
    // a post edited after we cached it must be reclassified
    const domTitle = norm(p.domTitle);
    if (domTitle && hit.title && domTitle !== hit.title) { todo.push(p); continue; }
    results[p.id] = Object.assign({}, hit, { fromCache: true });
  }
  if (!todo.length) return { results, errors };

  // 2. pull the authoritative post text from the archive
  let arctic = {};
  try {
    arctic = await fetchArctic(opts, todo.map((p) => p.id), errors);
  } catch (e) {
    errors['_arctic'] = e.message;
  }

  // 3. build the classification jobs, falling back to the on-page text
  // A post the archive did not return and a post the archive never had time for are
  // different situations, and the badge should not claim the archive was consulted
  // when it was not.
  const archiveDown = !!errors._arctic;
  const missText = archiveDown
    ? 'archive unavailable, classified from the page text'
    : 'not in arctic-shift archive, classified from the page text';
  const jobs = [];
  for (const p of todo) {
    const rec = arctic[bareId(p.id)];
    if (rec) {
      jobs.push({
        id: p.id,
        title: norm(rec.title),
        body: norm(rec.selftext),
        source: 'arctic-shift',
        arcticMiss: false
      });
    } else {
      const domTitle = norm(p.domTitle);
      const domBody = norm(p.domBody);
      if (domTitle || domBody) {
        jobs.push({ id: p.id, title: domTitle, body: domBody,
                    source: 'page-dom', arcticMiss: true });
        errors[p.id] = missText;
      } else {
        errors[p.id] = 'no text available';
      }
    }
  }
  // posts newer than the archive that the model has nothing to work with
  for (const p of todo) {
    if (!jobs.some((j) => j.id === p.id) && !errors[p.id]) errors[p.id] = 'no text available';
  }

  // 4. one predict call for everything
  const usable = jobs.filter((j) => j.title || j.body);
  if (!usable.length) return { results, errors };

  let preds;
  try {
    preds = await fetchPredictions(opts, usable);
  } catch (e) {
    const msg = e.status ? `${e.message}: ${e.body}` : e.message;
    for (const j of usable) errors[j.id] = 'classifier unavailable - ' + msg;
    return { results, errors };
  }

  // 5. store and reply
  for (const j of usable) {
    const p = preds[j.id];
    if (!p || !p.label) { errors[j.id] = 'no prediction returned'; continue; }
    const rec = {
      label: p.label,
      confidence: typeof p.confidence === 'number' ? p.confidence : null,
      probabilities: p.probabilities || null,
      emptyInput: !!p.empty_input,
      source: j.source,
      arcticMiss: j.arcticMiss,
      title: j.title,
      ts: now
    };
    results[j.id] = rec;
    toWrite[cacheKey(j.id)] = rec;
  }
  if (Object.keys(toWrite).length) await new Promise((res) => storageSet(toWrite, res));

  return { results, errors };
}

// ------------------------------------------------------------------ messaging

/*
 * One request router shared by both transports.
 *
 * A long-lived port (chrome.runtime.connect) is the primary transport. One-shot
 * sendMessage to an MV3 service worker fails outright on Chrome 131 with "the message
 * port closed before a response was received" - reproduced with a three-line control
 * extension and a hand-launched browser over CDP, so it is not a Playwright artefact.
 * A port also keeps the worker alive across a burst of classifications instead of
 * letting it be evicted and restarted between messages. sendMessage is still wired up
 * for any browser where it does work.
 */
async function dispatch(msg) {
  if (!msg || !msg.type) return null;
  const opts = await getOptionsP();

  if (msg.type === 'WSB_CLASSIFY') {
    if (!opts.enabled) return { results: {}, errors: { '_global': 'disabled' } };
    try {
      return await classify(msg.posts, opts);
    } catch (e) {
      return { results: {}, errors: { '_global': e.message } };
    }
  }

  if (msg.type === 'WSB_HEALTH') {
    try {
      const url = `${trimSlash(opts.apiBase)}/health`;
      return { ok: true, health: await fetchJson(url, {}, 6000) };
    } catch (e) {
      return { ok: false, error: `${e.message} (${opts.apiBase})` };
    }
  }

  if (msg.type === 'WSB_CLEAR_CACHE') {
    const all = await getStorageP(null);
    const drop = Object.keys(all).filter((k) => k.startsWith('c:'));
    await removeStorageP(drop);
    return { ok: true, removed: drop.length };
  }

  if (msg.type === 'WSB_CACHE_STATS') {
    const all = await getStorageP(null);
    return { ok: true, entries: Object.keys(all).filter((k) => k.startsWith('c:')).length };
  }

  return null;
}

const PORT_NAME = 'wsb-classifier';

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== PORT_NAME) return;
  port.onMessage.addListener(async (msg) => {
    let reply;
    try {
      reply = await dispatch(msg);
    } catch (e) {
      reply = { results: {}, errors: { '_global': e.message } };
    }
    if (reply) {
      // echo the sequence so the caller can match replies to requests
      try { port.postMessage(Object.assign({ seq: msg && msg.seq }, reply)); } catch (e) { /* port closed */ }
    }
  });
});

function handleMessage(msg, sendResponse) {
  if (!msg || !msg.type) return false;
  dispatch(msg).then((reply) => { if (reply) sendResponse(reply); });
  return true; // response is async
}

chrome.runtime.onMessage.addListener(handleMessage);

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(DEFAULTS, (cur) => {
    const missing = {};
    for (const k of Object.keys(DEFAULTS)) {
      if (cur[k] === undefined) missing[k] = DEFAULTS[k];
    }
    if (Object.keys(missing).length) chrome.storage.local.set(missing);
  });
});
