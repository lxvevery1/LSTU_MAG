/*
 * Regression tests for background.js network handling, driven by a fake fetch.
 *
 * These are the failure modes that actually showed up against the live archive and are
 * invisible in a dry run against a healthy server:
 *
 *  1. The archive returns 200 and then stalls part way through the body. fetch() resolves
 *     on the headers, so a timeout that is cancelled once fetch() settles never fires and
 *     res.text() waits forever.
 *  2. The archive returns a body that is cut off mid-object, so JSON.parse throws.
 *  3. One chunk of a multi-chunk fetch fails while the others succeed.
 *
 * A stalled request must degrade to classifying from the page text, not hang.
 *
 * Run: node tools/test-background.js
 */

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const BACKGROUND = path.join(__dirname, '..', 'background.js');
let failures = 0;
const check = (name, cond, detail) => {
  console.log(`  [${cond ? 'ok ' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`);
  if (!cond) failures++;
};
const tick = (ms) => new Promise((r) => setTimeout(r, ms));

/*
 * Build a background.js sandbox whose fetch is scripted by `handler(url, init)`.
 * The handler returns a fake Response: { ok, status, text() }. text() must reject with
 * an AbortError when the request's signal fires, which is what a real stalled body does.
 */
function loadBackground({ handler, opts = {} }) {
  const store = new Map(Object.entries(opts));
  const connectListeners = [];
  const calls = [];

  const fakeChrome = {
    storage: {
      local: {
        get(dk, cb) {
          const out = {};
          if (Array.isArray(dk)) {
            for (const k of dk) if (store.has(k)) out[k] = store.get(k);
          } else if (dk && typeof dk === 'object') {
            for (const [k, v] of Object.entries(dk)) out[k] = store.has(k) ? store.get(k) : v;
          }
          process.nextTick(() => cb(out));
        },
        set(obj, cb) { for (const [k, v] of Object.entries(obj)) store.set(k, v); if (cb) process.nextTick(cb); },
        remove(keys, cb) { for (const k of [].concat(keys)) store.delete(k); if (cb) process.nextTick(cb); }
      },
      onChanged: { addListener: () => {} }
    },
    runtime: {
      onMessage: { addListener: () => {} },
      onConnect: { addListener: (fn) => connectListeners.push(fn) },
      onInstalled: { addListener: () => {} },
      lastError: null
    }
  };

  const sandbox = {
    chrome: fakeChrome, console, AbortController, setTimeout, clearTimeout, setInterval,
    clearInterval, Object, JSON, Math, String, Number, Array, Error, Promise, Set, Map,
    Boolean, RegExp, Date, isNaN, URL,
    fetch: async (url, init) => {
      calls.push({ url, init });
      const r = await handler(url, init);
      return r;
    }
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(BACKGROUND, 'utf8'), sandbox, { filename: 'background.js' });

  const post = async (msg) => {
    const handlers = [];
    for (const fn of connectListeners) {
      fn({ name: 'wsb-classifier', onMessage: { addListener: (f) => handlers.push(f) },
           onDisconnect: { addListener: () => {} },
           postMessage: (m) => store.set('__reply', m) });
    }
    store.delete('__reply');
    for (const h of handlers) h(Object.assign({ seq: 1 }, msg));
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      const s = store.get('__reply');
      if (s) return s;
      await tick(20);
    }
    return { __timeout: true };
  };
  return { post, calls, store };
}

/*
 * A response that sends headers and then never finishes the body, the way the live
 * archive was observed behaving. The body only gives up if the request is aborted, so
 * this genuinely hangs unless the abort stays armed across the read.
 */
function stalledBody(signal) {
  return {
    ok: true, status: 200,
    text: () => new Promise((_, rej) => {
      if (!signal) return;                       // nothing will ever settle it
      if (signal.aborted) return rej(abortError());
      signal.addEventListener('abort', () => rej(abortError()));
    })
  };
}

function abortError() {
  const e = new Error('The user aborted a request.');
  e.name = 'AbortError';
  return e;
}

const POSTS = [
  { id: 't3_aaa', domTitle: 'Google is the AI arms dealer', domBody: '' },
  { id: 't3_bbb', domTitle: 'Jim Cramer is bearish on Nike', domBody: '' }
];

(async () => {
  // ------------------------------------------------------- 1. stalled archive body
  console.log('1. the archive returns headers then stalls mid-body');
  {
    const { post, calls } = loadBackground({
      // a 250ms archive budget keeps the test quick
      opts: { arcticTimeoutMs: 250 },
      handler: async (url, init) => {
        if (url.includes('/api/posts/ids')) return stalledBody(init && init.signal);
        // the model server must still be reached
        return { ok: true, status: 200, text: async () => JSON.stringify({
          predictions: [
            { label: 'ANALYSIS_DD', confidence: 0.7, probabilities: {
                NOISE_MEME: 0.1, GENUINE_OPINION: 0.1, FINANCIAL_CLAIM: 0.1, ANALYSIS_DD: 0.7 } },
            { label: 'NOISE_MEME', confidence: 0.6, probabilities: {
                NOISE_MEME: 0.6, GENUINE_OPINION: 0.2, FINANCIAL_CLAIM: 0.1, ANALYSIS_DD: 0.1 } }
          ]
        }) };
      }
    });
    const t0 = Date.now();
    const reply = await post({ type: 'WSB_CLASSIFY', posts: POSTS });
    const ms = Date.now() - t0;

    check('the request does not hang', !reply.__timeout, `${ms}ms`);
    check('it gave up on the archive quickly', ms < 5000, `${ms}ms`);
    check('both posts still got a label', Object.keys(reply.results || {}).length === 2,
          JSON.stringify(Object.keys(reply.results || {})));
    const sources = (reply.results ? Object.values(reply.results) : []).map((r) => r.source);
    check('labelled from the page text, not the archive',
          sources.every((s) => s === 'page-dom'), JSON.stringify(sources));
    check('the archive failure is recorded',
          !!(reply.errors && reply.errors._arctic), JSON.stringify(reply.errors && reply.errors._arctic));
    check('the badge is told the archive was unavailable, not that it had no record',
          !!(reply.errors && Object.values(reply.errors).some((v) => /archive unavailable/.test(v))),
          JSON.stringify(reply.errors));
    check('the model server was still called',
          calls.some((c) => c.url.includes('/predict')), String(calls.length));
  }

  // ------------------------------------------------------ 2. truncated JSON body
  console.log('2. the archive returns a body cut off mid-object');
  {
    let archiveCalls = 0;
    const { post } = loadBackground({
      opts: { arcticTimeoutMs: 5000 },
      handler: async (url, init) => {
        if (url.includes('/api/posts/ids')) {
          archiveCalls++;
          if (archiveCalls === 1) {
            // first attempt is truncated, exactly as observed live
            return { ok: true, status: 200,
                     text: async () => '{"data":[{"id":"t3_aaa","title":"Goog' };
          }
          return { ok: true, status: 200, text: async () => JSON.stringify({
            data: [{ id: 't3_aaa', title: 'Google is the AI arms dealer', selftext: '' }]
          }) };
        }
        return { ok: true, status: 200, text: async () => JSON.stringify({
          predictions: [
            { label: 'ANALYSIS_DD', confidence: 0.7, probabilities: {
                NOISE_MEME: 0.1, GENUINE_OPINION: 0.1, FINANCIAL_CLAIM: 0.1, ANALYSIS_DD: 0.7 } },
            { label: 'NOISE_MEME', confidence: 0.6, probabilities: {
                NOISE_MEME: 0.6, GENUINE_OPINION: 0.2, FINANCIAL_CLAIM: 0.1, ANALYSIS_DD: 0.1 } }
          ]
        }) };
      }
    });
    const reply = await post({ type: 'WSB_CLASSIFY', posts: POSTS });
    check('the archive was retried once', archiveCalls === 2, String(archiveCalls));
    check('the retried post is labelled from the archive',
          !!(reply.results && reply.results.t3_aaa &&
             reply.results.t3_aaa.source === 'arctic-shift'),
          JSON.stringify(reply.results && reply.results.t3_aaa && reply.results.t3_aaa.source));
  }

  // --------------------------------------------------------- 3. one chunk fails
  console.log('3. one chunk of a multi-chunk fetch fails, the others still land');
  {
    // ARCTIC_MAX_IDS is 20, so this needs more than 20 ids to actually split, and the
    // failing id is put in the second chunk to prove the first one's posts survive.
    const many = [];
    for (let i = 0; i < 21; i++) {
      many.push({ id: `t3_p${i}`, domTitle: `post number ${i}`, domBody: '' });
    }
    const { post, calls } = loadBackground({
      opts: { arcticTimeoutMs: 500 },
      handler: async (url, init) => {
        if (url.includes('/api/posts/ids')) {
          if (url.includes('t3_p20')) return { ok: false, status: 503, text: async () => 'upstream' };
          const ids = (url.split('ids=')[1] || '').split(',');
          return { ok: true, status: 200, text: async () => JSON.stringify({
            data: ids.map((id) => ({ id, title: 'title for ' + id, selftext: '' }))
          }) };
        }
        return { ok: true, status: 200, text: async () => JSON.stringify({
          predictions: many.map(() => ({ label: 'NOISE_MEME', confidence: 0.6,
            probabilities: { NOISE_MEME: 0.6, GENUINE_OPINION: 0.2,
                              FINANCIAL_CLAIM: 0.1, ANALYSIS_DD: 0.1 } }))
        }) };
      }
    });
    const reply = await post({ type: 'WSB_CLASSIFY', posts: many });
    const r = reply.results || {};
    const archiveCalls = calls.filter((c) => c.url.includes('/api/posts/ids')).length;
    check('the id list really was split into more than one request', archiveCalls > 1,
          String(archiveCalls));
    check('every post in the healthy chunk is labelled from the archive',
          many.slice(0, 20).every((p) => r[p.id] && r[p.id].source === 'arctic-shift'),
          JSON.stringify(many.slice(0, 20).map((p) => r[p.id] && r[p.id].source)
            .filter((s) => s !== 'arctic-shift')));
    check('only the post in the failed chunk falls back to the page text',
          r.t3_p20 && r.t3_p20.source === 'page-dom', r.t3_p20 && r.t3_p20.source);
  }

  // -------------------------------------------------- 4. a label is never invented
  console.log('4. the model is unreachable');
  {
    const { post } = loadBackground({
      opts: { arcticTimeoutMs: 5000 },
      handler: async (url, init) => {
        if (url.includes('/api/posts/ids')) {
          return { ok: true, status: 200, text: async () => JSON.stringify({ data: [] }) };
        }
        return { ok: false, status: 500, text: async () => 'boom' };
      }
    });
    const reply = await post({ type: 'WSB_CLASSIFY', posts: POSTS });
    check('no results are reported', Object.keys(reply.results || {}).length === 0,
          JSON.stringify(Object.keys(reply.results || {})));
    check('every post carries an error instead of a made-up label',
          !!(reply.errors && reply.errors.t3_aaa && reply.errors.t3_bbb),
          JSON.stringify(reply.errors));
  }

  console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})();
