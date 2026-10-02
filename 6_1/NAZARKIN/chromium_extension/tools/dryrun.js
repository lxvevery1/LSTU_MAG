/*
 * Dry-run harness for background.js.
 *
 * Loads the real background worker under a minimal fake `chrome` API and points it at
 * the real arctic-shift archive and the real api_model server, so the full pipeline
 * (batched archive fetch -> whitespace normalisation -> batched /predict -> cache) is
 * exercised exactly as the browser would. Run the server first:
 *
 *   python3 api_model/server.py --host 127.0.0.1 --port 8000
 *   node tools/dryrun.js t3_1wvjeke t3_1lqqrr5
 */

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const HERE = __dirname;
const store = new Map();
let listeners = [];
let connectListeners = [];

const fakeChrome = {
  storage: {
    local: {
      get(defaultsOrKeys, cb) {
        let out = {};
        if (Array.isArray(defaultsOrKeys)) {
          for (const k of defaultsOrKeys) if (store.has(k)) out[k] = store.get(k);
        } else if (typeof defaultsOrKeys === 'string') {
          if (store.has(defaultsOrKeys)) out[defaultsOrKeys] = store.get(defaultsOrKeys);
        } else if (defaultsOrKeys && typeof defaultsOrKeys === 'object') {
          for (const [k, v] of Object.entries(defaultsOrKeys)) {
            out[k] = store.has(k) ? store.get(k) : v;
          }
        }
        process.nextTick(() => cb(out));
      },
      set(obj, cb) {
        for (const [k, v] of Object.entries(obj)) store.set(k, v);
        if (cb) process.nextTick(cb);
      },
      remove(keys, cb) {
        for (const k of [].concat(keys)) store.delete(k);
        if (cb) process.nextTick(cb);
      }
    },
    onChanged: { addListener: () => {} }
  },
  runtime: {
    onMessage: { addListener: (fn) => listeners.push(fn) },
    onConnect: { addListener: (fn) => connectListeners.push(fn) },
    onInstalled: { addListener: (fn) => listeners.push(fn) },
    lastError: null
  }
};

const sandbox = { chrome: fakeChrome, fetch, console, AbortController, setTimeout,
                  clearTimeout, Object, JSON, Math, String, Number, Array, Error,
                  Promise, Set, Map, Boolean, RegExp, Date, isNaN };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(HERE, '..', 'background.js'), 'utf8'),
                sandbox, { filename: 'background.js' });

if (!connectListeners.length) {
  console.error('the worker registered no onConnect handler');
  process.exit(1);
}

// The port transport the extension actually uses, exercised end to end: grab the
// message handler the worker registers inside onConnect and call it directly, then
// check it really answers over the port shape.
async function sendOverPort(msg) {
  const captured = [];
  const posted = [];      // what the worker actually posts back on the port
  const mockPort = {
    name: 'wsb-classifier',
    onMessage: { addListener: (f) => captured.push(f) },
    onDisconnect: { addListener: () => {} },
    postMessage: (m) => posted.push(m)
  };
  for (const fn of connectListeners) fn(mockPort);
  if (!captured.length) return { __error: 'the worker registered no onConnect handler' };

  // hand the request to the worker's own listener, exactly as the port would
  for (const f of captured) f(Object.assign({ seq: 1 }, msg));

  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    if (posted.length) return posted.shift();
    await new Promise((r) => setTimeout(r, 25));
  }
  return { __timeout: true };
}

const LABELS = { NOISE_MEME: 1, GENUINE_OPINION: 1, FINANCIAL_CLAIM: 1, ANALYSIS_DD: 1 };

(async () => {
  const ids = process.argv.slice(2).map((a) => (/^t3_/.test(a) ? a : 't3_' + a));
  if (!ids.length) { console.error('usage: node tools/dryrun.js <id> [id...]'); process.exit(1); }

  console.log(`requesting ${ids.length} post(s): ${ids.join(', ')}\n`);
  const t0 = Date.now();
  const reply = await sendOverPort({ type: 'WSB_CLASSIFY', posts: ids.map((id) => ({ id, domTitle: '' })) });
  const ms = Date.now() - t0;

  const results = reply.results || {};
  const errors = reply.errors || {};
  let bad = 0;

  for (const id of ids) {
    const r = results[id];
    if (!r) {
      console.log(`  ${id}  NO RESULT   ${errors[id] || errors._global || ''}`);
      bad++;
      continue;
    }
    if (!LABELS[r.label]) { console.log(`  ${id}  BAD LABEL ${r.label}`); bad++; continue; }
    if (r.probabilities) {
      const sum = Object.values(r.probabilities).reduce((a, b) => a + b, 0);
      if (Math.abs(sum - 1) > 1e-3) { console.log(`  ${id}  PROBS SUM ${sum}`); bad++; }
      const top = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1])[0][0];
      if (top !== r.label) { console.log(`  ${id}  LABEL != ARGMAX (${r.label} vs ${top})`); bad++; }
    }
    const pct = r.probabilities
      ? Object.entries(r.probabilities).map(([k, v]) => `${k.replace('_', ' ').toLowerCase()} ${(v * 100).toFixed(0)}%`).join(', ')
      : '';
    console.log(`  ${id}  ${r.label.padEnd(16)} ${(r.confidence == null ? '  -  ' : r.confidence.toFixed(3))}  [${r.source}]`);
    console.log(`        ${pct}`);
    if (errors[id]) console.log(`        note: ${errors[id]}`);
  }

  const chunks = Math.max(1, Math.ceil(ids.length / 4));
  console.log(`\nlatency ${ms}ms for ${ids.length} post(s) ` +
              `(${chunks} arctic request(s) of max 4 ids + 1 model request)`);
  console.log(`cache entries written: ${[...store.keys()].filter((k) => k.startsWith('c:')).length}`);

  // second pass must be served entirely from cache
  const t1 = Date.now();
  const again = await sendOverPort({ type: 'WSB_CLASSIFY', posts: ids.map((id) => ({ id, domTitle: '' })) });
  const cached = Object.values(again.results || {}).filter((r) => r.fromCache).length;
  console.log(`cache pass: ${cached}/${ids.length} served from storage in ${Date.now() - t1}ms`);

  const health = await sendOverPort({ type: 'WSB_HEALTH' });
  console.log(`health: ${health.ok ? 'ok ' + JSON.stringify(health.health) : 'FAILED ' + health.error}`);

  console.log(bad ? `\nFAILED (${bad} problem(s))` : '\nOK');
  process.exit(bad ? 1 : 0);
})();
