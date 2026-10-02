/*
 * DOM test for content.js.
 *
 * Feeds the content script synthetic post cards that mirror the two Reddit layouts it
 * has to handle, then asserts that it extracts the right ids/titles, asks the
 * background for exactly those, and renders the returned label as a badge.
 *
 * Needs jsdom, which is not a project dependency:
 *   mkdir -p /tmp/wsb-domtest && cd /tmp/wsb-domtest && npm init -y && npm i jsdom
 *   NODE_PATH=/tmp/wsb-domtest/node_modules node tools/test-content.js
 */

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

let JSDOM;
try {
  ({ JSDOM } = require('jsdom'));
} catch (e) {
  console.error('jsdom not found - see the NODE_PATH note at the top of this file');
  process.exit(2);
}

const NEW_FEED = `
  <shreddit-post id="t3_1wvjeke" post-title="Google is the AI arms dealer"
                 permalink="/r/wallstreetbets/comments/1wvjeke/x/">
    <div slot="title"><a href="/r/wallstreetbets/comments/1wvjeke/x/">Google is the AI arms dealer</a></div>
    <div slot="text-body">Some long form body text.</div>
  </shreddit-post>`;

const OLD_FEED = `
  <div class="thing" data-fullname="t3_1wvqmur">
    <div class="entry">
      <p class="title"><a class="title" href="/r/wallstreetbets/comments/1wvqmur/nike/">What the fuck just happened to Nike?</a></p>
      <div class="usertext-body"><div class="md">tldr it is over</div></div>
    </div>
  </div>`;

/* no id attribute anywhere - the permalink anchor is the only way in */
const PERMALINK_ONLY = `
  <div data-testid="post-container">
    <a href="/r/wallstreetbets/comments/1wvoksp/rocket/">link</a>
    <h3>Some title in a heading</h3>
  </div>`;

/* a comments-page main post; its "N comments" link still points at the same thread */
const COMMENTS_PAGE = `
  <shreddit-post id="t3_1wvjeke" post-title="Google is the AI arms dealer">
    <div slot="title"><a href="/r/wallstreetbets/comments/1wvjeke/x/">Google is the AI arms dealer</a></div>
  </shreddit-post>
  <a href="/r/wallstreetbets/comments/1wvjeke/x/">128 Comments</a>`;

let failures = 0;
function check(name, cond, detail) {
  console.log(`  [${cond ? 'ok ' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`);
  if (!cond) failures++;
}

function makeEnv({ url, html, reply, autoReply = true }) {
  const dom = new JSDOM(html, { url, pretendToBeVisual: true });
  const sent = [];
  const ports = [];

  function makePort() {
    const listeners = { message: [], disconnect: [] };
    const port = {
      name: 'wsb-classifier',
      onMessage: { addListener: (f) => listeners.message.push(f) },
      onDisconnect: { addListener: (f) => listeners.disconnect.push(f) },
      postMessage(msg) {
        sent.push(msg);
        if (!autoReply) return;
        process.nextTick(() => {
          if (port.closed) return;
          for (const f of listeners.message) f(Object.assign({ seq: msg.seq }, reply(msg)));
        });
      },
      _disconnect() {
        port.closed = true;
        for (const f of listeners.disconnect) f();
      },
      closed: false
    };
    ports.push(port);
    return port;
  }

  const chrome = {
    storage: {
      local: {
        get(defaults, cb) {
          const out = {};
          for (const [k, v] of Object.entries(defaults || {})) {
            out[k] = k === 'subreddits' ? ['wallstreetbets'] : v;
          }
          process.nextTick(() => cb(out));
        }
      },
      onChanged: { addListener: () => {} }
    },
    runtime: {
      lastError: null,
      connect(info) {
        void info;
        return makePort();
      },
      sendMessage(msg, cb) {
        sent.push(msg);
        const p = Promise.resolve(reply(msg));
        if (typeof cb === 'function') p.then(cb);
        return p;
      }
    }
  };

  class IO {
    constructor(cb) { this.cb = cb; }
    observe(el) { process.nextTick(() => this.cb([{ isIntersecting: true, target: el }])); }
    unobserve() {}
    disconnect() {}
  }

  const win = dom.window;
  const sandbox = {
    chrome, console, setTimeout, clearTimeout, setInterval, clearInterval,
    document: win.document, location: win.location,
    IntersectionObserver: IO, MutationObserver: win.MutationObserver,
    Object, JSON, Math, String, Number, Array, Error, Promise, Set, Map,
    Boolean, RegExp, Date, parseInt, isNaN
  };
  sandbox.globalThis = sandbox;
  sandbox.window = win;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'content.js'), 'utf8'),
                  sandbox, { filename: 'content.js' });
  return { dom, sent, sandbox, ports };
}

const REC = (id, label, conf) => ({
  [id]: {
    id, label, confidence: conf, source: 'arctic-shift', arcticMiss: false,
    probabilities: { NOISE_MEME: 0.1, GENUINE_OPINION: 0.1,
                     FINANCIAL_CLAIM: 0.1, ANALYSIS_DD: conf },
    emptyInput: false
  }
});

const tick = (ms = 900) => new Promise((r) => setTimeout(r, ms));

(async () => {
  // ---------------------------------------------------------------- feed page
  console.log('1. www.reddit.com feed');
  {
    const { dom, sent } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${NEW_FEED}${OLD_FEED}${PERMALINK_ONLY}</body>`,
      reply: (msg) => ({ results: REC(msg.posts[0].id, 'ANALYSIS_DD', 0.71), errors: {} })
    });
    await tick();
    const asked = sent.flatMap((m) => (m.posts || []).map((p) => p.id));
    check('asked for all three ids', asked.length === 3, asked.join(','));
    check('shreddit-post id used', asked.includes('t3_1wvjeke'));
    check('old-reddit data-fullname used', asked.includes('t3_1wvqmur'));
    check('permalink fallback used', asked.includes('t3_1wvoksp'));
    const titles = sent.flatMap((m) => (m.posts || []).map((p) => p.domTitle));
    check('title read from post-title attr', titles.includes('Google is the AI arms dealer'));
    check('title read from a.title', titles.includes('What the fuck just happened to Nike?'));
    const bodies = sent.flatMap((m) => (m.posts || []).map((p) => p.domBody));
    check('body read from .usertext-body .md', bodies.includes('tldr it is over'));
    check('body read from slot=text-body', bodies.includes('Some long form body text.'));

    const badges = dom.window.document.querySelectorAll('.wsb-cls');
    check('a badge per post', badges.length === 3, String(badges.length));
    const one = dom.window.document.querySelector('#t3_1wvjeke .wsb-cls');
    check('badge carries the class', one && one.dataset.cls === 'ANALYSIS_DD',
          one && one.dataset.cls);
    check('badge shows confidence', one && one.textContent === 'analysis · 0.71',
          one && one.textContent);
    check('badge sits next to the title',
          !!dom.window.document.querySelector('[slot="title"] + .wsb-cls-slot'));

    // re-scanning must not multiply badges
    await tick(400);
    check('re-scan does not duplicate badges',
          dom.window.document.querySelectorAll('.wsb-cls').length === 3);
  }

  // -------------------------------------------------------------- comments page
  console.log('2. comments page');
  {
    const { dom, sent } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/comments/1wvjeke/google_is_the_ai_arms_dealer/',
      html: `<body>${COMMENTS_PAGE}</body>`,
      reply: (msg) => ({ results: REC('t3_1wvjeke', 'FINANCIAL_CLAIM', 0.83), errors: {} })
    });
    await tick();
    check('main post classified', sent.some((m) => m.posts.some((p) => p.id === 't3_1wvjeke')));
    const b = dom.window.document.querySelector('.wsb-cls');
    check('badge rendered', !!b);
    check('low confidence not flagged at 0.83', b && b.dataset.low !== '1');
  }

  // ------------------------------------------------------------ low confidence
  console.log('3. low confidence badge');
  {
    const { dom } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${NEW_FEED}</body>`,
      reply: () => ({ results: REC('t3_1wvjeke', 'GENUINE_OPINION', 0.31), errors: {} })
    });
    await tick();
    const b = dom.window.document.querySelector('.wsb-cls');
    check('uncertain label is outlined', b && b.dataset.low === '1', b && b.dataset.low);
  }

  // ------------------------------------------------------------- other subreddit
  console.log('4. not a watched subreddit');
  {
    const { sent } = makeEnv({
      url: 'https://www.reddit.com/r/stocks/',
      html: `<body>${NEW_FEED}</body>`,
      reply: () => ({ results: {}, errors: {} })
    });
    await tick();
    check('no request sent off-subreddit', sent.length === 0, String(sent.length));
  }

  // ---------------------------------------------------------------- offline api
  console.log('5. classifier offline');
  {
    const { dom } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${NEW_FEED}</body>`,
      reply: () => ({ results: {}, errors: { t3_1wvjeke: 'classifier unavailable - cannot reach host' } })
    });
    await tick();
    const b = dom.window.document.querySelector('.wsb-cls');
    check('error badge shown', b && b.dataset.state === 'error', b && b.dataset.state);
    check('no label invented', b && !b.dataset.cls, b && b.dataset.cls);
  }

  // ------------------------------------------------------------- archive miss
  console.log('6. post too new for the archive');
  {
    const { dom } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${NEW_FEED}</body>`,
      reply: () => ({ results: {
        t3_1wvjeke: { id: 't3_1wvjeke', label: 'NOISE_MEME', confidence: 0.6,
                      source: 'page-dom', arcticMiss: true, probabilities: null,
                      emptyInput: false }
      }, errors: { t3_1wvjeke: 'not in arctic-shift archive' } })
    });
    await tick();
    const b = dom.window.document.querySelector('.wsb-cls');
    check('dom-fallback label shown', b && b.dataset.cls === 'NOISE_MEME');
    check('tooltip discloses the fallback', b && /too new for the archive/.test(b.title),
          b && b.title.split('\n').pop());
  }

  // -------------------------------------------------------- unknown card markup
  // if Reddit renames shreddit-post, tier 1 finds nothing and tier 2 must carry it
  console.log('7. fallback tier for unknown post markup');
  {
    const UNKNOWN = `
      <div class="brand-new-post-card-v2">
        <a href="/r/wallstreetbets/comments/1wvpr04/mstr/">I bought MSTR at $532 in 2024</a>
        <span>4.2k upvotes</span>
      </div>
      <div class="comment">
        <a href="/r/wallstreetbets/comments/1wvpr04/mstr/">reply link</a>
      </div>`;
    const { dom, sent } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${UNKNOWN}</body>`,
      reply: (msg) => ({ results: REC(msg.posts[0].id, 'NOISE_MEME', 0.9), errors: {} })
    });
    await tick();
    const asked = sent.flatMap((m) => (m.posts || []).map((p) => p.id));
    check('permalink tier found the post', asked.includes('t3_1wvpr04'), asked.join(','));
    check('comment link not treated as a post', asked.length === 1, asked.join(','));
    const anchor = dom.window.document.querySelector(
      '.brand-new-post-card-v2 a[href*="/comments/"]');
    const nxt = anchor.nextElementSibling;
    check('badge placed next to the title', nxt && nxt.classList.contains('wsb-cls-slot'),
          nxt && nxt.className);
    check('badge rendered', !!dom.window.document.querySelector('.wsb-cls'));
  }

  // tier 1 must win when it is available, so badges land inside the real post element
  console.log('8. known markup takes precedence over the fallback');
  {
    const MIXED = `${NEW_FEED}
      <a href="/r/wallstreetbets/comments/1wvpr04/mstr/">MSTR</a>`;
    const { dom, sent } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${MIXED}</body>`,
      reply: (msg) => ({ results: REC(msg.posts[0].id, 'ANALYSIS_DD', 0.7), errors: {} })
    });
    await tick();
    const asked = sent.flatMap((m) => (m.posts || []).map((p) => p.id));
    check('only the shreddit-post was classified', asked.length === 1, asked.join(','));
    check('badge inside the post element',
          !!dom.window.document.querySelector('#t3_1wvjeke .wsb-cls'));
  }

  // a bot-check / login wall has no permalinks at all: say so instead of hanging
  console.log('9. no posts and no permalinks (bot wall)');
  {
    const { dom, sent } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: '<body><h1>Prove your humanity</h1><form></form></body>',
      reply: () => ({ results: {}, errors: {} })
    });
    await tick();
    check('no request sent', sent.length === 0, String(sent.length));
    check('no badge injected', dom.window.document.querySelectorAll('.wsb-cls').length === 0);
  }

  // ------------------------------------------------- background never answers
  console.log('10. background never answers (watchdog)');
  {
    const { dom } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${NEW_FEED}</body>`,
      autoReply: false                        // port never answers
    });
    await tick(1200);
    const early = dom.window.document.querySelector('.wsb-cls');
    check('shows a pending state while waiting', early && early.dataset.state === 'pending',
          early && early.dataset.state);

    await tick(24000);                        // RESPONSE_TIMEOUT_MS is 20s
    const late = dom.window.document.querySelector('.wsb-cls');
    check('stops claiming to be working', late && late.dataset.state === 'error',
          late && late.dataset.state);
    check('explains the timeout',
          late && /did not answer/.test(late.title), late && late.title);
  }

  // ------------------------------------------- message payload must be clonable
  // chrome.runtime port messaging serialises with the structured clone algorithm, which
  // throws DataCloneError on an HTMLElement. The queue holds DOM elements for later
  // badge updates, so the message must be a plain projection of it.
  console.log('11. outgoing port message is structured-cloneable');
  {
    const { sent } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${NEW_FEED}${OLD_FEED}</body>`,
      reply: () => ({ results: {}, errors: {} })
    });
    await tick();
    check('a classify message was posted', sent.length === 1, String(sent.length));
    let cloneErr = null;
    try {
      structuredClone(sent[0]);
    } catch (e) {
      cloneErr = e;
    }
    check('message survives structuredClone', !cloneErr,
          cloneErr ? `${cloneErr.name}: ${cloneErr.message}` : '');
    const p = sent[0].posts || [];
    check('no DOM node in the payload',
          p.every((x) => !x.host && !x.el && !x.node),
          JSON.stringify(Object.keys(p[0] || {})));
    check('payload has exactly the fields the background reads',
          p.every((x) => Object.keys(x).sort().join(',') === 'domBody,domTitle,id'),
          p[0] ? Object.keys(p[0]).sort().join(',') : '');
    check('carries a sequence number for reply matching',
          Number.isInteger(sent[0].seq) && sent[0].seq > 0, String(sent[0].seq));
  }

  // ------------------------------------------------ worker restart mid-request
  // An MV3 worker is evicted after ~30s idle and restarted on the next event, which
  // drops any port the content script was holding. The posts must be requeued and
  // answered by the new worker, not shown to the user as a failure.
  console.log('12. recovers when the worker restarts mid-request');
  {
    const { dom, sent, ports } = makeEnv({
      url: 'https://www.reddit.com/r/wallstreetbets/',
      html: `<body>${NEW_FEED}</body>`,
      autoReply: false
    });
    await tick();
    check('request is waiting on a port that never answers',
          sent.length === 1 && ports.length === 1,
          `sent=${sent.length} ports=${ports.length}`);

    ports[0]._disconnect();                       // worker evicted
    await tick(400);                             // reconnect backoff fires
    check('a replacement port was opened', ports.length === 2, String(ports.length));
    check('the waiting posts were requeued and resent',
          sent.length === 2, String(sent.length));
    check('the retry carries the same post, not a lost one',
          sent.length === 2 && sent[1].posts[0].id === sent[0].posts[0].id,
          sent.length === 2 ? sent[1].posts[0].id : '');

    // Wait past a watchdog tick (2s). A requeued post that is still marked inflight
    // with a stale timestamp gets reported as a 20s timeout almost immediately, which
    // is exactly what an evicted MV3 worker would look like to a user.
    await tick(3000);
    const badge = dom.window.document.querySelector('.wsb-cls');
    check('no error badge is shown for a worker restart',
          !badge || badge.dataset.state !== 'error',
          badge && badge.dataset.state);
  }

  console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})();
