'use strict';

/*
 * Content script: finds post cards, asks the background worker to classify the ones
 * currently on screen, and renders a label next to each.
 *
 * Reddit re-creates post elements on every infinite-scroll page, so nothing here
 * trusts a "already handled" flag: a post counts as done only while its badge node
 * is actually in the document. If Reddit drops the node, the badge is re-rendered
 * from the local result map without a new network call.
 */

const POST_SELECTOR = [
  'shreddit-post',                              // www.reddit.com
  'div.thing[data-fullname^="t3_"]',            // old.reddit.com
  'div[data-testid="post-container"]'
].join(', ');

// Anything that is a comment rather than a post; used to keep the fallback below honest
const COMMENT_SELECTOR = 'shreddit-comment, .comment, faceplate-tracker, [data-testid="comment"]';

const SLOT_CLASS = 'wsb-cls-slot';

/*
 * Only the keys the content script reads. The values must mirror DEFAULTS in
 * background.js, which is what seeds storage on install; the content script keeps
 * its own copy so a failed or skipped install still behaves.
 */
const DEFAULTS = {
  enabled: true,
  subreddits: ['wallstreetbets'],
  lowConfidence: 0.45
};

const LABEL_SHORT = {
  NOISE_MEME: 'noise',
  GENUINE_OPINION: 'opinion',
  FINANCIAL_CLAIM: 'claim',
  ANALYSIS_DD: 'analysis'
};

// results already returned this page session, so a re-render costs no request
const results = new Map();
// posts sent to the background, awaiting a response: id -> {host, tier, since, ...}
const inflight = new Map();
const queued = new Map();

// If the background never answers, the badge must stop claiming to be working.
const RESPONSE_TIMEOUT_MS = 20000;
let watchdog = null;

let opts = null;
let sendTimer = null;
let scanTimer = null;
let io = null;
let ro = null;

// ----------------------------------------------------------------- extraction

const idFromHref = (href) => {
  const m = String(href || '').match(/\/comments\/([a-z0-9]+)/);
  return m ? 't3_' + m[1] : null;
};

function postIdOf(el) {
  const id = el.getAttribute('id');
  if (id && /^t3_[a-z0-9]+$/.test(id)) return id;
  const fn = el.getAttribute('data-fullname');
  if (fn && /^t3_[a-z0-9]+$/.test(fn)) return fn;
  // last resort: the permalink. On a post card this is the thread itself, and on a
  // comments page the "N comments" link resolves to the same thread, so both are safe
  const a = el.querySelector('a[href*="/comments/"]');
  return a ? idFromHref(a.getAttribute('href')) : null;
}

function domTitleOf(el) {
  const attr = el.getAttribute('post-title');
  if (attr) return attr.trim();
  const t = el.querySelector('a.title');                    // old reddit
  if (t) return t.textContent.trim();
  const slot = el.querySelector('[slot="title"]');
  if (slot) return slot.textContent.trim();
  const h = el.querySelector('h1, h2, h3');
  if (h) return h.textContent.trim();
  // the fallback tier anchors the badge to the permalink link, which on both layouts
  // *is* the post title
  if (el.tagName === 'A') return el.textContent.trim();
  return '';
}

function domBodyOf(el) {
  const b = el.querySelector('[slot="text-body"], .usertext-body .md, .expando .usertext-body');
  return b ? b.textContent : '';
}

/*
 * Post cards, in two tiers.
 *
 * Tier 1 is the known markup: shreddit-post on www, div.thing on old.reddit. Tier 2
 * exists because those are custom/legacy element names and Reddit redesigns: if tier 1
 * finds nothing on a watched subreddit, fall back to the permalink links themselves and
 * hang the badge off the link. A post card always links to /comments/<id>/, so this
 * keeps working through a rename, at the cost of a title-only classification.
 */
function discoverCards() {
  const tier1 = [...document.querySelectorAll(POST_SELECTOR)]
    .map((el) => ({ host: el, id: postIdOf(el), tier: 1 }))
    .filter((c) => c.id);
  if (tier1.length) return tier1;

  const seen = new Set();
  const tier2 = [];
  for (const a of document.querySelectorAll('a[href*="/comments/"]')) {
    if (a.closest(COMMENT_SELECTOR)) continue;              // not a post
    const id = idFromHref(a.getAttribute('href'));
    if (!id || seen.has(id)) continue;                     // title + "N comments" links
    seen.add(id);
    tier2.push({ host: a, id, tier: 2 });
  }
  return tier2;
}

function isPostPage() {
  return /\/(?:comments|[^/]+\/comments)\//.test(location.pathname);
}

function isWatchedSub() {
  const m = location.pathname.match(/^\/r\/([^/]+)/);
  if (!m) return false;
  return opts.subreddits.map((s) => String(s).toLowerCase()).includes(m[1].toLowerCase());
}

// -------------------------------------------------------------------- render

function tooltipFor(rec) {
  const lines = [`label: ${rec.label}`];
  if (rec.confidence != null) lines.push(`confidence: ${rec.confidence.toFixed(3)}`);
  if (rec.probabilities) {
    lines.push('');
    for (const [cls, p] of Object.entries(rec.probabilities)) {
      lines.push(`  ${cls.padEnd(16)} ${(p * 100).toFixed(1)}%`);
    }
  }
  lines.push('');
  lines.push('text source: ' + rec.source);
  if (rec.arcticMiss) {
    lines.push('this post is too new for the archive - used the text on the page');
  }
  if (rec.fromCache) lines.push('(cached)');
  return lines.join('\n');
}

function buildBadge(state, rec, errText) {
  const slot = document.createElement('div');
  slot.className = SLOT_CLASS;

  const badge = document.createElement('span');
  badge.className = 'wsb-cls';

  if (state === 'pending') {
    badge.dataset.state = 'pending';
    badge.textContent = 'classifying…';
  } else if (state === 'error') {
    badge.dataset.state = 'error';
    badge.textContent = 'classifier offline';
    badge.title = errText || '';
  } else if (state === 'empty') {
    badge.dataset.state = 'empty';
    badge.textContent = 'no text';
    badge.title = 'Neither the archive nor the page had any text to classify.';
  } else {
    badge.dataset.state = 'ok';
    badge.dataset.cls = rec.label;
    if (rec.confidence != null && rec.confidence < opts.lowConfidence) {
      badge.dataset.low = '1';
    }
    const pct = rec.confidence != null ? ' · ' + rec.confidence.toFixed(2) : '';
    badge.textContent = (LABEL_SHORT[rec.label] || rec.label) + pct;
    badge.title = tooltipFor(rec);
  }

  slot.appendChild(badge);
  slot.addEventListener('click', (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    if (rec) console.log('[wsb-classifier]', rec.id || '', rec);
  });
  return slot;
}

/*
 * Places the badge as a normal-flow sibling right after the post title, so it cannot
 * collide with Reddit's own positioning. Falls back to appending to the card, and for
 * the permalink-anchor tier the anchor is already the title, so the badge goes right
 * after it as a sibling.
 */
function placeBadge(host, slot, tier) {
  if (tier === 2) {
    host.parentNode.insertBefore(slot, host.nextSibling);
    return;
  }
  const titleSlot = host.querySelector('[slot="title"]');
  if (titleSlot && titleSlot.parentNode) {
    titleSlot.parentNode.insertBefore(slot, titleSlot.nextSibling);
    return;
  }
  const oldTitle = host.querySelector('a.title');
  if (oldTitle) {
    const p = oldTitle.closest('p.title') || oldTitle;
    p.parentNode.insertBefore(slot, p.nextSibling);
    return;
  }
  host.appendChild(slot);
}

/*
 * Badge node per host element. A WeakMap rather than a querySelector because in the
 * permalink tier the badge is a *sibling* of the host, not a descendant, and because
 * Reddit re-creates cards on scroll: isConnected is the honest test of "still shown".
 */
const badgeNodes = new WeakMap();

function existingBadge(host) {
  const n = badgeNodes.get(host);
  return n && n.isConnected ? n : null;
}

function render(host, state, rec, errText, tier) {
  const old = badgeNodes.get(host);
  if (old && old.remove) old.remove();
  badgeNodes.delete(host);
  if (state === 'skip') return;
  const slot = buildBadge(state, rec, errText);
  placeBadge(host, slot, tier);
  badgeNodes.set(host, slot);
}

function applyRecord(card, rec) {
  if (!rec.emptyInput) results.set(card.id, rec);
  render(card.host, rec.emptyInput ? 'empty' : 'ok', rec, null, card.tier);
}

// ------------------------------------------------------------------ transport

/*
 * A long-lived port to the background worker, not one-shot sendMessage.
 *
 * Two reasons. One-shot sendMessage to an MV3 service worker never completes on Chrome
 * 131 - it fails with "the message port closed before a response was received" even for
 * a three-line control extension in a hand-launched browser, so it is not a test
 * artefact. And a port keeps the worker alive across a burst of classifications rather
 * than letting it be evicted and restarted mid-request.
 */
let port = null;
let seq = 0;
const waiters = new Map();      // seq -> {posts, resolve, reject}
let reconnectTimer = null;

function openPort() {
  if (port) return port;
  try {
    port = chrome.runtime.connect({ name: 'wsb-classifier' });
  } catch (e) {
    port = null;
    return null;
  }

  port.onMessage.addListener((reply) => {
    const w = waiters.get(reply && reply.seq);
    if (!w) return;
    waiters.delete(reply.seq);
    w.resolve(reply);
  });

  port.onDisconnect.addListener(() => {
    port = null;
    // Nothing is outstanding once the port is gone, so these posts are no longer
    // inflight: drop them from the map before requeueing. Leaving them there with a
    // stale `since` would let the watchdog report a 20s timeout for a request that was
    // only ever alive for a few hundred milliseconds, which is what an evicted MV3
    // worker looks like from here. flush() re-adds them with a fresh timestamp.
    for (const [s, w] of waiters) {
      waiters.delete(s);
      for (const p of w.posts) {
        inflight.delete(p.id);
        if (!queued.has(p.id)) queued.set(p.id, p);
      }
      w.reject(new Error('port disconnected'));
    }
    if (queued.size && !sendTimer) {
      sendTimer = setTimeout(flush, 150);
    }
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(() => { reconnectTimer = null; openPort(); }, 400);
    }
  });

  return port;
}

function call(msg, originals) {
  return new Promise((resolve, reject) => {
    const p = openPort();
    if (!p) { reject(new Error('cannot reach the extension background worker')); return; }
    const s = ++seq;
    waiters.set(s, { resolve, reject, posts: originals || [] });
    try {
      // the payload must be plain data: structured clone throws on an HTMLElement
      p.postMessage(Object.assign({ seq: s }, msg));
    } catch (e) {
      waiters.delete(s);
      reject(e);
    }
  });
}

// ------------------------------------------------------------------ batching

function flush() {
  sendTimer = null;
  if (!queued.size) return;

  const posts = [...queued.values()];
  queued.clear();
  const since = Date.now();
  for (const p of posts) { p.since = since; inflight.set(p.id, p); }

  // The queue entries carry the DOM element so the badge can be updated later, but the
  // port serialises with structured clone, which throws DataCloneError on an
  // HTMLElement. Send a plain projection and keep the elements local.
  const payload = posts.map((p) => ({
    id: p.id, domTitle: p.domTitle, domBody: p.domBody
  }));

  armWatchdog();
  call({ type: 'WSB_CLASSIFY', posts: payload }, posts).then((reply) => {
    for (const p of posts) inflight.delete(p.id);
    const byId = (reply && reply.results) || {};
    const errs = (reply && reply.errors) || {};
    const globalErr = errs['_global'];

    for (const p of posts) {
      if (!p.host.isConnected) continue;
      const rec = byId[p.id];
      if (rec) {
        applyRecord(p, rec);
      } else {
        // never invent a label: if the classifier did not answer, say so
        const why = globalErr || errs[p.id] || 'no response from the classifier';
        render(p.host, p.domTitle || p.domBody ? 'error' : 'empty', null, why, p.tier);
      }
    }
  }).catch((e) => {
    // A rejected request usually means onDisconnect already requeued these posts, and
    // the retry will answer them. Only surface a failure for a post that is genuinely
    // not queued and not in flight, so a worker restart does not flash an error.
    for (const p of posts) {
      if (queued.has(p.id) || inflight.has(p.id)) continue;
      if (p.host.isConnected) render(p.host, 'error', null, String(e && e.message), p.tier);
    }
  });
}

/*
 * The badge says "classifying…" until an answer lands, so a dropped response would
 * otherwise leave a permanent lie on the page. After RESPONSE_TIMEOUT_MS anything still
 * waiting is reported as a failure.
 */
function armWatchdog() {
  if (watchdog) return;
  watchdog = setInterval(() => {
    if (!inflight.size) { clearInterval(watchdog); watchdog = null; return; }
    const cutoff = Date.now() - RESPONSE_TIMEOUT_MS;
    const stuck = [...inflight.values()].filter((p) => p.since <= cutoff);
    for (const p of stuck) {
      inflight.delete(p.id);
      if (p.host.isConnected) {
        render(p.host, 'error', null,
               'the extension did not answer within ' + (RESPONSE_TIMEOUT_MS / 1000) +
               's - the model server may be down', p.tier);
      }
    }
  }, 2000);
}

function request(card) {
  const id = card.id;
  if (inflight.has(id) || queued.has(id)) return;

  const domTitle = domTitleOf(card.host);
  const domBody = card.tier === 2 ? '' : domBodyOf(card.host);

  if (!domTitle && !domBody) {
    render(card.host, 'empty', null, null, card.tier);
    return;
  }
  const known = results.get(id);
  if (known) {
    render(card.host, 'ok', known, null, card.tier);
    return;
  }

  queued.set(id, { id, host: card.host, tier: card.tier, domTitle, domBody });
  if (!existingBadge(card.host)) render(card.host, 'pending', null, null, card.tier);
  if (!sendTimer) sendTimer = setTimeout(flush, 300);
}

// --------------------------------------------------------------------- scan

// host element -> its card descriptor, so the observer callback can find the tier
const cardByHost = new WeakMap();
let warnedEmpty = false;

function scan() {
  if (!isWatchedSub()) return;
  const cards = discoverCards();
  for (const c of cards) {
    cardByHost.set(c.host, c);
    if (existingBadge(c.host)) continue;
    io.observe(c.host);
  }
  if (!cards.length && !warnedEmpty) {
    warnedEmpty = true;
    // one-shot diagnostic: if Reddit's markup changed, say so where it can be found
    const tags = [...new Set([...document.querySelectorAll('body *')]
      .map((e) => e.tagName.toLowerCase()))].slice(0, 30);
    console.warn('[wsb-classifier] no post cards found on a watched subreddit page, ' +
                 'and no /comments/ permalinks either. Most likely Reddit is showing a ' +
                 'bot check or a login wall. Tags present: ' + tags.join(', '));
  }
}

function onIntersect(entries) {
  if (!isWatchedSub()) return;
  for (const en of entries) {
    if (!en.isIntersecting) continue;
    const host = en.target;
    io.unobserve(host);
    if (existingBadge(host)) continue;
    request(cardByHost.get(host) || { host, id: postIdOf(host), tier: 1 });
  }
}

function scheduleScan() {
  if (scanTimer) return;
  scanTimer = setTimeout(() => { scanTimer = null; scan(); }, 250);
}

// --------------------------------------------------------------------- start

function start() {
  io = new IntersectionObserver(onIntersect, { rootMargin: '250px 0px' });

  ro = new MutationObserver(scheduleScan);
  ro.observe(document.body, { childList: true, subtree: true });

  // a comments page has one main post, which is already in the viewport
  if (isPostPage()) {
    for (const c of discoverCards().slice(0, 1)) {
      cardByHost.set(c.host, c);
      request(c);
    }
  }

  scan();

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local' || !changes.apiBase && !changes.subreddits && !changes.enabled
        && !changes.lowConfidence) {
      return;
    }
    const again = () => {
      // wipe rendered badges, then re-render with the new settings
      for (const s of document.querySelectorAll('.' + SLOT_CLASS)) s.remove();
      results.clear();
      scan();
    };
    chrome.storage.local.get(DEFAULTS, (next) => { opts = next; again(); });
  });
}

chrome.storage.local.get(DEFAULTS, (stored) => {
  opts = stored;
  if (!opts.enabled || !isWatchedSub()) return;
  start();
});
