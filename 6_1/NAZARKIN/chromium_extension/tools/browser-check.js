/*
 * Real-browser check for the extension.
 *
 * Loads the unpacked extension into a real Chromium and verifies the parts jsdom
 * cannot: that the manifest is accepted, the MV3 service worker starts, the content
 * script is injected on the reddit.com origin, the background worker can reach both
 * arctic-shift and the local model server, and badges actually render.
 *
 * By default it serves a Reddit-shaped fixture on the https://www.reddit.com origin via
 * request interception, so the test is hermetic and never touches Reddit. Pass --live to
 * point it at the real site instead (run this from your own machine - Reddit serves a
 * bot check to headless browsers and to logged-out clients, so --live will usually be
 * blocked here).
 *
 *   node tools/browser-check.js            # fixture
 *   node tools/browser-check.js --live     # real reddit.com
 *
 * Requires: playwright-core, a Chromium binary, and api_model/server.py on :8000
 */

'use strict';

const path = require('path');
const os = require('os');
const fs = require('fs');

const PW = process.env.PW || '/tmp/opencode/browsertest/node_modules/playwright-core';
let chromium;
try {
  ({ chromium } = require(PW));
} catch (e) {
  console.error('playwright-core not found; set PW=<path to playwright-core>');
  process.exit(2);
}

const LIVE = process.argv.includes('--live');
// Headed by default: there is a real display in this environment, and the messaging
// checks are the ones worth running. Pass --headless only to reproduce that limitation
// deliberately.
const HEADLESS = process.argv.includes('--headless');
const EXT = path.join(__dirname, '..');
const CHROME = process.env.CHROME_PATH || '/usr/bin/google-chrome-stable';

const POSTS = [
  ['1wvjeke', 'Google is the AI arms dealer'],
  ['1wvqnzs', 'Jim Cramer is bearish on Nike'],
  ['1wvoksp', 'I am the meme Stonk King and I say we all jump in rocket'],
  ['1wvoxxb', 'BTC back at 86k'],
  ['1wvpgqt', 'US stamp from 1974'],
  ['1wvpr04', 'I bought MSTR at $532 in 2024. All in. I am the most royal of stonks'],
  ['1wvqn61', 'Jim Cramer is Bearish on Nike'],
  ['1wvqmur', 'What the fuck just happened to Nike?']
];

let fails = 0;
const check = (name, cond, detail) => {
  console.log(`  [${cond ? 'ok ' : 'FAIL'}] ${name}${detail ? ' - ' + detail : ''}`);
  if (!cond) fails++;
};

const shredditPost = ([id, title]) => `
  <shreddit-post id="t3_${id}" post-title="${title}" permalink="/r/wallstreetbets/comments/${id}/x/"
      author="someone" score="1234" comment-count="56">
    <div slot="title"><a class="post-title-link" href="/r/wallstreetbets/comments/${id}/x/">${title}</a></div>
    <div slot="credit-bar"><span>u/someone</span></div>
  </shreddit-post>`;

const shell = (inner) => `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Reddit</title></head><body>${inner}</body></html>`;

const feedFixture = () => shell(`<shreddit-feed>${POSTS.map(shredditPost).join('')}</shreddit-feed>`);
const postFixture = (id) => shell(`
  ${shredditPost(POSTS.find((p) => p[0] === id) || POSTS[0])}
  <shreddit-comment-tree><faceplate-tracker>
    <shreddit-comment thingid="t1_abc"><div slot="comment">a comment</div></shreddit-comment>
    <a href="/r/wallstreetbets/comments/${id}/x/">42 Comments</a>
  </faceplate-tracker></shreddit-comment-tree>`);

async function badges(page) {
  return page.$$eval('.wsb-cls', (els) => els.map((e) => {
    // the element placed after the title is the slot wrapper, not the span inside it
    const slot = e.closest('.wsb-cls-slot') || e;
    return {
      text: e.textContent,
      cls: e.dataset.cls || null,
      state: e.dataset.state || null,
      inPost: !!(e.closest && e.closest('shreddit-post')),
      // the badge slot is inserted as the *next sibling* of the title element
      nextToTitle: !!(slot.previousElementSibling &&
        slot.previousElementSibling.matches('[slot="title"]')),
      firstLine: (e.title || '').split('\n')[0],
      // full tooltip: an error badge puts the underlying reason here
      why: e.title || ''
    };
  }));
}

(async () => {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'wsb-prof-'));
  const ctx = await chromium.launchPersistentContext(userDataDir, {
    executablePath: CHROME,
    headless: HEADLESS,
    args: [
      `--disable-extensions-except=${EXT}`,
      `--load-extension=${EXT}`,
      '--no-sandbox',
      '--disable-dev-shm-usage'
    ]
  });

  try {
    if (!LIVE) {
      // serve our own fixture on the reddit.com origin so the content script matches
      await ctx.route('https://www.reddit.com/**', async (route) => {
        const u = route.request().url();
        const m = u.match(/\/comments\/([a-z0-9]+)/);
        if (u.includes('/r/stocks')) return route.fulfill({ status: 200,
          contentType: 'text/html', body: shell(POSTS.map(shredditPost).join('')) });
        return route.fulfill({
          status: 200,
          contentType: 'text/html',
          body: m ? postFixture(m[1]) : feedFixture()
        });
      });
      console.log('mode: fixture (hermetic, reddit.com origin, no traffic to Reddit)');
    } else {
      console.log('mode: live reddit.com (expect a bot check unless you are logged in)');
    }

    // 1. manifest accepted + MV3 worker running
    console.log('1. extension loads');
    let worker = ctx.serviceWorkers()[0];
    if (!worker) worker = await ctx.waitForEvent('serviceworker', { timeout: 20000 });
    check('MV3 service worker started', !!worker, worker && worker.url());
    if (!worker) throw new Error('no worker; the manifest was probably rejected');

    // 2. the background can reach the model server (cross-origin from the worker)
    console.log('2. background can reach the model server');
    const health = await worker.evaluate(async () => {
      const r = await fetch('http://127.0.0.1:8000/health');
      return { status: r.status, body: await r.text() };
    });
    check('fetch to 127.0.0.1:8000 allowed by host_permissions',
          health.status === 200 && /"ok"/.test(health.body), health.body.slice(0, 80));

    // 3. the background can reach arctic-shift (CORS + Cloudflare, from a real browser)
    console.log('3. background can reach arctic-shift');
    const arc = await worker.evaluate(async () => {
      try {
        const r = await fetch('https://arctic-shift.photon-reddit.com/api/posts/ids?ids=t3_1wvjeke');
        const j = await r.json();
        return { status: r.status, n: (j.data || []).length,
                 title: (j.data || [])[0] ? j.data[0].title : null };
      } catch (e) { return { error: String(e) }; }
    });
    check('arctic-shift fetch succeeds from the extension origin',
          arc.status === 200 && arc.n === 1, JSON.stringify(arc).slice(0, 120));

    // 4. transport preflight.
    // The extension talks to the worker over a long-lived port. One-shot sendMessage is
    // broken on this Chrome build (a three-line control extension fails identically, even
    // in a hand-launched browser over CDP), so the port is the transport that matters.
    console.log('4. port transport preflight');
    const extPage = await ctx.newPage();
    await extPage.goto(`chrome-extension://${new URL(worker.url()).host}/options.html`);
    const ping = await extPage.evaluate(() => new Promise((res) => {
      let p;
      try { p = chrome.runtime.connect({ name: 'wsb-classifier' }); }
      catch (e) { return res({ connectThrew: String(e) }); }
      const t = setTimeout(() => res({ timeout: true }), 8000);
      p.onMessage.addListener((v) => { clearTimeout(t); res({ value: v }); });
      p.postMessage({ seq: 1, type: 'WSB_HEALTH' });
    }));
    const transportOk = !!(ping && ping.value && ping.value.ok);
    if (!transportOk) {
      console.log(`  [!!] the port did not round-trip: ${JSON.stringify(ping)}`);
      console.log('');
      console.log('  The messaging-dependent checks (post page, feed, cache, off-subreddit)');
      console.log('  cannot run. tools/test-content.js covers the same content-script logic');
      console.log('  in jsdom, and can be run in the meantime.');
      console.log('');
      console.log('  Verified so far: manifest accepted, MV3 worker started, and the worker');
      console.log('  reaching both arctic-shift and the model server cross-origin.');
      await ctx.close();
      process.exit(3);
    }
    check('service worker answers over the port', true,
          JSON.stringify(ping.value.health || {}).slice(0, 70));

    // 4. a real post page
    console.log('5. post page');
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', (e) => errs.push(e.message));
    await page.goto('https://www.reddit.com/r/wallstreetbets/comments/1wvjeke/google_is_the_ai_arms_dealer/',
                    { waitUntil: 'domcontentloaded', timeout: 60000 });
    check('content script injected on reddit.com',
          await page.evaluate(() => !!document.querySelector('shreddit-post')));

    await page.waitForSelector('.wsb-cls', { timeout: 45000 }).catch(() => {});
    await page.waitForFunction(
      () => document.querySelector('.wsb-cls') &&
            document.querySelector('.wsb-cls').dataset.state !== 'pending',
      null, { timeout: 30000 }).catch(() => {});
    let b = await badges(page);
    console.log('      ', JSON.stringify(b));
    check('no content-script errors', errs.length === 0, errs.join('; '));
    check('a badge appeared', b.length >= 1, String(b.length));
    const okBadge = b.find((x) => x.state === 'ok');
    check('badge has a real model label', !!(okBadge && okBadge.cls), okBadge && okBadge.cls);
    check('label and confidence rendered',
          !!(okBadge && /^[a-z ]+ · \d\.\d\d$/.test(okBadge.text)), okBadge && okBadge.text);
    check('badge sits next to the post title', !!(okBadge && okBadge.nextToTitle),
          okBadge && `parent=${okBadge.nextToTitle}`);
    check('badge is inside the post element', !!(okBadge && okBadge.inPost));

    // 5. the feed: one batch for the whole viewport
    console.log('6. feed');
    const feed = await ctx.newPage();
    feed.on('pageerror', (e) => errs.push('feed: ' + e.message));
    await feed.goto('https://www.reddit.com/r/wallstreetbets/',
                    { waitUntil: 'domcontentloaded', timeout: 60000 });
    await feed.waitForFunction(
      () => document.querySelectorAll('.wsb-cls').length >= 3 &&
            [...document.querySelectorAll('.wsb-cls')]
              .every((e) => e.dataset.state !== 'pending'),
      null, { timeout: 45000 }).catch(() => {});
    let fb = await badges(feed);
    const okAll = fb.filter((x) => x.state === 'ok');
    console.log('      ', JSON.stringify(fb.map((x) => x.text)));
    if (fb.some((x) => x.state === 'error')) {
      console.log('       reasons:', JSON.stringify([...new Set(fb.map((x) => x.why))]));
    }
    check('several posts labelled at once', okAll.length >= 3, `${okAll.length} labelled`);
    check('all labels are model classes',
          okAll.every((x) => ['NOISE_MEME', 'GENUINE_OPINION', 'FINANCIAL_CLAIM',
                              'ANALYSIS_DD'].includes(x.cls)));
    check('no badge stuck pending/error',
          fb.every((x) => x.state === 'ok' || x.state === 'empty'),
          fb.map((x) => x.state).join(','));
    check('every feed badge is inside its post element', okAll.every((x) => x.inPost));

    // 6. the cache: a reload must be instant and identical
    console.log('7. cache on reload');
    const t0 = Date.now();
    await feed.reload({ waitUntil: 'domcontentloaded' });
    await feed.waitForFunction(
      () => document.querySelectorAll('.wsb-cls').length >= 3 &&
            [...document.querySelectorAll('.wsb-cls')]
              .every((e) => e.dataset.state !== 'pending'),
      null, { timeout: 30000 }).catch(() => {});
    const cachedMs = Date.now() - t0;
    const fb2 = await badges(feed);
    check('reload reproduces the same labels',
          JSON.stringify(fb2.map((x) => x.text)) === JSON.stringify(fb.map((x) => x.text)),
          `${cachedMs}ms`);

    // 7. the crawler is quiet off-subreddit
    console.log('8. unwatched subreddit');
    const other = await ctx.newPage();
    await other.goto('https://www.reddit.com/r/stocks/', { waitUntil: 'domcontentloaded' });
    await other.waitForTimeout(5000);
    check('no badges on /r/stocks', (await badges(other)).length === 0);

    console.log('9. no uncaught errors overall', errs.length === 0, errs.join('; '));
  } catch (e) {
    console.error('ERROR', e.message);
    fails++;
  } finally {
    await ctx.close();
  }

  console.log(fails ? `\n${fails} check(s) FAILED` : '\nall browser checks passed');
  process.exit(fails ? 1 : 0);
})();
