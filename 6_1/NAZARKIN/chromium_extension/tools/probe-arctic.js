// Probe arctic-shift from the extension origin at several id counts, to find where it
// stalls. Run: node tools/probe-arctic.js
const path = require('path');
let chromium;
try { ({ chromium } = require(process.env.PW || 'playwright-core')); } catch (e) {
  console.error('playwright-core not found; set PW=<path>'); process.exit(2);
}
const fs = require('fs');
const os = require('os');
const EXT = path.join(__dirname, '..');
const CHROME = process.env.CHROME_PATH || '/usr/bin/google-chrome-stable';
const BASE = 'https://arctic-shift.photon-reddit.com/api/posts/ids?ids=';
const ALL = ['1wvjeke','1wvqnzs','1wvoksp','1wvoxxb','1wvpgqt','1wvpr04','1wvqn61','1wvqmur']
  .map((x) => 't3_' + x);

(async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wsb-probe-'));
  const ctx = await chromium.launchPersistentContext(dir, {
    executablePath: CHROME,
    headless: process.argv.includes('--headless'),
    args: [`--disable-extensions-except=${EXT}`, `--load-extension=${EXT}`,
           '--no-sandbox', '--disable-dev-shm-usage']
  });
  let worker = ctx.serviceWorkers()[0] || await ctx.waitForEvent('serviceworker', { timeout: 20000 });

  for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) {
    const ids = ALL.slice(0, n).join(',');
    const r = await worker.evaluate(async ({ url }) => {
      const t0 = Date.now();
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 20000);
      try {
        const res = await fetch(url, { headers: { Accept: 'application/json' }, signal: ctrl.signal });
        const text = await res.text();
        clearTimeout(timer);
        let n2 = null, bad = null;
        try { n2 = (JSON.parse(text).data || []).length; } catch (e) { bad = 'bad json'; }
        return { ms: Date.now() - t0, status: res.status, bytes: text.length, rows: n2, bad };
      } catch (e) {
        clearTimeout(timer);
        return { ms: Date.now() - t0, error: e.name + ': ' + e.message };
      }
    }, { url: BASE + ids });
    console.log(`n=${n}  ${JSON.stringify(r)}`);
  }
  await ctx.close();
})();
