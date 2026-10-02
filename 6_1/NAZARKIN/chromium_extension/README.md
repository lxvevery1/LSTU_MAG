# WSB Post Classifier — browser extension

A Chromium/Firefox extension that labels every r/WallStreetBets post with one of the
four classes the model in [`api_model/`](api_model/README.md) predicts:

| class | badge | meaning |
|---|---|---|
| `NOISE_MEME` | grey | joke, meme, reaction, P&L screenshot, process chatter |
| `GENUINE_OPINION` | blue | bullish/bearish/trading stance, no analysis |
| `FINANCIAL_CLAIM` | amber | concrete checkable fact about a company or market |
| `ANALYSIS_DD` | green | substantive reasoning, valuation, catalysts, risks |

The badge shows the short name plus the confidence (`analysis · 0.62`). Hovering
lists all four probabilities and where the text came from; clicking logs the full
record to the console.

## How a post gets classified

1. The content script finds post cards on screen and extracts the base-36 post id.
2. Only cards intersecting the viewport are queued, debounced 300 ms, so one
   scroll position produces **one** batch.
3. The background worker fetches every queued post in a single request to
   `https://arctic-shift.photon-reddit.com/api/posts/ids?ids=t3_a,t3_b,…`
4. It sends the whole batch in one `POST /predict` to the local server.
5. Results are cached in `chrome.storage.local` and rendered as badges.

Measured on a live feed: 12 posts → 2 network calls → 2.5 s total, and 0 ms for every
subsequent visit to that viewport.

```
reddit.com/r/wallstreetbets
        │  content.js   viewport scan, id extraction, badge injection
        ▼
   runtime.sendMessage  {posts:[{id, domTitle, domBody}]}
        │
   background.js        batched archive fetch  →  batched /predict  →  cache
        │                        │                        │
arctic-shift …/ids?ids=…   127.0.0.1:8000/predict   storage.local
                                 │
                        api_model/server.py  →  model.joblib
```

### Finding the post cards

Discovery is two-tier, because `shreddit-post` and `div.thing` are element names Reddit
can change without warning:

- **Tier 1** — the known markup: `shreddit-post` (www) or `div.thing[data-fullname]`
  (old.reddit), with the id read from `id="t3_…"`, `data-fullname`, or the permalink.
- **Tier 2** — if tier 1 finds nothing anywhere on the page, fall back to the
  `/comments/<id>/` permalink links themselves and hang the badge off the link, skipping
  anything inside a comment container. A post card always links to its thread, so this
  survives a rename; the cost is that the post is classified from its title only.

If neither tier finds anything, the content script logs a one-shot warning naming the
tags present on the page — that is the signal that you are looking at a bot check or a
login wall rather than a selector bug.

## Setup

### 1. Install the Python dependencies

```bash
python3 -m pip install scikit-learn joblib
```

`api_model/smoke_test.py` recomputes the model card's headline numbers on 1600
held-out posts, so it needs the labelled data one directory up:

```bash
cp ../wsb_classification/pickle_classification_for_learning.json .
python3 api_model/smoke_test.py     # 30 checks, exits non-zero on failure
```

### 2. Start the model server

```bash
python3 api_model/server.py --host 127.0.0.1 --port 8000
```

Keep it running — the extension shows a `classifier offline` badge otherwise.
Verify it:

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","version":"1.1.0","model":"model.joblib","classes":[...]}
```

If you use a different port, change it in the extension's options page.

### 3. Load the extension

Load this directory (`chromium_extension/`) as an unpacked extension.

**Chromium** — `chrome://extensions` → enable *Developer mode* → *Load unpacked* →
select this folder.

**Firefox** — `about:debugging#/runtime/this-firefox` → *Load Temporary Add-on* →
pick `manifest.json`. Firefox honours `background.scripts` and ignores
`background.service_worker`; Chromium does the reverse. Both are declared, so a single
manifest serves both. Firefox 121+ is required.

### 4. Open a post

<https://www.reddit.com/r/wallstreetbets/comments/1wvjeke/google_is_the_ai_arms_dealer/>

A badge appears under the title. On the subreddit listing, badges appear on every post
in view as you scroll.

## Options

Open the extension's options page to change:

- **Base URL of api_model/server.py** — default `http://127.0.0.1:8000`
- **Arctic Shift base URL** — the post text source
- **Subreddits** — defaults to `wallstreetbets`
- **Low-confidence threshold** — labels below it are drawn as an outline instead of a
  filled pill (default `0.45`)
- **Test API** / **Clear cache** buttons

## How the text is prepared

`background.js` normalises the title and body with exactly the training-time function
from `wsb_classification/build_sample_8k.py`:

```js
String(s).split(/\s+/).filter(Boolean).join(' ')   // == " ".join(s.split()) in Python
```

Nothing else is stripped. The training corpus still contained markdown links, bare
URLs and quote markers, so removing them here would shift the input distribution away
from what the model was fitted on. On the example post above, normalising raised
confidence from 0.38 to 0.62.

## Failure behaviour

| situation | what you see |
|---|---|
| post newer than the archive | classified from the on-page text, tooltip says so, dashed badge |
| server not running | `classifier offline`, no label invented |
| archive unreachable | same as above, falls back to page text |
| background never answers | `classifier offline` after 20 s, not an endless spinner |
| no text at all | `no text` |
| label below the threshold | dashed outline instead of a filled pill |
| bot check or login wall | nothing rendered, one console warning naming the tags on the page |

A label is never invented when the classifier did not answer.

## Tests

```bash
# model: 30 checks incl. the held-out score from the model card
python3 api_model/smoke_test.py

# background worker against the real archive + real server (server must be running)
node tools/dryrun.js t3_1wvjeke t3_1wvqnzs

# content script DOM behaviour; needs jsdom
mkdir -p /tmp/wsb-domtest && cd /tmp/wsb-domtest && npm init -y && npm i jsdom
cd - && NODE_PATH=/tmp/wsb-domtest/node_modules node tools/test-content.js

# real Chromium, real extension; needs playwright-core and a Chrome binary
node tools/browser-check.js            # serves a fixture on the reddit.com origin
node tools/browser-check.js --live     # point it at the real site
```

## What is verified, and what is not

| | status |
|---|---|
| model reproduces its own held-out score (acc 0.7106, macro-F1 0.6264) | verified, 30/30 |
| archive → normalise → batched `/predict` → cache, against the live APIs | verified, 12 posts in 2 calls / 2.5 s |
| DOM discovery, badge placement, cache, error and timeout paths | verified, 32/32 in jsdom |
| manifest accepted by Chrome 131, MV3 service worker starts | verified |
| worker fetches `127.0.0.1:8000` cross-origin without CORS headers | verified in Chrome |
| worker fetches arctic-shift from a `chrome-extension://` origin | verified in Chrome |
| **the live r/WallStreetBets markup** | **not verified** |

Two things are genuinely open, both environmental rather than code:

1. **The live Reddit DOM is unconfirmed.** Reddit serves a "Prove your humanity" bot
   challenge to every non-browser client and to headless browsers on this machine, so the
   real post markup could not be inspected. Tier 2 discovery exists precisely because of
   this. Load the extension and check the console for the tier-1-missed warning; if the
   badge lands somewhere odd, `placeBadge()` is the only function to change.
2. **The content-script → worker message hop is untested end-to-end.** Headless Chrome
   does not round-trip `runtime.sendMessage` to an MV3 service worker — a three-line
   control extension fails identically here, while calling the handler directly inside
   the worker succeeds. The same path is covered in jsdom, and
   `tools/browser-check.js` re-checks it as a preflight so a headed run verifies it.

## Limits worth knowing

- The model was fitted on posts, **not comments** — comments are never labelled.
- `GENUINE_OPINION` is the weak class (F1 0.42) and the model overfits
  (train accuracy 0.958 vs 0.711 held out), so treat a high-confidence badge as
  suggestive, not authoritative. Use macro-F1, not accuracy, to judge model changes.
- Real r/WallStreetBets posts are image- and link-heavy, so most have a short or empty
  body. Expect `NOISE_MEME` to dominate the feed, exactly as it dominates the
  training data (58%).
