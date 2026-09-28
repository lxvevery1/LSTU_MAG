#!/usr/bin/env python3
"""Fast concurrent WSB pool collector: 24 half-month windows per year, threaded paging."""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBREDDIT = "wallstreetbets"
FIELDS = "id,title,selftext,created_utc,score,num_comments,author,link_flair_text"
PER_YEAR = int(os.environ.get("PER_YEAR", "12000"))
WINDOW_PAGES = int(os.environ.get("WINDOW_PAGES", "5"))
WORKERS = int(os.environ.get("WORKERS", "12"))
OUT = os.environ.get("OUT", "pool_big_raw.json")

_local = threading.local()
_print_lock = threading.Lock()
_req_lock = threading.Lock()
_last_req = [0.0]
MIN_INTERVAL = float(os.environ.get("MIN_INTERVAL", "0.2"))


def _throttle():
    with _req_lock:
        wait = MIN_INTERVAL - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=WORKERS * 2))
        _local.s = s
    return _local.s


def fetch(after, before, retries=5):
    params = {"subreddit": SUBREDDIT, "after": after, "before": before, "limit": 100,
              "sort": "asc", "fields": FIELDS}
    for attempt in range(retries):
        try:
            _throttle()
            r = session().get(API, params=params, timeout=60)
            r.raise_for_status()
            return r.json().get("data") or []
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def is_valid(p):
    title = (p.get("title") or "").strip()
    body = (p.get("selftext") or "").strip()
    if body.lower() in ("[removed]", "[deleted]"):
        return False
    if not title or len(title) + len(body) < 20:
        return False
    return True


def window(a, b, label):
    posts, cursor, pages, seen = [], a, 0, set()
    while pages < WINDOW_PAGES and cursor < b:
        batch = fetch(cursor, b)
        if not batch:
            break
        pages += 1
        for p in batch:
            pid = p.get("id")
            if pid in seen or not is_valid(p):
                continue
            seen.add(pid)
            posts.append(p)
        cursor = max((p.get("created_utc", cursor) for p in batch), default=cursor) + 1
    with _print_lock:
        print(f"  {label}: {len(posts)}", file=sys.stderr, flush=True)
    return posts


def load_excluded():
    ids = set()
    for f in ("teachers_labeled_some_model.json", "teachers_to_label.json", "students_to_label.json"):
        if not os.path.exists(f):
            continue
        for y in json.load(open(f)).get("years", {}).values():
            for p in y:
                if p.get("id"):
                    ids.add(p["id"])
    return ids


def collect_year(year):
    tasks = []
    for half in range(24):
        m = half // 2 + 1          # month 1..12
        second = half % 2 == 1    # first or second half of the month
        a = ts(f"{year}-{m:02d}-{16 if second else 1:02d}")
        if second:
            b = ts(f"{year + (1 if m == 12 else 0)}-{(m % 12) + 1:02d}-01")
        else:
            b = ts(f"{year + (1 if m == 12 else 0)}-{(m % 12) + 1:02d}-01")
        if b <= a:
            continue
        tasks.append((a, b, f"{year}-h{half:02d}"))

    out, seen = [], set()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(window, a, b, lbl): lbl for a, b, lbl in tasks}
        for f in as_completed(futs):
            for p in f.result():
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                out.append(p)
    out.sort(key=lambda p: p.get("created_utc", 0))
    return out[:PER_YEAR]


def main():
    excluded = load_excluded()
    print(f"excluding {len(excluded)} labelled ids; target {PER_YEAR}/year", file=sys.stderr)
    data, total = {}, 0
    for year in range(2019, 2027):
        posts = [p for p in collect_year(year) if p["id"] not in excluded]
        data[str(year)] = posts
        total += len(posts)
        print(f"{year}: {len(posts)} posts (total {total})", file=sys.stderr, flush=True)
        json.dump(data, open(OUT, "w"), ensure_ascii=False)
    print(f"saved {total} -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
