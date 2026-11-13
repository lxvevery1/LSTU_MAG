#!/usr/bin/env python3
"""Collect extra teacher candidates for rare classes across 2019-2026."""
import json
import sys
import time
from datetime import datetime, timezone

import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBREDDIT = "wallstreetbets"
FIELDS = "id,title,selftext,created_utc,score,num_comments,author,link_flair_text"


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch(after, before, limit=100, sort="desc", retries=4):
    params = dict(subreddit=SUBREDDIT, after=after, before=before,
                  limit=limit, sort=sort, fields=FIELDS)
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, timeout=90)
            r.raise_for_status()
            return r.json().get("data") or []
        except (requests.exceptions.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return []


def clean(p):
    body = (p.get("selftext") or "").strip()
    if body.lower() in ("[removed]", "[deleted]"):
        body = ""
    title = (p.get("title") or "").strip()
    if title in ("[deleted by user]", "[ Removed by moderator ]"):
        title = ""
    return title, body


def main():
    seen = set(json.load(open("_seen_ids.json")))
    targets = {"DD", "Fundamentals", "Technical Analysis", "Technicals", "Options",
               "Discussion", "News", "Chart", "Earnings Thread"}
    out = {}
    for year in range(2019, 2027):
        picked = []
        for m in range(12, 0, -1):
            a = ts(f"{year}-{m:02d}-01")
            b = ts(f"{year+1}-01-01") if m == 12 else ts(f"{year}-{m+1:02d}-01")
            batch = fetch(a, b, limit=100, sort="desc")
            batch.sort(key=lambda p: (1 if clean(p)[1] else 0, p.get("score", 0)), reverse=True)
            for p in batch:
                if p["id"] in seen:
                    continue
                title, body = clean(p)
                if not title and not body:
                    continue
                if not body:
                    continue
                seen.add(p["id"])
                picked.append(p)
            if len(picked) >= 80:
                break
            time.sleep(0.1)
        out[str(year)] = picked[:80]
        print(f"{year}: {len(out[str(year)])}", file=sys.stderr)

    json.dump(out, open("teachers_extra_raw.json", "w"), ensure_ascii=False, indent=2)
    json.dump(sorted(seen), open("_seen_ids.json", "w"))
    print("saved", sum(len(v) for v in out.values()))


if __name__ == "__main__":
    main()
