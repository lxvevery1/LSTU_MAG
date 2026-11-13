#!/usr/bin/env python3
"""Collect 900 more WSB posts evenly across 2019-2026, excluding already-seen ids."""
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


def quality(p):
    title = (p.get("title") or "").strip()
    body = (p.get("selftext") or "").strip()
    if body.lower() in ("[removed]", "[deleted]"):
        body = ""
    if title in ("[deleted by user]", "[ Removed by moderator ]", ""):
        title = ""
    if not title and not body:
        return 0
    if body:
        return 2
    return 1


def sample_year(year, need, seen):
    """Pull posts across the year preferring ones with body text, no duplicates."""
    picked = []
    months = list(range(12, 0, -1))
    shuffle_order = months
    for m in shuffle_order:
        a = ts(f"{year}-{m:02d}-01")
        b = ts(f"{year+1}-01-01") if m == 12 else ts(f"{year}-{m+1:02d}-01")
        batch = fetch(a, b, limit=100, sort="desc")
        batch.sort(key=lambda p: (quality(p), p.get("score", 0)), reverse=True)
        for p in batch:
            if p["id"] in seen or quality(p) == 0:
                continue
            seen.add(p["id"])
            picked.append(p)
        if len(picked) >= need:
            break
        time.sleep(0.1)
    return picked


def main():
    total_need = 900
    start, end = 2019, 2026
    years = end - start + 1
    per_year = total_need // years
    remainder = total_need - per_year * years

    try:
        seen = set(json.load(open("_seen_ids.json")))
    except Exception:
        seen = set()

    out = {}
    for i, year in enumerate(range(start, end + 1)):
        need = per_year + (1 if i < remainder else 0)
        picked = sample_year(year, need, seen)
        out[str(year)] = picked[:need]
        print(f"{year}: {len(out[str(year)])}", file=sys.stderr)

    json.dump(out, open("students_raw.json", "w"), ensure_ascii=False, indent=2)
    json.dump(sorted(seen), open("_seen_ids.json", "w"))
    print("saved", sum(len(v) for v in out.values()))


if __name__ == "__main__":
    main()
