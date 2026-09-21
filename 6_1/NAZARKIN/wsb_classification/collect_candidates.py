#!/usr/bin/env python3
"""Collect teacher candidates: posts with meaningful text, spread across years."""
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
        except (requests.exceptions.RequestException, ValueError) as e:
            time.sleep(2 * (attempt + 1))
    return []


def clean(p):
    body = (p.get("selftext") or "").strip()
    if body.lower() in ("[removed]", "[deleted]"):
        body = ""
    return (p.get("title") or "").strip(), body


def quality(p):
    title, body = clean(p)
    if body:
        return 2
    if title and title not in ("[deleted by user]", "[ Removed by moderator ]"):
        return 1
    return 0


def month_windows(year, reverse=True):
    months = list(range(12, 0, -1)) if reverse else list(range(1, 13))
    for m in months:
        yield ts(f"{year}-{m:02d}-01"), ts(f"{year+1}-01-01") if m == 12 else ts(f"{year}-{m+1:02d}-01")


def collect(per_year=15, start=2019, end=2026):
    out = {}
    seen = set()
    for year in range(start, end + 1):
        selected = []
        for a, b in month_windows(year):
            batch = fetch(a, b, limit=100, sort="desc")
            batch.sort(key=lambda p: (quality(p), p.get("score", 0)), reverse=True)
            for p in batch:
                if quality(p) == 0 or p["id"] in seen:
                    continue
                seen.add(p["id"])
                selected.append(p)
            if len(selected) >= per_year * 3:
                break
            time.sleep(0.15)
        selected.sort(key=lambda p: (quality(p), p.get("score", 0)), reverse=True)
        out[year] = selected[:per_year]
        print(f"{year}: {len(out[year])}", file=sys.stderr)
    return out


if __name__ == "__main__":
    data = collect()
    with open("teachers_candidates.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("saved", sum(len(v) for v in data.values()))
