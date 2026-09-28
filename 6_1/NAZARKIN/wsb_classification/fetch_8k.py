#!/usr/bin/env python3
"""Fetch ~1000 WallStreetBets posts per year (2019-2026) from Arctic Shift API."""
import json
import sys
import time
from datetime import datetime, timezone

import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBREDDIT = "wallstreetbets"
FIELDS = "id,title,selftext,created_utc,score,num_comments,author,link_flair_text"
PER_YEAR = 1000
OUT = "posts_8k_raw.json"


def fetch(after, before, limit=100, sort="asc", retries=5):
    params = {
        "subreddit": SUBREDDIT,
        "after": after,
        "before": before,
        "limit": limit,
        "sort": sort,
        "fields": FIELDS,
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, timeout=120)
            r.raise_for_status()
            return r.json().get("data") or []
        except (requests.exceptions.RequestException, ValueError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    print(f"warning: gave up on window {after}-{before}: {last}", file=sys.stderr)
    return []


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def is_valid(p):
    title = (p.get("title") or "").strip()
    body = (p.get("selftext") or "").strip()
    if body.lower() in ("[removed]", "[deleted]"):
        body = ""
    if len(title) + len(body) < 15:
        return False
    return bool(title or body)


def collect_year(year, target=PER_YEAR):
    """Walk the year month by month (ascending) until target reached."""
    a = ts(f"{year}-01-01")
    b = ts(f"{year+1}-01-01")
    posts, seen = [], set()
    cursor = a
    guard = 0
    while cursor < b and len(posts) < target and guard < 200:
        guard += 1
        batch = fetch(cursor, b, limit=100, sort="asc")
        if not batch:
            # advance a day and retry a few times
            cursor += 86400 * 7
            continue
        added = 0
        for p in batch:
            if p.get("id") in seen or not is_valid(p):
                continue
            seen.add(p["id"])
            posts.append(p)
            added += 1
        if len(batch) < 100:
            # exhausted this window, jump forward
            last_t = max((p.get("created_utc", cursor) for p in batch), default=cursor)
            cursor = last_t + 1
        else:
            last_t = max((p.get("created_utc", cursor) for p in batch), default=cursor)
            cursor = last_t + 1
        time.sleep(0.2)
        if added == 0 and len(batch) == 100:
            # no valid posts in a full page, keep walking
            pass
    posts.sort(key=lambda p: p.get("created_utc", 0))
    return posts[:target]


def main():
    data = {}
    total = 0
    for year in range(2019, 2027):
        posts = collect_year(year)
        data[str(year)] = posts
        total += len(posts)
        print(f"{year}: {len(posts)} posts (running total {total})", file=sys.stderr)
    json.dump(data, open(OUT, "w"), ensure_ascii=False)
    print(f"saved {total} posts to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
