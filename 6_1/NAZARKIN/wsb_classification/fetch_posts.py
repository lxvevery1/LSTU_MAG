#!/usr/bin/env python3
"""Fetch WallStreetBets posts from Arctic Shift API."""
import json
import sys
import time
from datetime import datetime, timezone

import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBREDDIT = "wallstreetbets"
FIELDS = "id,title,selftext,created_utc,score,num_comments,author,link_flair_text"


def fetch(after, before, limit=100, sort="desc", retries=4):
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
            r = requests.get(API, params=params, timeout=90)
            r.raise_for_status()
            return r.json().get("data") or []
        except (requests.exceptions.RequestException, ValueError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    print(f"warning: giving up on {after}-{before}: {last}", file=sys.stderr)
    return []


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def year_ranges(start_year=2019, end_year=2026):
    for y in range(start_year, end_year + 1):
        yield y, ts(f"{y}-01-01"), ts(f"{y+1}-01-01")


def is_valid(p):
    title = (p.get("title") or "").strip()
    body = (p.get("selftext") or "").strip()
    if not title and not body:
        return False
    if body.lower() in ("[removed]", "[deleted]"):
        body = ""
    return bool(title or body)


def collect_per_year(per_year, start_year=2019, end_year=2026):
    out = {}
    for year, a, b in year_ranges(start_year, end_year):
        posts = []
        seen = set()
        # sample across the year using monthly windows, walking backwards
        for month in range(12, 0, -1):
            ma = ts(f"{year}-{month:02d}-01")
            mb = ts(f"{year + (month // 12)}-{month % 12 + 1:02d}-01") if month < 12 else ts(f"{year+1}-01-01")
            batch = fetch(ma, mb, limit=100, sort="desc")
            for p in batch:
                if not is_valid(p) or p["id"] in seen:
                    continue
                seen.add(p["id"])
                posts.append(p)
            time.sleep(0.2)
        posts.sort(key=lambda p: p["created_utc"], reverse=True)
        out[year] = posts[:per_year]
        print(f"{year}: fetched {len(out[year])} valid posts", file=sys.stderr)
    return out


def collect_pool(limit=1000, start_year=2019, end_year=2026):
    """Collect a larger mixed pool for student distillation/training."""
    pool = []
    for year, a, b in year_ranges(start_year, end_year):
        for month in range(12, 0, -1):
            ma = ts(f"{year}-{month:02d}-01")
            mb = ts(f"{year + (month // 12)}-{month % 12 + 1:02d}-01") if month < 12 else ts(f"{year+1}-01-01")
            batch = fetch(ma, mb, limit=100, sort="desc")
            for p in batch:
                if is_valid(p):
                    pool.append(p)
            time.sleep(0.15)
    # dedupe
    seen = set()
    uniq = []
    for p in pool:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        uniq.append(p)
    return uniq[:limit]


if __name__ == "__main__":
    per_year = 15
    data = collect_per_year(per_year)
    with open("teachers_raw.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    total = sum(len(v) for v in data.values())
    print(f"Saved {total} teacher candidate posts to teachers_raw.json", file=sys.stderr)
