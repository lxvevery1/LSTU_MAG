#!/usr/bin/env python3
"""Fetch a large WSB candidate pool per year for balanced 5-class dataset construction.

Excludes posts already used as labelled training data (teachers/students) to avoid leakage.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBREDDIT = "wallstreetbets"
FIELDS = "id,title,selftext,created_utc,score,num_comments,author,link_flair_text"
PER_YEAR = int(os.environ.get("PER_YEAR", "10000"))
OUT = os.environ.get("OUT", "pool_10k_raw.json")
EXCLUDE = [
    "teachers_labeled_some_model.json",
    "teachers_to_label.json",
    "students_to_label.json",
]


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def load_excluded():
    ids = set()
    for f in EXCLUDE:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for y in d.get("years", {}):
            for p in d["years"][y]:
                if p.get("id"):
                    ids.add(p["id"])
    return ids


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
    print(f"warning: window {after}-{before} failed: {last}", file=sys.stderr)
    return []


def is_valid(p):
    title = (p.get("title") or "").strip()
    body = (p.get("selftext") or "").strip()
    if body.lower() in ("[removed]", "[deleted]"):
        return False
    if len(title) + len(body) < 20:
        return False
    if not title:
        return False
    return True


def collect_year(year, target, excluded):
    a, b = ts(f"{year}-01-01"), ts(f"{year+1}-01-01")
    cursor, posts, seen, guard, stalls = a, [], set(), 0, 0
    while cursor < b and len(posts) < target and guard < 400:
        guard += 1
        batch = fetch(cursor, b, limit=100, sort="asc")
        if not batch:
            cursor += 86400 * 3
            stalls += 1
            if stalls > 12:
                break
            time.sleep(1.0)
            continue
        stalls = 0
        added = 0
        for p in batch:
            pid = p.get("id")
            if not pid or pid in seen or pid in excluded or not is_valid(p):
                continue
            seen.add(pid)
            posts.append(p)
            added += 1
        last_t = max((p.get("created_utc", cursor) for p in batch), default=cursor)
        cursor = last_t + 1
        time.sleep(0.1)
        if guard % 20 == 0:
            print(f"  {year}: {len(posts)}/{target}", file=sys.stderr)
    posts.sort(key=lambda p: p.get("created_utc", 0))
    return posts[:target]


def main():
    excluded = load_excluded()
    print(f"excluding {len(excluded)} already-labelled ids", file=sys.stderr)
    data, total = {}, 0
    for year in range(2019, 2027):
        posts = collect_year(year, PER_YEAR, excluded)
        data[str(year)] = posts
        total += len(posts)
        print(f"{year}: {len(posts)} posts (total {total})", file=sys.stderr)
        json.dump(data, open(OUT, "w"), ensure_ascii=False)
    print(f"saved {total} posts -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
