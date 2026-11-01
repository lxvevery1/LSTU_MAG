#!/usr/bin/env python3
"""Fetch r/WallStreetBets posts from the Arctic Shift API into JSON.

Arctic Shift is a Reddit archive with a public search API:
    https://arctic-shift.photon-reddit.com/api/posts/search

Example:
    python fetch_arctic_shift.py --after 2024-01-01 --before 2024-02-01 --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
FIELDS = "id,title,selftext,created_utc,score,num_comments,author,link_flair_text"


def timestamp(value: str) -> int:
    if value.isdigit():
        return int(value)
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch(params: dict, retries: int = 5) -> list:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(API, params=params, timeout=120)
            response.raise_for_status()
            return response.json().get("data") or []
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    print(f"warning: giving up after {retries} attempts: {last_error}", file=sys.stderr)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subreddit", default="wallstreetbets")
    ap.add_argument("--after", required=True, help="YYYY-MM-DD or unix timestamp")
    ap.add_argument("--before", required=True, help="YYYY-MM-DD or unix timestamp")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--out", default="data/arctic_posts.json")
    ap.add_argument("--fields", default=FIELDS)
    args = ap.parse_args()

    cursor = timestamp(args.after)
    end = timestamp(args.before)
    posts = []
    seen = set()
    while cursor < end and len(posts) < args.limit:
        batch = fetch({
            "subreddit": args.subreddit,
            "after": cursor,
            "before": end,
            "limit": args.page_size,
            "sort": "asc",
            "fields": args.fields,
        })
        if not batch:
            break
        added = 0
        for post in batch:
            post_id = post.get("id")
            if not post_id or post_id in seen:
                continue
            seen.add(post_id)
            posts.append(post)
            added += 1
            if len(posts) >= args.limit:
                break
        last = max(int(post.get("created_utc", cursor)) for post in batch)
        cursor = last + 1
        print(f"fetched {len(posts)}/{args.limit} (cursor {cursor})", file=sys.stderr)
        if added == 0 and len(batch) < args.page_size:
            break
        time.sleep(0.2)

    posts.sort(key=lambda post: int(post.get("created_utc", 0)))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(posts)} posts to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
