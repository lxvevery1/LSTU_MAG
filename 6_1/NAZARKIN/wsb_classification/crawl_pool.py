#!/usr/bin/env python3
"""Resumable serial WSB crawler.

Writes newline-delimited JSON (one post per line) plus a progress file of
completed (year, window) pairs, so an interrupted run resumes where it stopped.
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
JSONL = "pool_big.jsonl"
PROGRESS = "pool_big.progress"
WINDOW_PAGES = int(os.environ.get("WINDOW_PAGES", "3"))
REQ_SLEEP = float(os.environ.get("REQ_SLEEP", "2.5"))
EXCLUDE = ("teachers_labeled_some_model.json", "teachers_to_label.json", "students_to_label.json")


def ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def fetch(after, before, retries=2):
    params = {"subreddit": SUBREDDIT, "after": after, "before": before, "limit": 100,
              "sort": "asc", "fields": FIELDS}
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, timeout=25)
            r.raise_for_status()
            return r.json().get("data") or []
        except Exception as e:
            if attempt == retries - 1:
                print(f"    fail {after}-{before}: {type(e).__name__}", file=sys.stderr, flush=True)
            time.sleep(3 * (attempt + 1))
    return []


def is_valid(p):
    title = (p.get("title") or "").strip()
    body = (p.get("selftext") or "").strip()
    if body.lower() in ("[removed]", "[deleted]"):
        return False
    return bool(title) and len(title) + len(body) >= 20


def load_excluded():
    ids = set()
    for f in EXCLUDE:
        if not os.path.exists(f):
            continue
        for y in json.load(open(f)).get("years", {}).values():
            for p in y:
                if p.get("id"):
                    ids.add(p["id"])
    return ids


def build_tasks():
    tasks = []
    for year in range(2019, 2027):
        for half in range(24):
            m = half // 2 + 1
            second = half % 2 == 1
            a = ts(f"{year}-{m:02d}-{16 if second else 1:02d}")
            ny = year + (1 if m == 12 else 0)
            b = ts(f"{ny}-{(m % 12) + 1:02d}-01")
            if b > a:
                tasks.append((f"{year}-h{half:02d}", a, b))
    return tasks


def main():
    excluded = load_excluded()
    done = set()
    if os.path.exists(PROGRESS):
        done = {l.strip() for l in open(PROGRESS) if l.strip()}
    tasks = build_tasks()
    todo = [t for t in tasks if t[0] not in done]
    print(f"{len(tasks)} windows, {len(done)} done, {len(todo)} remaining, "
          f"excluding {len(excluded)} labelled ids", file=sys.stderr, flush=True)

    out = open(JSONL, "a", encoding="utf-8")
    for i, (label, a, b) in enumerate(todo, 1):
        cursor, pages, kept = a, 0, 0
        while pages < WINDOW_PAGES and cursor < b:
            batch = fetch(cursor, b)
            if not batch:
                break
            pages += 1
            for p in batch:
                if p.get("id") in excluded or not is_valid(p):
                    continue
                p["_window"] = label
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
                kept += 1
            cursor = max((p.get("created_utc", cursor) for p in batch), default=cursor) + 1
            time.sleep(REQ_SLEEP)
        out.flush()
        with open(PROGRESS, "a") as pf:
            pf.write(label + "\n")
        print(f"[{i}/{len(todo)}] {label}: kept {kept}", file=sys.stderr, flush=True)
        time.sleep(0.1)
    out.close()
    print("DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
