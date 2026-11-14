#!/usr/bin/env python3
"""Build the 16k WSB post corpus (TZ section 4.1).

The existing 8,000 hand-labelled posts are kept as-is.  For every year we add
new posts from a local pool that was collected from Arctic Shift, excluding
post IDs and near-duplicate texts that are already present.  New posts carry
no class label at this stage: ``label_posts.py`` applies the frozen FinBERT
classifier and creates the review queue.

Usage:
    python src/collect_posts.py
    python src/collect_posts.py --allow-live   # fetch missing years from API
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

from common import (
    load_config,
    normalize_text,
    project_path,
    read_json,
    read_jsonl,
    setup_logging,
    sha256_text,
    timestamp_to_utc,
    write_json,
)

LOG = logging.getLogger("collect_posts")
API = "https://arctic-shift.photon-reddit.com/api/posts/search"
FIELDS = "id,title,selftext,created_utc,score,num_comments,author,link_flair_text"


# ---------------------------------------------------------------- normalisation
def _iter_posts(obj) -> Iterable[dict]:
    """Walk the various pool JSON shapes and yield dicts that look like posts."""
    if isinstance(obj, dict):
        if "id" in obj and any(k in obj for k in ("created_utc", "created", "title")):
            yield obj
            return
        for value in obj.values():
            yield from _iter_posts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_posts(value)


def normalize_post(p: dict, source: str) -> dict | None:
    pid = str(p.get("id") or "").strip()
    created = p.get("created_utc", p.get("created"))
    if not pid or created is None:
        return None
    try:
        created = int(created)
    except (TypeError, ValueError):
        return None
    title = (p.get("title") or "").strip()
    body = (p.get("selftext") or p.get("body") or "").strip()
    if body.lower() in ("[deleted]", "[removed]", "[removed by moderator]"):
        body = ""
    author = (p.get("author") or "").strip()
    flair = (p.get("link_flair_text") or p.get("flair") or "").strip()
    return {
        "id": pid,
        "created_utc": created,
        "author_hash": sha256_text(author, "wsb16k-author:"),
        "score": int(p.get("score") or 0),
        "num_comments": int(p.get("num_comments") or 0),
        "flair": flair,
        "title": title,
        "body": body,
        "source_pool": source,
        "source_was_labelled": bool(p.get("label")),
    }


def is_valid(post: dict, bots: set[str], min_chars: int) -> bool:
    title = post["title"]
    body = post["body"]
    if not title and not body:
        return False
    if len(title) + len(body) < min_chars:
        return False
    # author is hashed in normalise_post; use the raw field only for rejection
    # before hashing is done in the caller where needed.
    return True


def _is_bot_raw(p: dict, bots: set[str]) -> bool:
    author = (p.get("author") or "").strip().lower()
    return author in bots or author == "automoderator"


# ---------------------------------------------------------------- near duplicates
SHINGLE_RE = re.compile(r"[a-z0-9$]+")


def _shingles(text: str, k: int = 3) -> list[str]:
    words = SHINGLE_RE.findall(text)
    if len(words) <= k:
        return [" ".join(words)]
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def simhash64(text: str, k: int = 3) -> int:
    vec = [0] * 64
    for sh in _shingles(text, k):
        h = int(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            vec[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if vec[bit] >= 0:
            out |= 1 << bit
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class NearDuplicateIndex:
    """Small SimHash LSH index; 4 bands of 16 bits."""

    def __init__(self, hamming: int = 3):
        self.hamming = hamming
        self.bands: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        self.hashes: list[int] = []

    @staticmethod
    def _bands(h: int) -> list[tuple[int, int]]:
        return [(i, (h >> (16 * i)) & 0xFFFF) for i in range(4)]

    def add(self, text: str) -> None:
        h = simhash64(text)
        self.hashes.append(h)
        for b in self._bands(h):
            self.bands[b].append(h)

    def is_near(self, text: str) -> bool:
        h = simhash64(text)
        seen = set()
        for b in self._bands(h):
            for other in self.bands.get(b, ()):
                if other in seen:
                    continue
                seen.add(other)
                if _hamming(h, other) <= self.hamming:
                    return True
        return False


# ---------------------------------------------------------------- loaders
def load_existing(cfg: dict) -> list[dict]:
    path = project_path(cfg["corpus"]["existing_posts"])
    obj = read_json(path)
    years = obj.get("years", obj)
    rows = []
    for arr in years.values():
        for p in arr:
            q = normalize_post({**p, "created_utc": p.get("created_utc", p.get("created"))}, "existing_manual")
            if q is None:
                continue
            q["class_label"] = p.get("label")
            q["label_source"] = "manual"
            q["class_probs"] = None
            q["confidence"] = 1.0
            q["review_status"] = "manual"
            rows.append(q)
    # dedupe by id (keep first)
    seen = set()
    out = []
    for p in rows:
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        out.append(p)
    return out


def load_pool_files(cfg: dict, extra_key: str = "pool_files") -> list[dict]:
    rows = []
    for rel in cfg["corpus"].get(extra_key, []):
        path = project_path(rel)
        if not path.exists():
            LOG.warning("pool file missing: %s", path)
            continue
        if path.suffix == ".jsonl":
            raw = read_jsonl(path)
        else:
            raw = list(_iter_posts(read_json(path)))
        for p in raw:
            q = normalize_post(p, path.stem)
            if q is not None:
                rows.append(q)
    return rows


def _valid_for_corpus(q: dict, bots: set[str], min_chars: int) -> bool:
    if q["author_hash"] == sha256_text("automoderator", "wsb16k-author:"):
        return False
    if q["author_hash"] == sha256_text("[deleted]", "wsb16k-author:"):
        return False
    return is_valid(q, bots, min_chars)


def fetch_window(session: requests.Session, after: int, before: int, limit: int = 20) -> list[dict]:
    params = {
        "subreddit": "wallstreetbets",
        "after": after,
        "before": before,
        "limit": limit,
        "sort": "asc",
        "fields": FIELDS,
    }
    for attempt in range(4):
        try:
            r = session.get(API, params=params, timeout=90)
            if r.status_code == 200:
                return r.json().get("data") or []
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return []
        except Exception as exc:  # noqa: BLE001
            LOG.debug("API retry: %s", exc)
            time.sleep(2 + attempt)
    return []


def collect_live(cfg: dict, deficits: dict[int, int], excluded_ids: set[str],
                 excluded_hashes: set[str], index: NearDuplicateIndex,
                 bots: set[str]) -> list[dict]:
    """Fetch random 6-hour windows for the years that still lack candidates."""
    rng = random.Random(int(cfg["project"]["seed"]) + 991)
    session = requests.Session()
    out = []
    for year, need in sorted(deficits.items()):
        if need <= 0:
            continue
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        if year == 2026:
            end = datetime(2026, 11, 15, tzinfo=timezone.utc)
        got = 0
        windows = max(8, int(need / 10) + 8)
        LOG.info("live collection %s: need %s, windows %s", year, need, windows)
        for _ in range(windows):
            if got >= need:
                break
            center = start + (end - start) * rng.random()
            before = center + __import__("datetime").timedelta(hours=3)
            after = center - __import__("datetime").timedelta(hours=3)
            batch = fetch_window(session, int(after.timestamp()), int(before.timestamp()))
            for p in batch:
                q = normalize_post(p, "api_extra")
                if q is None or q["id"] in excluded_ids:
                    continue
                if not _valid_for_corpus(q, bots, int(cfg["corpus"]["min_text_chars"])):
                    continue
                text = normalize_text(q["title"] + " " + q["body"])
                h = hashlib.sha256(text.encode()).hexdigest()
                if h in excluded_hashes or index.is_near(text):
                    continue
                excluded_ids.add(q["id"])
                excluded_hashes.add(h)
                index.add(text)
                out.append(q)
                got += 1
            time.sleep(0.25)
        LOG.info("live collection %s: got %s", year, got)
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-live", action="store_true",
                        help="fetch random windows from Arctic Shift if pools are short")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    c = cfg["corpus"]
    seed = int(cfg["project"]["seed"])
    bot_values = []
    for b in c.get("bots", []):
        if isinstance(b, list):
            bot_values.extend(str(x) for x in b)
        else:
            bot_values.append(str(b))
    bots = {b.lower() for b in bot_values}
    min_chars = int(c.get("min_text_chars", 20))

    existing = load_existing(cfg)
    LOG.info("existing manual posts: %s", len(existing))
    query_duplicates = len(existing) - len({p["id"] for p in existing})
    if query_duplicates:
        LOG.warning("duplicate IDs in existing corpus: %s", query_duplicates)

    selected: dict[str, list[dict]] = {str(y): [] for y in c["years"]}
    ids: set[str] = set()
    text_hashes: set[str] = set()
    index = NearDuplicateIndex(int(c.get("near_duplicate_hamming", 3)))
    dedupe = {"exact_text": 0, "near_text": 0, "duplicate_id": 0, "invalid": 0}

    def add(post: dict, y: int) -> bool:
        if post["id"] in ids:
            dedupe["duplicate_id"] += 1
            return False
        if not _valid_for_corpus(post, bots, min_chars):
            dedupe["invalid"] += 1
            return False
        text = normalize_text(post["title"] + " " + post["body"])
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in text_hashes:
            dedupe["exact_text"] += 1
            return False
        if index.is_near(text):
            dedupe["near_text"] += 1
            return False
        ids.add(post["id"])
        text_hashes.add(h)
        index.add(text)
        selected[str(y)].append(post)
        return True

    # Existing 8k: preserve them even if a title/body is short; their labels are
    # the ground truth for the classifier.  Skip exact duplicate IDs/text.
    for p in existing:
        y = timestamp_to_utc(p["created_utc"]).year
        if str(y) not in selected:
            continue
        text = normalize_text(p["title"] + " " + p["body"])
        h = hashlib.sha256(text.encode()).hexdigest()
        if p["id"] in ids:
            dedupe["duplicate_id"] += 1
            continue
        ids.add(p["id"])
        text_hashes.add(h)
        index.add(text)
        selected[str(y)].append(p)

    primary = load_pool_files(cfg, "pool_files")
    fallback = load_pool_files(cfg, "fallback_pool_files")
    LOG.info("primary pool candidates: %s; fallback: %s", len(primary), len(fallback))

    def fill(candidates: list[dict], label: str) -> None:
        by_year: dict[int, list[dict]] = collections.defaultdict(list)
        for p in candidates:
            y = timestamp_to_utc(p["created_utc"]).year
            if str(y) in selected:
                by_year[y].append(p)
        for y in c["years"]:
            need = int(c["per_year"]) - len(selected[str(y)])
            if need <= 0:
                continue
            rng = random.Random(seed + y + (0 if label == "primary" else 100_000))
            rng.shuffle(by_year.get(y, []))
            added = 0
            for p in by_year.get(y, []):
                if added >= need:
                    break
                p = dict(p)
                p["source_pool"] = f"{label}:{p.get('source_pool', '')}"
                if add(p, y):
                    added += 1
            LOG.info("%s %s: added %s / needed %s", label, y, added, need)

    fill(primary, "primary")
    deficits = {y: int(c["per_year"]) - len(selected[str(y)]) for y in c["years"]}
    if any(v > 0 for v in deficits.values()):
        fill(fallback, "fallback")
        deficits = {y: int(c["per_year"]) - len(selected[str(y)]) for y in c["years"]}
    if args.allow_live and any(v > 0 for v in deficits.values()):
        extra = collect_live(cfg, deficits, ids, text_hashes, index, bots)
        for p in extra:
            add(p, timestamp_to_utc(p["created_utc"]).year)
    deficits = {y: int(c["per_year"]) - len(selected[str(y)]) for y in c["years"]}
    if any(v > 0 for v in deficits.values()):
        LOG.warning("corpus is short for years: %s", {y: v for y, v in deficits.items() if v > 0})

    # Keep a stable order and attach a data version.
    for y in selected:
        selected[y].sort(key=lambda p: (p["created_utc"], p["id"]))
        for p in selected[y]:
            p["data_version"] = cfg["project"]["data_version"]

    report = {
        "data_version": cfg["project"]["data_version"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "target_per_year": int(c["per_year"]),
        "total": sum(len(v) for v in selected.values()),
        "per_year": {y: len(v) for y, v in selected.items()},
        "existing_manual": len(existing),
        "source_counts": dict(collections.Counter(
            p["source_pool"] for arr in selected.values() for p in arr)),
        "dedupe": dedupe,
        "deficits": {y: v for y, v in deficits.items() if v > 0},
    }
    write_json(project_path(cfg["paths"]["posts_raw"]), {
        "classes": cfg["labels"]["classes"],
        "influence_weights": cfg["labels"]["influence_weights"],
        "meta": report,
        "years": selected,
    })
    write_json(project_path(cfg["paths"]["reports"]) / "corpus_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
