#!/usr/bin/env python3
"""Build the random 1000-post-per-year (2019-2026) sample to be labelled by hand.

Reads pool_big.jsonl, groups by year, draws N=1000 posts per year with a fixed
seed so the sample is reproducible, and writes:

  sample_8k.json            - full sample, grouped by year (source of truth)
  sample_chunks/chunk_XX.json - flat readable chunks (200 posts each) for labelling
  sample_chunks/index.json  - chunk -> year mapping
"""
import json
import os
import random
from collections import defaultdict

POOL = "pool_big.jsonl"
OUT = "sample_8k.json"
CHUNK_DIR = "sample_chunks"
CHUNK_SIZE = 200
PER_YEAR = 1000
SEED = 20260928
YEARS = [str(y) for y in range(2019, 2027)]


def norm(s):
    return " ".join((s or "").split())


def main():
    by_year = defaultdict(list)
    seen = set()
    with open(POOL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = d.get("id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            year = str(d.get("_window", ""))[:4]
            if year not in YEARS:
                continue
            by_year[year].append(
                {
                    "id": pid,
                    "year": year,
                    "created_utc": d.get("created_utc"),
                    "author": d.get("author"),
                    "score": d.get("score"),
                    "num_comments": d.get("num_comments"),
                    "flair": d.get("link_flair_text"),
                    "title": norm(d.get("title")),
                    "body": norm(d.get("selftext")),
                }
            )

    rng = random.Random(SEED)
    sample = {}
    for y in YEARS:
        pool = by_year[y]
        rng.shuffle(pool)
        if len(pool) < PER_YEAR:
            raise SystemExit(f"year {y}: only {len(pool)} posts available")
        sample[y] = pool[:PER_YEAR]
        print(f"{y}: pool={len(pool)} sampled={PER_YEAR}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "per_year": PER_YEAR, "years": sample}, f,
                  ensure_ascii=False, indent=1)
    print(f"wrote {OUT}")

    os.makedirs(CHUNK_DIR, exist_ok=True)
    for old in os.listdir(CHUNK_DIR):
        if old.startswith("chunk_") and old.endswith(".json"):
            os.remove(os.path.join(CHUNK_DIR, old))

    flat = []
    for y in YEARS:
        for p in sample[y]:
            flat.append(p)

    index = []
    for n, start in enumerate(range(0, len(flat), CHUNK_SIZE)):
        rows = flat[start:start + CHUNK_SIZE]
        name = f"chunk_{n:03d}"
        with open(os.path.join(CHUNK_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        index.append({"chunk": name, "year": rows[0]["year"], "n": len(rows),
                      "ids": [r["id"] for r in rows]})
    with open(os.path.join(CHUNK_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(index)} chunks x {CHUNK_SIZE} = {len(flat)} posts into {CHUNK_DIR}/")


if __name__ == "__main__":
    main()
