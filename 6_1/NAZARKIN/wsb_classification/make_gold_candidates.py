#!/usr/bin/env python3
"""Build gold-set candidates for honest evaluation.

Sampling is deliberately INDEPENDENT of the trained classifier (flair + length
heuristics, plus uniform random) so that labelling them is not circular.
All training ids are excluded.
"""
import json
import os
import random
import re
from collections import Counter

import build_balanced as B

random.seed(20260928)

TRAIN_FILES = ["7k_posts_4class.json", "7k_posts.json", "teachers_labeled_some_model.json",
               "teachers_to_label.json", "students_to_label.json"]

NUM = re.compile(r"\$\s?\d|\d+\s?%|\b\d{2,}\b")
FIRST = re.compile(r"\b(i|my|me|we|our)\b", re.I)

HINTS = {
    "NOISE_MEME": lambda f, t, bl, n: (f or "").lower() in ("meme", "shitpost", "image", "video"),
    "GENUINE_OPINION": lambda f, t, bl, n: (f or "").lower() in ("options", "discussion", "yolo", "gain", "loss", "storytime", "first group buy"),
    "FINANCIAL_CLAIM": lambda f, t, bl, n: (f or "").lower() in ("news", "stocks", "fundamentals", "earnings thread", "earnings"),
    "ANALYSIS_DD": lambda f, t, bl, n: (f or "").lower() in ("dd", "fundamentals", "research", "chart", "technical analysis"),
}

PER_CLASS = 30
PER_NATURAL = 55


def main():
    train_ids = set()
    for f in TRAIN_FILES:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for y in d.get("years", {}).values():
            for p in y:
                if p.get("id"):
                    train_ids.add(p["id"])

    pool = [p for p in B.load_pool() if p["id"] not in train_ids]
    print("pool excluding training ids:", len(pool), dict(Counter(str(p['_year']) for p in pool)))

    def info(p):
        title = p.get("title") or ""
        body = p.get("selftext") or ""
        text = B.build_text(p)
        return p.get("link_flair_text"), title, body, len(text)

    chosen, used = [], set()

    # balanced candidates per class
    for cls, hint in HINTS.items():
        cands = []
        for p in pool:
            if p["id"] in used:
                continue
            f, t, b, n = info(p)
            if B.is_dead(t, b):
                continue
            if not hint(f, t, b, n):
                continue
            if cls == "ANALYSIS_DD" and (len(b) < 300 or n < 600):
                continue
            if cls == "NOISE_MEME" and n > 900:
                continue
            cands.append(p)
        random.shuffle(cands)
        take = cands[: PER_CLASS + 15]
        for p in take:
            used.add(p["id"])
            chosen.append({"hint_class": cls, "id": p["id"], "year": str(p["_year"]),
                           "flair": p.get("link_flair_text"), "title": p.get("title") or "",
                           "body": p.get("selftext") or ""})
        print(f"{cls}: pool {len(cands)} -> sampled {len(take)}")

    # natural-distribution candidates
    rest = [p for p in pool if p["id"] not in used]
    random.shuffle(rest)
    for p in rest[:PER_NATURAL]:
        used.add(p["id"])
        chosen.append({"hint_class": "NATURAL", "id": p["id"], "year": str(p["_year"]),
                       "flair": p.get("link_flair_text"), "title": p.get("title") or "",
                       "body": p.get("selftext") or ""})

    random.shuffle(chosen)
    json.dump(chosen, open("gold_candidates.json", "w"), ensure_ascii=False, indent=2)
    print("\ncandidates total:", len(chosen), dict(Counter(c["hint_class"] for c in chosen)))
    print("saved gold_candidates.json")


if __name__ == "__main__":
    main()
