#!/usr/bin/env python3
"""Build a class-balanced WSB dataset: ~200 posts per class per year for 2019-2026.

For every (year, class) we keep only posts where the classifier's top class is that
class, apply per-class quality gates, then pick the highest-scoring candidates
round-robin across the 12 months so the whole year is represented.

Output structure mirrors teachers_labeled_some_model.json.
"""
import datetime
import json
import os
import re
from collections import Counter, defaultdict

import joblib
import numpy as np

POOL = "pool_big.jsonl"
EXTRA_POOLS = ["posts_8k_raw.json"]
MODEL = "some_model/logreg_model.joblib"
OUT = "7k_posts.json"
PER_CLASS = int(os.environ.get("PER_CLASS", "200"))

CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]
INSTRUCTIONS = {
    "NOISE_MEME": "No meaningful financial/investment information. Pure joke, reaction, shitpost, image caption.",
    "GENUINE_OPINION": "Genuine bullish/bearish/trading opinion without substantial analysis.",
    "FINANCIAL_CLAIM": "Concrete factual claim about a company, market, event, price, earnings, metric.",
    "ANALYSIS_DD": "Detailed reasoning, DD, calculations, valuation, catalysts, risks, sources.",
    "UNCERTAIN": "Cannot confidently assign another label.",
}

# Reddit flair is a strong independent signal, so it is used to re-rank candidates.
FLAIR_BONUS = {
    "NOISE_MEME": {"shitpost": 0.30, "meme": 0.30, "gain": 0.10, "loss": 0.10, "yolo": 0.10, "discussion": -0.05},
    "GENUINE_OPINION": {"options": 0.15, "discussion": 0.10, "yolo": 0.10, "gain": 0.05, "loss": 0.05, "dd": -0.10},
    "FINANCIAL_CLAIM": {"stocks": 0.20, "fundamentals": 0.20, "earnings": 0.20, "news": 0.20, "dd": 0.05},
    "ANALYSIS_DD": {"dd": 0.35, "fundamentals": 0.20, "research": 0.20, "investing": 0.15, "stocks": 0.05},
    "UNCERTAIN": {"discussion": 0.10},
}


TICKER = re.compile(r"\$[A-Z]{1,5}\b")
MONEY = re.compile(r"\$\s?\d|\d+\s?%|\d+(\.\d+)?\s?(bn|billion|mn|million|trillion)\b")
QUESTION = re.compile(r"\?\s*$|^\s*(why|how|what|when|where|who|should|is|are|does|do|can|any)\b", re.I)
FIRST_PERSON = re.compile(r"\b(i|my|me|we|our)\b", re.I)


def build_text(p):
    return " ".join(x for x in [(p.get("title") or "").strip(), (p.get("selftext") or "").strip()] if x).strip()


def month_of(ts):
    return datetime.datetime.fromtimestamp(ts or 0, datetime.timezone.utc).month


def pick_balanced(cands, k):
    by_month = defaultdict(list)
    for c in cands:
        by_month[month_of(c["post"]["created"])].append(c)
    for m in by_month:
        by_month[m].sort(key=lambda c: -c["score"])
    months = sorted(by_month)
    out, i = [], 0
    while len(out) < k and any(by_month[m] for m in months):
        m = months[i % len(months)]
        if by_month[m]:
            out.append(by_month[m].pop(0))
        i += 1
    return out


DAILY_THREAD = re.compile(r"Daily Discussion Thread|Weekly Earnings Discussion Thread|What Are Your Moves Tomorrow", re.I)
DEAD_BODY = {"[removed]", "[deleted]"}
STORY_FLAIR = {"gain", "loss", "yolo", "meme", "shitpost", "storytime", "first group buy"}


def is_dead(title, body):
    """Only truly destroyed posts are dropped; title-only posts are still classifiable."""
    t, b = (title or "").strip(), (body or "").strip().lower()
    if b in DEAD_BODY:
        return True
    if not t and not b:
        return True
    # daily-thread boilerplate carries no content of its own
    return bool(DAILY_THREAD.search(t)) and len(t) + len(b) < 260


def flair_bonus(cls, flair):
    table = FLAIR_BONUS[cls]
    if not flair:
        return 0.0
    return table.get(flair.strip().lower(), 0.0)


def rules_ok(cls, text, body, flair, title):
    if is_dead(title, body):
        return False
    fl = (flair or "").strip().lower()
    n = len(text)
    links = text.count("http")
    blen = len((body or "").strip())
    t = (title or "").strip()

    if cls == "NOISE_MEME":
        return n <= 1200
    if cls == "GENUINE_OPINION":
        if fl in ("daily discussion", "meme", "shitpost"):
            return False
        return 60 <= n <= 3000 and bool(FIRST_PERSON.search(text) or QUESTION.search(text))
    if cls == "FINANCIAL_CLAIM":
        if fl in STORY_FLAIR:
            return False
        if blen < 50:
            # title-only post: the claim itself must be concrete and specific
            return len(t) >= 40 and bool(MONEY.search(t) or TICKER.search(t))
        return 80 <= n <= 3500 and bool(MONEY.search(text) or TICKER.search(text))
    if cls == "ANALYSIS_DD":
        if fl in STORY_FLAIR or fl == "news":
            return False
        if blen < 400:          # DD needs actual reasoning, not a headline
            return False
        if n < 500:
            return False
        if body.count("\n\n") + 1 < 3 and links < 2:
            return False
        return bool(MONEY.search(text) or TICKER.search(text) or links >= 2)
    if cls == "UNCERTAIN":
        return n < 400
    return True



def load_pool():
    posts, seen = [], set()
    labelled = set()
    for f in ("teachers_labeled_some_model.json", "teachers_to_label.json", "students_to_label.json"):
        if os.path.exists(f):
            for y in json.load(open(f)).get("years", {}).values():
                for p in y:
                    if p.get("id"):
                        labelled.add(p["id"])
    if os.path.exists(POOL):
        for line in open(POOL, encoding="utf-8"):
            p = json.loads(line)
            w = p.get("_window", "")
            if p.get("id") and p["id"] not in seen:
                seen.add(p["id"])
                p["_year"] = w[:4]
                posts.append(p)
    for f in EXTRA_POOLS:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for year, rows in d.items():
            for p in rows:
                if p.get("id") and p["id"] not in seen:
                    seen.add(p["id"])
                    p["_year"] = str(year)
                    posts.append(p)
    return [p for p in posts if p["id"] not in labelled]


def main():
    raw = load_pool()
    bundle = joblib.load(MODEL)
    feats, clf = bundle["features"], bundle["clf"]
    model_classes = [str(c) for c in clf.classes_]

    posts, texts = [], []
    for p in raw:
        posts.append({
            "id": p.get("id"), "created": p.get("created_utc"), "score": p.get("score"),
            "flair": p.get("link_flair_text"), "title": p.get("title") or "",
            "body": p.get("selftext") or "", "year": str(p["_year"]),
        })
        texts.append(build_text(p))

    print(f"scoring {len(posts)} candidate posts ...")
    P = clf.predict_proba(feats.transform(texts))
    col = {c: P[:, model_classes.index(c)] for c in CLASSES}
    for i, r in enumerate(posts):
        r["_probs"] = {c: float(col[c][i]) for c in CLASSES}
        r["_order"] = sorted(CLASSES, key=lambda c: -r["_probs"][c])
        r["_text"] = texts[i]

    years_out, report, seen_global = {}, {}, set()
    for year in sorted({r["year"] for r in posts}):
        rows = [r for r in posts if r["year"] == year and r["id"] not in seen_global]
        used, chosen, report[year] = set(), [], {}
        for cls in CLASSES:
            strict, loose = [], []
            for r in rows:
                if r["id"] in used:
                    continue
                if not rules_ok(cls, r["_text"], r["body"], r["flair"], r["title"]):
                    continue
                score = r["_probs"][cls] + flair_bonus(cls, r["flair"])
                cand = {"post": r, "score": score}
                if r["_order"][0] == cls:
                    strict.append(cand)
                elif r["_order"][1] == cls:
                    loose.append(cand)
            picked = pick_balanced(strict, PER_CLASS)
            fallback = 0
            if len(picked) < PER_CLASS:
                for c in pick_balanced(loose, PER_CLASS - len(picked)):
                    picked.append(c)
                    fallback += 1
            for c in picked:
                chosen.append((cls, c["post"]))
                used.add(c["post"]["id"])
                seen_global.add(c["post"]["id"])
            conf = [c["post"]["_probs"][cls] for c in picked]
            report[year][cls] = {
                "strict_cands": len(strict), "picked": len(picked), "top2_fallback": fallback,
                "avg_p": round(float(np.mean(conf)), 3) if conf else 0.0,
            }

        merged = []
        for cls in CLASSES:
            for r in [p for c, p in chosen if c == cls]:
                merged.append({**{k: r[k] for k in ("id", "created", "score", "flair", "title", "body")},
                               "label": cls})
        merged.sort(key=lambda r: r["created"] or 0, reverse=True)
        years_out[year] = merged

    result = {"classes": CLASSES, "instructions": INSTRUCTIONS, "years": years_out}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in years_out.values())
    print(f"\nwrote {OUT}: {total} posts")
    print("picked per class/year (fb = filled from top-2 fallback):")
    for year in sorted(report):
        cells = []
        for c in CLASSES:
            d = report[year][c]
            cells.append(f"{c[:4]}={d['picked']:3d}" + (f"(fb{d['top2_fallback']})" if d["top2_fallback"] else "     "))
        print(f"  {year}: " + " ".join(cells))
    print("\navg P(assigned class):")
    for year in sorted(report):
        print(f"  {year}: " + " ".join(f"{c[:4]}={report[year][c]['avg_p']:.2f}" for c in CLASSES))
    print("\noverall:", dict(Counter(p["label"] for v in years_out.values() for p in v)))



if __name__ == "__main__":
    main()
