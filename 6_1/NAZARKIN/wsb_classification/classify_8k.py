#!/usr/bin/env python3
"""Classify the ~8k fetched posts with the trained LogReg and save to pickle."""
import json
import pickle
from collections import Counter, defaultdict

import joblib

RAW = "posts_8k_raw.json"
MODEL = "some_model/logreg_model.joblib"
OUT_PICKLE = "posts_8k_labeled.pkl"
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]


def build_text(p):
    return " ".join(x for x in [(p.get("title") or "").strip(), (p.get("selftext") or "").strip()] if x).strip()


def main():
    raw = json.load(open(RAW))
    bundle = joblib.load(MODEL)
    feats, clf = bundle["features"], bundle["clf"]

    records = []
    texts = []
    for year in sorted(raw):
        for p in raw[year]:
            t = build_text(p)
            texts.append(t)
            records.append({
                "id": p.get("id"),
                "year": year,
                "created": p.get("created_utc"),
                "score": p.get("score"),
                "flair": p.get("link_flair_text"),
                "title": p.get("title") or "",
                "body": p.get("selftext") or "",
            })

    X = feats.transform(texts)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    classes = list(clf.classes_)

    for r, y, pr in zip(records, preds, probs):
        r["label"] = y
        r["probs"] = {c: round(float(v), 4) for c, v in zip(classes, pr)}
        r["confidence"] = round(float(max(pr)), 4)

    payload = {
        "classes": CLASSES,
        "source": RAW,
        "model": MODEL,
        "years": {y: [r for r in records if r["year"] == y] for y in sorted(raw)},
    }
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    n = len(records)
    print(f"classified {n} posts -> {OUT_PICKLE}")
    print("\noverall distribution:", dict(Counter(r["label"] for r in records)))
    print("\nper year:")
    for y in sorted(raw):
        c = Counter(r["label"] for r in payload["years"][y])
        print(f"  {y}: " + ", ".join(f"{k}={c.get(k,0)}" for k in CLASSES))
    low = sum(1 for r in records if r["confidence"] < 0.5)
    print(f"\nlow confidence (<0.5): {low} ({low/n:.1%})")


if __name__ == "__main__":
    main()
