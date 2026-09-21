#!/usr/bin/env python3
"""LogReg predictions over the unified pool (5-fold CV) with per-year accuracy."""
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

SEED = 42
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]


def load(path):
    d = json.load(open(path))
    rows = []
    for y, posts in d["years"].items():
        for p in posts:
            text = " ".join(x for x in [p.get("title") or "", p.get("body") or ""] if x).strip()
            rows.append({"text": text, "label": p["label"], "year": y, "post": p})
    return rows


def make_features():
    word = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=150000,
                           sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2,
                           max_features=150000, sublinear_tf=True, lowercase=True)
    return FeatureUnion([("word", word), ("char", char)])


def main():
    pool = load("teachers_to_label.json") + load("students_to_label.json")
    # dedupe by id, keep order
    seen = set(); uniq = []
    for r in pool:
        if r["post"]["id"] in seen:
            continue
        seen.add(r["post"]["id"]); uniq.append(r)
    pool = uniq
    print("pool:", len(pool), Counter(r["label"] for r in pool))

    X = [r["text"] for r in pool]
    y = [r["label"] for r in pool]
    years = [r["year"] for r in pool]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = [None] * len(pool)
    for tr, va in skf.split(X, y):
        feats = make_features()
        Xtr = feats.fit_transform([X[i] for i in tr])
        Xva = feats.transform([X[i] for i in va])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, [y[i] for i in tr])
        for idx, p in zip(va, clf.predict(Xva)):
            preds[idx] = p

    # build output
    out = {
        "model": "LogisticRegression (TF-IDF word+char), 5-fold CV OOF predictions",
        "classes": CLASSES,
        "overall_accuracy": round(accuracy_score(y, preds), 4),
        "per_year_accuracy": {},
        "per_year_counts": {},
        "years": {},
    }

    by_year = defaultdict(list)
    for r, pred in zip(pool, preds):
        by_year[r["year"]].append((r, pred))

    for year in sorted(by_year):
        items = by_year[year]
        correct = sum(1 for (r, p) in items if p == r["label"])
        acc = correct / len(items)
        out["per_year_accuracy"][year] = round(acc, 4)
        out["per_year_counts"][year] = {"total": len(items), "correct": correct}
        out["years"][year] = []
        for (r, p) in items:
            r["post"]["predicted_title"] = p
            out["years"][year].append(r["post"])

    json.dump(out, open("logreg_results.json", "w"), ensure_ascii=False, indent=2)

    print(f"\noverall accuracy: {out['overall_accuracy']}")
    for yr in sorted(out["per_year_accuracy"]):
        c = out["per_year_counts"][yr]
        print(f"  {yr}: {out['per_year_accuracy'][yr]:.4f}  ({c['correct']}/{c['total']})")
    print("saved logreg_results.json")


if __name__ == "__main__":
    main()
