#!/usr/bin/env python3
"""Train LogReg on teachers_labeled_some_model.json.

Saves model, metrics, predictions and per-year report into ./some_model/.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

SEED = 42
OUT_DIR = "some_model"
DATA = "teachers_labeled_some_model.json"
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
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load(DATA)
    seen = set(); uniq = []
    for r in rows:
        if r["post"]["id"] in seen:
            continue
        seen.add(r["post"]["id"]); uniq.append(r)
    rows = uniq
    print("posts:", len(rows), Counter(r["label"] for r in rows))

    X = [r["text"] for r in rows]
    y = [r["label"] for r in rows]

    # --- 5-fold CV out-of-fold predictions + metrics ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = [None] * len(rows)
    for tr, va in skf.split(X, y):
        feats = make_features()
        Xtr = feats.fit_transform([X[i] for i in tr])
        Xva = feats.transform([X[i] for i in va])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, [y[i] for i in tr])
        for idx, p in zip(va, clf.predict(Xva)):
            oof[idx] = p

    cv_acc = accuracy_score(y, oof)
    cv_f1 = f1_score(y, oof, average="macro")
    print(f"CV accuracy: {cv_acc:.4f}  macro-F1: {cv_f1:.4f}")

    # --- final model on all data ---
    feats = make_features()
    Xall = feats.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(Xall, y)

    joblib.dump({"features": feats, "clf": clf, "classes": CLASSES},
                os.path.join(OUT_DIR, "logreg_model.joblib"))

    # --- save predictions per post ---
    pred_map = {r["post"]["id"]: p for r, p in zip(rows, oof)}
    by_year = defaultdict(list)
    for r, p in zip(rows, oof):
        by_year[r["year"]].append((r, p))

    results = {
        "model": "LogisticRegression (TF-IDF word+char), 5-fold CV OOF",
        "classes": CLASSES,
        "overall_accuracy": round(cv_acc, 4),
        "macro_f1": round(cv_f1, 4),
        "per_year_accuracy": {},
        "per_year_counts": {},
        "class_distribution": {},
        "years": {},
    }

    for year in sorted(by_year):
        items = by_year[year]
        correct = sum(1 for (r, p) in items if p == r["label"])
        results["per_year_accuracy"][year] = round(correct / len(items), 4)
        results["per_year_counts"][year] = {"total": len(items), "correct": correct}
        actual = Counter(r["label"] for (r, p) in items)
        pred = Counter(p for (r, p) in items)
        results["class_distribution"][year] = {
            "actual": {c: actual.get(c, 0) for c in CLASSES},
            "predicted": {c: pred.get(c, 0) for c in CLASSES},
        }
        results["years"][year] = []
        for (r, p) in items:
            post = dict(r["post"])
            post["predicted_label"] = p
            results["years"][year].append(post)

    json.dump(results, open(os.path.join(OUT_DIR, "predictions.json"), "w"),
              ensure_ascii=False, indent=2)

    report = classification_report(y, oof, labels=CLASSES, target_names=CLASSES,
                                   zero_division=0)
    cm = confusion_matrix(y, oof, labels=CLASSES)
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"CV accuracy: {cv_acc:.4f}\nmacro-F1: {cv_f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true):\n")
        f.write(str(cm) + "\n")
        f.write("\nPer-year accuracy:\n")
        for yr in sorted(results["per_year_accuracy"]):
            c = results["per_year_counts"][yr]
            f.write(f"  {yr}: {results['per_year_accuracy'][yr]:.4f} ({c['correct']}/{c['total']})\n")

    print("\n" + report)
    print("Confusion matrix (rows=true):")
    print(cm)
    print("\nPer-year accuracy:")
    for yr in sorted(results["per_year_accuracy"]):
        c = results["per_year_counts"][yr]
        print(f"  {yr}: {results['per_year_accuracy'][yr]:.4f} ({c['correct']}/{c['total']})")
    print(f"\nsaved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
