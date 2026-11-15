#!/usr/bin/env python3
"""Train LogReg on the balanced 4-class dataset (UNCERTAIN removed).

Saves model, metrics and predictions into ./some_model_4c/.
"""
import json
import os
from collections import Counter, defaultdict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

SEED = 42
OUT_DIR = "some_model_4c"
TRAIN = "7k_posts_4class.json"
EXTERNAL = "teachers_labeled_some_model.json"
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]


def load(path, classes):
    d = json.load(open(path))
    rows = []
    for year in sorted(d["years"]):
        for p in d["years"][year]:
            if p["label"] not in classes:
                continue
            text = " ".join(x for x in [(p.get("title") or ""), (p.get("body") or "")] if x).strip()
            rows.append({"text": text, "label": p["label"], "year": year, "post": p})
    return rows


def make_features():
    word = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=150000,
                           sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2,
                           max_features=150000, sublinear_tf=True, lowercase=True)
    return FeatureUnion([("word", word), ("char", char)])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load(TRAIN, CLASSES)
    seen, uniq = set(), []
    for r in rows:
        if r["post"]["id"] in seen:
            continue
        seen.add(r["post"]["id"])
        uniq.append(r)
    rows = uniq
    X = [r["text"] for r in rows]
    y = [r["label"] for r in rows]
    print("train posts:", len(rows), dict(Counter(y)))

    # ---- 5-fold CV, out-of-fold predictions ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = [None] * len(rows)
    for tr, va in skf.split(X, y):
        feats = make_features()
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(feats.fit_transform([X[i] for i in tr]), [y[i] for i in tr])
        for idx, p in zip(va, clf.predict(feats.transform([X[i] for i in va]))):
            oof[idx] = p

    cv_acc = accuracy_score(y, oof)
    cv_f1 = f1_score(y, oof, average="macro")
    print(f"CV accuracy: {cv_acc:.4f}  macro-F1: {cv_f1:.4f}")

    # ---- final model on all data ----
    feats = make_features()
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(feats.fit_transform(X), y)
    joblib.dump({"features": feats, "clf": clf, "classes": CLASSES},
                os.path.join(OUT_DIR, "logreg_model.joblib"))

    # ---- per-year ----
    by_year = defaultdict(list)
    for r, p in zip(rows, oof):
        by_year[r["year"]].append((r, p))

    results = {
        "model": "LogisticRegression (TF-IDF word 1-3 + char_wb 2-6), 5-fold CV OOF",
        "classes": CLASSES,
        "train_file": TRAIN,
        "total_posts": len(rows),
        "cv_accuracy": round(cv_acc, 4),
        "macro_f1": round(cv_f1, 4),
        "per_year_accuracy": {},
        "per_year_counts": {},
        "class_distribution": {},
        "years": {},
    }
    for year in sorted(by_year):
        items = by_year[year]
        correct = sum(1 for r, p in items if p == r["label"])
        actual = Counter(r["label"] for r, _ in items)
        pred = Counter(p for _, p in items)
        results["per_year_accuracy"][year] = round(correct / len(items), 4)
        results["per_year_counts"][year] = {"total": len(items), "correct": correct}
        results["class_distribution"][year] = {
            "actual": {c: actual.get(c, 0) for c in CLASSES},
            "predicted": {c: pred.get(c, 0) for c in CLASSES},
        }
        results["years"][year] = []
        for r, p in items:
            post = dict(r["post"])
            post["predicted_label"] = p
            results["years"][year].append(post)

    # ---- external check on the teacher set (labels produced by another run) ----
    ext = load(EXTERNAL, CLASSES)
    ext_acc = ext_f1 = None
    if ext:
        preds = clf.predict(feats.transform([r["text"] for r in ext]))
        gold = [r["label"] for r in ext]
        ext_acc = accuracy_score(gold, preds)
        ext_f1 = f1_score(gold, preds, average="macro")
        results["external_eval"] = {
            "source": EXTERNAL, "n": len(ext),
            "accuracy": round(ext_acc, 4), "macro_f1": round(ext_f1, 4),
            "note": "gold labels here are another model's output, so this is a consistency check, not truth",
        }
        print(f"external ({len(ext)} posts from {EXTERNAL}): acc {ext_acc:.4f} macro-F1 {ext_f1:.4f}")

    with open(os.path.join(OUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    report = classification_report(y, oof, labels=CLASSES, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(y, oof, labels=CLASSES)
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("4-class LogReg (UNCERTAIN removed)\n")
        f.write(f"Train file: {TRAIN}  ({len(rows)} posts)\n\n")
        f.write(f"CV accuracy: {cv_acc:.4f}\nmacro-F1: {cv_f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true):\n")
        f.write(str(cm) + "\n")
        f.write("\nPer-year accuracy:\n")
        for yr in sorted(results["per_year_accuracy"]):
            c = results["per_year_counts"][yr]
            f.write(f"  {yr}: {results['per_year_accuracy'][yr]:.4f} ({c['correct']}/{c['total']})\n")
        if ext_acc is not None:
            f.write(f"\nExternal consistency check on {EXTERNAL} ({len(ext)} posts):\n")
            f.write(f"  accuracy {ext_acc:.4f}, macro-F1 {ext_f1:.4f}\n")
            f.write("  (these gold labels are another model's output, not human ground truth)\n")

    print()
    print(report)
    print("Confusion matrix (rows=true):")
    print(cm)
    print("\nPer-year accuracy:")
    for yr in sorted(results["per_year_accuracy"]):
        c = results["per_year_counts"][yr]
        print(f"  {yr}: {results['per_year_accuracy'][yr]:.4f} ({c['correct']}/{c['total']})")
    print(f"\nsaved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
