#!/usr/bin/env python3
"""Train on unified pool (teachers + students) with 80/20 split."""
import json
import random
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

SEED = 42
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]


def load(path):
    d = json.load(open(path))
    rows = []
    for y, posts in d["years"].items():
        for p in posts:
            text = " ".join(x for x in [p.get("title") or "", p.get("body") or ""] if x).strip()
            rows.append({"text": text, "label": p["label"]})
    return rows


def feats():
    word = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=150000,
                           sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2,
                           max_features=150000, sublinear_tf=True, lowercase=True)
    return FeatureUnion([("word", word), ("char", char)])


def main():
    random.seed(SEED); np.random.seed(SEED)
    teachers = load("teachers_to_label.json")
    students = load("students_to_label.json")
    pool = teachers + students
    print(f"pool={len(pool)} (teachers={len(teachers)}, students={len(students)})")
    print("class dist:", Counter(r["label"] for r in pool))

    train, test = train_test_split(pool, test_size=0.2, random_state=SEED,
                                   stratify=[r["label"] for r in pool])
    print("train", len(train), "test", len(test))

    f = feats()
    Xtr = f.fit_transform([r["text"] for r in train])
    Xte = f.transform([r["text"] for r in test])
    ytr = [r["label"] for r in train]
    yte = [r["label"] for r in test]

    models = {
        "logreg_C1_bal": LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
        "logreg_C4_bal": LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"),
        "logreg_C10_bal": LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced"),
        "svc_C1_bal": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3),
        "svc_C5_bal": CalibratedClassifierCV(LinearSVC(C=5.0, class_weight="balanced"), cv=3),
        "complementNB": ComplementNB(alpha=0.3),
    }

    best = None
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        p = clf.predict(Xte)
        f1 = f1_score(yte, p, average="macro")
        acc = accuracy_score(yte, p)
        print(f"{name:16s} acc {acc:.3f} macroF1 {f1:.3f}")
        if best is None or f1 > best[1]:
            best = (name, f1, clf)

    print("\nBEST:", best[0])
    p = best[2].predict(Xte)
    print(f"accuracy: {accuracy_score(yte, p):.3f}")
    print(classification_report(yte, p, labels=CLASSES, zero_division=0))
    print(confusion_matrix(yte, p, labels=CLASSES))

    import joblib
    joblib.dump({"features": f, "clf": best[2], "classes": CLASSES}, "model_pool.joblib")
    print("saved model_pool.joblib")


if __name__ == "__main__":
    main()
