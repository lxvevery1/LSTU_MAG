#!/usr/bin/env python3
"""Improve the baseline: compare several linear models and a small
char/word TF-IDF ensemble, report macro-F1 on val + held-out teachers test."""
import json
import random
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
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
    word = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=120000,
                           sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2,
                           max_features=120000, sublinear_tf=True, lowercase=True)
    return FeatureUnion([("word", word), ("char", char)])


def main():
    random.seed(SEED); np.random.seed(SEED)
    students = load("students_to_label.json")
    teachers = load("teachers_to_label.json")
    train, val = train_test_split(students, test_size=0.2, random_state=SEED,
                                  stratify=[r["label"] for r in students])

    f = feats()
    Xtr = f.fit_transform([r["text"] for r in train])
    Xval = f.transform([r["text"] for r in val])
    Xte = f.transform([r["text"] for r in teachers])
    ytr = [r["label"] for r in train]
    yval = [r["label"] for r in val]
    yte = [r["label"] for r in teachers]

    models = {
        "logreg_C1_bal": LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
        "logreg_C8_bal": LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced"),
        "logreg_C20_bal": LogisticRegression(max_iter=3000, C=20.0, class_weight="balanced"),
        "svc_C1_bal": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3),
        "svc_C5_bal": CalibratedClassifierCV(LinearSVC(C=5.0, class_weight="balanced"), cv=3),
        "complementNB": ComplementNB(alpha=0.3, class_prior=None),
    }

    best = None
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        vp = clf.predict(Xval)
        tp = clf.predict(Xte)
        vf1 = f1_score(yval, vp, average="macro")
        tf1 = f1_score(yte, tp, average="macro")
        va = accuracy_score(yval, vp)
        ta = accuracy_score(yte, tp)
        print(f"{name:16s} val acc {va:.3f} macroF1 {vf1:.3f} | test acc {ta:.3f} macroF1 {tf1:.3f}")
        if best is None or vf1 > best[1]:
            best = (name, vf1, clf)

    print("\nBEST by val macro-F1:", best[0])
    clf = best[2]
    tp = clf.predict(Xte)
    print(f"test accuracy: {accuracy_score(yte, tp):.3f}")
    print(classification_report(yte, tp, labels=CLASSES, zero_division=0))

    import joblib
    joblib.dump({"features": f, "clf": clf, "classes": CLASSES}, "model_best.joblib")
    print("saved model_best.joblib")


if __name__ == "__main__":
    main()
