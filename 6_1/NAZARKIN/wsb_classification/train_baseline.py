#!/usr/bin/env python3
"""Train a text classifier for WSB post classification.

Split: teachers = held-out 20% test set, students = train/val (80%).
Two models are compared:
  1) TF-IDF char+word -> LinearSVC / LogisticRegression
  2) optional transformer fine-tune (if transformers available)
"""
import json
import random
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

SEED = 42
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]


def load(path):
    d = json.load(open(path))
    rows = []
    for y, posts in d["years"].items():
        for p in posts:
            text = " ".join(x for x in [p.get("title") or "", p.get("body") or ""] if x).strip()
            rows.append({"text": text, "label": p["label"], "year": y, "id": p["id"]})
    return rows


def build_features():
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                           sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=60000, sublinear_tf=True, lowercase=True)
    return FeatureUnion([("word", word), ("char", char)])


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    teachers = load("teachers_to_label.json")
    students = load("students_to_label.json")
    print(f"teachers={len(teachers)} students={len(students)}")

    # students -> 80% train, remaining used as val; teachers -> held-out test
    train, val = train_test_split(students, test_size=0.2, random_state=SEED,
                                  stratify=[r["label"] for r in students])
    test = teachers

    print("train", len(train), Counter(r["label"] for r in train))
    print("val  ", len(val), Counter(r["label"] for r in val))
    print("test ", len(test), Counter(r["label"] for r in test))

    Xtr = [r["text"] for r in train]
    ytr = [r["label"] for r in train]
    Xval = [r["text"] for r in val]
    yval = [r["label"] for r in val]
    Xte = [r["text"] for r in test]
    yte = [r["label"] for r in test]

    feats = build_features()
    Xtr_f = feats.fit_transform(Xtr)
    Xval_f = feats.transform(Xval)
    Xte_f = feats.transform(Xte)

    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr_f, ytr)

    for name, X, y in [("val", Xval_f, yval), ("test(teachers)", Xte_f, yte)]:
        pred = clf.predict(X)
        print("=" * 60)
        print(f"{name} accuracy: {accuracy_score(y, pred):.3f}")
        print(classification_report(y, pred, labels=CLASSES, zero_division=0))
        print("confusion matrix (rows=true):")
        print(confusion_matrix(y, pred, labels=CLASSES))

    import joblib
    joblib.dump({"features": feats, "clf": clf, "classes": CLASSES}, "model_tfidf.joblib")
    print("saved model_tfidf.joblib")


if __name__ == "__main__":
    main()
