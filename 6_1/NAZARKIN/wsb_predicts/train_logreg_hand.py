#!/usr/bin/env python3
"""Train LogReg on the hand-labelled WSB dataset (pickle_classification_for_learning.json).

Split: 80% train / 20% test, stratified on (year, label) so both the year mix and
the class mix are preserved on both sides.

NOTE ON THE LEARNING RATE
-------------------------
sklearn >= 1.7 removed `learning_rate` / `eta0` from `LogisticRegression` (they only
ever applied to the sag/saga solvers, which lbfgs/liblinear/newton-* never used).
To honour an explicit fixed learning rate we therefore use

    SGDClassifier(loss="log_loss", learning_rate="constant", eta0=<lr>)

which minimises the same L2-regularised logistic objective with a constant step
size. `LogisticRegression` (lbfgs) is also trained as a no-lr reference point.

Configs:
  lr=0.01   (main, as requested)
  lr=0.001  (check, as requested)
  lbfgs     (reference: LogisticRegression, no explicit lr)
"""
import json
import os
import random
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion

SEED = 42
DATA = "pickle_classification_for_learning.json"
OUT_DIR = "logreg_hand"
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]
TEST_SIZE = 0.2
MAX_EPOCHS = 300


# --------------------------------------------------------------------------- data
def load():
    d = json.load(open(DATA, encoding="utf-8"))
    rows = []
    for y, posts in d["years"].items():
        for p in posts:
            text = " ".join(x for x in [(p.get("title") or "").strip(),
                                       (p.get("body") or "").strip()] if x).strip()
            rows.append({"text": text, "label": p["label"], "year": y, "id": p["id"],
                         "flair": p.get("flair")})
    return rows


def build_features():
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120000,
                           sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=120000, sublinear_tf=True, lowercase=True)
    return FeatureUnion([("word", word), ("char", char)])


def n_epochs(model):
    """sklearn returns n_iter_ as an array for older versions and a scalar in 1.9."""
    v = np.ravel(np.asarray(getattr(model, "n_iter_", 0), dtype=object))
    for x in v:
        try:
            return int(x)
        except (TypeError, ValueError):
            continue
    return 0


def sgd(lr, balanced=False, **kw):
    return SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                         class_weight="balanced" if balanced else None,
                         learning_rate="constant", eta0=lr, random_state=SEED,
                         max_iter=MAX_EPOCHS, tol=None, early_stopping=False, **kw)


# ----------------------------------------------------------------------- reporting
def report(name, yte, pred, ytr=None, ptr=None, extra=""):
    acc = accuracy_score(yte, pred)
    mf1 = f1_score(yte, pred, average="macro")
    lines = [f"--- {name} {extra}", f"    test accuracy : {acc:.4f}",
             f"    test macro-F1  : {mf1:.4f}"]
    if ytr is not None:
        lines.append(f"    train accuracy: {accuracy_score(ytr, ptr):.4f}"
                     f"   (fit {accuracy_score(ytr, ptr) - acc:+.4f} vs test)")
    lines.append(classification_report(yte, pred, labels=CLASSES, zero_division=0,
                                       digits=3))
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    lines.append("    confusion matrix (rows = true, cols = pred):")
    lines.append("      " + "".join(c[:12].rjust(14) for c in CLASSES))
    for c, row in zip(CLASSES, cm):
        lines.append("      " + c[:12].ljust(14) + "".join(str(v).rjust(14) for v in row))
    return lines, acc, mf1


def per_year(yte, pred, years):
    by = {}
    for y, t, p in zip(years, yte, pred):
        by.setdefault(y, ([], []))
        by[y][0].append(t)
        by[y][1].append(p)
    out = []
    for y in sorted(by):
        t, p = by[y]
        out.append((y, len(t), accuracy_score(t, p), f1_score(t, p, average="macro",
                                                             zero_division=0)))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    rows = load()
    print(f"loaded {len(rows)} hand-labelled posts, classes={sorted({r['label'] for r in rows})}")

    strat = [f"{r['year']}|{r['label']}" for r in rows]
    train, test = train_test_split(rows, test_size=TEST_SIZE, random_state=SEED,
                                   stratify=strat)
    print(f"train {len(train)}  test {len(test)}")
    print("train class mix:", dict(Counter(r['label'] for r in train)))
    print("test  class mix:", dict(Counter(r['label'] for r in test)))
    print("train year mix :", dict(Counter(r['year'] for r in train)))
    print("test  year mix :", dict(Counter(r['year'] for r in test)))

    feats = build_features()
    t0 = time.time()
    Xtr = feats.fit_transform([r["text"] for r in train])
    Xte = feats.transform([r["text"] for r in test])
    print(f"features: {Xtr.shape[1]} word+char tf-idf features, fit in {time.time() - t0:.1f}s")

    ytr = [r["label"] for r in train]
    yte = [r["label"] for r in test]
    yte_years = [r["year"] for r in test]

    results = {}
    all_lines = []
    header = ["=" * 78,
              f"LogReg on hand-labelled WSB data - {DATA}",
              f"train/test = {len(train)}/{len(test)} ({int((1 - TEST_SIZE) * 100)}/"
              f"{int(TEST_SIZE * 100)}), stratified on (year,label), seed {SEED}",
              f"max_iter={MAX_EPOCHS} epochs, constant learning rate, l2 penalty alpha=1e-4",
              "=" * 78]
    all_lines += header

    # ------------------------------------------------------------- the 2 lr configs
    # each lr is run with and without class_weight='balanced', so the effect of the
    # learning rate can be read independently of the effect of class weighting
    for lr in (0.01, 0.001):
        for balanced in (False, True):
            clf = sgd(lr, balanced)
            t0 = time.time()
            clf.fit(Xtr, ytr)
            dt = time.time() - t0
            pred = clf.predict(Xte)
            ptr = clf.predict(Xtr)
            name = f"logreg(lr={lr}, const, {'bal' if balanced else 'raw'})"
            lines, acc, mf1 = report(name, yte, pred, ytr, ptr,
                                     extra=f"[{dt:.1f}s, {clf.n_iter_} epochs]")
            all_lines += lines
            print("\n".join(lines))
            results[name] = {"model": clf, "acc": acc, "macro_f1": mf1,
                             "train_acc": accuracy_score(ytr, ptr),
                             "epochs": n_epochs(clf),
                             "per_year": per_year(yte, pred, yte_years)}
            joblib.dump({"features": feats, "clf": clf, "classes": CLASSES, "lr": lr,
                         "class_weight": "balanced" if balanced else None},
                        f"{OUT_DIR}/logreg_lr{lr}_{'bal' if balanced else 'raw'}.joblib")

    # ---------------------------------------------------------------- lbfgs reference
    ref = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
    t0 = time.time()
    ref.fit(Xtr, ytr)
    rpred = ref.predict(Xte)
    lines, acc, mf1 = report("logreg(lbfgs, C=4, balanced) - reference, no explicit lr",
                             yte, rpred, ytr, ref.predict(Xtr), extra=f"[{time.time() - t0:.1f}s]")
    all_lines += lines
    print("\n".join(lines))
    results["logreg(lbfgs, C=4, bal) ref"] = {
        "model": ref, "acc": acc, "macro_f1": mf1,
        "train_acc": accuracy_score(ytr, ref.predict(Xtr)),
        "epochs": n_epochs(ref),
        "per_year": per_year(yte, rpred, yte_years)}
    joblib.dump({"features": feats, "clf": ref, "classes": CLASSES},
                f"{OUT_DIR}/logreg_lbfgs.joblib")

    # ----------------------------------------------------------------- lr comparison
    all_lines += ["", "-" * 78, "LEARNING-RATE COMPARISON (same split, same features)",
                  "-" * 78,
                  f"{'config':34s}{'acc':>9s}{'macro-F1':>11s}{'train':>9s}{'epochs':>9s}"]
    for name, r in results.items():
        all_lines.append(f"{name:34s}{r['acc']:9.4f}{r['macro_f1']:11.4f}"
                         f"{r['train_acc']:9.4f}{r['epochs']:9d}")

    # ------------------------------------------------------- per-year breakdown
    all_lines += ["", "-" * 78, "PER-YEAR TEST BREAKDOWN", "-" * 78,
                  f"{'config':34s}" + "".join(y.rjust(16) for y in sorted({r['year'] for r in test}))]
    for name, r in results.items():
        cells = []
        for y, n, a, f1 in r["per_year"]:
            cells.append(f"{a:.3f}({n})".rjust(16))
        all_lines.append(f"{name:34s}" + "".join(cells))

    # ------------------------------------------------------- 5-fold CV stability
    all_lines += ["", "-" * 78, "5-FOLD CV (on all 8000, stratified) - stability check",
                  "-" * 78,
                  f"{'config':34s}{'acc mean':>11s}{'+-':>8s}{'macroF1':>11s}{'+-':>8s}"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    Xall = feats.transform([r["text"] for r in rows])
    yall = [r["label"] for r in rows]
    cv = {}
    for tag, mk in (("logreg(lr=0.01, raw)", lambda: sgd(0.01, False)),
                    ("logreg(lr=0.01, bal)", lambda: sgd(0.01, True)),
                    ("logreg(lr=0.001, raw)", lambda: sgd(0.001, False)),
                    ("logreg(lr=0.001, bal)", lambda: sgd(0.001, True)),
                    ("logreg(lbfgs ref)", lambda: LogisticRegression(
                        max_iter=3000, C=4.0, class_weight="balanced"))):
        accs, f1s = [], []
        for tr, te in skf.split(Xall, strat):
            m = mk()
            m.fit(Xall[tr], [yall[i] for i in tr])
            p = m.predict(Xall[te])
            yt = [yall[i] for i in te]
            accs.append(accuracy_score(yt, p))
            f1s.append(f1_score(yt, p, average="macro"))
        cv[tag] = (float(np.mean(accs)), float(np.std(accs)),
                   float(np.mean(f1s)), float(np.std(f1s)))
        print(f"{tag:20s} cv acc {np.mean(accs):.4f}+-{np.std(accs):.4f} "
              f"macroF1 {np.mean(f1s):.4f}+-{np.std(f1s):.4f}")
    for tag, (a, sa, f, sf) in cv.items():
        all_lines.append(f"{tag:34s}{a:11.4f}{sa:8.4f}{f:11.4f}{sf:8.4f}")

    with open(f"{OUT_DIR}/metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")
    with open(f"{OUT_DIR}/results.json", "w", encoding="utf-8") as f:
        json.dump({"split": {"train": len(train), "test": len(test), "seed": SEED},
                   "holdout": {k: {"acc": v["acc"], "macro_f1": v["macro_f1"],
                                   "per_year": v["per_year"]} for k, v in results.items()},
                   "cv5": cv}, f, indent=2)
    print(f"\nsaved {OUT_DIR}/metrics.txt, {OUT_DIR}/results.json, models in {OUT_DIR}/")


if __name__ == "__main__":
    main()
