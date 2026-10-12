#!/usr/bin/env python3
"""Smoke test for api_model/ - verifies the packaged FinBERT model.

Checks:
  1. the FinBERT model directory loads and exposes the 4 expected classes
  2. all three input shapes are accepted (str, (title, body), dict variants)
  3. empty input does not crash
  4. output shape/keys are correct, probabilities are sane and label == argmax
  5. the shipped model reproduces the reported holdout score on the same 80/20
     split the model was trained and selected with
  6. provenance: the base checkpoint is ProsusAI/finbert, the label map is ours
  7. the legacy logreg backend still loads (kept for reference)

Run:  .venv/bin/python smoke_test.py        (needs torch + transformers)
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from predictor import WSBClassifier  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "pickle_classification_for_learning.json")
EXPECTED_CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]
failures = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def main() -> int:
    meta = json.load(open(os.path.join(HERE, "metadata.json"), encoding="utf-8"))

    print("1. load")
    clf = WSBClassifier()
    check("model loads", True, clf.model_path)
    check("backend is finbert", clf.backend == "finbert", clf.backend)
    check("4 expected classes", clf.classes == EXPECTED_CLASSES, str(clf.classes))
    check("no forbidden class present", "UNCERTAIN" not in clf.classes)
    check("label order matches metadata", clf.classes == meta["classes"])

    print("2. input shapes")
    r_str = clf.predict_text("GME to the moon, calls 12/31 $80")
    r_tuple = clf.predict([("GME to the moon, calls 12/31 $80", "")])[0]
    r_dict = clf.predict([{"title": "GME to the moon, calls 12/31 $80", "body": ""}])[0]
    check("str accepted", isinstance(r_str.get("label"), str), r_str["label"])
    check("(title, body) accepted", r_tuple["label"] == r_str["label"])
    check("dict accepted", r_dict["label"] == r_str["label"])
    r_self = clf.predict([{"title": "x", "selftext": "y"}])[0]
    check("selftext alias accepted", isinstance(r_self.get("label"), str))

    print("3. edge cases")
    r_empty = clf.predict_one("", "")
    check("empty input does not crash", r_empty["empty_input"] is True)
    check("batch of empty still returns labels", isinstance(r_empty["label"], str))
    try:
        clf.predict([12345])
        check("bad type raises TypeError", False)
    except TypeError:
        check("bad type raises TypeError", True)
    check("empty batch -> []", clf.predict([]) == [])

    print("4. output shape")
    rec = clf.predict_one("NVDA reported Q3 revenue of $35.1B, up 17% YoY",
                          "Beat on the high end, margins 75%, guidance raised.")
    for k in ("index", "label", "confidence", "probabilities", "empty_input"):
        check(f"key '{k}' present", k in rec)
    probs = rec["probabilities"]
    check("probabilities cover all classes", set(probs) == set(EXPECTED_CLASSES))
    check("probabilities sum to 1", abs(sum(probs.values()) - 1.0) < 1e-4,
          f"{sum(probs.values()):.6f}")
    check("confidence == max prob",
          abs(rec["confidence"] - max(probs.values())) < 1e-6)
    check("label == argmax(probabilities)",
          rec["label"] == max(probs, key=probs.get), rec["label"])
    batch = clf.predict([
        {"title": "GME to the moon"},
        {"title": "NVDA Q3 revenue beat 17%"},
        {"title": "lol"},
        {"title": "My thesis: at 35x forward with 20% growth and 75% margins "
                  "the multiple compresses to 25x on any slowdown, so downside "
                  "to 100 from 180 is the base case", "body": "risks: customer "
                  "concentration, export controls, supply chain"},
    ])
    mismatch = [b["index"] for b in batch if b["label"] != max(b["probabilities"],
                                                              key=b["probabilities"].get)]
    check("label == argmax(prob) across a batch", not mismatch, f"bad={mismatch}")
    check("labels unique-ish across 4 diverse posts",
          len({b["label"] for b in batch}) >= 2,
          str([b["label"] for b in batch]))
    check("labels-only mode", clf.predict_labels(["GME to the moon"])[0]
          in EXPECTED_CLASSES)

    print("5. held-out score is this model's own")
    if os.path.exists(DATA):
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split

        d = json.load(open(DATA, encoding="utf-8"))
        rows = [p for y in sorted(d["years"]) for p in d["years"][y]]

        def text_of(p):
            return " ".join(x for x in [(p.get("title") or "").strip(),
                                        (p.get("body") or "").strip()] if x).strip()

        texts = [text_of(p) for p in rows]
        y = [p["label"] for p in rows]
        strat = [f"{p['year']}|{p['label']}" for p in rows]
        _, te = train_test_split(range(len(rows)), test_size=0.2, random_state=42,
                                 stratify=strat)

        pred = clf.predict_labels([{"title": texts[i]} for i in te])
        yt = [y[i] for i in te]
        acc = accuracy_score(yt, pred)
        f1 = f1_score(yt, pred, average="macro")
        rep = meta["metrics"]["holdout_20pct"]
        print(f"      recomputed on the 1600 held-out posts: acc {acc:.4f} "
              f"macroF1 {f1:.4f}")
        print(f"      stated in metadata.json              : acc {rep['accuracy']:.4f} "
              f"macroF1 {rep['macro_f1']:.4f}")
        check("held-out accuracy reproduces", abs(acc - rep["accuracy"]) < 1e-3,
              f"{acc:.4f}")
        check("held-out macro-F1 reproduces", abs(f1 - rep["macro_f1"]) < 2e-3,
              f"{f1:.4f}")

        # cross-check against the predictions recorded at training time
        pred_file = os.path.join(HERE, "finbert", "test_predictions.json")
        if os.path.exists(pred_file):
            stored = json.load(open(pred_file, encoding="utf-8"))
            agree = sum(1 for s, p in zip(stored, pred) if s["pred"] == p)
            check("labels match the training-time predictions exactly",
                  agree == len(stored), f"{agree}/{len(stored)}")
    else:
        print("      skipped (training data not found)")

    print("6. provenance")
    check("base checkpoint is ProsusAI/finbert",
          meta["provenance"]["base_model"] == "ProsusAI/finbert",
          meta["provenance"]["base_model"])
    pf = os.path.join(HERE, "finbert", "config.json")
    if os.path.exists(pf):
        cfg = json.load(open(pf, encoding="utf-8"))
        check("config model_type is bert", cfg.get("model_type") == "bert",
              str(cfg.get("model_type")))
        id2label = {int(k): v for k, v in cfg["id2label"].items()}
        check("label map is our 4 classes",
              [id2label[i] for i in sorted(id2label)] == EXPECTED_CLASSES,
              str(id2label))
        check("label map has 4 entries", len(id2label) == 4)

    print("7. legacy logreg backend still loads")
    legacy = os.path.join(HERE, "model.joblib")
    if os.path.exists(legacy):
        old = WSBClassifier(legacy)
        check("legacy backend loads", old.backend == "sklearn", old.backend)
        check("legacy classes match", old.classes == EXPECTED_CLASSES)
        check("legacy predicts", old.predict_text("GME to the moon")["label"]
              in EXPECTED_CLASSES)
    else:
        print("      skipped (model.joblib not present)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
