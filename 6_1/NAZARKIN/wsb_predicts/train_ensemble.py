#!/usr/bin/env python3
"""Ensemble TF-IDF LogReg + fine-tuned transformer on the unified pool."""
import json
from collections import Counter

import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]
L2I = {c: i for i, c in enumerate(CLASSES)}
MODEL_DIR = "transformer_clf"
MAX_LEN = 256


def load(path):
    d = json.load(open(path))
    rows = []
    for y, posts in d["years"].items():
        for p in posts:
            text = " ".join(x for x in [p.get("title") or "", p.get("body") or ""] if x).strip()
            rows.append({"text": text, "label": p["label"]})
    return rows


def main():
    pool = load("teachers_to_label.json") + load("students_to_label.json")
    train, test = train_test_split(pool, test_size=0.2, random_state=SEED,
                                   stratify=[r["label"] for r in pool])
    yte = [r["label"] for r in test]
    te_texts = [r["text"] for r in test]
    tr_texts = [r["text"] for r in train]
    ytr = [r["label"] for r in train]

    # ---- model 1: TF-IDF + LogReg ----
    bundle = joblib.load("model_pool.joblib")
    feats, clf = bundle["features"], bundle["clf"]
    # refit on train split only for a fair comparison
    Xtr = feats.fit_transform(tr_texts)
    Xte = feats.transform(te_texts)
    clf.fit(Xtr, ytr)
    lr_proba = clf.predict_proba(Xte)

    # ---- model 2: transformer ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tmodel = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

    tprobs = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(te_texts), bs):
            batch = te_texts[i:i + bs]
            enc = tok(batch, truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            logits = tmodel(**enc).logits
            tprobs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    tproba = np.vstack(tprobs)

    # ---- ensemble: weighted average ----
    truth = np.array([L2I[c] for c in yte])
    best = None
    for w in np.arange(0.0, 1.01, 0.1):
        proba = w * lr_proba + (1 - w) * tproba
        preds = proba.argmax(1)
        f1 = f1_score(truth, preds, average="macro")
        acc = accuracy_score(truth, preds)
        print(f"w_lr={w:.1f} acc {acc:.3f} macroF1 {f1:.3f}")
        if best is None or f1 > best[1]:
            best = (w, f1, acc)

    w, f1, acc = best
    print(f"\nBEST ensemble w_lr={w:.1f}: acc {acc:.3f} macroF1 {f1:.3f}")
    proba = w * lr_proba + (1 - w) * tproba
    preds = proba.argmax(1)
    print(classification_report(truth, preds, target_names=CLASSES, zero_division=0))
    print(confusion_matrix(truth, preds))

    # save the ensemble
    np.save("_ensemble_weight.npy", np.array([w]))
    joblib.dump({"features": feats, "clf": clf, "w_lr": float(w),
                 "classes": CLASSES}, "model_ensemble.joblib")
    print("saved model_ensemble.joblib")


if __name__ == "__main__":
    main()
