#!/usr/bin/env python3
"""Fine-tune FinBERT (ProsusAI/finbert) on the hand-labelled WSB dataset.

This replaces the TF-IDF + logistic-regression model. It uses the *exact same*
train/test split as train_logreg_hand.py

    strat = f"{year}|{label}"
    train_test_split(rows, test_size=0.2, random_state=42, stratify=strat)

so the two models are evaluated on identical held-out posts and their numbers
are directly comparable. The script asserts that the test ids match the logreg
split file when that file is present.

The 3-way sentiment head of FinBERT (positive/negative/neutral) is discarded and
replaced with a fresh 4-way classification head for our classes.

Env overrides:
    MODEL_NAME   base checkpoint      (default ProsusAI/finbert)
    MAX_LEN      truncation length    (default 256)
    EPOCHS       training epochs      (default 4)
    BATCH        train batch size     (default 16)
    LR           learning rate        (default 2e-5)
    OUT_DIR      output directory     (default finbert_out)
"""
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

SEED = 42
DATA = "pickle_classification_for_learning.json"
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]
L2I = {c: i for i, c in enumerate(CLASSES)}
TEST_SIZE = 0.2

MODEL_NAME = os.environ.get("MODEL_NAME", "ProsusAI/finbert")
MAX_LEN = int(os.environ.get("MAX_LEN", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BATCH = int(os.environ.get("BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR = float(os.environ.get("LR", "2e-5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
OUT_DIR = os.environ.get("OUT_DIR", "finbert_out")
LOGREG_SPLIT = "logreg_hand/split_test_ids.json"


def load():
    """Same row order and text join as train_logreg_hand.load()."""
    d = json.load(open(DATA, encoding="utf-8"))
    rows = []
    for y, posts in d["years"].items():
        for p in posts:
            text = " ".join(x for x in [(p.get("title") or "").strip(),
                                        (p.get("body") or "").strip()] if x).strip()
            rows.append({"text": text, "label": p["label"], "year": y, "id": p["id"]})
    return rows


def split(rows):
    strat = [f"{r['year']}|{r['label']}" for r in rows]
    return train_test_split(rows, test_size=TEST_SIZE, random_state=SEED,
                            stratify=strat)


class WsbDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=MAX_LEN)
        enc["labels"] = L2I[r["label"]]
        return enc


class WeightedTrainer(Trainer):
    """Cross-entropy with balanced class weights (matches the logreg 'bal' setting)."""

    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self._cw = class_weights

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self._cw.to(outputs.logits.device) if self._cw is not None else None
        loss = F.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")}


def report(name, yte_idx, pred, ytr_idx=None, ptr=None, extra=""):
    lines = [f"--- {name} {extra}", f"    test accuracy : {accuracy_score(yte_idx, pred):.4f}",
             f"    test macro-F1  : {f1_score(yte_idx, pred, average='macro'):.4f}"]
    if ytr_idx is not None:
        acc_tr = accuracy_score(ytr_idx, ptr)
        lines.append(f"    train accuracy: {acc_tr:.4f}"
                     f"   (fit {acc_tr - accuracy_score(yte_idx, pred):+.4f} vs test)")
    lines.append(classification_report(yte_idx, pred, labels=range(len(CLASSES)),
                                       target_names=CLASSES, zero_division=0, digits=3))
    cm = confusion_matrix(yte_idx, pred, labels=range(len(CLASSES)))
    lines.append("    confusion matrix (rows = true, cols = pred):")
    lines.append("      " + "".join(c[:12].rjust(14) for c in CLASSES))
    for c, row in zip(CLASSES, cm):
        lines.append("      " + c[:12].ljust(14) + "".join(str(v).rjust(14) for v in row))
    return lines


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} base={MODEL_NAME} max_len={MAX_LEN} epochs={EPOCHS} "
          f"batch={BATCH} lr={LR}")

    rows = load()
    train, test = split(rows)
    print(f"loaded {len(rows)} posts -> train {len(train)} / test {len(test)}")

    test_ids = sorted(r["id"] for r in test)
    if os.path.exists(LOGREG_SPLIT):
        ref = json.load(open(LOGREG_SPLIT))
        if sorted(ref.get("test_ids", [])) != test_ids:
            raise SystemExit("SPLIT MISMATCH vs logreg_hand/split_test_ids.json")
        print("split verified identical to the logreg split")
    with open(f"{OUT_DIR}/split_test_ids.json", "w") as f:
        json.dump({"test_ids": test_ids, "n_train": len(train), "n_test": len(test),
                   "seed": SEED, "stratify": "year|label"}, f)

    train_counts = Counter(r["label"] for r in train)
    print("train class mix:", dict(train_counts))
    weights = torch.tensor(
        [len(train) / (len(CLASSES) * train_counts[c]) for c in CLASSES],
        dtype=torch.float)
    print("class weights:", [round(w, 3) for w in weights.tolist()])

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASSES),
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id=L2I, ignore_mismatched_sizes=True)

    steps_per_epoch = max(1, len(train) // (BATCH * GRAD_ACCUM))
    total_steps = MAX_STEPS if MAX_STEPS > 0 else steps_per_epoch * EPOCHS
    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * total_steps)),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        fp16=False,
        max_steps=MAX_STEPS,
        seed=SEED,
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model, args=args,
        train_dataset=WsbDataset(train, tok),
        eval_dataset=WsbDataset(test, tok),
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_state()

    # ------------------------------------------------------------------ evaluate
    output = trainer.predict(WsbDataset(test, tok))
    logits = output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = probs.argmax(axis=-1)
    yte = np.array([L2I[r["label"]] for r in test])

    Ptr = torch.softmax(torch.tensor(
        trainer.predict(WsbDataset(train, tok)).predictions), dim=-1).numpy()
    ptr = Ptr.argmax(axis=-1)
    ytr = np.array([L2I[r["label"]] for r in train])

    lines = ["=" * 78,
             f"FinBERT ({MODEL_NAME}) on hand-labelled WSB data - {DATA}",
             f"train/test = {len(train)}/{len(test)} (80/20), stratified on "
             f"(year,label), seed {SEED}",
             f"max_len={MAX_LEN}, epochs={EPOCHS}, batch={BATCH}, lr={LR}, "
             f"balanced class weights, fp32",
             "=" * 78]
    lines += report("finbert(balanced)", yte, pred, ytr, ptr)

    # per-year breakdown
    by = {}
    for r, y, p in zip(test, yte, pred):
        by.setdefault(r["year"], ([], []))
        by[r["year"]][0].append(y)
        by[r["year"]][1].append(p)
    lines += ["", "-" * 78, "PER-YEAR TEST BREAKDOWN", "-" * 78,
              f"{'finbert':34s}" + "".join(y.rjust(16) for y in sorted(by))]
    cells = []
    for y in sorted(by):
        t, p = by[y]
        cells.append(f"{accuracy_score(t, p):.3f}({len(t)})".rjust(16))
    lines.append(f"{'finbert':34s}" + "".join(cells))

    text = "\n".join(lines)
    print("\n" + text)
    with open(f"{OUT_DIR}/metrics.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # ------------------------------------------------------------------- persist
    model.gradient_checkpointing_disable()
    model.config.gradient_checkpointing = False
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    with open(f"{OUT_DIR}/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME, "classes": CLASSES, "max_len": MAX_LEN,
            "epochs": EPOCHS, "batch": BATCH, "lr": LR,
            "class_weight": "balanced", "seed": SEED,
            "split": {"train": len(train), "test": len(test), "seed": SEED},
            "holdout": {
                "accuracy": float(accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro")),
                "train_accuracy": float(accuracy_score(ytr, ptr)),
                "per_year": {y: {"n": len(by[y][0]),
                                 "accuracy": float(accuracy_score(*by[y])),
                                 "macro_f1": float(f1_score(by[y][0], by[y][1],
                                                            average="macro",
                                                            zero_division=0))}
                             for y in sorted(by)},
            },
        }, f, indent=2)

    # keep the raw test predictions so smoke_test.py can recompute the metrics
    with open(f"{OUT_DIR}/test_predictions.json", "w", encoding="utf-8") as f:
        json.dump([{"id": r["id"], "year": r["year"], "true": r["label"],
                    "pred": CLASSES[int(p)], "probabilities":
                    {c: round(float(probs[i][j]), 6) for j, c in enumerate(CLASSES)}}
                   for i, (r, p) in enumerate(zip(test, pred))], f, indent=2)

    print(f"\nsaved {OUT_DIR}/ (model, tokenizer, metrics.txt, results.json, "
          f"test_predictions.json, split_test_ids.json)")


if __name__ == "__main__":
    main()
