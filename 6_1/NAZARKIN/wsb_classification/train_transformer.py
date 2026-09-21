#!/usr/bin/env python3
"""Fine-tune a transformer for WSB post classification on the unified pool."""
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)

SEED = 42
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]
L2I = {c: i for i, c in enumerate(CLASSES)}
MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
MAX_LEN = 256
OUT_DIR = "transformer_clf"


def load(path):
    d = json.load(open(path))
    rows = []
    for y, posts in d["years"].items():
        for p in posts:
            text = " ".join(x for x in [p.get("title") or "", p.get("body") or ""] if x).strip()
            rows.append({"text": text, "label": p["label"]})
    return rows


class WsbDataset(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(L2I[r["label"]], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "model:", MODEL_NAME)

    pool = load("teachers_to_label.json") + load("students_to_label.json")
    print("pool:", len(pool), Counter(r["label"] for r in pool))

    train, test = train_test_split(pool, test_size=0.2, random_state=SEED,
                                   stratify=[r["label"] for r in pool])
    print("train", len(train), "test", len(test))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CLASSES))

    # class weights for imbalance
    counts = Counter(r["label"] for r in train)
    weights = torch.tensor(
        [len(train) / (len(CLASSES) * counts[c]) for c in CLASSES], dtype=torch.float)
    print("class weights:", weights.tolist())

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=60,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        fp16=(device == "cuda"),
        seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=WsbDataset(train, tok),
        eval_dataset=WsbDataset(test, tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    pred = trainer.predict(WsbDataset(test, tok))
    preds = np.argmax(pred.predictions, axis=-1)
    yte = pred.label_ids
    print(f"\nTEST accuracy: {accuracy_score(yte, preds):.3f}  macro-F1: {f1_score(yte, preds, average='macro'):.3f}")
    print(classification_report(yte, preds, labels=range(len(CLASSES)),
                                target_names=CLASSES, zero_division=0))
    print(confusion_matrix(yte, preds, labels=range(len(CLASSES))))

    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("saved", OUT_DIR)


if __name__ == "__main__":
    main()
