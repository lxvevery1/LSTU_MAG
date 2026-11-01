#!/usr/bin/env python3
"""Fine-tune all-MiniLM-L6-v2 on WSB query/passage pairs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--output-dir", default=str(HERE / "model" / "wsb-minilm"))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--sampler", default="no_duplicates", choices=["default", "no_duplicates"])
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = {row["chunk_id"]: row for row in read_jsonl(data_dir / "corpus.jsonl")}
    rows = list(read_jsonl(data_dir / "train_pairs.jsonl"))
    if args.max_pairs > 0:
        random.shuffle(rows)
        rows = rows[: args.max_pairs]

    anchors, positives, negative1, negative2 = [], [], [], []
    for row in rows:
        positive = chunks.get(row["positive_id"])
        if not positive:
            continue
        negatives = [chunks.get(i) for i in row.get("negative_ids", [])]
        negatives = [n for n in negatives if n]
        if not negatives:
            continue
        anchors.append(row["query"])
        positives.append(positive["text"])
        negative1.append(negatives[0]["text"])
        negative2.append(negatives[1]["text"] if len(negatives) > 1 else negatives[0]["text"])

    if not anchors:
        raise SystemExit("no usable training rows")

    dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative1": negative1,
        "negative2": negative2,
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_length

    loss = MultipleNegativesRankingLoss(model, scale=20.0)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        fp16=args.fp16 and device == "cuda",
        optim=args.optim,
        batch_sampler=BatchSamplers.NO_DUPLICATES if args.sampler == "no_duplicates" else BatchSamplers.BATCH_SAMPLER,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()
    model.save(str(output_dir))

    summary = {
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "train_pairs": len(anchors),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "loss": "MultipleNegativesRankingLoss(scale=20)",
        "batch_sampler": "no_duplicates",
        "device": device,
        "seed": args.seed,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
