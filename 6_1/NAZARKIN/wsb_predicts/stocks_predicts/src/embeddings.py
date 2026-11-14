#!/usr/bin/env python3
"""Cache frozen FinBERT [CLS] embeddings for all posts (baseline B3)."""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import load_config, project_path, read_json, setup_logging

LOG = logging.getLogger("embeddings")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = project_path(cfg["labels"]["classifier_checkpoint"])
    obj = read_json(project_path(cfg["paths"]["posts_labeled"]))
    posts = [p for arr in obj["years"].values() for p in arr]
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModel.from_pretrained(str(ckpt)).to(device).eval()
    texts = [((p.get("title") or "") + "\n\n" + (p.get("body") or "")).strip() for p in posts]
    vectors = []
    t0 = time.time()
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        enc = tokenizer(batch, truncation=True, max_length=args.max_len,
                        padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out = model(**enc)
            vectors.append(out.last_hidden_state[:, 0, :].float().cpu().numpy())
        if (start // args.batch_size) % 20 == 0:
            LOG.info("encoded %s / %s", min(start + args.batch_size, len(texts)), len(texts))
    emb = np.vstack(vectors).astype(np.float32)
    out_path = project_path(cfg["paths"]["embeddings"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, post_ids=np.array([p["id"] for p in posts]),
                        embeddings=emb, max_len=args.max_len)
    LOG.info("saved %s shape=%s in %.1fs", out_path, emb.shape, time.time() - t0)


if __name__ == "__main__":
    main()
