#!/usr/bin/env python3
"""Encode the corpus and store vectors in FAISS, metadata in SQLite."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data" / "corpus.jsonl"))
    ap.add_argument("--model", default=str(HERE / "model" / "wsb-minilm"))
    ap.add_argument("--index-dir", default=str(HERE / "index"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(Path(args.corpus)))
    if not rows:
        raise SystemExit("empty corpus")

    model = SentenceTransformer(args.model, device=args.device)
    started = time.time()
    texts = [row["text"] for row in rows]
    vectors = model.encode(
        texts, batch_size=args.batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "wsb.faiss"))
    (index_dir / "chunk_ids.json").write_text(
        json.dumps([row["chunk_id"] for row in rows], ensure_ascii=False),
        encoding="utf-8")

    db_path = index_dir / "metadata.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE posts (
            post_id TEXT PRIMARY KEY,
            title TEXT,
            body TEXT,
            year TEXT,
            created_utc INTEGER,
            author TEXT,
            score INTEGER,
            num_comments INTEGER,
            flair TEXT,
            label TEXT,
            tickers TEXT,
            url TEXT
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            post_id TEXT,
            text TEXT,
            body TEXT,
            chunk_index INTEGER,
            n_chunks INTEGER,
            FOREIGN KEY(post_id) REFERENCES posts(post_id)
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )

    posts = {}
    chunks = []
    for row in rows:
        post_id = row["post_id"]
        if post_id not in posts:
            posts[post_id] = (
                post_id,
                row.get("title", ""),
                row.get("post_body", row.get("body", "")),
                row.get("year", ""),
                int(row.get("created_utc", 0) or 0),
                row.get("author", ""),
                int(row.get("score", 0) or 0),
                int(row.get("num_comments", 0) or 0),
                row.get("flair", ""),
                row.get("label", ""),
                json.dumps(row.get("tickers", []), ensure_ascii=False),
                f"https://www.reddit.com/r/wallstreetbets/comments/{post_id}/",
            )
        chunks.append((
            row["chunk_id"],
            post_id,
            row["text"],
            row.get("body", ""),
            int(row.get("chunk_index", 0) or 0),
            int(row.get("n_chunks", 1) or 1),
        ))

    conn.executemany(
        "INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", list(posts.values()))
    conn.executemany(
        "INSERT INTO chunks VALUES (?,?,?,?,?,?)", chunks)
    info = {
        "model": str(args.model),
        "corpus": str(args.corpus),
        "vectors": int(index.ntotal),
        "dimension": int(index.d),
        "posts": len(posts),
        "chunks": len(chunks),
        "metric": "inner product on L2-normalised vectors (cosine similarity)",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "encode_seconds": round(time.time() - started, 2),
    }
    conn.executemany("INSERT INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in info.items()])
    conn.commit()
    conn.close()
    (index_dir / "index_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
