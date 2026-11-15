#!/usr/bin/env python3
"""Searcher: sentence-transformer embeddings + FAISS index + SQLite metadata."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from . import wsb_data as wd

HERE = Path(__file__).resolve().parent
# lr3/backend/core -> repository root
_REPO_ROOT = HERE.parents[2]
DEFAULT_MODEL = os.environ.get(
    "LR3_MODEL_DIR", str(_REPO_ROOT / "semantic_search" / "model" / "wsb-minilm"))
DEFAULT_INDEX_DIR = os.environ.get(
    "LR3_INDEX_DIR", str(_REPO_ROOT / "semantic_search" / "index"))


class Searcher:
    def __init__(self, model_path: str = DEFAULT_MODEL, index_dir: str = DEFAULT_INDEX_DIR,
                 device: Optional[str] = None):
        self.model_path = model_path
        self.index_dir = Path(index_dir)
        self.device = device
        self.model = SentenceTransformer(model_path, device=device)
        self.index = faiss.read_index(str(self.index_dir / "wsb.faiss"))
        self.chunk_ids = json.loads((self.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        self.info: Dict[str, Any] = json.loads(
            (self.index_dir / "index_info.json").read_text(encoding="utf-8"))
        self.conn = sqlite3.connect(str(self.index_dir / "metadata.db"), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def close(self) -> None:
        self.conn.close()

    def encode_query(self, query: str, normalize: bool = True) -> tuple[str, np.ndarray]:
        normalized = wd.normalize_query(query) if normalize else query
        vector = self.model.encode(
            [normalized], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False)
        return normalized, np.asarray(vector, dtype="float32")

    def search(self, query: str, top_k: int = 10, normalize: bool = True,
               candidate_k: Optional[int] = None,
               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        normalized, vector = self.encode_query(query, normalize=normalize)
        if filters:
            # Metadata filters are applied after vector search.  The index is
            # small (11.5k chunks), so scanning all candidates is cheap and
            # guarantees that rare labels/tickers are not lost.
            candidate_k = int(self.index.ntotal)
        else:
            candidate_k = candidate_k or max(top_k * 8, 100)
        candidate_k = min(candidate_k, int(self.index.ntotal))

        with self._lock:
            scores, indices = self.index.search(vector, candidate_k)
            seen = set()
            results: List[Dict[str, Any]] = []
            rows = []
            for score, vector_id in zip(scores[0], indices[0]):
                if vector_id < 0 or vector_id >= len(self.chunk_ids):
                    continue
                chunk_id = self.chunk_ids[int(vector_id)]
                row = self.conn.execute(
                    """
                    SELECT c.chunk_id, c.post_id, c.text AS chunk_text, c.chunk_index, c.n_chunks,
                           p.title, p.body, p.year, p.created_utc, p.author, p.score,
                           p.num_comments, p.flair, p.label, p.tickers, p.url
                    FROM chunks c JOIN posts p ON p.post_id = c.post_id
                    WHERE c.chunk_id = ?
                    """,
                    (chunk_id,),
                ).fetchone()
                if row is not None and self._row_matches(row, filters):
                    rows.append((float(score), row))

            for score, row in rows:
                if row["post_id"] in seen:
                    continue
                seen.add(row["post_id"])
                results.append({
                    "rank": len(results) + 1,
                    "score": round(score, 6),
                    "post_id": row["post_id"],
                    "title": row["title"],
                    "snippet": self._snippet(row["chunk_text"]),
                    "body": row["body"],
                    "year": row["year"],
                    "created_utc": row["created_utc"],
                    "author": row["author"],
                    "post_score": row["score"],
                    "num_comments": row["num_comments"],
                    "flair": row["flair"],
                    "label": row["label"],
                    "tickers": json.loads(row["tickers"] or "[]"),
                    "chunk_index": row["chunk_index"],
                    "n_chunks": row["n_chunks"],
                    "url": row["url"],
                })
                if len(results) >= top_k:
                    break
        return {
            "query": query,
            "normalized_query": normalized,
            "top_k": top_k,
            "results": results,
        }

    @staticmethod
    def _row_matches(row: sqlite3.Row, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        labels = filters.get("labels") or []
        if labels and row["label"] not in labels:
            return False
        year_from = filters.get("year_from")
        year_to = filters.get("year_to")
        try:
            year = int(row["year"])
        except (TypeError, ValueError):
            year = 0
        if year_from is not None and year < int(year_from):
            return False
        if year_to is not None and year > int(year_to):
            return False
        if filters.get("min_post_score") is not None:
            try:
                post_score = int(row["post_score"] or 0)
            except (IndexError, KeyError, TypeError, ValueError):
                post_score = int(row["score"] or 0)
            if post_score < int(filters["min_post_score"]):
                return False
        tickers = filters.get("tickers") or []
        if tickers:
            present = {str(item).upper() for item in json.loads(row["tickers"] or "[]")}
            if not present.intersection({str(item).upper() for item in tickers}):
                return False
        return True

    def get_post(self, post_id: str, include_chunks: bool = True) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT post_id, title, body, year, created_utc, author, score,
                       num_comments, flair, label, tickers, url
                FROM posts WHERE post_id = ?
                """,
                (post_id,),
            ).fetchone()
            if row is None:
                return None
            post = {
                "post_id": row["post_id"],
                "title": row["title"],
                "body": row["body"],
                "year": row["year"],
                "created_utc": row["created_utc"],
                "author": row["author"],
                "post_score": row["score"],
                "num_comments": row["num_comments"],
                "flair": row["flair"],
                "label": row["label"],
                "tickers": json.loads(row["tickers"] or "[]"),
                "url": row["url"],
            }
            if include_chunks:
                chunks = self.conn.execute(
                    """
                    SELECT chunk_id, text, chunk_index, n_chunks
                    FROM chunks WHERE post_id = ? ORDER BY chunk_index
                    """,
                    (post_id,),
                ).fetchall()
                post["chunks"] = [
                    {
                        "chunk_id": item["chunk_id"],
                        "chunk_index": item["chunk_index"],
                        "n_chunks": item["n_chunks"],
                        "text": item["text"],
                    }
                    for item in chunks
                ]
            return post

    @staticmethod
    def _snippet(text: str, limit: int = 420) -> str:
        text = " ".join((text or "").split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rsplit(" ", 1)[0] + "…"

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            posts = self.conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "model": os.path.basename(str(self.model_path)),
            "index": str(self.index_dir / "wsb.faiss"),
            "vectors": int(self.index.ntotal),
            "dimension": int(self.index.d),
            "posts": posts,
            "chunks": chunks,
            "info": self.info,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"<Searcher model={self.model_path} vectors={self.index.ntotal}>"
