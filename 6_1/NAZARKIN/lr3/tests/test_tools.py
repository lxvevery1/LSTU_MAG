"""Unit tests for the backend tool gateway (no model/FAISS needed)."""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lr3"))

from backend.is_db import Database  # noqa: E402
from backend.tools.executor import ToolExecutor  # noqa: E402


class FakeSearcher:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE posts (
                post_id TEXT PRIMARY KEY, title TEXT, year TEXT, label TEXT,
                tickers TEXT, score INTEGER, num_comments INTEGER
            );
            INSERT INTO posts VALUES ('m5wj5q', 'GME squeeze', '2021', 'NOISE_MEME', '["GME"]', 10, 2);
            INSERT INTO posts VALUES ('k4coin', 'GME 2020', '2020', 'NOISE_MEME', '["GME"]', 5, 1);
            INSERT INTO posts VALUES ('aapl01', 'Apple earnings', '2024', 'ANALYSIS_DD', '["AAPL"]', 30, 7);
        """)
        self.conn.commit()

    def search(self, query, top_k=5, normalize=True, filters=None, candidate_k=None):
        rows = [
            {"rank": 1, "score": 0.9, "post_id": "m5wj5q", "title": "GME squeeze",
             "snippet": "GME squeeze", "year": "2021", "created_utc": 1, "author": "a",
             "post_score": 10, "num_comments": 2, "flair": "YOLO",
             "label": "NOISE_MEME", "tickers": ["GME"], "url": "https://reddit.com/x"},
            {"rank": 2, "score": 0.8, "post_id": "aapl01", "title": "Apple earnings",
             "snippet": "Apple earnings", "year": "2024", "created_utc": 2, "author": "b",
             "post_score": 30, "num_comments": 7, "flair": "DD",
             "label": "ANALYSIS_DD", "tickers": ["AAPL"], "url": "https://reddit.com/y"},
        ]
        return {"query": query, "normalized_query": query, "top_k": top_k,
                "results": rows[:top_k]}

    def get_post(self, post_id, include_chunks=True):
        if post_id != "m5wj5q":
            return None
        return {"post_id": post_id, "title": "GME squeeze", "body": "body",
                "year": "2021", "created_utc": 1, "author": "a", "post_score": 10,
                "num_comments": 2, "flair": "YOLO", "label": "NOISE_MEME",
                "tickers": ["GME"], "url": "https://reddit.com/x",
                "chunks": [{"chunk_id": "c1", "text": "GME squeeze"}] if include_chunks else []}

    def stats(self):
        return {"vectors": 1, "posts": 3}


class ToolGatewayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "is.db")
        self.executor = ToolExecutor(self.db, FakeSearcher())

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_search_and_get_thread(self):
        result = self.executor.call("search", {"query": "GME", "top_k": 2})
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["results"][0]["post_id"], "m5wj5q")
        thread = self.executor.call("get", {"entity_type": "thread", "id": "m5wj5q"})
        self.assertTrue(thread["ok"])
        self.assertEqual(thread["data"]["record"]["post_id"], "m5wj5q")

    def test_create_requires_confirmation_and_is_not_written_until_confirm(self):
        staged = self.executor.call("create", {
            "entity_type": "watch_item",
            "data": {"kind": "thread", "target": "m5wj5q", "note": "test"},
        }, "conv1")
        self.assertEqual(staged["status"], "pending_confirmation")
        self.assertEqual(self.db.fetchone("SELECT COUNT(*) n FROM watch_items")["n"], 0)

        applied = self.executor.confirm(staged["action_id"], "confirm", "conv1")
        self.assertTrue(applied["ok"])
        self.assertEqual(applied["data"]["record"]["target"], "m5wj5q")
        self.assertEqual(applied["data"]["record"]["status"], "active")

    def test_reject_does_not_write(self):
        staged = self.executor.call("create", {
            "entity_type": "watch_item",
            "data": {"kind": "ticker", "target": "TSLA"},
        })
        rejected = self.executor.confirm(staged["action_id"], "reject")
        self.assertTrue(rejected["ok"])
        self.assertEqual(self.db.fetchone("SELECT COUNT(*) n FROM watch_items")["n"], 0)

    def test_update_cancel_and_invalid_transition(self):
        staged = self.executor.call("create", {
            "entity_type": "watch_item",
            "data": {"kind": "ticker", "target": "GME"},
        })
        record = self.executor.confirm(staged["action_id"], "confirm")["data"]["record"]
        watch_id = record["id"]

        update = self.executor.call("update", {
            "entity_type": "watch_item", "id": watch_id,
            "patch": {"note": "new note", "priority": 5},
        })
        self.assertEqual(update["status"], "pending_confirmation")
        updated = self.executor.confirm(update["action_id"], "confirm")["data"]["record"]
        self.assertEqual(updated["note"], "new note")
        self.assertEqual(updated["priority"], 5)

        invalid = self.executor.call("change_status", {
            "entity_type": "watch_item", "id": watch_id, "to_status": "done",
        })
        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["error"]["code"], "VALIDATION_ERROR")

        cancel = self.executor.call("cancel", {
            "entity_type": "watch_item", "id": watch_id, "reason": "test",
        })
        cancelled = self.executor.confirm(cancel["action_id"], "confirm")["data"]["record"]
        self.assertEqual(cancelled["status"], "cancelled")

    def test_calculate_is_allowlisted(self):
        result = self.executor.call("calculate", {
            "metric": "count_posts",
            "filters": {"tickers": ["GME"]},
            "group_by": "year",
        })
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["total"], 2)
        self.assertEqual(result["data"]["rows"][0]["key"], "2021")

    def test_unknown_field_is_rejected(self):
        result = self.executor.call("search", {
            "query": "GME", "sql": "select * from posts",
        })
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "VALIDATION_ERROR")

    def test_pending_action_is_single_use(self):
        staged = self.executor.call("create", {
            "entity_type": "research_task",
            "data": {"title": "GME research", "query": "GME"},
        })
        first = self.executor.confirm(staged["action_id"], "confirm")
        second = self.executor.confirm(staged["action_id"], "confirm")
        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(second["error"]["code"], "CONFLICT")

    def test_invalid_confirmation_decision_is_structured(self):
        staged = self.executor.call("create", {
            "entity_type": "watch_item",
            "data": {"kind": "ticker", "target": "GME"},
        })
        result = self.executor.confirm(staged["action_id"], "maybe")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "VALIDATION_ERROR")


if __name__ == "__main__":
    unittest.main()
