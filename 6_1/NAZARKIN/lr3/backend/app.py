#!/usr/bin/env python3
"""HTTP backend of WSB Thread Radar.

This process owns the read-only search index and the operational IS database.
It is the only process allowed to execute SQL.  The LLM service talks to it via
the fixed tool endpoints below.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from backend import config  # noqa: E402
from backend.core.semantic_search import DEFAULT_INDEX_DIR, DEFAULT_MODEL, Searcher  # noqa: E402
from backend.is_db import Database  # noqa: E402
from backend.tools.executor import ToolExecutor  # noqa: E402
from backend.tools.schemas import TOOLS  # noqa: E402

DB: Database | None = None
SEARCHER: Searcher | None = None
EXECUTOR: ToolExecutor | None = None


def _bool(value, default=True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


class Handler(BaseHTTPRequestHandler):
    server_version = f"wsb-thread-radar/{config.VERSION}"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Internal-Token")

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            raise ValueError("empty request body")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def _authorized(self) -> bool:
        if not config.INTERNAL_TOKEN:
            return True
        return self.headers.get("X-Internal-Token", "") == config.INTERNAL_TOKEN

    def _need_auth(self) -> bool:
        if not self._authorized():
            self._send(401, {"ok": False, "error": {
                "code": "UNAUTHORIZED", "message": "Invalid internal token.",
                "details": {}, "retryable": False}})
            return False
        return True

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)
        try:
            if path in ("/", "/health"):
                self._send(200, {
                    "status": "ok",
                    "version": config.VERSION,
                    "model": os.path.basename(SEARCHER.model_path),
                    "vectors": int(SEARCHER.index.ntotal),
                    "db": str(DB.path),
                    "tools": len(TOOLS),
                })
            elif path == "/stats":
                self._send(200, self._stats())
            elif path == "/tools":
                self._send(200, {"tools": TOOLS})
            elif path == "/search":
                query = (params.get("q") or params.get("query") or [""])[0]
                if not query.strip():
                    raise ValueError("missing q parameter")
                top_k = int((params.get("top_k") or params.get("n") or ["10"])[0])
                normalize = _bool((params.get("normalize") or [None])[0])
                self._send(200, SEARCHER.search(query, top_k=top_k, normalize=normalize))
            elif path == "/watch-items":
                if not self._need_auth():
                    return
                rows = DB.fetchall("SELECT * FROM watch_items ORDER BY created_at DESC")
                self._send(200, {"items": [dict(row) for row in rows]})
            elif path == "/research-tasks":
                if not self._need_auth():
                    return
                rows = DB.fetchall("SELECT * FROM research_tasks ORDER BY created_at DESC")
                self._send(200, {"items": [dict(row) for row in rows]})
            elif path == "/audit":
                if not self._need_auth():
                    return
                limit = min(max(int((params.get("limit") or ["50"])[0]), 1), 200)
                rows = DB.fetchall(
                    "SELECT * FROM action_log ORDER BY created_at DESC LIMIT ?", (limit,))
                self._send(200, {"items": [dict(row) for row in rows]})
            else:
                self._send(404, {"ok": False, "error": {
                    "code": "NOT_FOUND", "message": f"Unknown endpoint {path}",
                    "details": {}, "retryable": False}})
        except ValueError as exc:
            self._send(400, {"ok": False, "error": {
                "code": "VALIDATION_ERROR", "message": str(exc),
                "details": {}, "retryable": False}})
        except Exception as exc:
            self._send(500, {"ok": False, "error": {
                "code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}",
                "details": {}, "retryable": False}})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            data = self._body()
        except Exception as exc:
            self._send(400, {"ok": False, "error": {
                "code": "VALIDATION_ERROR", "message": f"invalid JSON body: {exc}",
                "details": {}, "retryable": False}})
            return
        try:
            if path == "/search":
                query = str(data.get("query") or data.get("q") or "")
                if not query.strip():
                    raise ValueError("missing query")
                top_k = int(data.get("top_k", data.get("n", 10)))
                normalize = _bool(data.get("normalize"), default=True)
                self._send(200, SEARCHER.search(query, top_k=top_k, normalize=normalize))
                return
            if path == "/tools/call":
                if not self._need_auth():
                    return
                tool = str(data.get("tool") or "")
                arguments = data.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                result = EXECUTOR.call(tool, arguments,
                                       conversation_id=str(data.get("conversation_id") or ""))
                self._send(200, result)
                return
            if path == "/tools/confirm":
                if not self._need_auth():
                    return
                action_id = str(data.get("action_id") or "")
                decision = str(data.get("decision") or "")
                if not action_id:
                    raise ValueError("action_id is required")
                result = EXECUTOR.confirm(
                    action_id, decision,
                    conversation_id=str(data.get("conversation_id") or ""))
                self._send(200, result)
                return
            self._send(404, {"ok": False, "error": {
                "code": "NOT_FOUND", "message": f"Unknown endpoint {path}",
                "details": {}, "retryable": False}})
        except ValueError as exc:
            self._send(400, {"ok": False, "error": {
                "code": "VALIDATION_ERROR", "message": str(exc),
                "details": {}, "retryable": False}})
        except Exception as exc:
            self._send(500, {"ok": False, "error": {
                "code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}",
                "details": {}, "retryable": False}})

    @staticmethod
    def _stats() -> dict:
        stats = SEARCHER.stats()
        stats["is"] = {
            "watch_items": DB.fetchone("SELECT COUNT(*) AS n FROM watch_items")["n"],
            "research_tasks": DB.fetchone("SELECT COUNT(*) AS n FROM research_tasks")["n"],
            "pending_actions": DB.fetchone(
                "SELECT COUNT(*) AS n FROM pending_actions WHERE status='pending'")["n"],
            "action_log": DB.fetchone("SELECT COUNT(*) AS n FROM action_log")["n"],
        }
        return stats


def main() -> int:
    global DB, SEARCHER, EXECUTOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--model", default=config.MODEL_DIR or DEFAULT_MODEL)
    ap.add_argument("--index-dir", default=config.INDEX_DIR or DEFAULT_INDEX_DIR)
    ap.add_argument("--db", default=str(config.IS_DB))
    args = ap.parse_args()

    DB = Database(args.db)
    SEARCHER = Searcher(args.model, args.index_dir)
    EXECUTOR = ToolExecutor(DB, SEARCHER)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"WSB Thread Radar backend on http://{args.host}:{args.port}", flush=True)
    print(f"  index: {args.index_dir}", flush=True)
    print(f"  model: {args.model}", flush=True)
    print(f"  IS DB: {args.db}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
        SEARCHER.close()
        DB.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
