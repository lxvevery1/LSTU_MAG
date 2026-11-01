#!/usr/bin/env python3
"""Zero-dependency JSON HTTP API for WSB semantic search."""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantic_search import DEFAULT_INDEX_DIR, DEFAULT_MODEL, Searcher  # noqa: E402

SEARCHER: Searcher | None = None
VERSION = "1.0.0"


def _bool(value, default=True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


class Handler(BaseHTTPRequestHandler):
    server_version = f"wsb-semantic-search/{VERSION}"

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            raise ValueError("empty request body")
        return json.loads(self.rfile.read(n).decode("utf-8"))

    @property
    def searcher(self) -> Searcher:
        assert SEARCHER is not None, "searcher not initialised"
        return SEARCHER

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)
        try:
            if path in ("/", "/health"):
                self._send(200, {"status": "ok", "version": VERSION,
                                 "model": os.path.basename(self.searcher.model_path),
                                 "vectors": int(self.searcher.index.ntotal)})
            elif path == "/stats":
                self._send(200, self.searcher.stats())
            elif path == "/search":
                query = (params.get("q") or params.get("query") or [""])[0]
                if not query.strip():
                    self._send(400, {"error": "missing q parameter"})
                    return
                top_k = int((params.get("top_k") or params.get("n") or ["10"])[0])
                normalize = _bool((params.get("normalize") or [None])[0])
                self._send(200, self.searcher.search(query, top_k=top_k, normalize=normalize))
            else:
                self._send(404, {"error": f"unknown endpoint {path}"})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path != "/search":
            self._send(404, {"error": f"unknown endpoint {path}"})
            return
        try:
            data = self._body()
        except Exception as exc:
            self._send(400, {"error": f"invalid JSON body: {exc}"})
            return
        try:
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            query = str(data.get("query") or data.get("q") or "")
            if not query.strip():
                raise ValueError("missing query")
            top_k = int(data.get("top_k", data.get("n", 10)))
            normalize = _bool(data.get("normalize"), default=True)
            self._send(200, self.searcher.search(query, top_k=top_k, normalize=normalize))
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    global SEARCHER
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    args = ap.parse_args()

    SEARCHER = Searcher(args.model, args.index_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {SEARCHER!r} on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
        SEARCHER.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
