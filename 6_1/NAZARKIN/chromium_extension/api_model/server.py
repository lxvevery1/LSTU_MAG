#!/usr/bin/env python3
"""Zero-dependency JSON HTTP API around WSBClassifier (stdlib only).

    python3 server.py [--host 127.0.0.1] [--port 8000] [--model model.joblib]

Endpoints
    GET  /health    -> {"status":"ok","classes":[...],"version":...}
    GET  /classes   -> {"classes":[...]}
    POST /predict   -> body: {"posts":[{"title":"...","body":"..."}, ...]}
                  or  [{"title":...}, ...]  or  ["single string post", ...]
    POST /predict_one -> body: {"title":"...","body":"..."}
                  or  {"text":"..."}

Response of /predict:
    {"predictions":[{"index":0,"label":"GENUINE_OPINION",
                     "confidence":0.61,"probabilities":{...}}, ...]}

Errors return {"error": "..."} with HTTP 400 (bad input) or 500 (internal).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import WSBClassifier  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLF: WSBClassifier | None = None
VERSION = "1.1.0"


class Handler(BaseHTTPRequestHandler):
    server_version = f"wsb-classifier/{VERSION}"

    # -------------------------------------------------------------- plumbing
    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *a):        # quieter default logging
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % a))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            raise ValueError("empty request body")
        return json.loads(self.rfile.read(n).decode("utf-8"))

    @property
    def clf(self) -> WSBClassifier:
        assert CLF is not None, "classifier not initialised"
        return CLF

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/", "/health"):
            self._send(200, {"status": "ok", "version": VERSION,
                             "model": os.path.basename(self.clf.model_path),
                             "classes": self.clf.classes})
        elif path == "/classes":
            self._send(200, {"classes": self.clf.classes})
        else:
            self._send(404, {"error": f"unknown endpoint {path}"})

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            data = self._body()
        except Exception as e:
            self._send(400, {"error": f"invalid JSON body: {e}"})
            return

        try:
            if path == "/predict":
                posts = data.get("posts") if isinstance(data, dict) else data
                if posts is None:
                    posts = [data]
                self._send(200, {"predictions": self.clf.predict(posts)})
            elif path == "/predict_one":
                self._send(200, self.clf.predict([data]))
            else:
                self._send(404, {"error": f"unknown endpoint {path}"})
        except TypeError as e:
            self._send(400, {"error": f"bad input: {e}"})
        except Exception as e:                                  # pragma: no cover
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> int:
    global CLF
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=os.path.join(HERE, "model.joblib"))
    a = ap.parse_args()

    CLF = WSBClassifier(a.model)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"serving {CLF!r} on http://{a.host}:{a.port}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())