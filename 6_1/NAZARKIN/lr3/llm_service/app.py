#!/usr/bin/env python3
"""HTTP LLM gateway for WSB Thread Radar.

This process knows the LLM and the tool schemas, but it has no database access.
Every action is delegated to the backend through /tools/call and /tools/confirm.
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

from llm_service import config  # noqa: E402
from llm_service.backend_client import BackendClient  # noqa: E402
from llm_service.providers import MockProvider, OpenAICompatibleClient  # noqa: E402
from llm_service.service import AssistantService  # noqa: E402

SERVICE: AssistantService | None = None
BACKEND: BackendClient | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "wsb-thread-radar-llm/1.0"

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

    def _html(self, code: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            raise ValueError("empty request body")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)
        try:
            if path == "/":
                self._html(200, DEBUG_PAGE)
            elif path == "/health":
                backend_health = {}
                error = None
                try:
                    backend_health = BACKEND.health()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                self._send(200, {
                    "status": "ok" if not error else "degraded",
                    "service": "wsb-thread-radar-llm",
                    "provider": SERVICE.provider.name,
                    "model": getattr(SERVICE.provider, "model", "mock"),
                    "backend": backend_health,
                    "backend_error": error,
                })
            elif path == "/tools":
                self._send(200, {"tools": BACKEND.tools()})
            elif path == "/stats":
                self._send(200, BACKEND.stats())
            elif path == "/assistant/history":
                cid = (params.get("conversation_id") or [""])[0]
                if not cid:
                    raise ValueError("conversation_id is required")
                self._send(200, SERVICE.history(cid))
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
            if path == "/assistant/chat":
                message = str(data.get("message") or "")
                context = data.get("context") if isinstance(data.get("context"), dict) else None
                result = SERVICE.chat(
                    message,
                    conversation_id=str(data.get("conversation_id") or "") or None,
                    context=context,
                )
                self._send(200, result)
                return
            if path == "/assistant/confirm":
                action_id = str(data.get("action_id") or "")
                decision = str(data.get("decision") or "")
                if not action_id:
                    raise ValueError("action_id is required")
                result = SERVICE.confirm(
                    action_id, decision,
                    conversation_id=str(data.get("conversation_id") or "") or None,
                )
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


DEBUG_PAGE = """<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>WSB Thread Radar — LLM gateway</title>
<style>body{font:14px sans-serif;max-width:760px;margin:40px auto;padding:0 16px}
#log{border:1px solid #ddd;border-radius:8px;padding:12px;min-height:120px;white-space:pre-wrap}
input{width:75%;padding:8px}button{padding:8px 14px}</style>
<h1>WSB Thread Radar</h1>
<p>Отладочная страница LLM-шлюза. Основной интерфейс — расширение Chromium.</p>
<div id="log"></div><p><input id="q" placeholder="найди посты про GME 2021">
<button onclick="send()">Отправить</button></p>
<script>
let cid=localStorage.getItem('wsb_cid')||'';
function show(x){document.getElementById('log').textContent=JSON.stringify(x,null,2)}
async function send(){const q=document.getElementById('q').value;
 const r=await fetch('/assistant/chat',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({message:q,conversation_id:cid||undefined,context:{}})});
 const d=await r.json(); if(d.conversation_id){cid=d.conversation_id;localStorage.setItem('wsb_cid',cid)} show(d)}
</script></html>"""


def main() -> int:
    global SERVICE, BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--backend", default=config.BACKEND_URL)
    ap.add_argument("--provider", default=config.PROVIDER,
                    choices=["openai_compatible", "mock"])
    ap.add_argument("--base-url", default=config.BASE_URL)
    ap.add_argument("--api-key", default=config.API_KEY)
    ap.add_argument("--model", default=config.MODEL)
    args = ap.parse_args()

    BACKEND = BackendClient(args.backend)
    if args.provider == "mock":
        provider = MockProvider()
    else:
        provider = OpenAICompatibleClient(args.base_url, args.api_key, args.model)
    SERVICE = AssistantService(BACKEND, provider)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"WSB Thread Radar LLM gateway on http://{args.host}:{args.port}", flush=True)
    print(f"  backend: {args.backend}", flush=True)
    print(f"  provider: {provider.name} model={getattr(provider, 'model', '')}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
