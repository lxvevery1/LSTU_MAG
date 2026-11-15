"""Tool gateway: validation, permissions, read execution and staged writes.

The LLM never receives a database handle and never constructs SQL.  Every call
arrives here as a JSON object validated against a fixed schema.  Read tools run
immediately; write tools are converted into a pending action with a preview and
are only applied after a separate confirmation.
"""
from __future__ import annotations

import difflib
import json
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..config import (
    DANGEROUS_TOOLS,
    TASK_STATUSES,
    TASK_TRANSITIONS,
    WATCH_STATUSES,
    WATCH_TRANSITIONS,
    WRITE_TOOLS,
)
from ..is_db import Database, new_id, now_iso
from ..core import wsb_data as wd
from ..core.semantic_search import Searcher
from .errors import ToolError
from .schemas import LABELS, tool_by_name
from .validation import ValidationError, validate

POST_ID_RE = re.compile(r"/comments/([a-z0-9]+)", re.IGNORECASE)
PLAIN_POST_ID_RE = re.compile(r"^[a-z0-9]{3,20}$", re.IGNORECASE)


def _json(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def _clean(value: Any, limit: int = 1000) -> str:
    return " ".join(str(value or "").split())[:limit]


def _resolve_post_id(value: str) -> str:
    match = POST_ID_RE.search(value or "")
    candidate = match.group(1) if match else (value or "").strip()
    if not PLAIN_POST_ID_RE.match(candidate):
        raise ToolError("INVALID_THREAD_ID", "Не удалось определить id треда Reddit.",
                        {"value": value})
    return candidate.lower()


def _resolve_ticker(value: str) -> Optional[str]:
    raw = str(value or "").strip().lstrip("$")
    if not raw:
        return None
    upper = raw.upper()
    if upper in wd.TICKER_COMPANIES:
        return upper
    alias = wd.TICKER_ALIASES.get(raw.lower())
    if alias:
        return alias
    # A user may pass the company name with different capitalisation.
    alias = wd.TICKER_ALIASES.get(upper.lower())
    return alias


def _ticker_error(value: str) -> ToolError:
    keys = sorted(set(wd.TICKER_COMPANIES) | set(wd.TICKER_ALIASES))
    suggestions = difflib.get_close_matches(str(value).lower(), keys, n=5, cutoff=0.45)
    return ToolError(
        "UNKNOWN_TICKER",
        f"Тикер {value!r} не найден в справочнике.",
        {"value": value, "suggestions": suggestions},
    )


def _validate_filters(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ToolError("VALIDATION_ERROR", "filters должен быть объектом.")
    allowed = {"labels", "tickers", "year_from", "year_to", "min_post_score"}
    unknown = set(value) - allowed
    if unknown:
        raise ToolError("VALIDATION_ERROR", "Неизвестные фильтры.",
                        {"unknown": sorted(unknown), "allowed": sorted(allowed)})
    result: Dict[str, Any] = {}
    if "labels" in value:
        labels = value["labels"]
        if not isinstance(labels, list) or any(item not in LABELS for item in labels):
            raise ToolError("VALIDATION_ERROR", "Недопустимая метка.",
                            {"allowed": LABELS})
        result["labels"] = labels
    if "tickers" in value:
        tickers = value["tickers"]
        if not isinstance(tickers, list):
            raise ToolError("VALIDATION_ERROR", "tickers должен быть массивом.")
        resolved = []
        for item in tickers:
            ticker = _resolve_ticker(str(item))
            if not ticker:
                raise _ticker_error(str(item))
            resolved.append(ticker)
        result["tickers"] = sorted(set(resolved))
    for key in ("year_from", "year_to"):
        if key in value:
            year = value[key]
            if not isinstance(year, int) or year < 2019 or year > 2030:
                raise ToolError("VALIDATION_ERROR", f"{key} должен быть годом 2019..2030.")
            result[key] = year
    if result.get("year_from") and result.get("year_to"):
        if result["year_from"] > result["year_to"]:
            raise ToolError("VALIDATION_ERROR", "year_from больше year_to.")
    if "min_post_score" in value:
        score = value["min_post_score"]
        if not isinstance(score, int) or score < 0:
            raise ToolError("VALIDATION_ERROR", "min_post_score должен быть >= 0.")
        result["min_post_score"] = score
    return result


class ToolExecutor:
    def __init__(self, db: Database, searcher: Searcher):
        self.db = db
        self.searcher = searcher

    # ------------------------------------------------------------------ public
    def call(self, tool: str, arguments: Dict[str, Any], conversation_id: str = "") -> Dict[str, Any]:
        started = time.time()
        schema = tool_by_name(tool)
        if schema is None:
            return self._error(tool, ToolError("UNKNOWN_TOOL", f"Неизвестный инструмент {tool!r}."),
                               conversation_id, started)
        try:
            validate(arguments, schema["function"]["parameters"])
            if tool in WRITE_TOOLS:
                plan = self._plan_write(tool, dict(arguments))
                action_id = new_id("act")
                stored = dict(arguments)
                if plan.get("new_id"):
                    stored["__new_id"] = plan["new_id"]
                expires = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(
                    timespec="seconds")
                self.db.insert("pending_actions", {
                    "id": action_id,
                    "conversation_id": conversation_id or "",
                    "tool_name": tool,
                    "arguments": json.dumps(stored, ensure_ascii=False),
                    "preview": json.dumps(plan["preview"], ensure_ascii=False),
                    "status": "pending",
                    "created_at": now_iso(),
                    "expires_at": expires,
                })
                self.db.audit(
                    conversation_id=conversation_id, tool_name=tool,
                    arguments=arguments, decision="stage", status="pending_confirmation",
                    result={"action_id": action_id, "preview": plan["preview"]},
                )
                return {
                    "ok": True,
                    "tool": tool,
                    "status": "pending_confirmation",
                    "action_id": action_id,
                    "preview": plan["preview"],
                    "expires_at": expires,
                    "data": {"pending": True},
                    "meta": {
                        "read_only": False,
                        "dangerous": tool in DANGEROUS_TOOLS or bool(plan.get("dangerous")),
                        "duration_ms": round((time.time() - started) * 1000, 1),
                    },
                }

            data = self._execute_read(tool, arguments)
            result = {
                "ok": True,
                "tool": tool,
                "status": "ok",
                "data": data,
                "meta": {
                    "read_only": True,
                    "side_effect": tool == "check",
                    "duration_ms": round((time.time() - started) * 1000, 1),
                },
            }
            self.db.audit(
                conversation_id=conversation_id, tool_name=tool, arguments=arguments,
                decision="execute", status="ok", result=data,
            )
            return result
        except (ToolError, ValidationError) as exc:
            return self._error(tool, exc, conversation_id, started)
        except Exception as exc:  # defensive: never leak a traceback to the LLM
            wrapped = ToolError("TOOL_EXECUTION_ERROR", f"{type(exc).__name__}: {exc}")
            return self._error(tool, wrapped, conversation_id, started)

    def confirm(self, action_id: str, decision: str, conversation_id: str = "") -> Dict[str, Any]:
        if decision not in {"confirm", "reject"}:
            return {
                "ok": False,
                "status": "error",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "decision должен быть confirm или reject.",
                    "details": {"decision": decision},
                    "retryable": False,
                },
            }
        row = self.db.fetchone("SELECT * FROM pending_actions WHERE id = ?", (action_id,))
        if row is None:
            return {
                "ok": False,
                "status": "error",
                "error": {"code": "NOT_FOUND", "message": "Отложенная операция не найдена.",
                          "details": {"action_id": action_id}, "retryable": False},
            }
        if row["status"] != "pending":
            return {
                "ok": False,
                "status": "error",
                "error": {"code": "CONFLICT",
                          "message": f"Операция уже в состоянии {row['status']}.",
                          "details": {"action_id": action_id}, "retryable": False},
            }
        now = datetime.now(timezone.utc)
        try:
            expires = datetime.fromisoformat(row["expires_at"])
        except ValueError:
            expires = now - timedelta(seconds=1)
        if expires < now:
            self.db.update("pending_actions", action_id,
                           {"status": "expired", "decided_at": now_iso()})
            return {
                "ok": False,
                "status": "error",
                "error": {"code": "CONFIRMATION_EXPIRED",
                          "message": "Время подтверждения истекло. Повторите команду.",
                          "details": {"action_id": action_id}, "retryable": True},
            }

        tool = row["tool_name"]
        arguments = json.loads(row["arguments"])
        if decision == "reject":
            self.db.update("pending_actions", action_id,
                           {"status": "rejected", "decided_at": now_iso()})
            self.db.audit(conversation_id=conversation_id, tool_name=tool,
                          arguments=arguments, decision="reject", status="rejected")
            return {"ok": True, "tool": tool, "status": "rejected", "action_id": action_id,
                    "data": {"decision": "reject"}}

        try:
            plan = self._plan_write(tool, arguments)
            record = self._apply_plan(plan)
            self.db.update("pending_actions", action_id,
                           {"status": "confirmed", "decided_at": now_iso()})
            result = {
                "ok": True,
                "tool": tool,
                "status": "confirmed",
                "action_id": action_id,
                "data": {
                    "entity_type": plan["entity_type"],
                    "record": record,
                },
                "meta": {"read_only": False, "applied": True},
            }
            self.db.audit(conversation_id=conversation_id, tool_name=tool,
                          arguments=arguments, decision="confirm", status="ok",
                          result=result["data"])
            return result
        except (ToolError, ValidationError) as exc:
            self.db.update("pending_actions", action_id,
                           {"status": "failed", "decided_at": now_iso()})
            error = exc.as_dict() if hasattr(exc, "as_dict") else {
                "code": "VALIDATION_ERROR", "message": str(exc), "details": {},
                "retryable": False}
            self.db.audit(conversation_id=conversation_id, tool_name=tool,
                          arguments=arguments, decision="confirm", status="failed",
                          error=error)
            return {"ok": False, "tool": tool, "status": "error", "error": error}

    # ------------------------------------------------------------------ reads
    def _execute_read(self, tool: str, args: Dict[str, Any]) -> Any:
        if tool == "search":
            query = _clean(args.get("query"), 500)
            if not query:
                raise ToolError("VALIDATION_ERROR", "query не должен быть пустым.")
            raw_filters = {
                key: args[key] for key in ("tickers", "labels", "year_from", "year_to")
                if args.get(key) is not None
            }
            filters = _validate_filters(raw_filters)
            filters = {key: value for key, value in filters.items() if value not in (None, [], {})}
            payload = self.searcher.search(
                query, top_k=int(args.get("top_k", 5)), filters=filters or None)
            items = []
            for item in payload["results"]:
                items.append({
                    key: item.get(key) for key in (
                        "rank", "score", "post_id", "title", "snippet", "year",
                        "created_utc", "author", "post_score", "num_comments",
                        "flair", "label", "tickers", "url")
                })
            return {
                "query": query,
                "normalized_query": payload["normalized_query"],
                "filters": filters,
                "count": len(items),
                "results": items,
            }
        if tool == "get":
            entity_type = args["entity_type"]
            object_id = _clean(args["id"], 120)
            if entity_type == "thread":
                post = self.searcher.get_post(object_id, include_chunks=bool(args.get("include_chunks", True)))
                if post is None:
                    raise ToolError("NOT_FOUND", f"Тред {object_id!r} не найден.",
                                    {"entity_type": "thread", "id": object_id})
                return {"entity_type": "thread", "record": post}
            return self._get_entity(entity_type, object_id)
        if tool == "check":
            return self._check(args)
        if tool == "calculate":
            return self._calculate(args)
        raise ToolError("UNKNOWN_TOOL", f"Инструмент {tool!r} не является read-инструментом.")

    def _get_entity(self, entity_type: str, object_id: str) -> Dict[str, Any]:
        table = "watch_items" if entity_type == "watch_item" else "research_tasks"
        row = self.db.fetchone(f"SELECT * FROM {table} WHERE id = ?", (object_id,))
        if row is None:
            raise ToolError("NOT_FOUND", f"{entity_type} {object_id!r} не найден.",
                            {"entity_type": entity_type, "id": object_id})
        return {"entity_type": entity_type, "record": self._record(row, entity_type)}

    def _check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        target = args["target"]
        if target == "service":
            counts = {
                "watch_items": self.db.fetchone("SELECT COUNT(*) AS n FROM watch_items")["n"],
                "research_tasks": self.db.fetchone("SELECT COUNT(*) AS n FROM research_tasks")["n"],
                "pending_actions": self.db.fetchone(
                    "SELECT COUNT(*) AS n FROM pending_actions WHERE status='pending'")["n"],
            }
            return {"target": "service", "status": "ok", "stats": self.searcher.stats(),
                    "is_counts": counts}
        object_id = _clean(args.get("id"), 120)
        if not object_id:
            raise ToolError("VALIDATION_ERROR", "Для проверки нужен id.")
        if target == "research_task":
            row = self.db.fetchone("SELECT * FROM research_tasks WHERE id = ?", (object_id,))
            if row is None:
                raise ToolError("NOT_FOUND", "Задача не найдена.", {"id": object_id})
            return {"target": target, "record": self._record(row, "research_task")}

        row = self.db.fetchone("SELECT * FROM watch_items WHERE id = ?", (object_id,))
        if row is None:
            raise ToolError("NOT_FOUND", "Наблюдение не найдено.", {"id": object_id})
        if row["status"] in {"cancelled", "archived"}:
            raise ToolError("CONFLICT", "Наблюдение отменено или заархивировано.",
                            {"id": object_id, "status": row["status"]})
        watch = self._record(row, "watch_item")
        top_k = int(args.get("top_k", 5))
        previous = set(watch.get("last_seen_ids") or [])
        filters = watch.get("filters") or {}
        if watch["kind"] == "thread":
            post = self.searcher.get_post(watch["target"], include_chunks=False)
            matches = []
            if post:
                matches.append({
                    "post_id": post["post_id"], "title": post["title"],
                    "year": post["year"], "label": post["label"],
                    "tickers": post["tickers"], "url": post["url"],
                    "post_score": post["post_score"], "num_comments": post["num_comments"],
                })
        elif watch["kind"] == "ticker":
            search_filters = dict(filters)
            search_filters["tickers"] = [watch["target"]]
            payload = self.searcher.search(
                watch["target"], top_k=top_k, filters=search_filters)
            matches = payload["results"]
        else:
            payload = self.searcher.search(
                watch["target"], top_k=top_k, filters=filters or None)
            matches = payload["results"]
        current_ids = [item["post_id"] for item in matches]
        new_matches = [item for item in matches if item["post_id"] not in previous]
        seen = list(dict.fromkeys(current_ids + list(previous)))[:100]
        self.db.update("watch_items", object_id, {
            "last_seen_ids": json.dumps(seen, ensure_ascii=False),
            "last_checked_at": now_iso(),
            "updated_at": now_iso(),
        })
        return {
            "target": "watch_item",
            "watch_item": watch,
            "checked_at": now_iso(),
            "matches": matches,
            "new_matches": new_matches,
            "new_count": len(new_matches),
        }

    def _calculate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        metric = args["metric"]
        filters = _validate_filters(args.get("filters"))
        group_by = args.get("group_by")
        limit = int(args.get("limit", 20))
        where, params = [], []
        if filters.get("labels"):
            where.append(f"label IN ({','.join('?' for _ in filters['labels'])})")
            params.extend(filters["labels"])
        if filters.get("year_from"):
            where.append("CAST(year AS INTEGER) >= ?")
            params.append(filters["year_from"])
        if filters.get("year_to"):
            where.append("CAST(year AS INTEGER) <= ?")
            params.append(filters["year_to"])
        if filters.get("min_post_score") is not None:
            where.append("score >= ?")
            params.append(filters["min_post_score"])
        sql = ("SELECT post_id, title, year, label, tickers, score, num_comments "
               "FROM posts")
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = [dict(row) for row in self.searcher.conn.execute(sql, params).fetchall()]
        if filters.get("tickers"):
            wanted = set(filters["tickers"])
            rows = [
                row for row in rows
                if wanted.intersection(set(_json(row.get("tickers"), [])))
            ]
        for row in rows:
            row["tickers"] = _json(row.get("tickers"), [])

        if metric == "count_posts":
            if group_by in {"year", "label", "ticker"}:
                counter: Counter[str] = Counter()
                for row in rows:
                    if group_by == "ticker":
                        for ticker in row["tickers"]:
                            counter[ticker] += 1
                    else:
                        counter[str(row[group_by])] += 1
                return {
                    "metric": metric, "filters": filters,
                    "group_by": group_by,
                    "rows": [{"key": key, "count": value}
                             for key, value in counter.most_common(limit)],
                    "total": len(rows), "sample_size": len(rows),
                }
            return {"metric": metric, "filters": filters, "total": len(rows),
                    "sample_size": len(rows), "rows": [{"key": "all", "count": len(rows)}]}

        if metric == "label_distribution":
            if group_by == "year":
                counter: Counter[tuple[str, str]] = Counter(
                    (str(row["year"]), str(row["label"])) for row in rows)
                data = [{"year": year, "label": label, "count": count}
                        for (year, label), count in sorted(counter.items())]
                return {"metric": metric, "filters": filters, "rows": data,
                        "total": len(rows), "sample_size": len(rows)}
            counter = Counter(str(row["label"]) for row in rows)
            total = len(rows) or 1
            return {
                "metric": metric, "filters": filters,
                "rows": [{"label": label, "count": count,
                          "percent": round(count * 100 / total, 2)}
                         for label, count in counter.most_common()],
                "total": len(rows), "sample_size": len(rows),
            }

        if metric == "ticker_frequency":
            counter = Counter()
            for row in rows:
                for ticker in row["tickers"]:
                    counter[ticker] += 1
            return {
                "metric": metric, "filters": filters,
                "rows": [{"ticker": ticker, "count": count}
                         for ticker, count in counter.most_common(limit)],
                "total": len(rows), "sample_size": len(rows),
            }

        if metric == "engagement_stats":
            count = len(rows)
            total_score = sum(int(row["score"] or 0) for row in rows)
            total_comments = sum(int(row["num_comments"] or 0) for row in rows)
            return {
                "metric": metric, "filters": filters,
                "count": count,
                "avg_score": round(total_score / count, 2) if count else 0,
                "avg_comments": round(total_comments / count, 2) if count else 0,
                "total_score": total_score,
                "total_comments": total_comments,
                "max_score": max((int(row["score"] or 0) for row in rows), default=0),
                "sample_size": count,
            }
        raise ToolError("UNKNOWN_METRIC", f"Метрика {metric!r} не поддерживается.")

    # ------------------------------------------------------------------ writes
    def _plan_write(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        entity_type = args["entity_type"]
        if tool == "create":
            data = args.get("data")
            if not isinstance(data, dict):
                raise ToolError("VALIDATION_ERROR", "data должен быть объектом.")
            if entity_type == "watch_item":
                record = self._prepare_watch(
                    data, args.get("__new_id") or new_id("watch"))
                return {
                    "op": "create", "table": "watch_items", "entity_type": entity_type,
                    "new_id": record["id"], "record": record,
                    "preview": {
                        "summary": f"Создать наблюдение «{record['title'] or record['target']}» "
                                   f"({record['kind']}: {record['target']}).",
                        "entity_type": "watch_item", "entity_id": record["id"],
                        "changes": record,
                    },
                }
            if entity_type == "research_task":
                record = self._prepare_task(
                    data, args.get("__new_id") or new_id("task"))
                return {
                    "op": "create", "table": "research_tasks", "entity_type": entity_type,
                    "new_id": record["id"], "record": record,
                    "preview": {
                        "summary": f"Создать исследовательскую задачу «{record['title']}».",
                        "entity_type": "research_task", "entity_id": record["id"],
                        "changes": record,
                    },
                }
            raise ToolError("VALIDATION_ERROR", "Неизвестный entity_type.")

        object_id = _clean(args.get("id"), 120)
        table = "watch_items" if entity_type == "watch_item" else "research_tasks"
        row = self.db.fetchone(f"SELECT * FROM {table} WHERE id = ?", (object_id,))
        if row is None:
            raise ToolError("NOT_FOUND", f"{entity_type} {object_id!r} не найден.",
                            {"entity_type": entity_type, "id": object_id})
        current = self._record(row, entity_type)

        if tool == "update":
            patch = args.get("patch")
            if not isinstance(patch, dict) or not patch:
                raise ToolError("VALIDATION_ERROR", "patch должен быть непустым объектом.")
            allowed_patch = ({"title", "note", "filters", "priority"}
                             if entity_type == "watch_item"
                             else {"title", "query", "note"})
            unknown = set(patch) - allowed_patch
            if unknown:
                raise ToolError(
                    "VALIDATION_ERROR",
                    "Эти поля нельзя менять через update.",
                    {"unknown": sorted(unknown), "allowed": sorted(allowed_patch)},
                )
            if entity_type == "watch_item":
                updated = self._prepare_watch({**current, **patch}, object_id,
                                              existing=current)
            else:
                updated = self._prepare_task({**current, **patch}, object_id,
                                             existing=current)
            changes = {key: value for key, value in patch.items()
                       if current.get(key) != updated.get(key)}
            return {
                "op": "update", "table": table, "entity_type": entity_type,
                "record": updated, "preview": {
                    "summary": f"Обновить {entity_type} {object_id}: "
                               + ", ".join(changes) + ".",
                    "entity_type": entity_type, "entity_id": object_id,
                    "changes": changes,
                },
            }

        if tool == "cancel":
            if entity_type == "watch_item":
                if current["status"] in {"cancelled", "archived"}:
                    raise ToolError("CONFLICT", "Наблюдение уже отменено или заархивировано.",
                                    {"id": object_id, "status": current["status"]})
            else:
                if current["status"] in {"cancelled", "archived"}:
                    raise ToolError("CONFLICT", "Задача уже отменена или заархивирована.",
                                    {"id": object_id, "status": current["status"]})
            updated = dict(current)
            updated["status"] = "cancelled"
            updated["updated_at"] = now_iso()
            return {
                "op": "update", "table": table, "entity_type": entity_type,
                "record": updated, "dangerous": True,
                "preview": {
                    "summary": f"Отменить {entity_type} {object_id} "
                               f"({current['status']} → cancelled).",
                    "entity_type": entity_type, "entity_id": object_id,
                    "changes": {"status": "cancelled"},
                    "reason": _clean(args.get("reason"), 500),
                },
            }

        if tool == "change_status":
            to_status = args["to_status"]
            allowed = WATCH_STATUSES if entity_type == "watch_item" else TASK_STATUSES
            transitions = WATCH_TRANSITIONS if entity_type == "watch_item" else TASK_TRANSITIONS
            if to_status not in allowed:
                raise ToolError("VALIDATION_ERROR", f"Недопустимый статус {to_status!r}.",
                                {"allowed": list(allowed)})
            if to_status not in transitions.get(current["status"], set()):
                raise ToolError(
                    "INVALID_TRANSITION",
                    f"Переход {current['status']} → {to_status} запрещён.",
                    {"allowed": sorted(transitions.get(current["status"], set()))},
                )
            updated = dict(current)
            updated["status"] = to_status
            updated["updated_at"] = now_iso()
            return {
                "op": "update", "table": table, "entity_type": entity_type,
                "record": updated, "dangerous": to_status in {"cancelled", "archived"},
                "preview": {
                    "summary": f"Изменить статус {entity_type} {object_id}: "
                               f"{current['status']} → {to_status}.",
                    "entity_type": entity_type, "entity_id": object_id,
                    "changes": {"status": to_status},
                    "reason": _clean(args.get("reason"), 500),
                },
            }
        raise ToolError("UNKNOWN_TOOL", f"Инструмент {tool!r} не является write-инструментом.")

    def _apply_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(plan["record"])
        if plan["op"] == "create":
            self.db.insert(plan["table"], record)
            return self._record(
                self.db.fetchone(f"SELECT * FROM {plan['table']} WHERE id = ?",
                                 (record["id"],)),
                plan["entity_type"])
        table = plan["table"]
        values = {key: value for key, value in record.items()
                  if key not in {"id", "created_at"}}
        if table == "watch_items":
            for key in ("filters", "last_seen_ids"):
                if isinstance(values.get(key), (dict, list)):
                    values[key] = json.dumps(values[key], ensure_ascii=False)
        self.db.update(table, record["id"], values)
        row = self.db.fetchone(f"SELECT * FROM {table} WHERE id = ?", (record["id"],))
        return self._record(row, plan["entity_type"])

    def _prepare_watch(self, data: Dict[str, Any], record_id: str,
                       existing: Dict[str, Any] | None = None) -> Dict[str, Any]:
        kind = str(data.get("kind") or "").strip()
        if kind not in {"thread", "ticker", "query"}:
            raise ToolError("VALIDATION_ERROR", "kind должен быть thread, ticker или query.",
                            {"allowed": ["thread", "ticker", "query"]})
        target = _clean(data.get("target"), 500)
        if not target:
            raise ToolError("VALIDATION_ERROR", "target не должен быть пустым.")
        title = _clean(data.get("title"), 300)
        note = _clean(data.get("note"), 1000)
        priority = data.get("priority", 3)
        if not isinstance(priority, int) or not 1 <= priority <= 5:
            raise ToolError("VALIDATION_ERROR", "priority должен быть целым 1..5.")
        status = str(data.get("status") or (existing or {}).get("status") or "active")
        if status not in WATCH_STATUSES:
            raise ToolError("VALIDATION_ERROR", "Недопустимый статус наблюдения.",
                            {"allowed": list(WATCH_STATUSES)})
        filters = _validate_filters(data.get("filters"))
        if kind == "thread":
            target = _resolve_post_id(target)
            post = self.searcher.get_post(target, include_chunks=False)
            if post is None:
                raise ToolError("NOT_FOUND", f"Тред {target!r} не найден в индексе.",
                                {"post_id": target, "hint": "Сначала найдите тред через search."})
            if not title:
                title = post["title"]
        elif kind == "ticker":
            ticker = _resolve_ticker(target)
            if not ticker:
                raise _ticker_error(target)
            target = ticker
            if not title:
                title = f"Мониторинг ${ticker}"
        else:
            if not title:
                title = target[:120]
        now = now_iso()
        previous = existing or {}
        return {
            "id": record_id,
            "kind": kind,
            "target": target,
            "title": title,
            "note": note,
            "filters": json.dumps(filters, ensure_ascii=False),
            "priority": priority,
            "status": status,
            "last_seen_ids": json.dumps(previous.get("last_seen_ids") or [], ensure_ascii=False),
            "last_checked_at": previous.get("last_checked_at"),
            "created_at": previous.get("created_at") or now,
            "updated_at": now,
        }

    def _prepare_task(self, data: Dict[str, Any], record_id: str,
                      existing: Dict[str, Any] | None = None) -> Dict[str, Any]:
        title = _clean(data.get("title"), 300)
        query = _clean(data.get("query"), 500)
        note = _clean(data.get("note"), 1000)
        if not title:
            raise ToolError("VALIDATION_ERROR", "title задачи не должен быть пустым.")
        if not query:
            raise ToolError("VALIDATION_ERROR", "query задачи не должен быть пустым.")
        status = str(data.get("status") or (existing or {}).get("status") or "draft")
        if status not in TASK_STATUSES:
            raise ToolError("VALIDATION_ERROR", "Недопустимый статус задачи.",
                            {"allowed": list(TASK_STATUSES)})
        now = now_iso()
        previous = existing or {}
        return {
            "id": record_id,
            "title": title,
            "query": query,
            "note": note,
            "status": status,
            "result": _clean(data.get("result") or previous.get("result") or "", 4000),
            "created_at": previous.get("created_at") or now,
            "updated_at": now,
        }

    @staticmethod
    def _record(row, entity_type: str) -> Dict[str, Any]:
        record = dict(row)
        if entity_type == "watch_item":
            record["filters"] = _json(record.get("filters"), {})
            record["last_seen_ids"] = _json(record.get("last_seen_ids"), [])
        return record

    def _error(self, tool: str, exc: Exception, conversation_id: str, started: float) -> Dict[str, Any]:
        if isinstance(exc, ToolError):
            error = exc.as_dict()
        elif isinstance(exc, ValidationError):
            error = exc.as_dict()
        else:
            error = {"code": "INTERNAL_ERROR", "message": str(exc),
                     "details": {}, "retryable": False}
        self.db.audit(conversation_id=conversation_id, tool_name=tool,
                      arguments={}, decision="execute", status="error", error=error)
        return {
            "ok": False,
            "tool": tool,
            "status": "error",
            "error": error,
            "meta": {"duration_ms": round((time.time() - started) * 1000, 1)},
        }
