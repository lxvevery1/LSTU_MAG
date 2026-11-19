"""LLM providers: OpenAI-compatible HTTP client and deterministic mock."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from . import config


class LLMError(Exception):
    def __init__(self, code: str, message: str, status: int = 0,
                 retryable: bool = False, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable
        self.details = details or {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
            "status": self.status,
        }


class OpenAICompatibleClient:
    name = "openai_compatible"

    def __init__(self, base_url: str = config.BASE_URL, api_key: str = config.API_KEY,
                 model: str = config.MODEL, timeout: float = config.TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": config.TEMPERATURE,
        }
        if config.MAX_TOKENS > 0:
            payload["max_tokens"] = config.MAX_TOKENS
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "wsb-thread-radar/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except ValueError:
                parsed = {}
            error = parsed.get("error") if isinstance(parsed, dict) else {}
            message = (error or {}).get("message") or body[:500] or str(exc)
            code = (error or {}).get("type") or "LLM_HTTP_ERROR"
            if exc.code == 429:
                code = "LLM_RATE_LIMITED"
            elif exc.code in {401, 403}:
                code = "LLM_AUTH_ERROR"
            raise LLMError(code, message, status=exc.code,
                           retryable=exc.code == 429 or exc.code >= 500,
                           details={"provider": self.name, "model": self.model})
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError("LLM_UNAVAILABLE", f"{type(exc).__name__}: {exc}",
                           retryable=True, details={"provider": self.name})
        except Exception as exc:
            raise LLMError("LLM_ERROR", f"{type(exc).__name__}: {exc}",
                           details={"provider": self.name})

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("LLM_BAD_RESPONSE", "Provider returned no message.",
                           details={"response": str(data)[:1000]})
        return {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
        }


def _tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"call_{name}_{abs(hash(json.dumps(arguments, sort_keys=True))) % 1000000}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _message_text(messages: List[Dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def _last_tool(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Only look at tool messages that belong to the current user turn.  A tool
    # result from a previous turn must not make the mock ignore a new command.
    for message in reversed(messages):
        if message.get("role") == "user":
            return None
        if message.get("role") == "tool":
            try:
                return json.loads(message.get("content") or "{}")
            except ValueError:
                return None
    return None


def _last_tool_name(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return ""
        if message.get("role") == "tool":
            return str(message.get("name") or "")
    return ""


TICKER_MAP = {
    "apple": "AAPL", "эпл": "AAPL", "эппл": "AAPL", "аппл": "AAPL",
    "tesla": "TSLA", "тесла": "TSLA", "tesla": "TSLA",
    "gamestop": "GME", "геймстоп": "GME", "gme": "GME",
    "nvidia": "NVDA", "нвидиа": "NVDA", "нвидия": "NVDA",
    "microsoft": "MSFT", "майкрософт": "MSFT",
    "amazon": "AMZN", "амазон": "AMZN",
    "google": "GOOGL", "гугл": "GOOGL",
    "meta": "META", "мета": "META",
    "netflix": "NFLX", "нетфликс": "NFLX",
    "amd": "AMD", "spy": "SPY", "qqq": "QQQ",
}
STOP_TICKERS = {
    "THE", "AND", "FOR", "WITH", "POST", "POSTS", "FROM", "THIS", "THAT",
    "SEARCH", "FIND", "SHOW", "SAVE", "ADD", "CHECK", "CANCEL", "UPDATE",
    "STATUS", "YEAR", "YEARS", "TOP", "ALL",
}


def _extract_tickers(text: str) -> List[str]:
    found = []
    lower = text.lower()
    for word, ticker in TICKER_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", lower) and ticker not in found:
            found.append(ticker)
    for match in re.findall(r"\$?([A-Z]{2,5})\b", text):
        if match not in STOP_TICKERS and match not in found:
            found.append(match)
    return found[:10]


def _extract_years(text: str) -> tuple[Optional[int], Optional[int]]:
    years = [int(item) for item in re.findall(r"\b(20\d\d)\b", text)]
    if not years:
        return None, None
    return min(years), max(years)


def _extract_id(text: str, prefix: str) -> Optional[str]:
    match = re.search(rf"\b({prefix}_[a-z0-9]+)\b", text, re.IGNORECASE)
    return match.group(1) if match else None


def _extract_post_id(text: str) -> Optional[str]:
    match = re.search(r"/comments/([a-z0-9]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    # Reddit post ids are short base-36 tokens such as ``m5wj5q``.  Require at
    # least one digit so ordinary words ("покажи", "thread") are not mistaken
    # for ids.
    for token in re.findall(r"(?<![a-z0-9_])[a-z0-9]{5,10}(?![a-z0-9_])", text, re.IGNORECASE):
        if re.search(r"\d", token) and re.search(r"[a-z]", token, re.IGNORECASE):
            return token.lower()
    return None


def _extract_top_k(text: str) -> int:
    match = re.search(r"\b(\d{1,2})\s*(?:пост|поста|постов|результат|результата|шт|штук|тред|треда)", text, re.IGNORECASE)
    if match:
        return max(1, min(20, int(match.group(1))))
    match = re.search(r"\btop\s*(\d{1,2})\b", text, re.IGNORECASE)
    if match:
        return max(1, min(20, int(match.group(1))))
    return 5


class MockProvider:
    """Deterministic fallback that uses the same backend tools as the real LLM.

    It is not a replacement for an LLM in the report; it exists so tests and a
    demo do not fail when the free provider quota is exhausted.
    """
    name = "mock"

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        # The service appends a context block to the user message.  Commands are
        # parsed only from the user's own text; context is read separately.
        user = _message_text(messages, "user").split("\n\nКонтекст", 1)[0]
        lowered = user.lower()
        last_tool = _last_tool(messages)
        last_tool_name = _last_tool_name(messages)

        if last_tool is not None:
            if last_tool_name == "search" and last_tool.get("ok") and (
                    "сохран" in lowered or "добав" in lowered or "монитор" in lowered):
                results = (last_tool.get("data") or {}).get("results") or []
                if results:
                    return self._call("create", {
                        "entity_type": "watch_item",
                        "data": {
                            "kind": "thread",
                            "target": results[0]["post_id"],
                            "title": results[0].get("title") or "",
                            "note": "Сохранено через ассистента",
                        },
                    })
            return {"role": "assistant", "content": self._summarize(last_tool_name, last_tool),
                    "tool_calls": []}

        if any(word in lowered for word in ("найди", "поищи", "search", "find", "покажи посты")):
            query = re.sub(
                r"^(?:найди|поищи|search|find|покажи)\s+(?:посты|пост|треды|тред|posts?)?\s*",
                "", user.strip(), flags=re.IGNORECASE).strip(" .:—-")
            if not query:
                return self._ask("Уточните, что именно искать.")
            args: Dict[str, Any] = {"query": query, "top_k": _extract_top_k(user)}
            tickers = _extract_tickers(user)
            if tickers:
                args["tickers"] = tickers
            year_from, year_to = _extract_years(user)
            if year_from:
                args["year_from"] = year_from
                args["year_to"] = year_to
            return self._call("search", args)

        if any(word in lowered for word in ("отмени", "отмена", "cancel")):
            object_id = (_extract_id(user, "watch") or _extract_id(user, "task")
                         or self._context(messages).get("last_entity_id"))
            if not object_id:
                return self._ask("Уточните id наблюдения или задачи, которую нужно отменить.")
            entity = "research_task" if object_id.startswith("task_") else "watch_item"
            return self._call("cancel", {"entity_type": entity, "id": object_id,
                                         "reason": "Отменено пользователем через ассистента"})

        if (re.search(r"\b(?:обнови|обновить|update)\b", lowered)
                and ("заметк" in lowered or "note" in lowered
                     or "приоритет" in lowered or "priority" in lowered)):
            object_id = (_extract_id(user, "watch") or _extract_id(user, "task")
                         or self._context(messages).get("last_entity_id"))
            if not object_id:
                return self._ask("Уточните id объекта, который нужно обновить.")
            patch: Dict[str, Any] = {}
            note = re.search(r"(?:заметк[ау]?|note)\s*[:=—-]\s*(.+)$", user, re.IGNORECASE)
            if note:
                note_text = re.split(
                    r",?\s*(?:приоритет|priority)\s*[:= ]\s*",
                    note.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
                patch["note"] = note_text.strip(" ,")
            priority = re.search(r"(?:приоритет|priority)\s*[:= ]\s*([1-5])", lowered)
            if priority:
                patch["priority"] = int(priority.group(1))
            if not patch:
                return self._ask("Укажите новую заметку или приоритет (1–5).")
            entity = "research_task" if object_id.startswith("task_") else "watch_item"
            if entity == "research_task" and "priority" in patch:
                patch.pop("priority")
            return self._call("update", {"entity_type": entity, "id": object_id,
                                         "patch": patch})

        if any(word in lowered for word in ("проверь", "проверить", "check")):
            context = self._context(messages)
            task_id = _extract_id(user, "task")
            if task_id:
                return self._call("check", {"target": "research_task", "id": task_id})
            object_id = _extract_id(user, "watch") or context.get("last_entity_id")
            if object_id and object_id.startswith("task_"):
                return self._call("check", {"target": "research_task", "id": object_id})
            if object_id and object_id.startswith("watch_"):
                return self._call("check", {"target": "watch_item", "id": object_id})
            return self._call("check", {"target": "service"})

        if "задач" in lowered and re.search(r"(?:созда|добав|завед|нов)", lowered):
            match = re.search(r"(?:про|по|о)\s+(.+)$", user, re.IGNORECASE)
            subject = (match.group(1) if match else user).strip(" .")
            subject = subject[:200]
            return self._call("create", {
                "entity_type": "research_task",
                "data": {"title": f"Исследование: {subject[:100]}", "query": subject},
            })

        if (re.search(r"(?:сохран|добав|монитор)", lowered)
                or re.search(r"\b(?:watch|save)\b", lowered)):
            context = self._context(messages)
            tickers = _extract_tickers(user)
            explicit_thread = _extract_post_id(user)
            current_ref = any(word in lowered for word in
                              ("этот", "эту", "это", "текущ", "this", "current"))
            # An explicit post id or ticker wins over the automatically attached
            # page context; context is used only for "this thread".
            target = explicit_thread or (context.get("post_id") if current_ref else None)
            if target and not (tickers and not current_ref):
                data = {"kind": "thread", "target": target,
                        "note": "Сохранено через ассистента"}
                if context.get("title") and current_ref:
                    data["title"] = context["title"]
                return self._call("create", {"entity_type": "watch_item", "data": data})
            if tickers:
                return self._call("create", {
                    "entity_type": "watch_item",
                    "data": {"kind": "ticker", "target": tickers[0]},
                })
            return self._ask("Уточните, что сохранить: конкретный тред (post_id/URL), тикер или поисковый запрос.")

        if any(word in lowered for word in ("сколько", "статистик", "распределен", "calculate", "count")):
            filters: Dict[str, Any] = {}
            tickers = _extract_tickers(user)
            if tickers:
                filters["tickers"] = tickers
            year_from, year_to = _extract_years(user)
            if year_from:
                filters["year_from"] = year_from
                filters["year_to"] = year_to
            if "метк" in lowered or "класс" in lowered or "label" in lowered:
                metric = "label_distribution"
            elif "тикер" in lowered or "ticker" in lowered:
                metric = "ticker_frequency"
            elif "коммент" in lowered or "score" in lowered or "вовлеч" in lowered:
                metric = "engagement_stats"
            else:
                metric = "count_posts"
            args: Dict[str, Any] = {"metric": metric, "filters": filters}
            if "по годам" in lowered or "by year" in lowered:
                args["group_by"] = "year"
            elif "по меткам" in lowered:
                args["group_by"] = "label"
            return self._call("calculate", args)

        if any(word in lowered for word in
               ("статус", "переведи", "перевести", "change status",
                "пауза", "паузу", "приостанов", "возобнов", "архив",
                "очеред", "заверш")):
            object_id = (_extract_id(user, "watch") or _extract_id(user, "task")
                         or self._context(messages).get("last_entity_id"))
            if not object_id:
                return self._ask("Уточните id объекта и целевой статус.")
            status = "in_progress"
            for candidate in ("active", "paused", "queued", "in_progress", "done",
                              "cancelled", "archived", "draft"):
                if candidate in lowered:
                    status = candidate
                    break
            if "пауз" in lowered or "приостанов" in lowered:
                status = "paused"
            if "возобнов" in lowered or "активир" in lowered:
                status = "active"
            if "архив" in lowered:
                status = "archived"
            if "очеред" in lowered:
                status = "queued"
            if "работ" in lowered:
                status = "in_progress"
            if "готов" in lowered or "выполн" in lowered or "заверш" in lowered:
                status = "done"
            entity = "research_task" if object_id.startswith("task_") else "watch_item"
            return self._call("change_status", {
                "entity_type": entity, "id": object_id, "to_status": status,
            })

        if any(word in lowered for word in ("покажи", "открой", "get", "show")):
            context = self._context(messages)
            post_id = context.get("post_id") or _extract_post_id(user) or _extract_id(user, "watch")
            if post_id:
                return self._call("get", {"entity_type": "thread", "id": post_id})
            return self._ask("Уточните, какой тред показать (post_id или ссылку).")

        return self._ask(
            "Я умею искать треды, показывать их, создавать наблюдения и задачи, "
            "проверять их, считать статистику, менять статус и отменять объекты. "
            "Сформулируйте команду, например: «найди посты про GME 2021»."
        )

    @staticmethod
    def _context(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = _message_text(messages, "user")
        context: Dict[str, Any] = {}
        for key in ("post_id", "last_entity_id"):
            match = re.search(rf"{key}\s*[:=]\s*([a-zA-Z0-9_]+)", text)
            if match:
                context[key] = match.group(1)
        match = re.search(r"(?:заголовок треда|title)\s*[:=]\s*([^\n]+)", text)
        if match:
            context["title"] = match.group(1).strip()
        return context

    @staticmethod
    def _call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"role": "assistant", "content": "", "tool_calls": [_tool_call(name, args)]}

    @staticmethod
    def _ask(text: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": text, "tool_calls": []}

    @staticmethod
    def _summarize(tool_name: str, result: Dict[str, Any]) -> str:
        if not result.get("ok"):
            error = result.get("error") or {}
            message = str(error.get("message") or "неизвестная ошибка").rstrip(".")
            return f"Ошибка {error.get('code', 'ERROR')}: {message}."
        data = result.get("data") or {}
        if tool_name == "search":
            items = data.get("results") or []
            if not items:
                return "По запросу ничего не найдено."
            first = items[0]
            return (f"Найдено {len(items)} постов. Лучший результат: «{first.get('title', '')}» "
                    f"(post_id {first.get('post_id', '')}, {first.get('year', '')}).")
        if tool_name == "get":
            record = data.get("record") or {}
            return f"Получен объект: {record.get('title') or record.get('id') or 'без названия'}."
        if tool_name == "calculate":
            return f"Расчёт готов: {json.dumps(data, ensure_ascii=False)[:500]}"
        if tool_name == "check":
            if data.get("target") == "service":
                return "Сервис доступен, статистика получена."
            record = data.get("record") or {}
            if data.get("target") == "research_task":
                return (f"Задача «{record.get('title', '')}»: "
                        f"статус {record.get('status', '')}.")
            return f"Проверка завершена. Новых совпадений: {data.get('new_count', 0)}."
        return "Готово."
