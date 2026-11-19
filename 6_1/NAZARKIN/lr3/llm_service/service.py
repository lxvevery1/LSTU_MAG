"""Assistant service: conversation state, LLM tool loop and confirmations."""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config
from .backend_client import BackendClient, BackendError
from .prompts import SYSTEM_PROMPT, context_block
from .providers import LLMError, MockProvider, OpenAICompatibleClient


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Conversation:
    id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    pending_action: Optional[Dict[str, Any]] = None
    last_result: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=_now)


class ConversationStore:
    def __init__(self):
        self._items: Dict[str, Conversation] = {}
        self._lock = threading.RLock()

    def get(self, conversation_id: str | None) -> Conversation:
        with self._lock:
            if conversation_id and conversation_id in self._items:
                return self._items[conversation_id]
            cid = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
            conversation = Conversation(id=cid)
            self._items[cid] = conversation
            return conversation

    def history(self, conversation_id: str) -> List[Dict[str, str]]:
        conversation = self.get(conversation_id)
        result = []
        for message in conversation.messages:
            if message.get("role") in {"user", "assistant"}:
                content = str(message.get("content") or "")
                if "Контекст страницы" in content:
                    content = content.split("\n\nКонтекст страницы", 1)[0]
                if content:
                    result.append({"role": message["role"], "content": content})
        return result


class AssistantService:
    def __init__(self, backend: BackendClient | None = None,
                 provider: Any | None = None):
        self.backend = backend or BackendClient()
        if provider is not None:
            self.provider = provider
        elif config.PROVIDER == "mock":
            self.provider = MockProvider()
        else:
            self.provider = OpenAICompatibleClient()
        self.mock = MockProvider()
        self.store = ConversationStore()

    # ------------------------------------------------------------------ public
    def chat(self, message: str, conversation_id: str | None = None,
             context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        message = str(message or "").strip()
        if not message:
            return self._error(conversation_id or "", "VALIDATION_ERROR",
                               "Пустая команда.")
        conversation = self.store.get(conversation_id)
        if context:
            conversation.context.update({key: value for key, value in context.items()
                                         if value not in (None, "")})
        augmented = message
        block = context_block(conversation.context)
        if block:
            augmented = f"{message}\n\n{block}"
        conversation.messages.append({"role": "user", "content": augmented})
        return self._run_tool_loop(conversation, message)

    def confirm(self, action_id: str, decision: str,
                conversation_id: str | None = None) -> Dict[str, Any]:
        conversation = self.store.get(conversation_id)
        if decision not in {"confirm", "reject"}:
            return self._error(conversation.id, "VALIDATION_ERROR",
                               "decision должен быть confirm или reject.")
        try:
            result = self.backend.confirm(action_id, decision, conversation.id)
        except BackendError as exc:
            return self._error(conversation.id, exc.code, exc.message, exc.details)
        trace = [{"tool": "confirm", "arguments": {"action_id": action_id,
                                                   "decision": decision},
                  "status": result.get("status"), "result": result}]
        conversation.pending_action = None
        if result.get("ok") and decision == "confirm":
            data = result.get("data") or {}
            record = data.get("record") or {}
            if record.get("id"):
                conversation.context["last_entity_id"] = record["id"]
                if data.get("entity_type") == "watch_item":
                    conversation.context["last_watch_id"] = record["id"]
            conversation.last_result = result
            reply = self._confirm_text(result)
        elif decision == "reject":
            reply = "Операция отменена пользователем; изменения в ИС не внесены."
        else:
            error = result.get("error") or {}
            reply = f"Не удалось подтвердить операцию: {error.get('message', 'ошибка backend')}."
        conversation.messages.append({"role": "assistant", "content": reply})
        return {
            "ok": bool(result.get("ok")),
            "conversation_id": conversation.id,
            "status": result.get("status") or ("ok" if result.get("ok") else "error"),
            "reply": reply,
            "actual_result": result,
            "trace": trace,
            "error": result.get("error"),
        }

    def history(self, conversation_id: str) -> Dict[str, Any]:
        return {"conversation_id": conversation_id,
                "messages": self.store.history(conversation_id)}

    # ------------------------------------------------------------------ loop
    def _run_tool_loop(self, conversation: Conversation, user_text: str) -> Dict[str, Any]:
        try:
            tools = self.backend.tools()
        except BackendError as exc:
            return self._error(conversation.id, exc.code, exc.message, exc.details)
        if not tools:
            return self._error(conversation.id, "NO_TOOLS",
                               "Backend не вернул ни одного инструмента.")

        trace: List[Dict[str, Any]] = []
        provider_name = "mock" if isinstance(self.provider, MockProvider) else "openai_compatible"
        notice = ""
        last_response: Optional[Dict[str, Any]] = None

        for _step in range(config.MAX_TOOL_STEPS):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation.messages[-24:]
            try:
                response, provider_name, notice = self._provider_chat(messages, tools)
            except LLMError as exc:
                return self._error(conversation.id, exc.code, exc.message, exc.details)

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                reply = str(response.get("content") or "").strip()
                conversation.messages.append({"role": "assistant", "content": reply})
                if not reply:
                    reply = self._summarize_last(trace)
                last_result = trace[-1]["result"] if trace else conversation.last_result
                last_ok = bool(last_result.get("ok")) if last_result else True
                return {
                    "ok": last_ok,
                    "conversation_id": conversation.id,
                    "status": "ok" if last_ok else "error",
                    "reply": reply,
                    "actual_result": last_result,
                    "trace": trace,
                    "provider": provider_name,
                    "notice": notice,
                    "error": None if last_ok or not last_result else last_result.get("error"),
                }

            normalized_calls = []
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name") or call.get("name") or ""
                arguments = function.get("arguments", call.get("arguments", {}))
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments or "{}")
                    except ValueError:
                        parsed = {}
                elif isinstance(arguments, dict):
                    parsed = arguments
                else:
                    parsed = {}
                normalized_calls.append({
                    "id": call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps(parsed, ensure_ascii=False)},
                })
            conversation.messages.append({
                "role": "assistant",
                "content": response.get("content") or "",
                "tool_calls": normalized_calls,
            })
            for call in normalized_calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except ValueError:
                    arguments = {}
                try:
                    result = self.backend.call(name, arguments, conversation.id)
                except BackendError as exc:
                    result = {"ok": False, "tool": name, "status": "error",
                              "error": exc.as_dict()}
                conversation.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
                trace.append({
                    "tool": name,
                    "arguments": arguments,
                    "status": result.get("status"),
                    "result": result,
                })
                self._remember(conversation, result)
                if result.get("status") == "pending_confirmation":
                    conversation.pending_action = result
                    reply = (response.get("content") or "").strip()
                    preview = result.get("preview") or {}
                    if not reply:
                        reply = (preview.get("summary") or "Операция подготовлена.") + \
                                " Подтвердите выполнение."
                    conversation.messages.append({"role": "assistant", "content": reply})
                    return {
                        "ok": True,
                        "conversation_id": conversation.id,
                        "status": "needs_confirmation",
                        "reply": reply,
                        "pending_action": result,
                        "actual_result": result,
                        "trace": trace,
                        "provider": provider_name,
                        "notice": notice,
                    }
            last_response = response

        reply = "Не удалось завершить команду за отведённое число шагов. Уточните запрос."
        conversation.messages.append({"role": "assistant", "content": reply})
        return {
            "ok": False,
            "conversation_id": conversation.id,
            "status": "error",
            "reply": reply,
            "actual_result": trace[-1]["result"] if trace else None,
            "trace": trace,
            "provider": provider_name,
            "notice": notice,
            "error": {"code": "TOOL_LOOP_LIMIT", "message": reply,
                      "details": {"steps": config.MAX_TOOL_STEPS}, "retryable": True},
        }

    def _provider_chat(self, messages: List[Dict[str, Any]],
                       tools: List[Dict[str, Any]]):
        try:
            return self.provider.chat(messages, tools), self.provider.name, ""
        except LLMError as exc:
            if not config.FALLBACK_MOCK or isinstance(self.provider, MockProvider):
                raise
            response = self.mock.chat(messages, tools)
            notice = (f"LLM-провайдер недоступен ({exc.code}); использован "
                      "детерминированный fallback, тот же backend и те же tools.")
            return response, "mock_fallback", notice

    def _remember(self, conversation: Conversation, result: Dict[str, Any]) -> None:
        if not result.get("ok"):
            return
        conversation.last_result = result
        data = result.get("data") or {}
        if result.get("tool") == "search":
            items = data.get("results") or []
            if len(items) == 1:
                conversation.context["post_id"] = items[0]["post_id"]
                conversation.context["title"] = items[0].get("title", "")
            elif items:
                conversation.context["last_entity_id"] = items[0]["post_id"]
        elif result.get("tool") in {"get", "create", "update", "cancel", "change_status"}:
            record = data.get("record") or {}
            if record.get("id"):
                conversation.context["last_entity_id"] = record["id"]
            if record.get("post_id"):
                conversation.context["post_id"] = record["post_id"]
                conversation.context["title"] = record.get("title", "")
            if data.get("entity_type") == "watch_item" and record.get("id"):
                conversation.context["last_watch_id"] = record["id"]

    @staticmethod
    def _summarize_last(trace: List[Dict[str, Any]]) -> str:
        if not trace:
            return "Готово."
        last = trace[-1]["result"]
        if not last.get("ok"):
            error = last.get("error") or {}
            return f"Ошибка {error.get('code', 'ERROR')}: {error.get('message', '')}."
        return f"Инструмент {trace[-1]['tool']} выполнен."

    @staticmethod
    def _confirm_text(result: Dict[str, Any]) -> str:
        data = result.get("data") or {}
        record = data.get("record") or {}
        entity = data.get("entity_type") or "объект"
        name = record.get("title") or record.get("id") or ""
        tool = result.get("tool") or ""
        if tool == "cancel":
            return f"Отменено: {name}. Статус: {record.get('status', 'cancelled')}."
        if tool == "change_status":
            return f"Статус изменён: {name} → {record.get('status', '')}."
        if tool == "update":
            return f"Обновлено: {name}."
        if entity == "watch_item":
            return f"Наблюдение сохранено: {name}."
        if entity == "research_task":
            return f"Задача сохранена: {name}."
        return "Операция выполнена."

    @staticmethod
    def _error(conversation_id: str, code: str, message: str,
               details: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "ok": False,
            "conversation_id": conversation_id,
            "status": "error",
            "reply": f"Ошибка: {message}",
            "error": {"code": code, "message": message,
                      "details": details or {}, "retryable": False},
            "trace": [],
        }
