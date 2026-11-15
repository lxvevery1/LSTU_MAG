"""Tests for the assistant tool loop using the deterministic mock provider."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lr3"))

from backend.tools.schemas import TOOLS  # noqa: E402
from llm_service.providers import MockProvider  # noqa: E402
from llm_service.service import AssistantService  # noqa: E402


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.pending = None

    def tools(self):
        return TOOLS

    def call(self, tool, arguments, conversation_id=""):
        self.calls.append((tool, arguments))
        if tool == "search":
            return {"ok": True, "tool": "search", "status": "ok", "data": {
                "results": [{"post_id": "m5wj5q", "title": "GME squeeze", "year": "2021"}]}}
        if tool == "create":
            self.pending = {"action_id": "act_1", "preview": {
                "summary": "Создать наблюдение.", "changes": arguments}}
            return {"ok": True, "tool": "create", "status": "pending_confirmation",
                    "action_id": "act_1", "preview": self.pending["preview"]}
        if tool == "check":
            return {"ok": True, "tool": "check", "status": "ok", "data": {
                "target": "service", "status": "ok"}}
        if tool == "calculate":
            return {"ok": True, "tool": "calculate", "status": "ok", "data": {
                "metric": "count_posts", "total": 2, "rows": [{"key": "2021", "count": 2}]}}
        return {"ok": True, "tool": tool, "status": "ok", "data": {}}

    def confirm(self, action_id, decision, conversation_id=""):
        self.calls.append(("confirm", {"action_id": action_id, "decision": decision}))
        if decision == "reject":
            return {"ok": True, "tool": "create", "status": "rejected",
                    "action_id": action_id, "data": {"decision": "reject"}}
        return {"ok": True, "tool": "create", "status": "confirmed",
                "action_id": action_id, "data": {
                    "entity_type": "watch_item",
                    "record": {"id": "watch_1", "title": "GME squeeze", "status": "active"}}}


class AssistantFlowTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.service = AssistantService(self.backend, MockProvider())

    def test_search_is_executed_and_result_is_returned(self):
        result = self.service.chat("найди посты про GME", context={})
        self.assertTrue(result["ok"])
        self.assertEqual(result["trace"][0]["tool"], "search")
        self.assertEqual(result["actual_result"]["data"]["results"][0]["post_id"], "m5wj5q")

    def test_create_is_staged_and_then_confirmed(self):
        result = self.service.chat(
            "добавь этот тред в мониторинг",
            context={"post_id": "m5wj5q", "title": "GME squeeze"},
        )
        self.assertEqual(result["status"], "needs_confirmation")
        action_id = result["pending_action"]["action_id"]
        confirmed = self.service.confirm(action_id, "confirm", result["conversation_id"])
        self.assertTrue(confirmed["ok"])
        self.assertEqual(confirmed["actual_result"]["data"]["record"]["id"], "watch_1")

    def test_reject_does_not_apply(self):
        result = self.service.chat(
            "добавь TSLA в мониторинг", context={})
        rejected = self.service.confirm(
            result["pending_action"]["action_id"], "reject", result["conversation_id"])
        self.assertEqual(rejected["status"], "rejected")

    def test_unclear_command_asks_for_clarification(self):
        result = self.service.chat("сделай что-нибудь", context={})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["trace"], [])
        self.assertIn("умею", result["reply"].lower())

    def test_new_turn_after_confirmation_is_not_stuck_on_old_tool_result(self):
        first = self.service.chat(
            "добавь этот тред в мониторинг",
            context={"post_id": "m5wj5q", "title": "GME squeeze"},
        )
        cid = first["conversation_id"]
        self.service.confirm(first["pending_action"]["action_id"], "confirm", cid)
        second = self.service.chat("проверь сервис", conversation_id=cid)
        self.assertEqual(second["trace"][0]["tool"], "check")

    def test_calculate(self):
        result = self.service.chat("сколько постов про GME по годам", context={})
        self.assertEqual(result["trace"][0]["tool"], "calculate")
        self.assertEqual(result["actual_result"]["data"]["total"], 2)

    def test_plain_post_id_is_understood(self):
        result = self.service.chat("покажи m5wj5q", context={})
        self.assertEqual(result["trace"][0]["tool"], "get")
        self.assertEqual(result["trace"][0]["arguments"]["id"], "m5wj5q")

    def test_explicit_ticker_wins_over_page_context(self):
        result = self.service.chat(
            "добавь TSLA в мониторинг",
            context={"post_id": "m5wj5q", "title": "GME squeeze"},
        )
        self.assertEqual(result["status"], "needs_confirmation")
        tool, arguments = self.backend.calls[-1]
        self.assertEqual(tool, "create")
        self.assertEqual(arguments["data"]["kind"], "ticker")
        self.assertEqual(arguments["data"]["target"], "TSLA")

    def test_status_command_with_watch_id_is_not_treated_as_save(self):
        result = self.service.chat("поставь на паузу watch_abc123", context={})
        self.assertEqual(result["trace"][0]["tool"], "change_status")
        self.assertEqual(result["trace"][0]["arguments"]["id"], "watch_abc123")
        self.assertEqual(result["trace"][0]["arguments"]["to_status"], "paused")

    def test_update_note_and_priority(self):
        result = self.service.chat(
            "обнови watch_abc123 заметка: проверить отчёт, приоритет 5", context={})
        self.assertEqual(result["trace"][0]["tool"], "update")
        patch = result["trace"][0]["arguments"]["patch"]
        self.assertEqual(patch["note"], "проверить отчёт")
        self.assertEqual(patch["priority"], 5)

    def test_create_research_task(self):
        result = self.service.chat(
            "создай исследовательскую задачу про GME short squeeze", context={})
        self.assertEqual(result["trace"][0]["tool"], "create")
        data = result["trace"][0]["arguments"]["data"]
        self.assertEqual(result["trace"][0]["arguments"]["entity_type"], "research_task")
        self.assertEqual(data["query"], "GME short squeeze")


if __name__ == "__main__":
    unittest.main()
