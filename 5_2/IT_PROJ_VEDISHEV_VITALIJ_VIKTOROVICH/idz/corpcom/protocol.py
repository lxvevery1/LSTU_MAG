"""JSON-протокол поверх WebSocket."""

from __future__ import annotations

import base64
import json
from enum import StrEnum
from typing import Any


class MsgType(StrEnum):
    AUTH = "auth"
    AUTH_OK = "auth_ok"
    AUTH_ERR = "auth_err"
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    PING = "ping"
    PONG = "pong"
    SYS = "sys"
    CLIENT_LIST = "client_list"


def dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def loads(raw: str) -> dict[str, Any]:
    return json.loads(raw)


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def auth_message(token: str, role: str, name: str, room: str) -> str:
    return dumps(
        {
            "type": MsgType.AUTH,
            "token": token,
            "role": role,
            "name": name,
            "room": room,
        }
    )


def text_message(room: str, sender: str, content: str) -> str:
    return dumps(
        {
            "type": MsgType.TEXT,
            "room": room,
            "sender": sender,
            "content": content,
        }
    )


def voice_message(room: str, sender: str, pcm: bytes) -> str:
    return dumps(
        {
            "type": MsgType.VOICE,
            "room": room,
            "sender": sender,
            "data": b64_encode(pcm),
        }
    )


def image_message(room: str, sender: str, filename: str, image: bytes) -> str:
    return dumps(
        {
            "type": MsgType.IMAGE,
            "room": room,
            "sender": sender,
            "filename": filename,
            "data": b64_encode(image),
        }
    )
