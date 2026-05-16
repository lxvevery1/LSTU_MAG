#!/usr/bin/env python3
"""
CorpCom — сервер реального времени для корпоративной LAN.
Без внешних облачных сервисов; только ваш хост в сети предприятия.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import websockets
from websockets.server import WebSocketServerProtocol

from corpcom import config
from corpcom.protocol import MsgType, dumps, loads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("corpcom.server")


@dataclass
class Peer:
    ws: WebSocketServerProtocol
    name: str
    role: str  # user | projector
    room: str


@dataclass
class Hub:
    peers: dict[WebSocketServerProtocol, Peer] = field(default_factory=dict)

    def add(self, ws: WebSocketServerProtocol, peer: Peer) -> None:
        self.peers[ws] = peer

    def remove(self, ws: WebSocketServerProtocol) -> Peer | None:
        return self.peers.pop(ws, None)

    def by_room(self, room: str, *, role: str | None = None) -> list[Peer]:
        out: list[Peer] = []
        for p in self.peers.values():
            if p.room != room:
                continue
            if role is not None and p.role != role:
                continue
            out.append(p)
        return out

    async def broadcast(
        self,
        room: str,
        payload: str,
        *,
        exclude: WebSocketServerProtocol | None = None,
        roles: set[str] | None = None,
    ) -> None:
        tasks: list[asyncio.Task[None]] = []
        for ws, peer in list(self.peers.items()):
            if peer.room != room:
                continue
            if exclude is not None and ws is exclude:
                continue
            if roles is not None and peer.role not in roles:
                continue
            tasks.append(asyncio.create_task(ws.send(payload)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def client_list_payload(self, room: str) -> str:
        names = sorted({p.name for p in self.by_room(room)})
        return dumps({"type": MsgType.CLIENT_LIST, "room": room, "clients": names})


hub = Hub()


async def handle(ws: WebSocketServerProtocol) -> None:
    peer: Peer | None = None
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
        msg = loads(raw)  # type: ignore[arg-type]
        if msg.get("type") != MsgType.AUTH:
            await ws.send(dumps({"type": MsgType.AUTH_ERR, "reason": "Ожидается auth"}))
            return
        if msg.get("token") != config.AUTH_TOKEN:
            await ws.send(dumps({"type": MsgType.AUTH_ERR, "reason": "Неверный токен"}))
            return

        role = msg.get("role", "user")
        if role not in ("user", "projector"):
            await ws.send(dumps({"type": MsgType.AUTH_ERR, "reason": "role: user|projector"}))
            return

        name = (msg.get("name") or "anonymous").strip()[:64] or "anonymous"
        room = (msg.get("room") or config.DEFAULT_ROOM).strip()[:64] or config.DEFAULT_ROOM

        peer = Peer(ws=ws, name=name, role=role, room=room)
        hub.add(ws, peer)
        await ws.send(
            dumps(
                {
                    "type": MsgType.AUTH_OK,
                    "name": name,
                    "role": role,
                    "room": room,
                    "host": config.HOST,
                    "port": config.PORT,
                }
            )
        )
        await hub.broadcast(
            room,
            dumps(
                {
                    "type": MsgType.SYS,
                    "content": f"{name} ({role}) подключился",
                }
            ),
            exclude=ws,
        )
        await hub.broadcast(room, hub.client_list_payload(room))
        log.info("+%s [%s] room=%s", name, role, room)

        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            data = loads(raw)
            mtype = data.get("type")

            if mtype == MsgType.PING:
                await ws.send(dumps({"type": MsgType.PONG}))
                continue

            if mtype == MsgType.TEXT:
                content = (data.get("content") or "").strip()
                if not content:
                    continue
                payload = dumps(
                    {
                        "type": MsgType.TEXT,
                        "room": peer.room,
                        "sender": peer.name,
                        "content": content[:8000],
                    }
                )
                await hub.broadcast(peer.room, payload, exclude=ws, roles={"user"})
                continue

            if mtype in (MsgType.VOICE, MsgType.VOICE_START, MsgType.VOICE_END):
                if mtype == MsgType.VOICE:
                    blob = data.get("data")
                    if not blob:
                        continue
                    payload = dumps(
                        {
                            "type": MsgType.VOICE,
                            "room": peer.room,
                            "sender": peer.name,
                            "seq": int(data.get("seq") or 0),
                            "data": blob,
                        }
                    )
                else:
                    payload = dumps(
                        {
                            "type": mtype,
                            "room": peer.room,
                            "sender": peer.name,
                        }
                    )
                await hub.broadcast(peer.room, payload, exclude=ws, roles={"user"})
                continue

            if mtype == MsgType.IMAGE:
                blob = data.get("data")
                filename = (data.get("filename") or "image.png")[:255]
                if not blob:
                    continue
                # Проекторы + отправитель (подтверждение)
                payload = dumps(
                    {
                        "type": MsgType.IMAGE,
                        "room": peer.room,
                        "sender": peer.name,
                        "filename": filename,
                        "data": blob,
                    }
                )
                await hub.broadcast(
                    peer.room,
                    payload,
                    roles={"user", "projector"},
                )
                log.info("image %s -> room %s (%s)", filename, peer.room, peer.name)

    except asyncio.TimeoutError:
        log.warning("auth timeout")
    except websockets.ConnectionClosed:
        pass
    finally:
        if peer is not None:
            hub.remove(ws)
            await hub.broadcast(
                peer.room,
                dumps({"type": MsgType.SYS, "content": f"{peer.name} отключился"}),
            )
            await hub.broadcast(peer.room, hub.client_list_payload(peer.room))
            log.info("-%s", peer.name)


async def main() -> None:
    log.info("CorpCom server %s:%s (token из CORPCOM_TOKEN)", config.HOST, config.PORT)
    async with websockets.serve(
        handle,
        config.HOST,
        config.PORT,
        ping_interval=20,
        ping_timeout=20,
        max_size=config.MAX_IMAGE_BYTES + 512 * 1024,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
