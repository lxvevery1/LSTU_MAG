#!/usr/bin/env python3
"""CorpCom — клиент проектора: полноэкранный вывод изображений из LAN."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import tkinter as tk
from tkinter import ttk

import websockets
from PIL import Image, ImageTk

from corpcom import config
from corpcom.protocol import MsgType, auth_message, loads

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("corpcom.projector")


class ProjectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CorpCom — проектор")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", False)

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._photo: ImageTk.PhotoImage | None = None

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="Сервер").pack(side=tk.LEFT)
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(top, textvariable=self.host_var, width=14).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Порт").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value=str(config.PORT))
        ttk.Entry(top, textvariable=self.port_var, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Комната").pack(side=tk.LEFT)
        self.room_var = tk.StringVar(value=config.DEFAULT_ROOM)
        ttk.Entry(top, textvariable=self.room_var, width=12).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Токен").pack(side=tk.LEFT)
        self.token_var = tk.StringVar(value=config.AUTH_TOKEN)
        ttk.Entry(top, textvariable=self.token_var, width=18, show="*").pack(side=tk.LEFT, padx=4)

        self.btn = ttk.Button(top, text="Старт", command=self.toggle)
        self.btn.pack(side=tk.LEFT, padx=8)

        ttk.Button(top, text="На весь экран (F11)", command=self.toggle_fullscreen).pack(
            side=tk.LEFT, padx=4
        )
        self.root.bind("<F11>", lambda _e: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda _e: self.root.attributes("-fullscreen", False))

        self.canvas = tk.Label(self.root, bg="black", fg="#666", text="Ожидание кадра…")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.info = ttk.Label(self.root, text="")
        self.info.pack(fill=tk.X, padx=8, pady=4)

        self.running = False

    def toggle_fullscreen(self) -> None:
        cur = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not cur)

    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self.info.configure(text="Некорректный порт")
            return
        room = self.room_var.get().strip()
        token = self.token_var.get()
        self.running = True
        self.btn.configure(text="Стоп")

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            loop.run_until_complete(self._run(host, port, token, room))

        threading.Thread(target=runner, daemon=True).start()

    def stop(self) -> None:
        self.running = False
        if self.loop and self.ws:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        self.btn.configure(text="Старт")

    async def _run(self, host: str, port: int, token: str, room: str) -> None:
        uri = f"ws://{host}:{port}"
        try:
            async with websockets.connect(
                uri,
                ping_interval=20,
                max_size=config.MAX_IMAGE_BYTES + 512 * 1024,
            ) as ws:
                self.ws = ws
                await ws.send(auth_message(token, "projector", "projector-1", room))
                raw = await ws.recv()
                msg = loads(raw)  # type: ignore[arg-type]
                if msg.get("type") == MsgType.AUTH_ERR:
                    self.root.after(0, lambda: self.info.configure(text=msg.get("reason")))
                    return
                self.root.after(
                    0,
                    lambda: self.info.configure(text=f"Проектор в комнате «{room}»"),
                )
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    data = loads(raw)
                    if data.get("type") == MsgType.IMAGE:
                        from corpcom.protocol import b64_decode

                        img_bytes = b64_decode(data["data"])
                        sender = data.get("sender", "?")
                        fname = data.get("filename", "")
                        self.root.after(
                            0,
                            lambda b=img_bytes, s=sender, f=fname: self.show_frame(b, s, f),
                        )
        except Exception as exc:
            self.root.after(0, lambda: self.info.configure(text=str(exc)))
        finally:
            self.ws = None
            self.running = False
            self.root.after(0, lambda: self.btn.configure(text="Старт"))

    def show_frame(self, data: bytes, sender: str, filename: str) -> None:
        img = Image.open(io.BytesIO(data))
        w = max(self.canvas.winfo_width(), 640)
        h = max(self.canvas.winfo_height(), 360)
        img.thumbnail((w, h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self._photo, text="")
        self.info.configure(text=f"{filename} — от {sender}")


def main() -> None:
    root = tk.Tk()
    ProjectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
