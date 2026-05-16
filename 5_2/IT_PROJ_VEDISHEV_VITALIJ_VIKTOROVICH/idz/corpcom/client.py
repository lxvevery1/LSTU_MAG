#!/usr/bin/env python3
"""CorpCom — клиент ПК: текст, голос (PTT), отправка изображения на проектор."""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import websockets
from PIL import Image, ImageTk

from corpcom import config
from corpcom.protocol import MsgType, auth_message, dumps, image_message, loads, text_message

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("corpcom.client")

try:
    import numpy as np
    import sounddevice as sd

    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class CorpComClient:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CorpCom — корпоративный клиент")
        self.root.geometry("720x560")

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.connected = False
        self.my_name = ""
        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.ptt_active = False

        self._build_ui()
        self.root.after(100, self._drain_ui_queue)

    def _build_ui(self) -> None:
        conn = ttk.LabelFrame(self.root, text="Подключение (LAN)")
        conn.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(conn, text="Сервер").grid(row=0, column=0, padx=4, pady=4)
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(conn, textvariable=self.host_var, width=16).grid(row=0, column=1, padx=4)

        ttk.Label(conn, text="Порт").grid(row=0, column=2, padx=4)
        self.port_var = tk.StringVar(value=str(config.PORT))
        ttk.Entry(conn, textvariable=self.port_var, width=8).grid(row=0, column=3, padx=4)

        ttk.Label(conn, text="Имя").grid(row=0, column=4, padx=4)
        self.name_var = tk.StringVar(value="user-1")
        ttk.Entry(conn, textvariable=self.name_var, width=12).grid(row=0, column=5, padx=4)

        ttk.Label(conn, text="Комната").grid(row=1, column=0, padx=4)
        self.room_var = tk.StringVar(value=config.DEFAULT_ROOM)
        ttk.Entry(conn, textvariable=self.room_var, width=16).grid(row=1, column=1, padx=4)

        ttk.Label(conn, text="Токен").grid(row=1, column=2, padx=4)
        self.token_var = tk.StringVar(value=config.AUTH_TOKEN)
        ttk.Entry(conn, textvariable=self.token_var, width=24, show="*").grid(
            row=1, column=3, columnspan=2, padx=4, sticky=tk.W
        )

        self.btn_connect = ttk.Button(conn, text="Подключить", command=self.toggle_connect)
        self.btn_connect.grid(row=1, column=5, padx=4)

        self.status_var = tk.StringVar(value="Отключено")
        ttk.Label(self.root, textvariable=self.status_var).pack(anchor=tk.W, padx=10)

        chat_frame = ttk.LabelFrame(self.root, text="Чат")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.chat = scrolledtext.ScrolledText(chat_frame, state=tk.DISABLED, height=16)
        self.chat.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        row = ttk.Frame(chat_frame)
        row.pack(fill=tk.X, padx=4, pady=4)
        self.msg_var = tk.StringVar()
        ent = ttk.Entry(row, textvariable=self.msg_var)
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ent.bind("<Return>", lambda _e: self.send_text())
        ttk.Button(row, text="Отправить", command=self.send_text).pack(side=tk.LEFT, padx=4)

        tools = ttk.LabelFrame(self.root, text="Медиа")
        tools.pack(fill=tk.X, padx=8, pady=4)

        self.btn_ptt = ttk.Button(
            tools,
            text="Голос (удерживать)" if HAS_AUDIO else "Голос (нет sounddevice)",
            state=tk.NORMAL if HAS_AUDIO else tk.DISABLED,
        )
        self.btn_ptt.pack(side=tk.LEFT, padx=4, pady=4)
        if HAS_AUDIO:
            self.btn_ptt.bind("<ButtonPress-1>", lambda _e: self.start_ptt())
            self.btn_ptt.bind("<ButtonRelease-1>", lambda _e: self.stop_ptt())

        ttk.Button(tools, text="Изображение на проектор", command=self.send_image).pack(
            side=tk.LEFT, padx=4, pady=4
        )

        self.preview_label = ttk.Label(self.root, text="Превью последнего кадра")
        self.preview_label.pack(pady=4)
        self._preview_photo: ImageTk.PhotoImage | None = None

    def _append_chat(self, line: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, line + "\n")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def toggle_connect(self) -> None:
        if self.connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self) -> None:
        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный порт")
            return
        name = self.name_var.get().strip() or "user"
        room = self.room_var.get().strip() or config.DEFAULT_ROOM
        token = self.token_var.get()

        self.my_name = name
        self.btn_connect.configure(state=tk.DISABLED)

        def runner() -> None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._session(host, port, token, name, room))

        self.thread = threading.Thread(target=runner, daemon=True)
        self.thread.start()

    async def _session(
        self, host: str, port: int, token: str, name: str, room: str
    ) -> None:
        uri = f"ws://{host}:{port}"
        try:
            async with websockets.connect(
                uri,
                ping_interval=20,
                max_size=config.MAX_IMAGE_BYTES + 512 * 1024,
            ) as ws:
                self.ws = ws
                await ws.send(
                    auth_message(token, "user", name, room)
                )
                raw = await ws.recv()
                msg = loads(raw)  # type: ignore[arg-type]
                if msg.get("type") == MsgType.AUTH_ERR:
                    self.ui_queue.put(("error", msg.get("reason", "auth failed")))
                    return
                self.ui_queue.put(("connected", msg))
                recv_task = asyncio.create_task(self._recv_loop(ws))
                await recv_task
        except Exception as exc:
            self.ui_queue.put(("error", str(exc)))
        finally:
            self.ws = None
            self.ui_queue.put(("disconnected", None))

    async def _recv_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            self.ui_queue.put(("message", loads(raw)))

    def disconnect(self) -> None:
        if self.loop and self.ws:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "connected":
                    self.connected = True
                    self.btn_connect.configure(text="Отключить", state=tk.NORMAL)
                    self.status_var.set(f"В сети: {payload.get('room')}")
                    self._append_chat("[система] Подключено к CorpCom")
                elif kind == "disconnected":
                    self.connected = False
                    self.btn_connect.configure(text="Подключить", state=tk.NORMAL)
                    self.status_var.set("Отключено")
                    self._append_chat("[система] Отключено")
                elif kind == "error":
                    messagebox.showerror("CorpCom", str(payload))
                    self.btn_connect.configure(state=tk.NORMAL)
                elif kind == "message":
                    self._handle_message(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        self.root.after(100, self._drain_ui_queue)

    def _handle_message(self, msg: dict) -> None:
        t = msg.get("type")
        if t == MsgType.TEXT:
            sender = msg.get("sender", "?")
            if sender == self.my_name:
                return
            self._append_chat(f"[{sender}] {msg.get('content', '')}")
        elif t == MsgType.SYS:
            self._append_chat(f"[система] {msg.get('content', '')}")
        elif t == MsgType.CLIENT_LIST:
            clients = ", ".join(msg.get("clients", []))
            self.status_var.set(f"В сети | участники: {clients}")
        elif t == MsgType.VOICE and HAS_AUDIO:
            sender = msg.get("sender", "?")
            if sender == self.my_name:
                return
            from corpcom.protocol import b64_decode

            pcm = b64_decode(msg["data"])
            arr = np.frombuffer(pcm, dtype=np.int16)
            sd.play(arr, config.SAMPLE_RATE, blocking=False)
        elif t == MsgType.IMAGE:
            from corpcom.protocol import b64_decode

            data = b64_decode(msg["data"])
            sender = msg.get("sender", "?")
            self._show_preview(data)
            self._append_chat(f"[изображение] от {sender}: {msg.get('filename', '')}")

    def _show_preview(self, data: bytes) -> None:
        img = Image.open(io.BytesIO(data))
        img.thumbnail((320, 180))
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._preview_photo, text="")

    def send_text(self) -> None:
        if not self.connected or not self.ws or not self.loop:
            return
        text = self.msg_var.get().strip()
        if not text:
            return
        room = self.room_var.get().strip()
        payload = text_message(room, self.my_name, text)
        asyncio.run_coroutine_threadsafe(self.ws.send(payload), self.loop)
        self._append_chat(f"[я] {text}")
        self.msg_var.set("")

    def send_image(self) -> None:
        if not self.connected or not self.ws or not self.loop:
            messagebox.showinfo("CorpCom", "Сначала подключитесь к серверу")
            return
        path = filedialog.askopenfilename(
            title="Изображение для проектора",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return
        with open(path, "rb") as f:
            data = f.read()
        if len(data) > config.MAX_IMAGE_BYTES:
            messagebox.showerror("CorpCom", "Файл слишком большой")
            return
        import os

        room = self.room_var.get().strip()
        payload = image_message(room, self.my_name, os.path.basename(path), data)
        asyncio.run_coroutine_threadsafe(self.ws.send(payload), self.loop)
        self._append_chat(f"[я] отправлено на проектор: {os.path.basename(path)}")

    def start_ptt(self) -> None:
        if not HAS_AUDIO or not self.connected or self.ptt_active:
            return
        self.ptt_active = True
        threading.Thread(target=self._ptt_capture_loop, daemon=True).start()

    def stop_ptt(self) -> None:
        self.ptt_active = False

    def _ptt_capture_loop(self) -> None:
        from corpcom.protocol import voice_message

        room = self.room_var.get().strip()
        while self.ptt_active and self.ws and self.loop:
            rec = sd.rec(
                config.VOICE_CHUNK_SAMPLES,
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype="int16",
            )
            sd.wait()
            if not self.ptt_active:
                break
            pcm = rec.tobytes()
            payload = voice_message(room, self.my_name, pcm)
            asyncio.run_coroutine_threadsafe(self.ws.send(payload), self.loop)


def main() -> None:
    root = tk.Tk()
    CorpComClient(root)
    root.mainloop()


if __name__ == "__main__":
    main()
