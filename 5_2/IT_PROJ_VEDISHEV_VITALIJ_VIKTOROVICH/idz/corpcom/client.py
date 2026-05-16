#!/usr/bin/env python3
"""CorpCom — клиент ПК: текст, голос в реальном времени, изображение на проектор."""

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
from corpcom.protocol import (
    MsgType,
    auth_message,
    image_message,
    loads,
    text_message,
    voice_end_message,
    voice_message,
    voice_start_message,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("corpcom.client")

try:
    from corpcom.voice import VoiceEngine

    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class CorpComClient:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CorpCom — корпоративный клиент")
        self.root.geometry("760x600")

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.connected = False
        self.my_name = ""
        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.voice_streaming = False
        self.ptt_active = False
        self.voice_seq = 0
        self.voice_out: asyncio.Queue[bytes] | None = None
        self._voice_sender_task: asyncio.Task[None] | None = None
        self.voice: VoiceEngine | None = None
        if HAS_AUDIO:
            self.voice = VoiceEngine(self._on_voice_captured)

        self._build_ui()
        self.root.after(80, self._drain_ui_queue)

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

        voice_frame = ttk.LabelFrame(self.root, text="Голос в реальном времени")
        voice_frame.pack(fill=tk.X, padx=8, pady=4)

        self.voice_status_var = tk.StringVar(
            value="Микрофон выкл" if HAS_AUDIO else "Установите: pip install sounddevice numpy (через ./setup.sh)"
        )
        ttk.Label(voice_frame, textvariable=self.voice_status_var).pack(side=tk.LEFT, padx=8)

        self.btn_mic = ttk.Button(
            voice_frame,
            text="Включить микрофон",
            command=self.toggle_voice_stream,
            state=tk.NORMAL if HAS_AUDIO else tk.DISABLED,
        )
        self.btn_mic.pack(side=tk.LEFT, padx=4, pady=4)

        self.btn_ptt = ttk.Button(
            voice_frame,
            text="PTT (удерживать)",
            state=tk.NORMAL if HAS_AUDIO else tk.DISABLED,
        )
        self.btn_ptt.pack(side=tk.LEFT, padx=4, pady=4)
        if HAS_AUDIO:
            self.btn_ptt.bind("<ButtonPress-1>", lambda _e: self._start_ptt())
            self.btn_ptt.bind("<ButtonRelease-1>", lambda _e: self._stop_ptt())

        self.listen_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            voice_frame,
            text="Слышать собеседников",
            variable=self.listen_var,
            command=self._toggle_listen,
            state=tk.NORMAL if HAS_AUDIO else tk.DISABLED,
        ).pack(side=tk.LEFT, padx=8)

        chat_frame = ttk.LabelFrame(self.root, text="Чат")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.chat = scrolledtext.ScrolledText(chat_frame, state=tk.DISABLED, height=14)
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

    def _toggle_listen(self) -> None:
        if self.voice:
            self.voice.set_listening(self.listen_var.get())

    def _on_voice_captured(self, pcm: bytes) -> None:
        if not self.loop or not self.voice_out:
            return
        try:
            self.loop.call_soon_threadsafe(self.voice_out.put_nowait, pcm)
        except asyncio.QueueFull:
            pass

    def _room(self) -> str:
        return self.room_var.get().strip() or config.DEFAULT_ROOM

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
        room = self._room()
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
        self.voice_out = asyncio.Queue(maxsize=64)
        try:
            async with websockets.connect(
                uri,
                ping_interval=20,
                max_size=config.MAX_IMAGE_BYTES + 512 * 1024,
            ) as ws:
                self.ws = ws
                await ws.send(auth_message(token, "user", name, room))
                raw = await ws.recv()
                msg = loads(raw)  # type: ignore[arg-type]
                if msg.get("type") == MsgType.AUTH_ERR:
                    self.ui_queue.put(("error", msg.get("reason", "auth failed")))
                    return
                self.ui_queue.put(("connected", msg))
                self._voice_sender_task = asyncio.create_task(self._voice_sender(ws))
                await self._recv_loop(ws)
        except Exception as exc:
            self.ui_queue.put(("error", str(exc)))
        finally:
            if self._voice_sender_task:
                self._voice_sender_task.cancel()
            self._voice_sender_task = None
            self.voice_out = None
            self.ws = None
            self.ui_queue.put(("disconnected", None))

    async def _voice_sender(self, ws: websockets.WebSocketClientProtocol) -> None:
        assert self.voice_out is not None
        room = self._room()
        while True:
            pcm = await self.voice_out.get()
            if not self.ws:
                continue
            self.voice_seq += 1
            await ws.send(
                voice_message(room, self.my_name, pcm, self.voice_seq)
            )

    async def _recv_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            self.ui_queue.put(("message", loads(raw)))

    def disconnect(self) -> None:
        self._stop_voice_stream()
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
                    if self.voice and self.listen_var.get():
                        self.voice.set_listening(True)
                elif kind == "disconnected":
                    self.connected = False
                    self._stop_voice_stream()
                    if self.voice:
                        self.voice.shutdown()
                    self.btn_connect.configure(text="Подключить", state=tk.NORMAL)
                    self.btn_mic.configure(text="Включить микрофон")
                    self.status_var.set("Отключено")
                    self.voice_status_var.set("Микрофон выкл")
                    self._append_chat("[система] Отключено")
                elif kind == "error":
                    messagebox.showerror("CorpCom", str(payload))
                    self.btn_connect.configure(state=tk.NORMAL)
                elif kind == "message":
                    self._handle_message(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        self.root.after(80, self._drain_ui_queue)

    def _handle_message(self, msg: dict) -> None:
        t = msg.get("type")
        sender = msg.get("sender", "?")
        if t == MsgType.TEXT:
            if sender == self.my_name:
                return
            self._append_chat(f"[{sender}] {msg.get('content', '')}")
        elif t == MsgType.SYS:
            self._append_chat(f"[система] {msg.get('content', '')}")
        elif t == MsgType.CLIENT_LIST:
            clients = ", ".join(msg.get("clients", []))
            self.status_var.set(f"В сети | участники: {clients}")
        elif t == MsgType.VOICE_START and sender != self.my_name:
            self._append_chat(f"[голос] {sender} говорит…")
        elif t == MsgType.VOICE_END and sender != self.my_name:
            self._append_chat(f"[голос] {sender} закончил")
        elif t == MsgType.VOICE and sender != self.my_name and self.voice:
            from corpcom.protocol import b64_decode

            self.voice.play_remote(b64_decode(msg["data"]))
        elif t == MsgType.IMAGE:
            from corpcom.protocol import b64_decode

            data = b64_decode(msg["data"])
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
        payload = text_message(self._room(), self.my_name, text)
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

        payload = image_message(
            self._room(), self.my_name, os.path.basename(path), data
        )
        asyncio.run_coroutine_threadsafe(self.ws.send(payload), self.loop)
        self._append_chat(f"[я] отправлено на проектор: {os.path.basename(path)}")

    def _send_voice_signal(self, factory) -> None:
        if self.ws and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.ws.send(factory(self._room(), self.my_name)), self.loop
            )

    def toggle_voice_stream(self) -> None:
        if not HAS_AUDIO or not self.connected:
            messagebox.showinfo("CorpCom", "Подключитесь к серверу")
            return
        if self.voice_streaming:
            self._stop_voice_stream()
        else:
            self._start_voice_stream()

    def _start_voice_stream(self) -> None:
        if not self.voice or self.voice_streaming:
            return
        self.voice_streaming = True
        self.voice.start_tx()
        self._send_voice_signal(voice_start_message)
        self.btn_mic.configure(text="Выключить микрофон")
        self.voice_status_var.set("Микрофон вкл — идёт передача")
        self._append_chat("[голос] Микрофон включён")

    def _stop_voice_stream(self) -> None:
        if not self.voice_streaming:
            return
        self.voice_streaming = False
        if self.voice:
            self.voice.stop_tx()
        self._send_voice_signal(voice_end_message)
        self.btn_mic.configure(text="Включить микрофон")
        self.voice_status_var.set("Микрофон выкл")

    def _start_ptt(self) -> None:
        if not HAS_AUDIO or not self.connected or self.ptt_active or self.voice_streaming:
            return
        if not self.voice:
            return
        self.ptt_active = True
        self.voice.start_tx()
        self._send_voice_signal(voice_start_message)
        self.voice_status_var.set("PTT: говорите…")

    def _stop_ptt(self) -> None:
        if not self.ptt_active:
            return
        self.ptt_active = False
        if self.voice:
            self.voice.stop_tx()
        self._send_voice_signal(voice_end_message)
        self.voice_status_var.set("Микрофон выкл")


def main() -> None:
    root = tk.Tk()
    CorpComClient(root)
    root.mainloop()


if __name__ == "__main__":
    main()
