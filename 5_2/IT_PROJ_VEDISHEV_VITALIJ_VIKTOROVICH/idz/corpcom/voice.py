"""Потоковый захват и воспроизведение голоса (PCM) в реальном времени."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from corpcom import config

log = logging.getLogger("corpcom.voice")


class VoicePlayer:
    """Непрерывное воспроизведение входящих фрагментов без наложения sd.play()."""

    def __init__(self) -> None:
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self._pending = np.array([], dtype=np.int16)
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._stream is not None:
            return

        block = config.VOICE_CHUNK_SAMPLES

        def callback(outdata, frames, _time, status) -> None:
            if status:
                log.debug("playback: %s", status)
            out = np.zeros(frames, dtype=np.int16)
            pos = 0
            with self._lock:
                while pos < frames:
                    if self._pending.size == 0:
                        try:
                            self._pending = self._q.get_nowait()
                        except queue.Empty:
                            break
                    take = min(frames - pos, self._pending.size)
                    out[pos : pos + take] = self._pending[:take]
                    self._pending = self._pending[take:]
                    pos += take
            outdata[:, 0] = out

        self._stream = sd.OutputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="int16",
            blocksize=block,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            self._pending = np.array([], dtype=np.int16)
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def feed(self, pcm: bytes) -> None:
        if not pcm:
            return
        chunk = np.frombuffer(pcm, dtype=np.int16)
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(chunk)


class VoiceCapture:
    """Захват с микрофона фиксированными блоками через InputStream."""

    def __init__(self, on_chunk: Callable[[bytes], None]) -> None:
        self._on_chunk = on_chunk
        self._active = threading.Event()
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        if self._stream is not None:
            self._active.set()
            return

        block = config.VOICE_CHUNK_SAMPLES

        def callback(indata, _frames, _time, status) -> None:
            if status:
                log.debug("capture: %s", status)
            if self._active.is_set():
                self._on_chunk(indata[:, 0].tobytes())

        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="int16",
            blocksize=block,
            callback=callback,
        )
        self._active.set()
        self._stream.start()

    def stop(self) -> None:
        self._active.clear()

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class VoiceEngine:
    """Захват + воспроизведение для клиента CorpCom."""

    def __init__(self, on_captured: Callable[[bytes], None]) -> None:
        self._capture = VoiceCapture(on_captured)
        self._player = VoicePlayer()
        self._listen = True

    @property
    def listening(self) -> bool:
        return self._listen

    def set_listening(self, enabled: bool) -> None:
        self._listen = enabled
        if not enabled:
            self._player.stop()
        elif enabled:
            self._player.start()

    def start_tx(self) -> None:
        self._capture.start()

    def stop_tx(self) -> None:
        self._capture.stop()

    def close_tx(self) -> None:
        self._capture.close()

    def ensure_rx(self) -> None:
        if self._listen:
            self._player.start()

    def play_remote(self, pcm: bytes) -> None:
        if self._listen:
            self.ensure_rx()
            self._player.feed(pcm)

    def shutdown(self) -> None:
        self._capture.close()
        self._player.stop()
