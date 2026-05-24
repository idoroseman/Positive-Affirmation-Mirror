from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from pathlib import Path


class AudioPlayer:
    def __init__(self, playback_command: str | None = None) -> None:
        self._queue: queue.Queue[Path] = queue.Queue()
        self._playback_command = playback_command or self._detect_default_command()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def enqueue(self, audio_path: Path) -> None:
        if audio_path.exists():
            self._queue.put(audio_path)

    def _detect_default_command(self) -> str | None:
        for command in ("ffplay", "mpg123"):
            if shutil.which(command):
                return command
        return None

    def _build_command(self, audio_path: Path) -> list[str] | None:
        if self._playback_command == "ffplay":
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)]
        if self._playback_command == "mpg123":
            return ["mpg123", "-q", str(audio_path)]
        return None

    def _run(self) -> None:
        while True:
            audio_path = self._queue.get()
            try:
                command = self._build_command(audio_path)
                if command is not None:
                    subprocess.run(command, check=False)
            finally:
                self._queue.task_done()
