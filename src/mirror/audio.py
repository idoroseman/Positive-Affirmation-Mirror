from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
import shlex
from pathlib import Path


class AudioPlayer:
    def __init__(self, playback_command: str | None = None) -> None:
        self._queue: queue.Queue[Path] = queue.Queue()
        self._playback_command = playback_command or self._detect_default_command()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self.now_playing: Path | None = None
        logging.info("AudioPlayer initialized with playback command: %s", self._playback_command)

    def enqueue(self, audio_path: Path) -> None:
        if audio_path.exists():
            self._queue.put(audio_path)

    def _detect_default_command(self) -> str | None:
        # Prefer ALSA-native path first for headless systemd setups.
        if shutil.which("ffmpeg") and shutil.which("aplay"):
            return "ffmpeg-aplay"
        for command in ("mpg123", "ffplay"):
            if shutil.which(command):
                return command
        return None

    def _build_command(self, audio_path: Path) -> list[str] | None:
        if self._playback_command and "{" in self._playback_command:
            return [token.format(path=str(audio_path)) for token in shlex.split(self._playback_command)]
        if self._playback_command and " " in self._playback_command:
            return shlex.split(self._playback_command) + [str(audio_path)]
        if self._playback_command == "ffplay":
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)]
        if self._playback_command == "mpg123":
            # Force ALSA output to avoid JACK backend crashes in headless service environments.
            return ["mpg123", "-q", "-o", "alsa", str(audio_path)]
        if self._playback_command == "ffmpeg-aplay":
            return ["ffmpeg-aplay", str(audio_path)]
        return None

    def _play_with_ffmpeg_aplay(self, audio_path: Path) -> tuple[int, str]:
        preferred_device = os.environ.get("MIRROR_APLAY_DEVICE", "").strip()
        # Candidate order prefers analog/USB outputs before HDMI in typical Pi setups.
        device_candidates: list[str | None] = [
            None,
            "plughw:1,0",
            "plughw:3,0",
            "plughw:0,0",
            "plughw:2,0",
            "hw:1,0",
            "hw:3,0",
            "hw:0,0",
            "hw:2,0",
        ]
        if preferred_device:
            device_candidates.insert(0, preferred_device)

        seen: set[str | None] = set()
        ordered_candidates: list[str | None] = []
        for candidate in device_candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered_candidates.append(candidate)

        last_error = ""
        last_exit_code = 1
        for aplay_device in ordered_candidates:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(audio_path),
                    "-f",
                    "wav",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )

            aplay_command = ["aplay", "-q"]
            if aplay_device:
                aplay_command.extend(["-D", aplay_device])

            try:
                aplay = subprocess.Popen(
                    aplay_command,
                    stdin=ffmpeg.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            finally:
                if ffmpeg.stdout is not None:
                    ffmpeg.stdout.close()

            _, ffmpeg_stderr = ffmpeg.communicate()
            _, aplay_stderr = aplay.communicate()

            if aplay.returncode != 0:
                last_exit_code = aplay.returncode
                tail = (aplay_stderr or "").strip().splitlines()[-1:] or [""]
                device_label = aplay_device or "default"
                last_error = f"aplay[{device_label}]: {tail[0]}"
                continue

            ffmpeg_stderr_text = (ffmpeg_stderr or b"").decode("utf-8", errors="ignore").strip()
            if ffmpeg.returncode != 0 and "Broken pipe" not in ffmpeg_stderr_text:
                last_exit_code = ffmpeg.returncode
                tail = ffmpeg_stderr_text.splitlines()[-1:] or [""]
                device_label = aplay_device or "default"
                last_error = f"ffmpeg[{device_label}]: {tail[0]}"
                continue

            if aplay_device:
                logging.info("Audio playback using ALSA device: %s", aplay_device)
            return 0, ""

        return last_exit_code, last_error

    def _run_command(self, command: list[str], audio_path: Path) -> tuple[int, str]:
        if command and command[0] == "ffmpeg-aplay":
            return self._play_with_ffmpeg_aplay(audio_path)

        result = subprocess.run(command, check=False, capture_output=True, text=True)
        stderr_tail = (result.stderr or "").strip().splitlines()[-1:] or [""]
        return result.returncode, stderr_tail[0]

    def _run(self) -> None:
        while True:
            audio_path = self._queue.get()
            try:
                command = self._build_command(audio_path)
                if command is not None:
                    self.now_playing = audio_path
                    exit_code, stderr_tail = self._run_command(command, audio_path)
                    if exit_code != 0:
                        logging.error(
                            "Audio playback failed (command=%s, exit=%s, file=%s): %s",
                            command[0],
                            exit_code,
                            audio_path,
                            stderr_tail,
                        )
                        fallbacks: list[list[str]] = []
                        if command[0] == "mpg123" and shutil.which("ffplay"):
                            fallbacks.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)])
                        if command[0] != "ffmpeg-aplay" and shutil.which("ffmpeg") and shutil.which("aplay"):
                            fallbacks.append(["ffmpeg-aplay", str(audio_path)])

                        for fallback in fallbacks:
                            fallback_exit_code, fallback_stderr_tail = self._run_command(fallback, audio_path)
                            if fallback_exit_code == 0:
                                logging.info(
                                    "Audio playback fallback succeeded (command=%s): %s",
                                    fallback[0],
                                    audio_path.name,
                                )
                                break
                            logging.error(
                                "Audio playback fallback failed (command=%s, exit=%s, file=%s): %s",
                                fallback[0],
                                fallback_exit_code,
                                audio_path,
                                fallback_stderr_tail,
                            )
                    else:
                        logging.info("Audio playback completed: %s", audio_path.name)
                else:
                    logging.error("No supported playback command found (tried ffmpeg+aplay, ffplay, mpg123)")
            finally:
                self.now_playing = None
                self._queue.task_done()
