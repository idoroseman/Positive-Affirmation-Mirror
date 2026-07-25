from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import face_recognition
import numpy as np


@dataclass(slots=True)
class KnownFace:
    name: str
    encoding: np.ndarray


@dataclass(slots=True)
class MatchResult:
    person_name: str | None
    distance: float | None


class FaceLibrary:
    SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def __init__(self, known_faces_dir: Path, tolerance: float) -> None:
        self._known_faces_dir = known_faces_dir
        self._tolerance = tolerance
        self._faces: list[KnownFace] = []
        self._dataset_fingerprint: tuple[tuple[str, int, int], ...] = ()
        self._last_reload_check_at: float = 0.0
        self._reload_faces()

    def maybe_reload(self, now: float, check_interval_seconds: float = 2.0) -> bool:
        if check_interval_seconds > 0 and (now - self._last_reload_check_at) < check_interval_seconds:
            return False
        self._last_reload_check_at = now

        image_paths, fingerprint = self._scan_image_paths_and_fingerprint()
        if fingerprint == self._dataset_fingerprint:
            return False

        previous_count = len(self._faces)
        self._reload_faces(image_paths=image_paths, fingerprint=fingerprint)
        logging.info(
            "[Faces] Reloaded known-face dataset: %s -> %s encodings (%s image files)",
            previous_count,
            len(self._faces),
            len(image_paths),
        )
        return True

    def _reload_faces(
        self,
        image_paths: list[Path] | None = None,
        fingerprint: tuple[tuple[str, int, int], ...] | None = None,
    ) -> None:
        if image_paths is None or fingerprint is None:
            image_paths, fingerprint = self._scan_image_paths_and_fingerprint()
        self._faces = self._load_faces(image_paths)
        self._dataset_fingerprint = fingerprint

    def _scan_image_paths_and_fingerprint(self) -> tuple[list[Path], tuple[tuple[str, int, int], ...]]:
        image_paths: list[Path] = []
        fingerprint_entries: list[tuple[str, int, int]] = []
        if not self._known_faces_dir.exists():
            return image_paths, ()

        for person_dir in sorted(
            path for path in self._known_faces_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        ):
            for image_path in sorted(person_dir.glob("*")):
                if image_path.name.startswith("."):
                    continue
                if image_path.suffix.lower() not in self.SUPPORTED_IMAGE_SUFFIXES:
                    continue
                try:
                    stat = image_path.stat()
                except OSError:
                    continue
                image_paths.append(image_path)
                relative_path = image_path.relative_to(self._known_faces_dir).as_posix()
                fingerprint_entries.append((relative_path, stat.st_mtime_ns, stat.st_size))
        return image_paths, tuple(fingerprint_entries)

    def _load_faces(self, image_paths: list[Path]) -> list[KnownFace]:
        faces: list[KnownFace] = []
        for image_path in image_paths:
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
            except OSError:
                continue
            if encodings:
                faces.append(KnownFace(name=image_path.parent.name, encoding=encodings[0]))
        return faces

    def match(self, encoding: np.ndarray) -> MatchResult:
        if not self._faces:
            return MatchResult(person_name=None, distance=None)

        known_encodings = [face.encoding for face in self._faces]
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance <= self._tolerance:
            return MatchResult(person_name=self._faces[best_index].name, distance=best_distance)
        return MatchResult(person_name=None, distance=best_distance)
