from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    def __init__(self, known_faces_dir: Path, tolerance: float) -> None:
        self._known_faces_dir = known_faces_dir
        self._tolerance = tolerance
        self._faces = self._load_faces()

    def _load_faces(self) -> list[KnownFace]:
        faces: list[KnownFace] = []
        if not self._known_faces_dir.exists():
            return faces

        for person_dir in sorted(
            path for path in self._known_faces_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        ):
            for image_path in sorted(person_dir.glob("*")):
                if image_path.name.startswith("."):
                    continue
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    faces.append(KnownFace(name=person_dir.name, encoding=encodings[0]))
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
