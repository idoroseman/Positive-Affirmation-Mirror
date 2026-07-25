from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    encoding: np.ndarray
    person_name: str | None
    demographics_label: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        top, right, bottom, left = self.bbox
        return ((left + right) / 2.0, (top + bottom) / 2.0)


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    encoding: np.ndarray
    first_seen_at: float
    last_seen_at: float
    person_name: str | None
    demographics_label: str | None
    greeted: bool = False
    dwell_logged: bool = False
    visible: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


class TrackTable:
    def __init__(self, match_distance: float, centroid_match_px: int, absence_seconds: float) -> None:
        self._match_distance = match_distance
        self._centroid_match_px = centroid_match_px
        self._absence_seconds = absence_seconds
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1

    def update(self, detections: list[Detection], now: float | None = None) -> list[Track]:
        now = now or time.monotonic()

        for track in self._tracks.values():
            track.visible = False

        remaining_track_ids = set(self._tracks)
        for detection in detections:
            track = self._match_detection(detection, remaining_track_ids)
            if track is None:
                track = self._create_track(detection, now)
            else:
                remaining_track_ids.discard(track.track_id)
                self._refresh_track(track, detection, now)

        stale_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.last_seen_at > self._absence_seconds
        ]
        for track_id in stale_track_ids:
            track = self._tracks[track_id]
            logging.info(
                "[Track] timed out: id=%s label=%s absent_for=%.1fs",
                track.track_id,
                self._describe_track(track),
                now - track.last_seen_at,
            )
            del self._tracks[track_id]

        return sorted(self._tracks.values(), key=lambda track: track.track_id)

    def visible_tracks(self) -> list[Track]:
        return [track for track in self._tracks.values() if track.visible]

    def _match_detection(self, detection: Detection, candidate_track_ids: set[int]) -> Track | None:
        best_track: Track | None = None
        best_score = math.inf

        for track_id in candidate_track_ids:
            track = self._tracks[track_id]
            centroid_distance = self._centroid_distance(track.centroid, detection.centroid)
            if centroid_distance > self._centroid_match_px:
                continue

            embedding_distance = float(np.linalg.norm(track.encoding - detection.encoding))
            if embedding_distance > self._match_distance:
                continue

            if detection.person_name and track.person_name and detection.person_name != track.person_name:
                continue

            score = embedding_distance + (centroid_distance / max(self._centroid_match_px, 1))
            if score < best_score:
                best_score = score
                best_track = track

        return best_track

    def _create_track(self, detection: Detection, now: float) -> Track:
        track = Track(
            track_id=self._next_track_id,
            bbox=detection.bbox,
            centroid=detection.centroid,
            encoding=detection.encoding,
            first_seen_at=now,
            last_seen_at=now,
            person_name=detection.person_name,
            demographics_label=detection.demographics_label,
            metadata=dict(detection.metadata),
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        logging.info("[Track] added: id=%s label=%s bbox=%s", track.track_id, self._describe_track(track), track.bbox)
        return track

    def _refresh_track(self, track: Track, detection: Detection, now: float) -> None:
        track.bbox = detection.bbox
        track.centroid = detection.centroid
        track.encoding = detection.encoding
        track.last_seen_at = now
        track.visible = True
        track.person_name = detection.person_name or track.person_name
        track.demographics_label = detection.demographics_label or track.demographics_label
        track.metadata.update(detection.metadata)

    @staticmethod
    def _centroid_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return math.dist(first, second)

    @staticmethod
    def _describe_track(track: Track) -> str:
        if track.person_name:
            return track.person_name
        age_bucket = track.metadata.get("age_bucket")
        gender = track.metadata.get("gender") or track.demographics_label
        if age_bucket and gender:
            return f"{age_bucket}_{gender}"
        return gender or age_bucket or "unknown"
