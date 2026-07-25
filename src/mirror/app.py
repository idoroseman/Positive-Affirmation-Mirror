from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np

from mirror.audio import AudioPlayer
from mirror.config import AppConfig, load_config
from mirror.recognizer import FaceLibrary
from mirror.tracker import Detection, Track, TrackTable


GENDER_SCORE_DECAY = 0.85
MIN_GENDER_OBSERVATIONS = 3
MIN_GENDER_STABLE_CONFIDENCE = 0.90


@dataclass(slots=True)
class recordingEvent:
    audio_path: Path
    reason: str
    track_id: int | None = None


@dataclass(slots=True)
class DemographicsResult:
    gender: str | None = None
    gender_confidence: float | None = None
    gender_scores: dict[str, float] | None = None


@dataclass(slots=True)
class ClassifierPrediction:
    label: str | None
    confidence: float
    scores: dict[str, float]


@dataclass(slots=True)
class OpenCVCaffeAnalyzer:
    net: Any
    labels: tuple[str, ...]
    min_confidence: float

    def describe(self, face_crop) -> ClassifierPrediction | None:
        blob = cv2.dnn.blobFromImage(
            face_crop,
            scalefactor=1.0,
            size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        result = self.net.forward()
        if result.ndim != 2 or result.shape[1] < 2:
            return None

        label_index = int(result[0].argmax())
        confidence = float(result[0][label_index])
        if label_index >= len(self.labels):
            return None
        scores = {
            label: float(result[0][index])
            for index, label in enumerate(self.labels)
            if index < result.shape[1]
        }
        label = self.labels[label_index] if confidence >= self.min_confidence else None
        return ClassifierPrediction(label=label, confidence=confidence, scores=scores)


@dataclass(slots=True)
class FairFaceGenderAnalyzer:
    net: Any
    min_confidence: float

    def describe(self, face_crop) -> ClassifierPrediction | None:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        face_rgb = face_rgb.astype(np.float32) / 255.0
        face_rgb = (face_rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        blob = np.transpose(face_rgb, (2, 0, 1))[np.newaxis, :, :, :]
        self.net.setInput(blob)
        result = self.net.forward().reshape(-1)
        if result.size < 9:
            return None

        gender_logits = result[7:9]
        gender_logits -= np.max(gender_logits)
        probabilities = np.exp(gender_logits)
        probabilities /= probabilities.sum()

        man_score = float(probabilities[0])
        woman_score = float(probabilities[1])
        scores = {
            "man": man_score,
            "woman": woman_score,
        }
        if man_score >= woman_score:
            label = "man"
            confidence = man_score
        else:
            label = "woman"
            confidence = woman_score

        return ClassifierPrediction(
            label=label if confidence >= self.min_confidence else None,
            confidence=confidence,
            scores=scores,
        )


class DemographicsService:
    FACE_PADDING_PX = 20

    def __init__(
        self,
        enabled: bool,
        gender_config_path: Path | None,
        gender_model_path: Path | None,
        gender_min_confidence: float,
    ) -> None:
        self._gender_analyzer = self._load_gender_analyzer(
            gender_config_path,
            gender_model_path,
            gender_min_confidence,
        ) if enabled else None

    def _load_gender_analyzer(
        self,
        config_path: Path | None,
        model_path: Path | None,
        min_confidence: float,
    ) -> FairFaceGenderAnalyzer | None:
        if model_path is None:
            return None
        if not model_path.exists():
            return None

        try:
            if model_path.suffix.lower() == ".onnx":
                net = cv2.dnn.readNetFromONNX(str(model_path))
            elif config_path is not None and config_path.exists():
                net = cv2.dnn.readNet(str(model_path), str(config_path))
            else:
                return None
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return FairFaceGenderAnalyzer(net=net, min_confidence=min_confidence)
        except cv2.error as exc:
            logging.error(f"[Error] Could not load demographics model {model_path.name}: {exc}")
            return None

    def needs_description(self, track: Track) -> bool:
        if self._gender_analyzer is None:
            return False

        existing_gender = track.metadata.get("gender") or track.demographics_label
        return existing_gender not in {"man", "woman"}

    def describe(self, frame_bgr, bbox: tuple[int, int, int, int]) -> DemographicsResult | None:
        if self._gender_analyzer is None:
            return None

        top, right, bottom, left = bbox
        face_crop = frame_bgr[
            max(top - self.FACE_PADDING_PX, 0):min(bottom + self.FACE_PADDING_PX, frame_bgr.shape[0] - 1),
            max(left - self.FACE_PADDING_PX, 0):min(right + self.FACE_PADDING_PX, frame_bgr.shape[1] - 1),
        ]
        if face_crop.size == 0:
            return None
        if face_crop.shape[0] < 32 or face_crop.shape[1] < 32:
            return None

        try:
            gender_prediction = self._gender_analyzer.describe(face_crop)
        except cv2.error:
            return None

        gender = gender_prediction.label if gender_prediction else None
        if gender is None and not (gender_prediction and gender_prediction.scores):
            return None
        return DemographicsResult(
            gender=gender,
            gender_confidence=gender_prediction.confidence if gender_prediction else None,
            gender_scores=gender_prediction.scores if gender_prediction else None,
        )


class recordingSelector:
    AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}
    UNKNOWN_DIRECTORY_BY_GENDER = {
        "man": "unknown_m",
        "woman": "unknown_f",
    }

    def __init__(self, recordings_dir: Path) -> None:
        self._recordings_dir = recordings_dir

    def choose(self, visible_tracks: list[Track], dwell_seconds: float, now: float) -> recordingEvent | None:
        qualified_tracks = [
            track for track in visible_tracks if not track.greeted and now - track.first_seen_at >= dwell_seconds
        ]
        if not qualified_tracks:
            return None

        for track in qualified_tracks:
            if track.dwell_logged:
                continue
            logging.info(
                f"[Track] dwelled: id={track.track_id} label={describe_track(track)} "
                f"seen_for={now - track.first_seen_at:.1f}s"
            )
            track.dwell_logged = True
        reason = "unknown"
        selected_track_id: int | None = None
        
        if len(qualified_tracks) >= 3:
            audio_path = self._pick_from_directory(self._recordings_dir / "group")
            reason = "group"
            selected_track_id = qualified_tracks[0].track_id
        elif len(qualified_tracks) == 2:
            audio_path = self._pick_from_directory(self._recordings_dir / "couple")
            reason = "two_people"
            selected_track_id = qualified_tracks[0].track_id
        elif len(qualified_tracks) == 1:
            track = qualified_tracks[0]
            selected_track_id = track.track_id
            gender = track.metadata.get("gender") or track.demographics_label

            if track.person_name:
                audio_path = self._pick_from_directory(self._recordings_dir / track.person_name)
                reason = track.person_name
            elif gender == "man":
                audio_path = self._pick_from_directory(self._recordings_dir / "unknown_m")
                reason = "unknown_m"
            elif gender == "woman":
                audio_path = self._pick_from_directory(self._recordings_dir / "unknown_f")
                reason = "unknown_f"
            else:                
                audio_path = self._pick_from_directory(self._recordings_dir / "unknown")  
                reason = "unknown"

        if audio_path is None:
            logging.info(f"[Greet]: id={selected_track_id} {reason} -> no audio file found")
            return None

        for track in qualified_tracks:
            track.greeted = True
        return recordingEvent(audio_path=audio_path, reason=reason, track_id=selected_track_id)

    def _pick_from_directory(self, directory: Path) -> Path | None:
        if not directory.is_dir():
            return None

        candidates = sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in self.AUDIO_SUFFIXES
        )
        if not candidates:
            return None
        return random.choice(candidates)


def describe_track(track: Track) -> str:
    if track.person_name:
        return track.person_name
    gender = track.metadata.get("gender") or track.demographics_label
    return gender or "unknown"



def save_unknown_snapshots(
    frame_bgr,
    visible_tracks: list[Track],
    dwell_seconds: float,
    now: float,
    snapshot_dir: Path | None,
) -> None:
    if snapshot_dir is None:
        return

    for track in visible_tracks:
        if track.person_name is not None:
            continue
        if track.metadata.get("snapshot_saved") == "1":
            continue
        if now - track.first_seen_at < dwell_seconds:
            continue

        top, right, bottom, left = track.bbox
        face_crop = frame_bgr[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
        if face_crop.size == 0:
            continue

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-track-{track.track_id}.jpg"
        snapshot_path = snapshot_dir / filename
        if cv2.imwrite(str(snapshot_path), face_crop):
            track.metadata["snapshot_saved"] = "1"
            logging.debug(f"[Debug] saved Snapshot: id={track.track_id} path={filename}")


def build_detections(
    frame_bgr,
    face_library: FaceLibrary,
) -> list[Detection]:
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detections: list[Detection] = []
    for bbox, encoding in zip(face_locations, encodings):
        match = face_library.match(encoding)
        detections.append(
            Detection(
                bbox=bbox,
                encoding=encoding,
                person_name=match.person_name,
            )
        )
    return detections


def _update_gender_metadata(track: Track, result: DemographicsResult) -> None:
    if not result.gender_scores or result.gender is None or result.gender_confidence is None:
        return

    for label, score in result.gender_scores.items():
        score_key = f"gender_score_{label}"
        previous_score = float(track.metadata.get(score_key, "0") or 0)
        track.metadata[score_key] = f"{(previous_score * GENDER_SCORE_DECAY) + score:.4f}"

    observation_count = int(track.metadata.get("gender_observations", "0") or 0) + 1
    track.metadata["gender_observations"] = str(observation_count)

    man_score = float(track.metadata.get("gender_score_man", "0") or 0)
    woman_score = float(track.metadata.get("gender_score_woman", "0") or 0)
    total_score = man_score + woman_score
    if total_score <= 0:
        return

    if man_score >= woman_score:
        label = "man"
        confidence = man_score / total_score
    else:
        label = "woman"
        confidence = woman_score / total_score

    if observation_count < MIN_GENDER_OBSERVATIONS or confidence < MIN_GENDER_STABLE_CONFIDENCE:
        track.demographics_label = None
        track.metadata.pop("gender", None)
        track.metadata.pop("gender_confidence", None)
        return

    track.demographics_label = label
    track.metadata["gender"] = label
    track.metadata["gender_confidence"] = f"{confidence:.2f}"


def describe_visible_tracks(frame_bgr, visible_tracks: list[Track], demographics: DemographicsService) -> None:
    for track in visible_tracks:
        if not demographics.needs_description(track):
            continue
        result = demographics.describe(frame_bgr, track.bbox)
        if result is None:
            continue
        _update_gender_metadata(track, result)


def annotate_frame(frame_bgr, tracks: list[Track], now: float, fps: float | None, now_playing: Path | None) -> None:
    for track in tracks:
        if not track.visible:
            continue

        top, right, bottom, left = track.bbox
        color = (40, 200, 40) if track.greeted else (0, 180, 255)
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

        state = f"seen {now - track.first_seen_at:.1f}s"
        label = f"{track.person_name or 'Unknown'} {track.track_id}"

        gender = track.metadata.get("gender") or track.demographics_label
        gender_confidence = track.metadata.get("gender_confidence")

        gender_color = (0, 180, 255)
        if gender == "man":
            gender_color = (255, 50, 50)
        if gender == "woman":
            gender_color = (255, 50, 255)

        cv2.putText(
            frame_bgr,
            f"{label}",
            (left, max(15, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            gender_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"{state}",
            (left, max(15, bottom + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

    faces = len([track for track in tracks if track.visible])
    fps_text = f"{fps:.1f}" if fps is not None else "N/A"
    cv2.putText(
        frame_bgr,
        f"fps: {fps_text} active: {faces:2}/{len(tracks)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"{now_playing.name if now_playing else ''}",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )


def resize_frame(frame_bgr, target_width: int):
    height, width = frame_bgr.shape[:2]
    if width <= target_width:
        return frame_bgr
    scale = target_width / float(width)
    new_size = (target_width, int(height * scale))
    return cv2.resize(frame_bgr, new_size)




def run_app(config: AppConfig, show_window: bool = True) -> int:
    base_dir = Path.cwd()
    known_faces_dir = base_dir / config.known_faces_dir
    recordings_dir = base_dir / config.recordings_dir
    unknown_snapshot_dir = base_dir / config.unknown_snapshot_dir if config.unknown_snapshot_dir else None
    gender_config_path = (
        base_dir / config.demographics_gender_model_config if config.demographics_gender_model_config else None
    )
    gender_model_path = base_dir / config.demographics_gender_model if config.demographics_gender_model else None

    audio_player = AudioPlayer(playback_command=config.playback_command)
    face_library = FaceLibrary(known_faces_dir=known_faces_dir, tolerance=config.recognition_tolerance)
    tracker = TrackTable(
        match_distance=config.track_match_distance,
        centroid_match_px=config.centroid_match_px,
        absence_seconds=config.absence_seconds,
    )
    selector = recordingSelector(recordings_dir=recordings_dir)
    demographics = DemographicsService(
        enabled=config.allow_optional_demographics,
        gender_config_path=gender_config_path,
        gender_model_path=gender_model_path,
        gender_min_confidence=config.demographics_gender_min_confidence,
    )

    capture = None
    picam = None

    if config.camera_index == "picam":
        try:
            from picamera2 import Picamera2
        except ModuleNotFoundError as exc:
            logging.error(f"[Error] {exc}")
            logging.error("[Error] Raspberry Pi camera support comes from OS packages, not pip.")
            logging.error("[Error] Install:")
            logging.error("  sudo apt install -y python3-picamera2 python3-libcamera libcamera-dev libudev-dev")
            logging.error("[Error] If you use a virtual environment, recreate it with:")
            logging.error("  python3 -m venv --system-site-packages .venv")
            return 1
        picam = Picamera2()
        picam.configure(picam.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        picam.start()
    else:
        capture = cv2.VideoCapture(config.camera_index)

    if capture is not None and not capture.isOpened():
        if config.camera_index is not None:
            logging.error(f"[Error] Could not open video source: {config.camera_index}")
        else:
            logging.error("[Error] Could not open webcam")
        return 1

    previous_frame_at: float | None = None
    last_tracks: list[Track] = []
    try:
        while True:
            if config.camera_index == "picam":
                frame_bgr = picam.capture_array()
            else:
                ok, frame_bgr = capture.read()
                if not ok:
                    if config.camera_index is not None:
                        break
                    logging.error("[Error] Camera frame read failed")
                    return 1

                frame_bgr = resize_frame(frame_bgr, config.frame_resize_width)
            
            now = time.monotonic()
            fps = None if previous_frame_at is None else 1.0 / max(now - previous_frame_at, 1e-6)
            previous_frame_at = now

            if not tracker.visible_tracks():
                face_library.maybe_reload(now, config.known_faces_reload_interval_seconds)
            detections = build_detections(frame_bgr, face_library)
            last_tracks = tracker.update(detections, now=now)
            describe_visible_tracks(frame_bgr, tracker.visible_tracks(), demographics)
            if config.save_unknown_snapshots:
                save_unknown_snapshots(
                    frame_bgr,
                    tracker.visible_tracks(),
                    config.dwell_seconds,
                    now,
                    unknown_snapshot_dir,
                )
            event = selector.choose(tracker.visible_tracks(), config.dwell_seconds, now)
            if event is not None:
                if event.track_id is None:
                    logging.info(f"[Greet]: {event.reason} -> {event.audio_path.name}")
                else:
                    logging.info(f"[Greet]: id={event.track_id} {event.reason} -> {event.audio_path.name}")
                audio_player.enqueue(event.audio_path)

            annotate_frame(frame_bgr, last_tracks, now, fps, audio_player.now_playing)

            if show_window:
                cv2.imshow("mirror", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if os.environ.get("MIRROR_ONE_SHOT") == "1":
                    break
    finally:
        if capture is not None:
            capture.release()
        if picam is not None:
            picam.stop()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face-aware recording app for Raspberry Pi")
    parser.add_argument("--config", default="config.json", help="Path to the JSON config file")
    parser.add_argument("--headless", action="store_true", help="Run without opening a preview window")
    parser.add_argument("--video", help="Path to a test video file instead of using the live camera")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    args = parse_args()
    logging.info("Starting vibe-mirror")
    config = load_config(Path(args.config))
    logging.info("Loaded config from %s", args.config)
    if args.video:
        config.camera_index = args.video
        logging.info("Overriding camera input with video: %s", args.video)
    return run_app(config, show_window=not args.headless)
