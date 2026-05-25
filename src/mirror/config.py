from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    camera_index: int = 0
    video_source: str | None = None
    frame_resize_width: int = 640
    dwell_seconds: float = 5.0
    absence_seconds: float = 30.0
    recognition_tolerance: float = 0.47
    track_match_distance: float = 0.52
    centroid_match_px: int = 180
    playback_command: str | None = None
    known_faces_dir: str = "data/known_faces"
    recordings_dir: str = "data/recordings"
    unknown_snapshot_dir: str | None = "data/unknown_snapshots"
    save_unknown_snapshots: bool = False,
    allow_optional_demographics: bool = True
    demographics_gender_min_confidence: float = 0.7
    demographics_gender_model_config: str | None = None
    demographics_gender_model: str | None = "models/fairface_gender/fairface.onnx"


def load_config(config_path: Path) -> AppConfig:
    if not config_path.exists():
        return AppConfig()

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if "demographics_gender_model_config" not in raw and "demographics_gender_prototxt" in raw:
        raw["demographics_gender_model_config"] = raw["demographics_gender_prototxt"]
    raw.pop("demographics_gender_prototxt", None)
    return AppConfig(**raw)
