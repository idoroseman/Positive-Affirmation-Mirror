# Project Guidelines

## Architecture

- Keep the app split by responsibility: [main.py](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/main.py) is only the entrypoint, [src/mirror/app.py](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/src/mirror/app.py) owns the main loop and greeting decisions, [src/mirror/tracker.py](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/src/mirror/tracker.py) owns dwell and absence tracking, [src/mirror/recognizer.py](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/src/mirror/recognizer.py) owns known-face matching, and [src/mirror/audio.py](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/src/mirror/audio.py) owns playback.
- Preserve the current runtime model for Raspberry Pi: CPU-friendly face detection, frame-throttled recognition, and asynchronous audio playback.
- Keep config-driven behavior in [config.example.json](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/config.example.json) and [src/mirror/config.py](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/src/mirror/config.py) rather than hardcoding thresholds or paths.

## Build And Test

- Use the project virtual environment in `.venv` when running Python commands.
- Install dependencies with `pip install -r requirements.txt`.
- Run the app with `PYTHONPATH=src python main.py`, `PYTHONPATH=src python main.py --headless`, or `PYTHONPATH=src python main.py --video path/to/file.mp4`.
- There is no test suite yet. For validation, prefer narrow checks such as Python syntax/errors on touched files and targeted runs in headless mode.

## Conventions

- Treat `dwell_seconds`, `absence_seconds`, `recognition_tolerance`, `track_match_distance`, `centroid_match_px`, `demographics_gender_min_confidence`, and `demographics_age_min_confidence` as the main behavior-tuning surfaces. Do not change defaults casually.
- Paths in config are resolved from the current working directory. If code depends on files under `data/`, keep that assumption consistent or update both code and docs together.
- Unknown-face demographic routing is optional. Keep the default path working without DeepFace or TensorFlow Lite installed, using the OpenCV Caffe model files under `data/models/opencv_gender/` when demographics are enabled.
- `video_source` is optional and should preserve the live-camera path when unset.
- Prefer small, local changes. If you add behavior, update [README.md](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/README.md) when setup, runtime flow, dataset layout, greeting asset expectations, or config knobs change.
- Link to existing docs instead of duplicating them. The operational setup and dataset layout live in [README.md](/home/ido/Dropbox/Hobby Projects/vibe/vibe-mirror/README.md).