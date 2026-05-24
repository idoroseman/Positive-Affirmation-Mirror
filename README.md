# Vibe-Mirror

Raspberry Pi greeter for a live camera or test video that:

- detects faces from a live webcam feed,
- recognizes known people from a local photo dataset,
- waits until a face is present for a dwell threshold before greeting,
- avoids greeting the same person again until they have been absent long enough,
- uses couple and group greetings when multiple people are visible together.

## Behavior

- A face must remain visible for `dwell_seconds` before any greeting is triggered. The default is `2.0` seconds.
- Once greeted, a tracked face will not be greeted again while it remains in view.
- If a face disappears for more than `absence_seconds`, its track is removed. The default is `30.0` seconds.
- Two visible faces trigger the couple greeting.
- Three or more visible faces trigger the group greeting.
- A single visible face triggers:
  - a personal greeting chosen from any audio file in `data/recordings/<name>/` when the face matches a person in the dataset,
  - otherwise an unknown-person greeting chosen from `data/recordings/unknown/`, or from `data/recordings/unknown_m/` / `data/recordings/unknown_f/` when a stable gender label is available.

## Project layout

```text
data/
  known_faces/
    alice/
      1.jpg
      2.jpg
    bob/
      1.jpg
  recordings/
    alice/
      0.mp3
      1.mp3
    bob/
      0.mp3
    couple/
      0.mp3
    group/
      0.mp3
    unknown/
      0.mp3
    unknown_f/
      0.mp3
    unknown_m/
      0.mp3
  unknown_snapshots/
    track-7-20260512-101530.jpg
  models/
    fairface_gender/
      fairface.onnx
src/mirror/
main.py
config.json
```

## Setup on Raspberry Pi

1. Install system packages needed by OpenCV and audio playback.
2. Create a Python virtual environment.
3. Install Python dependencies.
4. Copy `config.example.json` to `config.json` and adjust paths if needed.
5. Add face images and greeting audio files.

Example:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev cmake build-essential libatlas-base-dev libopenblas-dev liblapack-dev libjpeg-dev aplay ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp config.example.json config.json
```

If you want gender-based unknown greetings:

1. Download a FairFace gender ONNX file to `data/models/fairface_gender/fairface.onnx`.
2. Set `allow_optional_demographics` to `true` in `config.json`.

The app only runs this model for unknown tracks that are still missing demographic metadata, which is much lighter than calling a full face-analysis stack on every detection frame.
If the defaults are too aggressive or too noisy, tune `demographics_gender_min_confidence` in `config.json`.
If you want to test with a video file by default, set `video_source` in `config.json`.
If you want to keep snapshots of new unknown faces after they pass dwell, leave `unknown_snapshot_dir` set or point it at another folder.

## Running

```bash
PYTHONPATH=src python main.py
```

Run against a test video file instead of the live camera:

```bash
PYTHONPATH=src python main.py --video path/to/test-video.mp4
```

You can also set `video_source` in `config.json` to make file playback the default input.

Headless mode:

```bash
PYTHONPATH=src python main.py --headless
```

Headless mode with a test video:

```bash
PYTHONPATH=src python main.py --headless --video path/to/test-video.mp4
```

Press `q` in the preview window to quit.

## Notes

- The default detector uses the HOG model from `face_recognition`, which is reasonable for Raspberry Pi CPU usage.
- Recognition accuracy improves when each known person has multiple clear front-facing photos.
- `video_source` takes precedence over `camera_index` when it is set.
- Unknown faces are tracked using face embeddings and centroid proximity, so short-lived frame drops do not immediately reset the greeting state.
- When an unknown face passes dwell for the first time, the app saves one cropped face snapshot to `unknown_snapshot_dir`.
- Optional demographics are loaded from the shipped model paths if present; if the files are missing or no stable label is available, unknown greetings use `data/recordings/unknown/`.
- Unknown greeting lookup checks the gender-matched recordings directory first, then falls back to `data/recordings/unknown/`.
- Low-confidence gender predictions are discarded before greeting selection, so the app falls back to a less specific file when the model is unsure.
- You can extend `DemographicsService` to add smile detection, uniforms, or other business-specific routing logic.