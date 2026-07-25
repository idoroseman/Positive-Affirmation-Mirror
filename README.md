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
- The app periodically rescans `known_faces_dir` and automatically reloads known-face encodings when files are added, removed, or changed.
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
sudo apt install -y python3-venv python3-dev cmake build-essential libatlas-base-dev libopenblas-dev liblapack-dev libjpeg-dev libcap-dev python3-picamera2 python3-libcamera libcamera-dev libudev-dev aplay ffmpeg
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp config.example.json config.json
```

If you use `"camera_index": "picam"`, install `picamera2` and `libcamera` from `apt`, not `pip`. Those Python modules are shipped as Raspberry Pi OS packages and are often invisible from a plain venv unless you create it with `--system-site-packages`.

## Setup on macOS

```bash
brew install python3 libopenblas lapack
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp config.example.json config.json
```

Note: On macOS, use `--video path/to/test-video.mp4` or a standard OpenCV webcam index (default `0`). The Raspberry Pi `picamera2` library is not available on macOS.

If you want gender-based unknown greetings:

1. Download a FairFace gender ONNX file to `models/fairface_gender/fairface.onnx`.
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
PYTHONPATH=src .venv/bin/python main.py --video data/videos/VIDEO-2026-04-19-14-40-41.mp4 
```

Press `q` in the preview window to quit.

## Run as systemd service

The repository includes a starter unit file at `vibe-mirror.service`.

1. Edit `vibe-mirror.service` and set `User`, `Group`, `WorkingDirectory`, and `ExecStart` for your Pi.
2. Install the service file.
3. Reload systemd and enable the service.
4. Start the service and check logs.

```bash
sudo cp vibe-mirror.service /etc/systemd/system/vibe-mirror.service
sudo systemctl daemon-reload
sudo systemctl enable vibe-mirror.service
sudo systemctl start vibe-mirror.service
sudo systemctl status vibe-mirror.service
journalctl -u vibe-mirror.service -f
```

Stop or restart later:

```bash
sudo systemctl stop vibe-mirror.service
sudo systemctl restart vibe-mirror.service
```

If the service runs but no audio is heard:

1. Ensure a player exists for the service user (`mpg123` is preferred):

```bash
sudo apt install -y mpg123
sudo -u admin which mpg123
```

2. Confirm the service user can play a file directly:

```bash
sudo -u admin mpg123 -q /home/admin/vibe-mirror/data/recordings/unknown/0.mp3
```

If `mpg123` or `ffplay` still fail under systemd, force the ALSA-native pipeline in `config.json`:

```json
"playback_command": "ffmpeg-aplay"
```

This uses `ffmpeg` to decode and `aplay` to send PCM directly to ALSA.

If ALSA default still fails (for example `audio open error: Unknown error 524`), set a specific output device in the service:

```ini
Environment=MIRROR_APLAY_DEVICE=plughw:0,0
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart vibe-mirror.service
```

To discover device ids:

```bash
aplay -l
```

3. Make sure the service has audio-group access (already set in `vibe-mirror.service`):

```bash
sudo systemctl cat vibe-mirror.service | grep -E 'User=|Group=|SupplementaryGroups='
```

4. Restart and inspect playback errors from the app logs:

```bash
sudo systemctl daemon-reload
sudo systemctl restart vibe-mirror.service
journalctl -u vibe-mirror.service -f -o short-iso
```

## Notes

- The default detector uses the HOG model from `face_recognition`, which is reasonable for Raspberry Pi CPU usage.
- Recognition accuracy improves when each known person has multiple clear front-facing photos.
- `video_source` takes precedence over `camera_index` when it is set.
- Unknown faces are tracked using face embeddings and centroid proximity, so short-lived frame drops do not immediately reset the greeting state.
- `known_faces_reload_interval_seconds` controls how often the app checks for known-face dataset changes while running.
- When an unknown face passes dwell for the first time, the app saves one cropped face snapshot to `unknown_snapshot_dir`.
- Optional demographics are loaded from the shipped model paths if present; if the files are missing or no stable label is available, unknown greetings use `data/recordings/unknown/`.
- Unknown greeting lookup checks the gender-matched recordings directory first, then falls back to `data/recordings/unknown/`.
- Low-confidence gender predictions are discarded before greeting selection, so the app falls back to a less specific file when the model is unsure.
- You can extend `DemographicsService` to add smile detection, uniforms, or other business-specific routing logic.

## TODO

1. Add do-not-disturb hours where the mirror stays silent.
2. Detect repeating guests and suggest adding them as known people.
3. Ensure unknown greeting routing always falls back from `unknown_f` / `unknown_m` to `unknown` when needed.
4. if face is passing dwell and there are more that are almost there, group them together
5. ~~show "now playing" on debug screen~~
6. backend server
