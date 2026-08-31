# Live Audience Attention Tracker

Reads a webcam pointed at an audience, detects every face in frame, and
classifies each as **attentive**, **partial**, or **away** based on head pose
and gaze — surfaced as a live engagement percentage on a browser dashboard,
logged to CSV, and summarized in an end-of-session report.

Attention-only: engagement is driven by head pose (yaw/pitch) and gaze
direction. No frames, face crops, or per-person identities are ever written
to disk — only aggregate per-second counts.

## Setup

```bash
py -m pip install -r requirements.txt
py download_model.py
```

`download_model.py` fetches `models/face_landmarker.task` (~3.75 MB) from
Google's model store. `server.py` will also do this automatically on first
run if the file is missing.

## Run

```bash
py server.py
```

Then open `http://127.0.0.1:8000`.

Useful flags:

```bash
py server.py --source 0              # camera index (default 0)
py server.py --source path/to.mp4    # test against a recorded video, looped
py server.py --source path/to.jpg    # test against a still group photo
py server.py --max-faces 40          # raise the tracked-face cap for a large room
py server.py --port 8080 --host 0.0.0.0
```

## Using it

1. **Position the camera** at or near the screen the audience is meant to be
   watching. Attention is measured relative to the camera's viewpoint.
2. **Auto-calibration** happens automatically: the first ~2 seconds after any
   face is seen, the tracker samples that head pose as the "looking at the
   screen" origin. Make sure whoever is in frame right after startup is
   actually facing the screen normally — this matters because MediaPipe's raw
   yaw/pitch for a typical webcam angle (e.g. looking slightly up from below
   eye level) is *not* near zero for a neutral frontal face, so without this
   step everyone reads as "away" by default.
   Click **Calibrate** any time afterward to re-sample the origin — e.g. if
   the camera moves, or the auto-calibration sampled a bad moment.
3. **Start Session** to begin logging; **Stop Session** to end it and get a
   link to the generated report.

## What it measures

- **Head pose**: decomposed from MediaPipe's facial transformation matrix
  (yaw = turning, pitch = nodding).
- **Gaze**: from eye-direction blendshapes, so a face pointed at the screen
  with eyes darting elsewhere is still caught.
- **Eyes closed**: sustained blink blendshapes above threshold count as away
  regardless of head pose.
- **Engagement %** = `(attentive + 0.5 × partial) / total faces detected`,
  smoothed with an exponential moving average (`config.ema_alpha`).

All thresholds live in [config.py](config.py) — `yaw_threshold_deg`,
`pitch_threshold_deg`, `gaze_threshold`, `eye_closed_threshold`, etc.

## Known limits

- **No per-person tracking.** Metrics are per-frame crowd aggregates: "42% of
  the room is attentive right now" is accurate; "person 3 was distracted for
  4 minutes" is not available. Adding a centroid tracker in `aggregate.py`
  would unlock per-person timelines.
- **Sentiment/expression is not scored.** The model already returns the
  blendshapes (`mouthSmile*`, `browDown*`, etc.) a sentiment score would need
  — see the hook documented at the top of `attention.py` — but no scoring
  function is wired up.
- **num_faces** (`config.max_faces`, default 25) caps how many faces MediaPipe
  will track at once; raise it for bigger rooms.

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunables: thresholds, camera source, resolution, calibration state |
| `download_model.py` | Fetches the face landmarker model asset |
| `capture.py` | Threaded, latest-frame-wins camera/video/image reader |
| `attention.py` | Head-pose + gaze math → per-face attention state |
| `engine.py` | Owns the camera thread + FaceLandmarker, produces annotated frames |
| `aggregate.py` | Crowd rollup, EMA smoothing, calibration sampling |
| `session.py` | 1 Hz CSV logging of aggregate metrics only |
| `report.py` | Builds a standalone HTML report with an inline SVG chart |
| `server.py` | FastAPI app: dashboard, `/video` MJPEG stream, `/ws` metrics |
| `static/` | Dashboard HTML/CSS/JS |

## Output

- `logs/session_<timestamp>.csv` — one row per second:
  `timestamp_iso, elapsed_s, faces_detected, attentive, partial, away, engagement_pct, mean_yaw, mean_pitch, inference_fps`
- `reports/session_<timestamp>.html` — headline stats, an engagement-over-time
  chart, and any dip windows (engagement below `dip_threshold_pct` for at
  least `dip_min_seconds`).
