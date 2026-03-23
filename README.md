# FaceTrack Pro — Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-driven unique visitor counter that processes a video stream to detect,
track, and recognize faces in real-time using YOLOv8 + InsightFace + SQLite.

---

## Demo Video


> **Important:** A video walkthrough is required for submission review.

🎬 **[Watch the Demo on Loom ](https://www.loom.com/share/e56df1a9ff0b49f89bcd602a9cb4782d?t=57)**
   > **continuation**🎬 **[Watch the Demo on Loom](https://www.loom.com/share/a21211806af445d69c1ecbb67ca84187)**


---

## Table of Contents

- [Features](#features)
- [Assumptions](#assumptions)
- [AI Planning Document](#ai-planning-document)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Sample config.json](#sample-configjson)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Output](#output)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)

---

## Features

- Real-time face detection using YOLOv8
- Face recognition and embedding using InsightFace (buffalo_l / ArcFace)
- Automatic registration of new faces with unique UUIDs
- IoU-based centroid tracker with active / pending-exit / exited states
- Persistent re-identification — same person always gets the same UUID
- Entry and exit logging with timestamped cropped face images
- Unique visitor counting stored persistently in SQLite
- Switchable between video file (dev) and live RTSP stream (production)
- Streamlit web dashboard for live monitoring
- All parameters configurable via config.json — no code changes needed

---

## Assumptions

The following assumptions were made during the design and development of this system:

1. **Single camera feed** — the system processes one video stream at a time,
   either a local video file or a single RTSP stream.

2. **Frontal or near-frontal faces** — InsightFace (ArcFace) performs best
   on faces that are reasonably visible to the camera. Heavily occluded,
   masked, or profile-only faces may not generate reliable embeddings.

3. **Adequate lighting** — the system assumes standard indoor surveillance
   lighting. Extremely dark or overexposed frames may reduce detection accuracy.

4. **NVIDIA GPU available** — the pipeline is optimized for CUDA inference
   via onnxruntime-gpu and PyTorch. CPU fallback is supported but significantly
   slower and not suitable for real-time processing.

5. **Minimum face size** — faces smaller than 40×40 pixels in the frame
   are ignored by default. This filters out distant or blurry faces that
   would produce unreliable embeddings. Configurable via config.json.

6. **One identity per person** — the system assigns one UUID per unique face.
   Identical twins or near-identical faces may occasionally be treated as
   the same person depending on the similarity threshold setting.

7. **Exit is confirmed after 180 frames of absence** — approximately 6 seconds
   at 30fps. This grace period prevents false exits during brief occlusions
   or temporary frame drops.

8. **InsightFace buffalo_l model** — this model is downloaded automatically
   from the InsightFace model zoo on first run. Internet access is required
   for the initial download only.

9. **Video file format** — tested with .mp4, .avi, and .mov files using
   OpenCV VideoCapture. Other formats may work but are not guaranteed.

10. **Single-session SQLite** — the database is accessed from the main thread
    only. No concurrent writes from multiple processes are assumed.

---

## AI Planning Document

### Problem Statement

Traditional visitor counting systems rely on manual tallying or simple
motion-based counters that cannot distinguish unique individuals.
The challenge is to build a system that:
- Detects every face in a video stream
- Assigns a permanent unique identity to each person
- Recognizes returning visitors without re-registering them
- Logs every entry and exit accurately with a visual record
- Maintains an accurate unique visitor count across the entire session

### Approach

The solution is a four-stage AI pipeline:

**Stage 1 — Detect**
YOLOv8 (yolov8n-face.pt) runs on every Nth frame (configurable via skip_frames)
to detect face bounding boxes. Between detection frames, the IoU tracker
maintains existing bounding boxes without the full inference cost.
This keeps the pipeline real-time on GPU.

**Stage 2 — Embed**
For each detected face, InsightFace crops the ROI from the frame,
adds 30px padding for context, and generates a 512-dimensional
L2-normalized embedding using the ArcFace model (buffalo_l).
This embedding is a unique numerical fingerprint of that face.

**Stage 3 — Recognize**
The new embedding is compared against all stored embeddings using
cosine similarity: similarity = 1 - scipy.spatial.distance.cosine(a, b).
If the best match exceeds the threshold (default 0.30) — it is a known visitor.
Same UUID is reused. No new registration.
If no match is found — a new UUID4 is generated, the embedding is stored
in SQLite, and the unique visitor count increments.

**Stage 4 — Track**
An IoU-based centroid tracker maintains track IDs across frames.
Each track has three states:
  - active       : face is currently visible
  - pending_exit : face has been absent for > max_disappeared frames
  - exited       : exit confirmed, event logged

When a face in pending_exit state reappears and its embedding matches
the stored UUID — it is restored to active and a fresh ENTRY is logged.
This prevents false exits and duplicate registrations for the same person.

### Key Design Decisions

| Decision | Reasoning |
|---|---|
| InsightFace over face_recognition library | ArcFace embeddings are production-grade. The face_recognition library uses dlib HOG which is significantly less accurate for re-identification |
| Cosine similarity at 0.30 threshold | Tuned empirically. Too high = same person gets multiple IDs. Too low = different people collapse to same ID |
| IoU tracker over DeepSort/ByteTrack | Keeps dependencies minimal. DeepSort requires a separate reid model. IoU tracking is sufficient when combined with embedding-based re-ID |
| SQLite over PostgreSQL/MongoDB | No server setup. Portable. Sufficient for single-camera single-session use |
| Pending exit state over immediate exit | Real-world faces disappear briefly due to occlusion, head turns, and frame drops. Immediate exit logging produces noisy duplicate entry/exit pairs |
| Frame skipping | Running full YOLO + InsightFace inference on every frame at 30fps is computationally expensive. Skipping N-1 frames between detections gives real-time performance without sacrificing accuracy |

### Unique Visitor Counting Logic

```
New face detected
       │
       ▼
Generate 512-d embedding (InsightFace)
       │
       ▼
Compare with all stored embeddings (cosine similarity)
       │
   ┌───┴───┐
Match ≥ 0.30?  No match?
   │               │
Reuse UUID     New UUID4
Update last    Register in DB
seen           Increment count
   │               │
   └───────┬───────┘
           │
     Log ENTRY event
     Save face image
           │
     Track across frames
           │
     Face disappears?
           │
     pending_exit state
           │
  Reappears?    Gone > 180 frames?
     │                  │
  Restore active    Log EXIT event
  Log new ENTRY     Release track
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│         Video File (.mp4/.avi)  OR  RTSP Live Stream            │
│                  Switched via config.json                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FRAME READER                                │
│   detector.py — reads frames, skips N frames per config        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FACE DETECTION                                │
│        detector.py — YOLOv8 (yolov8n-face.pt)                  │
│        Returns: bounding boxes + confidence scores             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FACE EMBEDDING                                 │
│     embedder.py — InsightFace buffalo_l (ArcFace)              │
│     Returns: 512-d L2-normalized embedding per face            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐   ┌─────────────────────────────────────┐
│    FACE DB          │   │        IDENTITY RECOGNITION          │
│  (SQLite)           │◄──│  recognizer.py — cosine similarity   │
│  Embeddings+UUIDs   │──►│  Match found → reuse UUID           │
└─────────────────────┘   │  No match   → register new UUID     │
                          └───────────────┬─────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      IoU TRACKER                                │
│   tracker.py — maintains track states across frames            │
│   States: active → pending_exit → exited                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PIPELINE ORCHESTRATOR                           │
│   pipeline.py — entry/exit decisions, frame loop               │
└────────────┬──────────────┬──────────────────┬─────────────────┘
             │              │                  │
             ▼              ▼                  ▼
┌────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
│  EVENT LOGGER  │ │ SQLite DATABASE │ │   VISITOR COUNTER    │
│  logger.py     │ │ database.py     │ │ COUNT DISTINCT UUIDs │
│  events.log    │ │ faces / events  │ │ from faces table     │
│  + face images │ │ / summary       │ └──────────────────────┘
└────────────────┘ └─────────────────┘
             │              │
             ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                            │
│   app.py — KPI cards, charts, event feed, face gallery,        │
│   visitor table, system log, source config, settings           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
face_tracker/
├── app.py                  # Streamlit dashboard (UI)
├── main.py                 # CLI entry point
├── config.json             # All configuration parameters
├── requirements.txt        # Python dependencies
├── db/
│   └── face_tracker.db     # SQLite database (auto-created on run)
├── logs/
│   ├── events.log          # System event log (auto-created)
│   ├── entries/
│   │   └── YYYY-MM-DD/     # Cropped face images on entry
│   └── exits/
│       └── YYYY-MM-DD/     # Cropped face images on exit
├── models/
│   └── yolov8n-face.pt     # YOLO face model (download manually)
└── src/
    ├── __init__.py
    ├── config_loader.py    # Loads and validates config.json
    ├── database.py         # SQLite schema + all DB operations
    ├── detector.py         # YOLOv8 face detection
    ├── embedder.py         # InsightFace 512-d embeddings
    ├── recognizer.py       # Cosine similarity matcher + registration
    ├── tracker.py          # IoU-based centroid tracker
    ├── logger.py           # File + DB event logging
    └── pipeline.py         # Main orchestration pipeline
```

---

## Sample config.json

```json
{
  "video_source": "sample.mp4",
  "use_rtsp": false,
  "rtsp_url": "rtsp://username:password@192.168.1.100:554/stream",

  "skip_frames": 3,

  "detection": {
    "yolo_model": "models/yolov8n-face.pt",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  },

  "recognition": {
    "similarity_threshold": 0.30,
    "embedding_model": "buffalo_l",
    "min_face_size": 40
  },

  "tracking": {
    "max_disappeared": 60,
    "exit_confirm_frames": 180,
    "iou_threshold": 0.3
  },

  "logging": {
    "log_file": "logs/events.log",
    "image_quality": 95,
    "save_exit_images": true
  },

  "database": {
    "path": "db/face_tracker.db"
  },

  "display": {
    "show_video": true,
    "draw_landmarks": false,
    "window_width": 1280,
    "window_height": 720
  }
}
```

### Configuration Parameters

| Key | Default | Description |
|-----|---------|-------------|
| `video_source` | `"sample.mp4"` | Path to video file for dev/testing |
| `use_rtsp` | `false` | Set to `true` to use live RTSP stream |
| `rtsp_url` | — | RTSP camera URL (used only when use_rtsp is true) |
| `skip_frames` | `3` | Run full detection every N frames |
| `detection.confidence_threshold` | `0.5` | YOLO minimum confidence (0.0–1.0) |
| `detection.iou_threshold` | `0.45` | YOLO NMS IoU threshold |
| `recognition.similarity_threshold` | `0.30` | Cosine similarity cutoff for re-ID |
| `recognition.min_face_size` | `40` | Minimum face bbox size in pixels |
| `tracking.max_disappeared` | `60` | Frames before marking pending exit |
| `tracking.exit_confirm_frames` | `180` | Frames before confirming exit |
| `display.show_video` | `true` | Show live annotated video window |

---

## Setup Instructions

### Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with CUDA 12.1+
- ~4GB disk space for models

### Step 1 — Navigate to project folder

```powershell
cd AI_Face_Tracker/face_tracker
```

### Step 2 — Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get a permissions error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Step 3 — Upgrade pip

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Step 4 — Install PyTorch with CUDA

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA is working:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### Step 5 — Install all dependencies

```powershell
pip install numpy --only-binary=:all:
pip install opencv-python --only-binary=:all:
pip install insightface onnxruntime-gpu ultralytics Pillow scipy tqdm
pip install streamlit plotly lapx
```

### Step 6 — Download YOLO face model

- Go to: https://github.com/akanametov/yolo-face/releases
- Download `yolov8n-face.pt`
- Place it inside the `models/` folder:

```
face_tracker/models/yolov8n-face.pt
```

> InsightFace's `buffalo_l` model downloads automatically on first run.
> Requires internet access for initial download only.

---

## Running the Project

### Option A — Command line

```powershell
python main.py
```

Press `Q` to quit the video window.

### Option B — Streamlit dashboard

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

### Switching to RTSP (live camera)

Edit `config.json`:
```json
{
  "use_rtsp": true,
  "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream"
}
```
Then run `python main.py` as normal. No code changes needed.

---

## Output

| Output | Location | Description |
|--------|----------|-------------|
| Live video window | Screen | Bounding boxes, face UUIDs, visitor count overlay |
| Event log | `logs/events.log` | Every entry/exit timestamped |
| Entry images | `logs/entries/YYYY-MM-DD/` | Cropped face images on entry |
| Exit images | `logs/exits/YYYY-MM-DD/` | Last good frame crop on exit |
| Database | `db/face_tracker.db` | Faces, events, visitor summary |

---

## Tech Stack

| Module | Technology |
|--------|------------|
| Face Detection | YOLOv8 (ultralytics) |
| Face Recognition | InsightFace buffalo_l (ArcFace) |
| Tracking | IoU centroid tracker (pure Python) |
| Database | SQLite3 (built-in, no server needed) |
| Dashboard | Streamlit + Plotly |
| GPU Inference | CUDA via onnxruntime-gpu + PyTorch |
| Configuration | JSON (config.json) |
| Language | Python 3.10+ |

---

## Troubleshooting

**CUDA not detected**
```powershell
python -c "import torch; print(torch.cuda.is_available())"
# If False — reinstall PyTorch with correct CUDA version for your GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**InsightFace install fails on Windows**
```powershell
pip install insightface --prefer-binary
```

**InsightFace model download fails on first run**
- Ensure internet access
- Model downloads to: `C:\Users\<you>\.insightface\models\buffalo_l\`

**lap install fails**
```powershell
pip install lapx
```

**Video not opening**
- Check `video_source` path in config.json
- Use forward slashes: `"videos/sample.mp4"`

**numpy build error**
```powershell
pip install numpy --only-binary=:all:
```

---



> This project is a part of a hackathon run by https://katomaran.com
