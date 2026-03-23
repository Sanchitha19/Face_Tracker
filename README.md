# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-driven unique visitor counter that processes a video stream to detect,
track, and recognize faces in real-time using YOLOv8 + InsightFace + SQLite.

---

## Features

- Real-time face detection using YOLOv8
- Face recognition and embedding using InsightFace (buffalo_l)
- Automatic registration of new faces with unique UUIDs
- IoU-based face tracking across frames
- Entry and exit logging with timestamped cropped images
- Unique visitor counting stored persistently in SQLite
- Switchable between video file and live RTSP stream via config.json
- Streamlit web dashboard (app.py) for live monitoring

---

## Project Structure

```
face_tracker/
├── app.py                  # Streamlit dashboard (UI)
├── main.py                 # CLI entry point
├── config.json             # All configuration parameters
├── requirements.txt        # Python dependencies
├── db/
│   └── face_tracker.db     # SQLite database (auto-created)
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

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with CUDA 12.1+
- ~4GB disk space for models

---

## Installation

### 1. Clone / open the project
```powershell
cd AI_Face_Tracker/face_tracker
```

### 2. Create and activate virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Upgrade pip
```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch with CUDA
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is working:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 5. Install all other dependencies
```powershell
pip install onnxruntime-gpu==1.18.0
pip install insightface==0.7.3
pip install ultralytics==8.2.18 opencv-python==4.9.0.80 Pillow==10.3.0 scipy==1.13.0 tqdm==4.66.4
pip install streamlit==1.35.0 plotly==5.22.0
```

### 6. Install lap (tracker dependency)
```powershell
pip install lap==0.4.0
# If the above fails on Windows, use:
pip install lapx
```

### 7. Download YOLO face model
- Go to: https://github.com/akanametov/yolo-face/releases
- Download `yolov8n-face.pt`
- Place it in the `models/` folder:
  ```
  face_tracker/models/yolov8n-face.pt
  ```

> InsightFace's `buffalo_l` model downloads automatically on first run.

---

## Configuration (config.json)

| Key | Description |
|-----|-------------|
| `video_source` | Path to video file for development/testing |
| `use_rtsp` | Set to `true` to switch to live RTSP stream |
| `rtsp_url` | RTSP camera URL (used only when `use_rtsp` is true) |
| `skip_frames` | Run full detection every N frames (3 = good balance) |
| `detection.confidence_threshold` | YOLO confidence cutoff (0.0–1.0) |
| `recognition.similarity_threshold` | Cosine similarity cutoff for re-ID (0.0–1.0) |
| `recognition.min_face_size` | Minimum face bbox size in pixels to process |
| `tracking.max_disappeared` | Frames before a lost track is removed |
| `display.show_video` | Show live annotated video window |

---

## Running

### Option A — Command line (headless)
```powershell
python main.py
```
Press `Q` to quit the video window.

### Option B — Streamlit dashboard
```powershell
streamlit run app.py
```
Opens at `http://localhost:8501` in your browser.

---

## Output

- **Live video window** — bounding boxes, face UUIDs, visitor count overlay
- **logs/events.log** — every entry/exit event timestamped
- **logs/entries/YYYY-MM-DD/** — cropped face images on entry
- **logs/exits/YYYY-MM-DD/** — cropped face images on exit
- **db/face_tracker.db** — persistent SQLite database with:
  - All registered faces + embeddings
  - Full event history
  - Daily visitor summaries

---

## Switching to RTSP (Live Camera)

Edit `config.json`:
```json
{
  "use_rtsp": true,
  "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream"
}
```
Then run `python main.py` as normal.

---

## Tech Stack

| Module | Technology |
|--------|-----------|
| Face Detection | YOLOv8 (ultralytics) |
| Face Recognition | InsightFace buffalo_l (ArcFace) |
| Tracking | IoU centroid tracker (pure Python) |
| Database | SQLite3 (built-in) |
| Dashboard | Streamlit + Plotly |
| GPU Inference | CUDA via onnxruntime-gpu + PyTorch |

---

## Troubleshooting

**CUDA not detected**
```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Must print True — if False, reinstall PyTorch with correct CUDA version
```

**InsightFace download fails**
- Ensure you have internet access on first run
- Model downloads to `C:\Users\<you>\.insightface\models\buffalo_l\`

**lap install fails**
```powershell
pip install lapx   # Drop-in replacement for Windows
```

**Video not opening**
- Check `video_source` path in config.json is correct
- Use forward slashes or double backslashes: `"videos/sample.mp4"`
