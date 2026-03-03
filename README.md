# Phantom Tracker 👻

**Multi-Object Tracking with Predictive Intelligence**

Open-vocabulary detection • Occlusion reasoning • Re-identification • Predictive trajectories

---

## Quick Start

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd phantom_tracker

# 2. Create environment (recommended: conda)
conda create -n phantom python=3.10 -y
conda activate phantom

# 3. Install PyTorch with CUDA (check your CUDA version first)
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
# pip install torch torchvision

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Run with webcam
python main.py

# 6. Run with video file
python main.py --input path/to/video.mp4

# 7. Save output
python main.py --input video.mp4 --output result.mp4
```

## Keyboard Controls

| Key | Toggle |
|-----|--------|
| `q` | Quit |
| `t` | Trajectory trails |
| `g` | Ghost outlines (occluded objects) |
| `p` | Predicted future path |
| `i` | ID labels |
| `f` | FPS counter |
| `h` | Heatmap overlay |
| `d` | Analytics dashboard |

## Project Structure

```
phantom_tracker/
├── main.py                  # Pipeline orchestrator
├── core/
│   └── interfaces.py        # ⚠️ SHARED DATA CONTRACTS — do not modify without team agreement
├── detection/
│   └── detector.py          # Owner: Divyansh — YOLO + Grounding DINO
├── tracking/
│   └── tracker.py           # Owner: Dhruvish — BoT-SORT + Kalman + state machine
├── reid/
│   └── reidentifier.py      # Owner: Dharmik — OSNet + feature bank + Re-ID matching
├── visualization/
│   └── visualizer.py        # Owner: Agastya — overlays + dashboard
├── utils/
│   ├── fps_counter.py
│   ├── logger.py
│   └── colors.py
├── configs/                 # YAML configs for different scenarios
├── demos/                   # Demo scripts and recorded videos
├── data/                    # Downloaded datasets (gitignored)
└── requirements.txt
```

## Team

| Member | Module | Branch |
|--------|--------|--------|
| Divyansh | Detection & Open-Vocab | `feat/detection` |
| Dhruvish | Tracking & Motion Prediction | `feat/tracker` |
| Dharmik | Re-ID & GenAI | `feat/reid` |
| Agastya | Visualization & Dashboard | `feat/viz` |

## Git Workflow

```bash
# Create your feature branch
git checkout -b feat/detection   # (use your module name)

# Work, commit, push
git add .
git commit -m "feat(detection): add YOLOv11 inference pipeline"
git push origin feat/detection

# Merge to main ONLY at checkpoints after team testing
```
