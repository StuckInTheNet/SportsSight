# SportsSight

Real-time sports video analytics — fatigue detection, play prediction, and biomechanical analysis for NBA basketball.

## What It Does

SportsSight processes NBA game footage (broadcast or proprietary cameras) and produces per-player fatigue scores in real-time. It tracks 13 biomechanical indicators — speed decline, jump height loss, defensive stance breakdown, recovery time — and alerts coaching staff when players cross fatigue thresholds.

```
Video Feed → Player Detection → Tracking → Re-ID → Pose Estimation → Feature Extraction → Fatigue Model → Dashboard
```

### Pipeline

| Stage | Technology | What It Does |
|-------|-----------|-------------|
| Detection | YOLOv8x | Detects players in each frame |
| Tracking | ByteTrack | Maintains identity within continuous footage |
| Team Classification | K-means on jersey HSV | Splits players into two teams automatically |
| Re-ID | Jersey OCR + color histograms | Matches players across camera cuts |
| Pose | YOLOv8x-Pose | Extracts 17 COCO keypoints per player |
| Court Mapping | OpenCV homography | Converts pixel positions to real-world feet |
| Features | Rolling biomechanics | 16 metrics: speed, stride, stance, jumps, posture |
| Fatigue | Rule-based + Transformer | Scores 0-100 with contributing factors |
| Delivery | FastAPI + WebSocket + React | Real-time dashboard with alerts |

### Results (720p NBA Highlights)

- **54 core players** tracked across the game
- **6.7 players per frame** detected on average
- **13 contributing factors** per fatigue score
- **Fatigue range**: 0 (fresh) → 88 (critical)
- **Processing speed**: ~1:1.5 ratio (12 min video in 18 min) on Apple M4 Max

## Quick Start

```bash
# Clone and setup
git clone https://github.com/itsme-fish/SportsSight.git
cd SportsSight
make dev

# Download models
sportssight download-models

# Analyze a game
sportssight analyze path/to/game.mp4 --game-id my-game -o results.json

# Start the API + dashboard
docker compose up -d          # Postgres + Redis
sportssight serve              # API on :8000
cd dashboard && npm install && npm run dev  # Dashboard on :3000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VIDEO INGESTION                         │
│  Broadcast (RTMP/HLS) │ Proprietary Cameras (RTSP) │ Files  │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   VISION PIPELINE                            │
│  YOLOv8 Detection → ByteTrack → Team Classifier → Re-ID     │
│  YOLOv8-Pose Skeleton → Court Homography                     │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION + FATIGUE MODEL              │
│  16 biomechanical features → Temporal Transformer / Rules    │
│  Per-player fatigue score (0-100) + contributing factors     │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  API + DASHBOARD                             │
│  FastAPI REST + WebSocket → React + Recharts                 │
│  Multi-tenant, API key auth, configurable alerts             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/
├── ingestion/        # Video sources (file, RTMP, RTSP, HLS)
├── vision/
│   ├── detector.py   # YOLOv8 player detection
│   ├── tracker.py    # ByteTrack multi-object tracking
│   ├── team_classifier.py  # Auto-calibrating team color classification
│   ├── reid.py       # Jersey OCR + appearance matching
│   ├── jersey.py     # EasyOCR jersey number detection
│   ├── pose.py       # YOLOv8-Pose skeleton extraction
│   ├── court.py      # Court homography (pixel → feet)
│   ├── track_merger.py  # Post-game identity consolidation
│   └── pipeline.py   # Orchestrator
├── features/         # 16 biomechanical feature extraction
├── models/           # Fatigue transformer + rule-based scoring
├── realtime/         # Stream processor + alert system
├── api/              # FastAPI REST + WebSocket + auth
└── cli.py            # CLI interface
dashboard/            # React 19 + Tailwind + Recharts
```

## Tech Stack

**Backend**: Python 3.13, PyTorch (MPS/CUDA), YOLOv8, OpenCV, FastAPI, Redis Streams, PostgreSQL

**Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Recharts, WebSocket

**Infrastructure**: Docker Compose, Alembic migrations, TimescaleDB

## Development

```bash
make test          # Run 93 tests
make lint          # Ruff linter
make test-cov      # Coverage report
```

## License

MIT
