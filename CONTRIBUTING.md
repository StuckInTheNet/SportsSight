# Contributing to SportsSight

## Development Setup

```bash
git clone <repo-url>
cd SportsSight
make dev                # Creates venv, installs deps, copies .env
docker compose up -d    # Starts Postgres + Redis
make models             # Downloads ML models
```

## Running Tests

```bash
make test               # All tests
make test-unit          # Unit tests only (no GPU / no DB required)
make test-integration   # Integration tests
make test-cov           # With coverage report
```

## Code Quality

```bash
make lint               # Ruff linter
make format             # Ruff formatter
make typecheck          # Mypy strict mode
```

## Architecture

The project follows a pipeline architecture:

```
Video Source → Detection → Tracking → Re-ID → Pose → Features → Fatigue Model → API/Dashboard
```

Each stage is independently testable. The `RealtimeEngine` orchestrates the full pipeline.

### Key modules

| Module | Purpose |
|--------|---------|
| `src/ingestion/` | Video source adapters (file, RTMP, RTSP) |
| `src/vision/` | CV pipeline (YOLO, ByteTrack, OSNet, RTMPose) |
| `src/features/` | Biomechanical feature extraction |
| `src/models/` | Fatigue scoring (rule-based + transformer) |
| `src/realtime/` | Stream processing engine + alerts |
| `src/api/` | FastAPI REST + WebSocket server |
| `dashboard/` | React frontend |

### Adding a new feature metric

1. Add the field to `PlayerFeatures` in `src/features/extractor.py`
2. Update `to_array()` and increment `FEATURE_DIM`
3. Add computation logic in `FeatureExtractor`
4. Update the weight in `FatigueModel._score_rule_based()`
5. Add tests in `tests/unit/test_features.py`

### Adding a new sport

1. Create a new court/field module in `src/vision/` (e.g., `field.py` for soccer)
2. Adjust detection filters in `PlayerDetector` (aspect ratios, min area)
3. Add sport-specific feature computations
4. Update `configs/default.yaml` with sport-specific parameters

## Commit Messages

Use imperative mood: "Add feature" not "Added feature". Keep the first line under 50 characters.

## Pull Requests

Include: what changed, why, and how to test it.
