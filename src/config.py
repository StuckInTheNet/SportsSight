"""Centralized configuration — loads YAML defaults, overrides from env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"


def _resolve_device(requested: str = "auto") -> str:
    """Pick the best available compute device."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    """Runtime configuration assembled from YAML + env overrides."""

    # Paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    raw_dir: Path = DATA_DIR / "raw"
    processed_dir: Path = DATA_DIR / "processed"
    model_dir: Path = DATA_DIR / "models"

    # Infrastructure
    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me-in-production"

    # Device
    device: str = "auto"

    # Full YAML config tree for subsystem access
    _yaml: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.device = _resolve_device(self.device)

    # Subsystem access
    @property
    def detection(self) -> dict[str, Any]:
        return self._yaml.get("detection", {})

    @property
    def tracking(self) -> dict[str, Any]:
        return self._yaml.get("tracking", {})

    @property
    def reid(self) -> dict[str, Any]:
        return self._yaml.get("reid", {})

    @property
    def pose(self) -> dict[str, Any]:
        return self._yaml.get("pose", {})

    @property
    def court(self) -> dict[str, Any]:
        return self._yaml.get("court", {})

    @property
    def features(self) -> dict[str, Any]:
        return self._yaml.get("features", {})

    @property
    def fatigue(self) -> dict[str, Any]:
        return self._yaml.get("fatigue", {})

    @property
    def realtime(self) -> dict[str, Any]:
        return self._yaml.get("realtime", {})

    @property
    def pipeline(self) -> dict[str, Any]:
        return self._yaml.get("pipeline", {})


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from YAML file with env var overrides."""
    path = config_path or CONFIG_DIR / "default.yaml"

    yaml_data: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

    return Config(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://sportssight:sportssight@localhost:5432/sportssight",
        ),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        api_secret_key=os.getenv("API_SECRET_KEY", "change-me-in-production"),
        device=os.getenv("DEVICE", yaml_data.get("pipeline", {}).get("device", "auto")),
        _yaml=yaml_data,
    )
