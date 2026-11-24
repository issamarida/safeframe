"""Runtime configuration, read from ``SAFEFRAME_*`` environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from safeframe.pipeline.age import DEFAULT_AGE_MODEL
from safeframe.pipeline.detection import DEFAULT_YOLO_WEIGHTS
from safeframe.pipeline.wearers import DEFAULT_POSE_WEIGHTS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAFEFRAME_", env_file=".env", extra="ignore")

    data_dir: Path = Field(Path("data"), description="Where job files live while processing")
    max_upload_mb: int = Field(4096, ge=1)
    # Every job (including kept uploads) is deleted after this long without activity.
    job_ttl_hours: float = Field(24.0, gt=0)
    sweep_interval_seconds: int = Field(60, ge=5)

    yolo_weights: str = DEFAULT_YOLO_WEIGHTS
    # Wrist keypoints for band attribution; empty disables the pose model.
    pose_weights: str = DEFAULT_POSE_WEIGHTS
    age_model: str = DEFAULT_AGE_MODEL
    age_model_revision: str | None = None
    device: str | None = None
    warm_up_models: bool = True

    # Guarded-zone registry JSON; the packaged sample is used when unset.
    zone_registry: Path | None = None
    # Escrow public key (from `safeframe trustees init`). When set, the original
    # of every photo or video in which a band was honoured is sealed to it.
    escrow_public_key: Path | None = None
    seal_retention_days: float = Field(30.0, gt=0)

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Built React app; served at / when present.
    frontend_dist: Path = Path(__file__).resolve().parents[3] / "frontend" / "dist"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
