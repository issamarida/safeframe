"""Shared fixtures. All test video is synthesised by FFmpeg at runtime —
no footage is ever stored in the repository."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from safeframe.pipeline.masks import Mask, build_coverage
from safeframe.pipeline.types import Box, FaceDetection, PersonObservation

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        name: str = "input.mp4",
        size: str = "320x240",
        rate: str = "25",
        seconds: float = 2.0,
        audio: bool = True,
        audio_codec: str = "aac",
        metadata: dict[str, str] | None = None,
    ) -> Path:
        out = tmp_path / name
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}:rate={rate}:duration={seconds}",
        ]
        if audio:
            cmd += [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={seconds}",
                "-c:a",
                audio_codec,
            ]
        for key, value in (metadata or {}).items():
            cmd += ["-metadata", f"{key}={value}"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(out)]
        subprocess.run(cmd, check=True)
        return out

    return _make


class ScriptedTracker:
    """Replays a fixed {frame: [PersonObservation]} script."""

    def __init__(self, script: dict[int, list[PersonObservation]]) -> None:
        self.script = script
        self.frame = 0

    def update(self, frame) -> list[PersonObservation]:
        people = self.script.get(self.frame, [])
        self.frame += 1
        return people


class HeadFaceDetector:
    """Reports a face at the top-centre of each person the tracker just returned."""

    def __init__(self, tracker: ScriptedTracker, face_sizes: dict[int, float]) -> None:
        self.tracker = tracker
        self.face_sizes = face_sizes

    def detect(self, frame) -> list[FaceDetection]:
        faces = []
        for person in self.tracker.script.get(self.tracker.frame - 1, []):
            size = self.face_sizes.get(person.track_id)
            if size is None:
                continue  # this person never shows their face
            cx = (person.box.x1 + person.box.x2) / 2
            y = person.box.y1 + 2
            faces.append(FaceDetection(Box(cx - size / 2, y, cx + size / 2, y + size), 0.95))
        return faces


class SizeAgeEstimator:
    """Small face crops are 'children', large ones 'adults'."""

    def __init__(self, child_below_px: int) -> None:
        self.child_below_px = child_below_px
        self.calls = 0

    def estimate(self, faces):
        self.calls += 1
        return [(0.9, 8.0) if f.shape[1] < self.child_below_px else (0.05, 35.0) for f in faces]


def person(
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    conf: float = 0.9,
    mask: Mask | None = None,
):
    return PersonObservation(track_id, Box(x1, y1, x2, y2), conf, mask)


def ellipse_raw(box: Box, width: int, height: int) -> np.ndarray:
    """Full-frame binary 'silhouette': an upright ellipse inside ``box``."""
    raw = np.zeros((height, width), np.uint8)
    cx, cy = box.center
    axes = (round(box.width * 0.35), round(box.height * 0.45))
    cv2.ellipse(raw, (round(cx), round(cy)), axes, 0, 0, 360, 1, -1)
    return raw


def ellipse_mask(box: Box, width: int, height: int) -> Mask:
    mask = build_coverage(ellipse_raw(box, width, height), box)
    assert mask is not None
    return mask
