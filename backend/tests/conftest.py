"""Shared fixtures. All test video is synthesised by FFmpeg at runtime —
no footage is ever stored in the repository.

Every synthetic frame carries its own index as 12 black/white 8x8 blocks
along the top edge, so scripted fakes know which frame they are looking at
even when the pipeline decodes a video twice (pass 2 never calls the tracker).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from safeframe.pipeline.analyze import Components
from safeframe.pipeline.band import BandDetection
from safeframe.pipeline.masks import Mask, build_coverage
from safeframe.pipeline.types import Box, FaceDetection, PersonObservation
from safeframe.pipeline.wearers import PoseObservation, Wrist

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

INDEX_BITS = 12
_STAMP = f"lt(Y,8)*lt(X,{8 * INDEX_BITS})"
STAMP_FILTER = (
    f"geq=lum='if({_STAMP},255*mod(floor(N/pow(2,floor(X/8))),2),lum(X,Y))'"
    f":cb='if({_STAMP},128,cb(X,Y))':cr='if({_STAMP},128,cr(X,Y))'"
)


def frame_index(frame: np.ndarray) -> int:
    """Decode the index stamped into a synthetic frame by ``make_video``."""
    return sum(int(frame[4, 8 * b + 4].mean() > 128) << b for b in range(INDEX_BITS))


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
        cmd += [
            "-vf",
            STAMP_FILTER,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            str(out),
        ]
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
    """Reports a face at the top-centre of each scripted person in this frame."""

    def __init__(self, tracker: ScriptedTracker, face_sizes: dict[int, float]) -> None:
        self.tracker = tracker
        self.face_sizes = face_sizes
        self.calls = 0

    def detect(self, frame) -> list[FaceDetection]:
        self.calls += 1
        faces = []
        for person in self.tracker.script.get(frame_index(frame), []):
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
        self.crops = 0

    def estimate(self, faces):
        self.calls += 1
        self.crops += len(faces)
        return [(0.9, 8.0) if f.shape[1] < self.child_below_px else (0.05, 35.0) for f in faces]


def marker_at(x: float, y: float, side: float = 12.0, marker_id: int = 7) -> BandDetection:
    h = side / 2
    return BandDetection(
        marker_id, ((x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h))
    )


class ScriptedBands:
    """Reports band markers at scripted centres: {frame: [(x, y), ...]}."""

    def __init__(self, script: dict[int, list[tuple[float, float]]]) -> None:
        self.script = script
        self.rois_seen: list = []

    def detect(self, frame, rois=()):
        self.rois_seen.extend(rois)
        return [marker_at(x, y) for x, y in self.script.get(frame_index(frame), [])]


class ScriptedWrists:
    """Pose model fake: {frame: [PoseObservation, ...]}."""

    def __init__(self, script: dict[int, list[PoseObservation]]) -> None:
        self.script = script

    def locate(self, frame):
        return self.script.get(frame_index(frame), [])


def pose(box: tuple[float, float, float, float], *wrists: tuple[float, float]) -> PoseObservation:
    return PoseObservation(Box(*box), tuple(Wrist(x, y, 0.9) for x, y in wrists))


def components(tracker, faces, ages, bands=None, wrists=None) -> Components:
    return Components(
        tracker=tracker,
        band_detector=bands or ScriptedBands({}),
        face_detector=faces,
        age_estimator=ages,
        wrist_locator=wrists,
    )


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
