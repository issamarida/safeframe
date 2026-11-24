"""Construct the real ML components (lazily, so importing SafeFrame stays cheap)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from safeframe.models import YUNET, fetch
from safeframe.pipeline.age import DEFAULT_AGE_MODEL
from safeframe.pipeline.analyze import Components
from safeframe.pipeline.detection import DEFAULT_YOLO_WEIGHTS
from safeframe.pipeline.wearers import DEFAULT_POSE_WEIGHTS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS
    # Wrist keypoints for band attribution; "" disables the pose model.
    pose_weights: str = DEFAULT_POSE_WEIGHTS
    age_model: str = DEFAULT_AGE_MODEL
    age_model_revision: str | None = None
    device: str | None = None  # None = auto (CUDA if available)


class _LazyAgeEstimator:
    """Loads the age model on first use: most analyses never need it."""

    def __init__(self, bundle: ModelBundle) -> None:
        self._bundle = bundle

    def estimate(self, faces: Sequence[np.ndarray]) -> list[tuple[float, float]]:
        return self._bundle.age_estimator().estimate(faces)


class _LazyFaceDetector:
    def __init__(self) -> None:
        self._detector = None

    def detect(self, frame: np.ndarray):
        if self._detector is None:
            from safeframe.pipeline.faces import YuNetFaceDetector

            self._detector = YuNetFaceDetector(fetch(YUNET))
        return self._detector.detect(frame)


class ModelBundle:
    """Holds models that are safe to share between jobs (age estimator, pose
    model, band detector) and builds the per-video stateful tracker."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._lock = threading.Lock()
        self._age = None
        self._pose = None
        self._band = None

    def new_tracker(self, photo: bool = False):
        from safeframe.pipeline.detection import YoloPersonTracker

        return YoloPersonTracker(
            self.config.yolo_weights, device=self.config.device, tracking=not photo
        )

    def band_detector(self):
        with self._lock:
            if self._band is None:
                from safeframe.pipeline.band import ArucoBandDetector

                self._band = ArucoBandDetector()
            return self._band

    def wrist_locator(self):
        if not self.config.pose_weights:
            return None
        with self._lock:
            if self._pose is None:
                from safeframe.pipeline.wearers import YoloWristLocator

                self._pose = YoloWristLocator(self.config.pose_weights, device=self.config.device)
            return self._pose

    def age_estimator(self):
        with self._lock:
            if self._age is None:
                from safeframe.pipeline.age import ViTAgeEstimator

                log.info("loading age model %s", self.config.age_model)
                self._age = ViTAgeEstimator(
                    self.config.age_model,
                    device=self.config.device,
                    revision=self.config.age_model_revision,
                )
            return self._age

    def components(self, photo: bool = False) -> Components:
        return Components(
            tracker=self.new_tracker(photo),
            band_detector=self.band_detector(),
            face_detector=_LazyFaceDetector(),
            age_estimator=_LazyAgeEstimator(self),
            wrist_locator=self.wrist_locator(),
        )

    def warm_up(self) -> None:
        """Download/load the always-needed models once so the first job doesn't wait.

        The age model is deliberately not loaded: it is needed only in guarded
        zones and event mode.
        """
        self.new_tracker()
        self.band_detector()
        self.wrist_locator()
