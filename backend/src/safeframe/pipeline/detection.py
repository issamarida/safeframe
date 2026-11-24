"""Person detection + instance segmentation (YOLO) and tracking (BoT-SORT with ReID).

Silhouettes come from the *same* tracked result as the boxes. After tracking,
Ultralytics re-indexes the whole ``Results`` object with the detection index
of every surviving track (``result[idx]``), which reorders ``boxes`` and
``masks`` together, so row ``i`` of ``masks.data`` belongs to row ``i`` of
``boxes`` (verified against ultralytics 8.4.176; see docs/ARCHITECTURE.md).

With ``retina_masks=True`` masks are produced at the decoded source frame size
with the letterbox removed by Ultralytics itself, so no rescaling happens here.
A mask whose shape differs from the frame is discarded rather than stretched.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import numpy as np

from safeframe.models import DEFAULT_YOLO_WEIGHTS, yolo_weights_path
from safeframe.pipeline.masks import build_coverage
from safeframe.pipeline.types import Box, PersonObservation

log = logging.getLogger(__name__)

PERSON_CLASS = 0  # COCO "person"
__all__ = ["DEFAULT_YOLO_WEIGHTS", "PERSON_CLASS", "YoloPersonTracker", "observations_from_result"]


def _np(t) -> np.ndarray:
    """Tensor or array -> numpy, without importing torch."""
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def observations_from_result(result, frame_shape: tuple[int, ...]) -> list[PersonObservation]:
    """Convert one tracked Ultralytics result into observations with masks.

    Every tracked box becomes an observation even when its mask is missing or
    cannot be used: the box is then the fallback coverage.
    """
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    xyxy = _np(boxes.xyxy)
    ids = [int(i) for i in _np(boxes.id).astype(int).tolist()]
    confs = [float(c) for c in _np(boxes.conf).tolist()]
    masks = None
    raw_masks = getattr(result, "masks", None)
    if raw_masks is not None:
        try:
            data = _np(raw_masks.data)
            if data.ndim == 3 and data.shape[0] == len(ids) and data.shape[1:] == frame_shape[:2]:
                masks = data
            else:
                log.warning("mask stack %s does not match boxes/frame; using boxes", data.shape)
        except Exception:
            log.exception("could not read masks; using boxes")
    out = []
    for row, (b, tid, c) in enumerate(zip(xyxy, ids, confs, strict=True)):
        box = Box(*map(float, b))
        mask = None
        if masks is not None:
            try:
                mask = build_coverage(masks[row], box)
            except Exception:
                log.exception("mask extraction failed for track %d; using its box", tid)
        out.append(PersonObservation(track_id=tid, box=box, confidence=c, mask=mask))
    return out


def tracker_config() -> Path:
    """Tuned BoT-SORT config: long track buffer + appearance ReID so identities
    survive head turns and occlusions of a second or two."""
    return Path(str(resources.files("safeframe.pipeline") / "botsort_safeframe.yaml"))


class YoloPersonTracker:
    """Ultralytics YOLO detector + BoT-SORT tracker, restricted to people.

    Detection confidence is intentionally low: BoT-SORT's second association
    stage uses low-score boxes to keep tracks alive through blur and partial
    occlusion, and for a privacy tool recall matters more than precision.
    """

    def __init__(
        self,
        weights: str = DEFAULT_YOLO_WEIGHTS,
        device: str | None = None,
        conf: float = 0.1,
        iou: float = 0.6,
        imgsz: int = 960,
    ) -> None:
        from ultralytics import YOLO

        path = Path(weights)
        if not path.is_absolute() and path.parent == Path("."):
            # Bare model name: keep the download out of the working directory.
            path = yolo_weights_path(weights)
        self._model = YOLO(str(path))
        self._kwargs = dict(
            persist=True,
            tracker=str(tracker_config()),
            classes=[PERSON_CLASS],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            # Masks at decoded source resolution, letterbox undone by Ultralytics.
            retina_masks=True,
        )
        if device:
            self._kwargs["device"] = device
        self._checked_feats = False

    def update(self, frame: np.ndarray) -> list[PersonObservation]:
        result = self._model.track(frame, **self._kwargs)[0]
        if result.masks is not None and not self._checked_feats:
            self._checked_feats = True
            # The segment predictor drops detections whose mask is empty unless
            # native ReID features are being captured. Our tracker config keeps
            # ReID on, so no person box is lost; warn if that ever changes.
            if getattr(self._model.predictor, "_feats", None) is None:
                log.warning(
                    "segmentation without native ReID: Ultralytics may drop person "
                    "detections that have an empty mask"
                )
        return observations_from_result(result, frame.shape)
