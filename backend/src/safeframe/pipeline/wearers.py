"""Attributing band sightings to tracked people.

A band is worn on a wrist, so the strongest evidence is a pose model's wrist
keypoint right next to the marker. When no wrist is close (the pose model
missed the arm, or isn't installed) the marker is attributed to the person
whose silhouette contains it, and failing that to the person whose box does.

When a sighting fits two or more people about equally well, for example two
children holding hands, every one of them gets an *ambiguous* hit. They are
all protected by default, because leaving the real wearer exposed is the
failure that can't be undone; the reviewer can keep anyone who clearly isn't
wearing it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from safeframe.models import yolo_weights_path
from safeframe.pipeline.band import BandDetection
from safeframe.pipeline.types import BandHit, BandVia, Box, PersonObservation

# A wrist "holds" a marker whose centre is within this fraction of the
# person's box height (a forearm is roughly 0.15 of standing height).
WRIST_RADIUS = 0.1
# A second candidate this close (in the same normalised units) makes a wrist
# attribution ambiguous.
WRIST_AMBIGUITY = 0.03
# Wrist keypoints below this confidence are ignored.
MIN_KEYPOINT_CONF = 0.3
# Pose people are matched to tracked people by box IoU.
MIN_POSE_IOU = 0.4
# Size of the zoom crop around each wrist, as a fraction of person height.
WRIST_ROI = 0.22
MIN_WRIST_ROI_PX = 40.0

DEFAULT_POSE_WEIGHTS = "yolo11n-pose.pt"
# COCO keypoint indices.
LEFT_WRIST, RIGHT_WRIST = 9, 10


@dataclass(frozen=True, slots=True)
class Wrist:
    x: float
    y: float
    conf: float


@dataclass(frozen=True, slots=True)
class PoseObservation:
    box: Box
    wrists: tuple[Wrist, ...]


@dataclass(frozen=True, slots=True)
class Attribution:
    detection: BandDetection
    track_ids: tuple[int, ...]  # one id = clean hit; several = ambiguous; none = orphan
    via: BandVia | None


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def match_poses(
    poses: Sequence[PoseObservation], people: Sequence[PersonObservation]
) -> dict[int, PoseObservation]:
    """Greedy one-to-one match of pose detections to tracked people by IoU."""
    pairs = sorted(
        ((_iou(pose.box, p.box), i, p.track_id) for i, pose in enumerate(poses) for p in people),
        reverse=True,
    )
    used_pose: set[int] = set()
    out: dict[int, PoseObservation] = {}
    for iou, i, tid in pairs:
        if iou < MIN_POSE_IOU:
            break
        if i in used_pose or tid in out:
            continue
        used_pose.add(i)
        out[tid] = poses[i]
    return out


def wrist_rois(poses: Sequence[PoseObservation]) -> list[Box]:
    """Square crops around every confident wrist, for zoomed band detection."""
    rois = []
    for pose in poses:
        half = max(MIN_WRIST_ROI_PX, WRIST_ROI * pose.box.height) / 2
        for w in pose.wrists:
            if w.conf >= MIN_KEYPOINT_CONF:
                rois.append(Box(w.x - half, w.y - half, w.x + half, w.y + half))
    return rois


def _mask_contains(person: PersonObservation, x: float, y: float) -> bool:
    m = person.mask
    if m is None:
        return False
    cx, cy = int(x) - m.x, int(y) - m.y
    if not (0 <= cx < m.width and 0 <= cy < m.height):
        return False
    return bool(m.decode()[cy, cx])


def attribute(
    det: BandDetection,
    people: Sequence[PersonObservation],
    poses_by_track: dict[int, PoseObservation],
) -> Attribution:
    cx, cy = det.center
    wrist = []
    for p in people:
        pose = poses_by_track.get(p.track_id)
        if pose is None:
            continue
        height = max(pose.box.height, 1.0)
        dists = [
            float(np.hypot(w.x - cx, w.y - cy)) / height
            for w in pose.wrists
            if w.conf >= MIN_KEYPOINT_CONF
        ]
        if dists and (d := min(dists)) <= WRIST_RADIUS:
            wrist.append((d, p.track_id))
    if wrist:
        wrist.sort()
        best = wrist[0][0]
        ids = tuple(tid for d, tid in wrist if d - best < WRIST_AMBIGUITY)
        return Attribution(det, ids, BandVia.WRIST)

    body = tuple(p.track_id for p in people if _mask_contains(p, cx, cy))
    if body:
        return Attribution(det, body, BandVia.BODY)

    boxed = tuple(p.track_id for p in people if p.box.expand(0.05).contains(cx, cy))
    if boxed:
        return Attribution(det, boxed, BandVia.BOX)
    return Attribution(det, (), None)


_STRENGTH = {BandVia.WRIST: 3, BandVia.BODY: 2, BandVia.BOX: 1}


def frame_hits(
    frame: int,
    detections: Sequence[BandDetection],
    people: Sequence[PersonObservation],
    poses: Sequence[PoseObservation] = (),
) -> tuple[dict[int, BandHit], bool]:
    """Best hit per track in one frame, and whether any marker was an orphan.

    A clean hit beats an ambiguous one, then stronger evidence (wrist > body >
    box) beats weaker, then the larger marker wins.
    """
    poses_by_track = match_poses(poses, people) if poses else {}
    best: dict[int, BandHit] = {}
    orphan = False
    for det in detections:
        att = attribute(det, people, poses_by_track)
        if not att.track_ids or att.via is None:
            orphan = True
            continue
        ambiguous = len(att.track_ids) > 1
        for tid in att.track_ids:
            hit = BandHit(frame, att.via, ambiguous, det.side_px)
            cur = best.get(tid)
            if cur is None or _rank(hit) > _rank(cur):
                best[tid] = hit
    return best, orphan


def _rank(hit: BandHit) -> tuple[int, int, float]:
    return (0 if hit.ambiguous else 1, _STRENGTH[hit.via], hit.side_px)


class YoloWristLocator:
    """Ultralytics YOLO pose model, used only for wrist keypoints."""

    def __init__(
        self,
        weights: str = DEFAULT_POSE_WEIGHTS,
        device: str | None = None,
        conf: float = 0.25,
        imgsz: int = 960,
    ) -> None:
        from ultralytics import YOLO

        path = Path(weights)
        if not path.is_absolute() and path.parent == Path("."):
            path = yolo_weights_path(weights)
        self._model = YOLO(str(path))
        self._kwargs = dict(conf=conf, imgsz=imgsz, verbose=False)
        if device:
            self._kwargs["device"] = device

    def locate(self, frame: np.ndarray) -> list[PoseObservation]:
        result = self._model.predict(frame, **self._kwargs)[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        xy = result.keypoints.xy.cpu().numpy()
        kconf = result.keypoints.conf
        kconf = kconf.cpu().numpy() if kconf is not None else np.ones(xy.shape[:2])
        out = []
        for b, pts, cf in zip(boxes, xy, kconf, strict=True):
            wrists = tuple(
                Wrist(float(pts[k][0]), float(pts[k][1]), float(cf[k]))
                for k in (LEFT_WRIST, RIGHT_WRIST)
            )
            out.append(PoseObservation(Box(*map(float, b)), wrists))
        return out
