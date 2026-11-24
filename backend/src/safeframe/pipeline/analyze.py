"""Analysis stage: detect, track and age-estimate every person in a video.

Identity comes from the *body* tracker, not the face, so a person keeps the
same track ID while they turn their head away or are briefly occluded. Age
evidence accumulates on the track whenever their face is visible, and the
decision for a track applies to every frame of it — including frames where
the face is hidden.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from safeframe.pipeline.faces import assign_faces, crop_face
from safeframe.pipeline.interfaces import (
    AgeEstimator,
    Cancelled,
    CancelToken,
    FaceDetector,
    PersonTracker,
)
from safeframe.pipeline.masks import MAX_RUNS_TOTAL, WIRE_KEY, Budget, load_track_masks
from safeframe.pipeline.policy import aggregate
from safeframe.pipeline.smoothing import smooth_track
from safeframe.pipeline.types import AgeSample, Analysis, Track, VideoInfo
from safeframe.video.io import iter_frames, probe

log = logging.getLogger(__name__)

ProgressFn = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    # Run face detection + age estimation every N frames.
    age_every: int = 3
    # Stop sampling a track after this many usable face samples.
    max_samples_per_track: int = 40
    # Faces smaller than this (pixels, shorter side) are too unreliable to age.
    min_face_px: int = 32
    # Gaps up to this long (seconds) inside a track are interpolated.
    max_gap_seconds: float = 1.5
    # Hold boxes this long (seconds) before a track starts and after it ends.
    pad_seconds: float = 0.25
    # Longest side of saved per-track review thumbnails.
    thumbnail_px: int = 160

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class AnalysisResult:
    analysis: Analysis
    # track_id -> JPEG bytes of the best person crop, for the review screen.
    thumbnails: dict[int, bytes]


def _sample_weight(face_score: float, face_px: float) -> float:
    # Larger, more confidently detected faces carry more weight; saturate at 112 px.
    return face_score * min(1.0, face_px / 112.0)


def _thumbnail(frame: np.ndarray, track_box, max_px: int) -> bytes | None:
    h, w = frame.shape[:2]
    b = track_box.clip(w, h)
    x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2]
    scale = max_px / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def analyze_video(
    path: Path,
    tracker: PersonTracker,
    face_detector: FaceDetector,
    age_estimator: AgeEstimator,
    settings: AnalysisSettings | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
    info: VideoInfo | None = None,
) -> AnalysisResult:
    settings = settings or AnalysisSettings()
    info = info or probe(path)
    tracks: dict[int, Track] = {}
    thumbs: dict[int, bytes] = {}
    thumb_conf: dict[int, float] = {}
    total = max(info.frame_count, 1)
    last_index = -1
    mask_runs = 0

    for index, frame in iter_frames(path):
        last_index = index
        if cancel is not None and cancel.is_set():
            raise Cancelled()

        people = tracker.update(frame)
        for person in people:
            track = tracks.setdefault(person.track_id, Track(track_id=person.track_id))
            track.boxes[index] = person.box
            # Same zero-based decoded frame index as the box. Stop collecting
            # once the payload budget is clearly blown (the rest use boxes).
            if person.mask is not None and mask_runs < MAX_RUNS_TOTAL:
                track.masks[index] = person.mask
                mask_runs += person.mask.run_count
            track.peak_confidence = max(track.peak_confidence, person.confidence)
            if person.confidence > thumb_conf.get(person.track_id, -1.0):
                thumb = _thumbnail(frame, person.box, settings.thumbnail_px)
                if thumb is not None:
                    thumbs[person.track_id] = thumb
                    thumb_conf[person.track_id] = person.confidence

        wanting = [
            p for p in people if len(tracks[p.track_id].samples) < settings.max_samples_per_track
        ]
        if wanting and index % settings.age_every == 0:
            _sample_ages(frame, index, wanting, tracks, face_detector, age_estimator, settings)

        if progress is not None and index % 10 == 0:
            progress(min(0.999, (index + 1) / total))

    frame_count = max(info.frame_count, last_index + 1)
    if frame_count != info.frame_count:
        info = VideoInfo(**{**info.to_dict(), "frame_count": frame_count})

    max_gap = max(1, round(settings.max_gap_seconds * info.fps))
    pad = max(0, round(settings.pad_seconds * info.fps))
    for track in tracks.values():
        track.p_minor, track.expected_age = aggregate(track.samples)
        smooth_track(track, max_gap=max_gap, pad=pad, frame_count=frame_count)

    if progress is not None:
        progress(1.0)
    ordered = sorted(tracks.values(), key=lambda t: (t.first_frame, t.track_id))
    _apply_mask_budget(ordered, info)
    masked = sum(len(t.masks) for t in ordered)
    observed = sum(len(t.boxes) - len(t.interpolated) for t in ordered)
    log.info(
        "analysed %d frames, %d tracks, silhouettes on %d of %d observed person-frames",
        frame_count,
        len(ordered),
        masked,
        observed,
    )
    return AnalysisResult(
        analysis=Analysis(
            video=info,
            tracks=ordered,
            settings={**settings.to_dict(), "mask_format": WIRE_KEY},
        ),
        thumbnails=thumbs,
    )


def _apply_mask_budget(tracks: list[Track], info: VideoInfo) -> None:
    """Keep exactly the masks the loaders will accept, in the order they read them,
    so analysis, saved JSON, export and review all agree on mask vs box frames."""
    budget = Budget()
    for track in tracks:
        if not track.masks:
            continue
        observed = set(track.boxes) - track.interpolated
        rows = [track.masks[f].to_row(f) for f in sorted(track.masks)]
        track.masks = load_track_masks(rows, observed, info.width, info.height, budget)


def _sample_ages(frame, index, people, tracks, face_detector, age_estimator, settings) -> None:
    faces = face_detector.detect(frame)
    if not faces:
        return
    matches = assign_faces(faces, people)
    crops, owners = [], []
    for track_id, face in matches.items():
        if min(face.box.width, face.box.height) < settings.min_face_px:
            continue
        crop = crop_face(frame, face.box)
        if crop is None:
            continue
        crops.append(crop)
        owners.append((track_id, face))
    if not crops:
        return
    for (track_id, face), (p_minor, expected) in zip(
        owners, age_estimator.estimate(crops), strict=True
    ):
        tracks[track_id].samples.append(
            AgeSample(
                frame=index,
                p_minor=float(p_minor),
                expected_age=float(expected),
                weight=_sample_weight(face.score, min(face.box.width, face.box.height)),
            )
        )
