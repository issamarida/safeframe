"""Analysis stage: find every person, and find who is wearing a SafeFrame Band.

Pass 1 tracks people (identity comes from the *body*, so a person keeps one
track ID through head turns and brief occlusion) and looks for band markers,
attributing each sighting to a track. One sighting protects the whole track,
including the frames where the band is behind someone's back.

Pass 2 runs only when a decision depends on age: for band wearers in a
guarded zone (is this really a child?) and for everyone in event mode. Faces
of anyone else are never age-estimated. This is data minimisation by
construction, not by policy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from safeframe.pipeline.context import CaptureContext
from safeframe.pipeline.faces import assign_faces, crop_face
from safeframe.pipeline.interfaces import (
    AgeEstimator,
    BandDetector,
    Cancelled,
    CancelToken,
    FaceDetector,
    PersonTracker,
    WristLocator,
)
from safeframe.pipeline.masks import MAX_RUNS_TOTAL, WIRE_KEY, Budget, load_track_masks
from safeframe.pipeline.policy import aggregate, needs_age
from safeframe.pipeline.smoothing import smooth_track
from safeframe.pipeline.types import AgeSample, Analysis, PersonObservation, Track, VideoInfo
from safeframe.pipeline.wearers import frame_hits, wrist_rois
from safeframe.video.io import VideoError, iter_frames, probe

log = logging.getLogger(__name__)

ProgressFn = Callable[[float], None]
Frames = Callable[[], Iterable[tuple[int, np.ndarray]]]


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    # Look for bands every N frames (always 1 for photos).
    band_every: int = 2
    # Run face detection + age estimation every N frames (pass 2 only).
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
class Components:
    """The models one analysis needs. Tracker state is per video."""

    tracker: PersonTracker
    band_detector: BandDetector
    face_detector: FaceDetector
    age_estimator: AgeEstimator
    # Optional: without it, bands are attributed by silhouette or box only.
    wrist_locator: WristLocator | None = None


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


def read_photo(path: Path) -> np.ndarray:
    """Decode a still image upright (EXIF orientation applied), as BGR."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise VideoError(f"could not read {path.name} as an image")
    return frame


def analyze_video(
    path: Path,
    components: Components,
    context: CaptureContext | None = None,
    settings: AnalysisSettings | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
    info: VideoInfo | None = None,
) -> AnalysisResult:
    return _analyze(
        lambda: iter_frames(path),
        info or probe(path),
        components,
        context or CaptureContext(),
        settings or AnalysisSettings(),
        progress,
        cancel,
    )


def analyze_photo(
    path: Path,
    components: Components,
    context: CaptureContext | None = None,
    settings: AnalysisSettings | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelToken | None = None,
) -> AnalysisResult:
    frame = read_photo(path)
    settings = settings or AnalysisSettings()
    photo_settings = AnalysisSettings(**{**settings.to_dict(), "band_every": 1, "age_every": 1})
    return _analyze(
        lambda: iter([(0, frame)]),
        VideoInfo.photo(frame.shape[1], frame.shape[0]),
        components,
        context or CaptureContext(),
        photo_settings,
        progress,
        cancel,
    )


def _analyze(
    frames: Frames,
    info: VideoInfo,
    components: Components,
    context: CaptureContext,
    settings: AnalysisSettings,
    progress: ProgressFn | None,
    cancel: CancelToken | None,
) -> AnalysisResult:
    tracks: dict[int, Track] = {}
    thumbs: dict[int, bytes] = {}
    thumb_conf: dict[int, float] = {}
    orphans: list[int] = []
    total = max(info.frame_count, 1)
    may_need_age = context.guarded or context.protect_minors
    pass1_share = 0.75 if may_need_age else 1.0
    last_index = -1
    mask_runs = 0

    for index, frame in _checked(frames(), cancel):
        last_index = index
        people = components.tracker.update(frame)
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

        if index % settings.band_every == 0 and _look_for_bands(
            index, frame, people, components, tracks
        ):
            orphans.append(index)

        if progress is not None and index % 10 == 0:
            progress(min(0.999, pass1_share * (index + 1) / total))

    frame_count = max(info.frame_count, last_index + 1)
    if frame_count != info.frame_count:
        info = VideoInfo(**{**info.to_dict(), "frame_count": frame_count})

    wanted = {
        t.track_id for t in tracks.values() if needs_age(t, context.guarded, context.protect_minors)
    }
    if wanted:
        _estimate_ages(frames, tracks, wanted, components, settings, cancel, progress, total)
    log.info(
        "%d tracks, %d with a band, ages estimated for %d",
        len(tracks),
        sum(t.wears_band for t in tracks.values()),
        len(wanted),
    )

    max_gap = max(1, round(settings.max_gap_seconds * info.fps))
    pad = max(0, round(settings.pad_seconds * info.fps))
    for track in tracks.values():
        track.p_minor, track.expected_age = aggregate(track.samples)
        smooth_track(track, max_gap=max_gap, pad=pad, frame_count=frame_count)

    if progress is not None:
        progress(1.0)
    ordered = sorted(tracks.values(), key=lambda t: (t.first_frame, t.track_id))
    _apply_mask_budget(ordered, info)
    return AnalysisResult(
        analysis=Analysis(
            video=info,
            tracks=ordered,
            settings={**settings.to_dict(), "mask_format": WIRE_KEY},
            context=context.to_dict(),
            orphan_band_frames=orphans,
        ),
        thumbnails=thumbs,
    )


def _checked(frames: Iterable, cancel: CancelToken | None) -> Iterator:
    for item in frames:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        yield item


def _look_for_bands(
    index: int,
    frame: np.ndarray,
    people: list[PersonObservation],
    components: Components,
    tracks: dict[int, Track],
) -> bool:
    """Record band hits for this frame. Returns True if a band had no owner."""
    poses = (
        components.wrist_locator.locate(frame)
        if components.wrist_locator is not None and people
        else []
    )
    detections = components.band_detector.detect(frame, wrist_rois(poses))
    if not detections:
        return False
    hits, orphan = frame_hits(index, detections, people, poses)
    for track_id, hit in hits.items():
        tracks[track_id].add_band_hit(hit)
    return orphan


def _estimate_ages(
    frames: Frames,
    tracks: dict[int, Track],
    wanted: set[int],
    components: Components,
    settings: AnalysisSettings,
    cancel: CancelToken | None,
    progress: ProgressFn | None,
    total: int,
) -> None:
    """Pass 2: age evidence for the ``wanted`` tracks only, from their pass-1 boxes."""
    for index, frame in _checked(frames(), cancel):
        if progress is not None and index % 10 == 0:
            progress(min(0.999, 0.75 + 0.25 * (index + 1) / total))
        if index % settings.age_every:
            continue
        # Everyone present is passed for face *attribution*, so a face is never
        # credited to a wearer when it fits someone else better.
        present = [
            PersonObservation(t.track_id, t.boxes[index], 1.0)
            for t in tracks.values()
            if index in t.boxes
        ]
        if not any(
            p.track_id in wanted
            and len(tracks[p.track_id].samples) < settings.max_samples_per_track
            for p in present
        ):
            continue
        _sample_ages(frame, index, present, wanted, tracks, components, settings)


def _sample_ages(frame, index, people, wanted, tracks, components, settings) -> None:
    faces = components.face_detector.detect(frame)
    if not faces:
        return
    matches = assign_faces(faces, people)
    crops, owners = [], []
    for track_id, face in matches.items():
        if track_id not in wanted:
            continue
        if len(tracks[track_id].samples) >= settings.max_samples_per_track:
            continue
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
        owners, components.age_estimator.estimate(crops), strict=True
    ):
        tracks[track_id].samples.append(
            AgeSample(
                frame=index,
                p_minor=float(p_minor),
                expected_age=float(expected),
                weight=_sample_weight(face.score, min(face.box.width, face.box.height)),
            )
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
