"""Plain data types shared by the analysis, review and render stages.

Everything here is JSON-serialisable via ``to_dict``/``from_dict`` so an
analysis can be persisted, edited in the review UI and rendered later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

from safeframe.pipeline.masks import WIRE_KEY, Budget, Mask, load_track_masks


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def expand(self, ratio: float) -> Box:
        """Grow by ``ratio`` of width/height on every side."""
        dx, dy = self.width * ratio, self.height * ratio
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def clip(self, width: int, height: int) -> Box:
        return Box(
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
            min(max(self.x2, 0.0), width),
            min(max(self.y2, 0.0), height),
        )

    def lerp(self, other: Box, t: float) -> Box:
        return Box(
            self.x1 + (other.x1 - self.x1) * t,
            self.y1 + (other.y1 - self.y1) * t,
            self.x2 + (other.x2 - self.x2) * t,
            self.y2 + (other.y2 - self.y2) * t,
        )

    def as_list(self) -> list[float]:
        return [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)]


@dataclass(frozen=True, slots=True)
class PersonObservation:
    """One tracked person in one frame, as reported by the tracker."""

    track_id: int
    box: Box
    confidence: float
    # Final protected coverage for this person in this frame, or None when the
    # model gave no usable silhouette (the box is then used instead).
    mask: Mask | None = None


@dataclass(frozen=True, slots=True)
class FaceDetection:
    box: Box
    score: float


@dataclass(frozen=True, slots=True)
class AgeSample:
    """Age evidence from one face crop of one track in one frame."""

    frame: int
    p_minor: float
    expected_age: float
    weight: float


class TrackStatus(StrEnum):
    MINOR = "minor"  # enough evidence, probability of being under 18 >= threshold
    ADULT = "adult"  # enough evidence, probability below threshold
    UNCERTAIN = "uncertain"  # not enough face evidence: blurred by default (fail closed)


@dataclass(slots=True)
class Track:
    """A person followed across frames, with accumulated age evidence."""

    track_id: int
    # frame index -> box. Includes interpolated frames after smoothing.
    boxes: dict[int, Box] = field(default_factory=dict)
    interpolated: set[int] = field(default_factory=set)
    # Sparse frame index -> silhouette coverage, only on observed frames. A
    # frame with a box but no mask is blurred with the padded box.
    masks: dict[int, Mask] = field(default_factory=dict)
    samples: list[AgeSample] = field(default_factory=list)
    p_minor: float | None = None
    expected_age: float | None = None
    peak_confidence: float = 0.0

    @property
    def first_frame(self) -> int:
        return min(self.boxes)

    @property
    def last_frame(self) -> int:
        return max(self.boxes)

    def mask_for(self, frame: int) -> Mask | None:
        """Silhouette for ``frame``, only where the box there was actually observed."""
        if frame in self.interpolated:
            return None
        return self.masks.get(frame)

    def to_dict(self) -> dict:
        data = {
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "p_minor": None if self.p_minor is None else round(self.p_minor, 4),
            "expected_age": None if self.expected_age is None else round(self.expected_age, 1),
            "face_samples": len(self.samples),
            "peak_confidence": round(self.peak_confidence, 3),
            # Compact per-frame boxes: [frame, x1, y1, x2, y2, interpolated(0/1)]
            "boxes": [
                [f, *self.boxes[f].as_list(), int(f in self.interpolated)]
                for f in sorted(self.boxes)
            ],
            "samples": [asdict(s) for s in self.samples],
        }
        masks = [
            self.masks[f].to_row(f)
            for f in sorted(self.masks)
            if f in self.boxes and f not in self.interpolated
        ]
        if masks:
            data[WIRE_KEY] = masks
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict,
        frame_size: tuple[int, int] | None = None,
        budget: Budget | None = None,
    ) -> Track:
        """Load a track. Masks are read only when ``frame_size`` (width, height)
        is known; invalid or over-budget mask entries are dropped (box fallback)."""
        track = cls(track_id=int(data["track_id"]))
        for f, x1, y1, x2, y2, interp in data["boxes"]:
            track.boxes[int(f)] = Box(x1, y1, x2, y2)
            if interp:
                track.interpolated.add(int(f))
        if frame_size is not None and WIRE_KEY in data:
            observed = set(track.boxes) - track.interpolated
            track.masks = load_track_masks(
                data[WIRE_KEY], observed, frame_size[0], frame_size[1], budget or Budget()
            )
        track.samples = [AgeSample(**s) for s in data.get("samples", [])]
        track.p_minor = data.get("p_minor")
        track.expected_age = data.get("expected_age")
        track.peak_confidence = data.get("peak_confidence", 0.0)
        return track


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    # Exact rational frame rate as reported by FFmpeg, e.g. "30000/1001".
    fps_rational: str
    frame_count: int
    duration: float
    has_audio: bool
    audio_codec: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VideoInfo:
        return cls(**data)


@dataclass(slots=True)
class ManualRegion:
    """A rectangle the reviewer added by hand, blurred over [start_frame, end_frame]."""

    region_id: str
    start_frame: int
    end_frame: int
    box: Box

    def covers(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "box": self.box.as_list(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ManualRegion:
        return cls(
            region_id=str(data["region_id"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            box=Box(*data["box"]),
        )


@dataclass(slots=True)
class Analysis:
    """Output of the analysis stage: everything needed to review and render."""

    video: VideoInfo
    tracks: list[Track]
    settings: dict

    def to_dict(self) -> dict:
        return {
            "video": self.video.to_dict(),
            "settings": self.settings,
            "tracks": [t.to_dict() for t in self.tracks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Analysis:
        video = VideoInfo.from_dict(data["video"])
        budget = Budget()  # one run budget shared by every track, in file order
        return cls(
            video=video,
            tracks=[
                Track.from_dict(t, (video.width, video.height), budget) for t in data["tracks"]
            ],
            settings=data.get("settings", {}),
        )

    @property
    def mask_count(self) -> int:
        return sum(len(t.masks) for t in self.tracks)
