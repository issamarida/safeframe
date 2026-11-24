"""Person silhouette masks: construction, wire format and validation.

A mask is the *final protected coverage* of one tracked person in one frame,
already dilated by a small safety margin, stored as a binary crop in source
pixel coordinates.

Wire format (``masks_v1`` on each track): ``[frame, x, y, width, height, runs]``

* ``frame`` is the zero-based decoded frame index, the same key as the
  observed box in ``boxes``. Masks only exist on observed (non-interpolated)
  frames.
* ``x, y, width, height`` are integers in source pixels describing the
  half-open crop ``[x, x + width) x [y, y + height)``, fully inside the frame.
* ``runs`` is a row-major run-length encoding of the crop: alternating
  background / foreground run lengths, starting with background. The first
  run may be 0 (crop starts in foreground); every later run is positive. The
  runs sum to exactly ``width * height``.

Binarisation rule: a source pixel is foreground when the model's mask logit
is > 0 (probability > 0.5), as returned by Ultralytics with
``retina_masks=True``. Coverage is that foreground dilated by
:func:`margin_px` pixels.

Anything that does not satisfy these rules is treated as *missing*: the
renderer and the review UI then fall back to the padded bounding box, which
always covers at least as much as the old rectangular blur did.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # types.py imports this module
    from safeframe.pipeline.types import Box

log = logging.getLogger(__name__)

WIRE_KEY = "masks_v1"

# --- Coverage margin --------------------------------------------------------
# Dilation radius as a fraction of the person's longer box side, clamped so
# tiny people still get a margin and huge ones don't balloon.
MARGIN_RATIO = 0.03
MIN_MARGIN_PX = 3
MAX_MARGIN_PX = 24

# --- Geometry quality gates (on the raw model mask, before dilation) --------
# Fewer foreground pixels than this is noise, not a person.
MIN_MASK_PIXELS = 24
# A person silhouette normally covers ~35-75% of its box. Below this the mask
# is implausibly small (e.g. only a hand was segmented).
MIN_BOX_FILL = 0.12
# Share of foreground that must fall inside the (generously expanded) box.
MIN_INSIDE_FRACTION = 0.9
BOX_TOLERANCE = 0.15
# The mask's own extent must overlap the box this much (IoU of the two rects).
MIN_EXTENT_IOU = 0.3

# --- Payload budgets (both loaders apply the same ones, in the same order) --
MAX_RUNS_PER_MASK = 8_192
MAX_RUNS_PER_TRACK = 1_500_000
MAX_RUNS_TOTAL = 4_000_000


@dataclass(frozen=True, slots=True)
class Mask:
    """Binary coverage crop in source pixels (see module docstring)."""

    x: int
    y: int
    width: int
    height: int
    runs: tuple[int, ...]

    @property
    def run_count(self) -> int:
        return len(self.runs)

    def decode(self) -> np.ndarray:
        """Return a ``(height, width)`` bool array."""
        flat = np.zeros(self.width * self.height, dtype=bool)
        pos = 0
        fg = False
        for run in self.runs:
            if fg:
                flat[pos : pos + run] = True
            pos += run
            fg = not fg
        return flat.reshape(self.height, self.width)

    def to_row(self, frame: int) -> list:
        return [frame, self.x, self.y, self.width, self.height, list(self.runs)]


def margin_px(box: Box) -> int:
    scaled = round(MARGIN_RATIO * max(box.width, box.height))
    return int(min(MAX_MARGIN_PX, max(MIN_MARGIN_PX, scaled)))


def encode(crop: np.ndarray, x: int, y: int) -> Mask | None:
    """Encode a bool crop whose top-left is at (x, y). Trims empty borders."""
    rows = np.flatnonzero(crop.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(crop.any(axis=0))
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    tight = np.ascontiguousarray(crop[r0:r1, c0:c1]).reshape(-1)
    # Run boundaries: positions where the value changes.
    changes = np.flatnonzero(tight[1:] != tight[:-1]) + 1
    bounds = np.concatenate(([0], changes, [tight.size]))
    runs = np.diff(bounds).tolist()
    if tight[0]:
        runs.insert(0, 0)
    return Mask(x + c0, y + r0, c1 - c0, r1 - r0, tuple(int(r) for r in runs))


def _rect_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_coverage(raw: np.ndarray, box: Box) -> Mask | None:
    """Turn a full-frame model mask into stored coverage, or None if untrustworthy.

    ``raw`` must be a 2-D array the size of the decoded source frame, already
    binarised or holding values where > 0.5 means foreground.
    """
    if raw.ndim != 2:
        return None
    height, width = raw.shape
    if not all(map(math.isfinite, (box.x1, box.y1, box.x2, box.y2))):
        return None
    # Judge the mask against the visible part of the box.
    box = box.clip(width, height)
    fg = raw > 0.5
    ys, xs = np.nonzero(fg)
    count = xs.size
    if count < MIN_MASK_PIXELS or box.area <= 0:
        return None
    if count < MIN_BOX_FILL * box.area:
        return None
    tol = box.expand(BOX_TOLERANCE)
    inside = (xs >= tol.x1) & (xs < tol.x2) & (ys >= tol.y1) & (ys < tol.y2)
    if inside.mean() < MIN_INSIDE_FRACTION:
        return None
    extent = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    if _rect_iou(extent, (box.x1, box.y1, box.x2, box.y2)) < MIN_EXTENT_IOU:
        return None

    m = margin_px(box)
    # Work on the occupied crop grown by the margin, clipped to the frame.
    x0, y0 = max(0, int(extent[0]) - m), max(0, int(extent[1]) - m)
    x1, y1 = min(width, int(extent[2]) + m), min(height, int(extent[3]) + m)
    crop = fg[y0:y1, x0:x1].astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1))
    dilated = cv2.dilate(crop, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    mask = encode(dilated, x0, y0)
    if mask is None or mask.run_count > MAX_RUNS_PER_MASK:
        return None
    return mask


# --- Loading ------------------------------------------------------------------


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class Budget:
    """Shared run budget across one analysis load. Order of calls matters."""

    def __init__(self, total: int | None = None) -> None:
        self.remaining = MAX_RUNS_TOTAL if total is None else total

    def take(self, runs: int) -> bool:
        if runs > self.remaining:  # exhausted for good: later callers get nothing
            self.remaining = 0
            return False
        self.remaining -= runs
        return True


def parse_row(row: object, frame_width: int, frame_height: int) -> tuple[int, Mask] | None:
    """Validate one wire row. Returns ``(frame, Mask)`` or None if invalid.

    All checks happen before anything proportional to the mask area is
    allocated.
    """
    if not isinstance(row, list | tuple) or len(row) != 6:
        return None
    frame, x, y, w, h, runs = row
    if not all(_is_int(v) for v in (frame, x, y, w, h)):
        return None
    if frame < 0 or x < 0 or y < 0 or w < 1 or h < 1:
        return None
    if x + w > frame_width or y + h > frame_height:
        return None
    if not isinstance(runs, list | tuple) or not 2 <= len(runs) <= MAX_RUNS_PER_MASK:
        return None
    total = 0
    for i, r in enumerate(runs):
        if not _is_int(r) or r < 0 or (r == 0 and i > 0):
            return None
        total += r
    if total != w * h:
        return None
    return frame, Mask(x, y, w, h, tuple(runs))


def load_track_masks(
    rows: object,
    observed_frames: set[int],
    frame_width: int,
    frame_height: int,
    budget: Budget,
) -> dict[int, Mask]:
    """Parse a track's ``masks_v1`` list. Invalid entries become missing (box fallback).

    A frame listed twice is ambiguous, so neither entry is used.
    """
    if not isinstance(rows, list):
        return {}
    out: dict[int, Mask] = {}
    duplicates: set[int] = set()
    track_runs = 0
    dropped = 0
    exhausted = False
    for row in rows:
        if exhausted:
            dropped += 1
            continue
        parsed = parse_row(row, frame_width, frame_height)
        if parsed is None:
            dropped += 1
            continue
        frame, mask = parsed
        if frame not in observed_frames:
            dropped += 1
            continue
        if frame in out or frame in duplicates:
            duplicates.add(frame)
            out.pop(frame, None)
            dropped += 1
            continue
        if track_runs + mask.run_count > MAX_RUNS_PER_TRACK or not budget.take(mask.run_count):
            # Once a budget is exceeded every later entry is dropped too, so the
            # result never depends on which later masks happen to be small.
            exhausted = True
            dropped += 1
            continue
        track_runs += mask.run_count
        out[frame] = mask
    if dropped:
        log.warning("ignored %d invalid or over-budget mask entries; using box fallback", dropped)
    return out
