"""Region anonymisation: strong Gaussian blur, then greyscale.

A region is protected either by its silhouette mask (only pixels inside the
person's coverage change) or, when no usable mask exists, by its padded box.
Every region's effect is computed from the *unmodified* frame and written only
where that region's coverage is true, in plan order. Overlapping regions
therefore never restore original pixels and never blur the same source pixels
twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from safeframe.pipeline.masks import Mask
from safeframe.pipeline.types import Box

# The blurred region is reduced to roughly this many pixels along its longest
# side before blurring. Detail finer than ~1/BLUR_RESOLUTION of the region is
# destroyed regardless of resolution, so a face 900 px tall is just as
# unrecognisable as one 90 px tall.
BLUR_RESOLUTION = 24
# Fraction of the box width/height added on every side, so hair, hands and
# small tracker misalignments are still covered.
DEFAULT_PADDING = 0.08


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def gaussian_blur_strong(roi: np.ndarray) -> np.ndarray:
    """Gaussian-blur ``roi`` with a sigma proportional to its size.

    A direct Gaussian with sigma ~ size/12 needs kernels hundreds of pixels
    wide on HD video. Blurring a downscaled copy and upscaling is equivalent
    for this purpose (the result is a smooth, low-frequency field) and is
    orders of magnitude faster.
    """
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return roi.copy()
    scale = min(1.0, BLUR_RESOLUTION / max(h, w))
    small_w, small_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    k = _odd(max(3, round(max(small_w, small_h) / 4)))
    small = cv2.GaussianBlur(small, (k, k), 0)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # Second pass at full resolution removes any interpolation blockiness.
    k_full = _odd(max(3, round(max(w, h) / BLUR_RESOLUTION)))
    return cv2.GaussianBlur(up, (k_full, k_full), 0)


def to_greyscale(roi: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def pixel_rect(
    box: Box, width: int, height: int, padding: float = DEFAULT_PADDING
) -> tuple[int, int, int, int] | None:
    """Padded, clipped integer rectangle (x1, y1, x2, y2) or None if empty."""
    b = box.expand(padding).clip(width, height)
    x1, y1 = int(np.floor(b.x1)), int(np.floor(b.y1))
    x2, y2 = int(np.ceil(b.x2)), int(np.ceil(b.y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class ProtectedRegion:
    """One thing to obscure in one frame.

    ``box`` is always present: the person's box (or a manual rectangle). It
    sets the blur strength and is the coverage when ``mask`` is None.
    """

    box: Box
    mask: Mask | None = None
    kind: str = "track"  # "track" | "manual"
    track_id: int | None = None


def _mask_fits(mask: Mask, width: int, height: int) -> bool:
    return (
        mask.x >= 0
        and mask.y >= 0
        and mask.x + mask.width <= width
        and mask.y + mask.height <= height
    )


def coverage(
    region: ProtectedRegion, width: int, height: int, padding: float = DEFAULT_PADDING
) -> tuple[tuple[int, int, int, int], np.ndarray | None] | None:
    """Effect rectangle and coverage for ``region``.

    Returns ``(rect, mask)`` where ``rect`` is the ROI the blur is computed on
    and ``mask`` is a bool array placed at ``(region.mask.x, region.mask.y)``,
    or None meaning "the whole rect". Returns None if nothing can be covered.
    """
    rect = pixel_rect(region.box, width, height, padding)
    m = region.mask
    if m is None or not _mask_fits(m, width, height):
        return (rect, None) if rect is not None else None
    # Blur context: the padded box grown to include the whole silhouette, so
    # limbs outside the box are still protected and blur strength matches the
    # rectangular effect.
    if rect is None:
        rect = (m.x, m.y, m.x + m.width, m.y + m.height)
    else:
        rect = (
            min(rect[0], m.x),
            min(rect[1], m.y),
            max(rect[2], m.x + m.width),
            max(rect[3], m.y + m.height),
        )
    return rect, m.decode()


def anonymize_frame(
    frame: np.ndarray, regions: list[ProtectedRegion], padding: float = DEFAULT_PADDING
) -> np.ndarray:
    """Obscure every region in ``frame`` (in place). Returns ``frame``."""
    if not regions:
        return frame
    height, width = frame.shape[:2]
    source = frame.copy()
    for region in regions:
        cov = coverage(region, width, height, padding)
        if cov is None:
            continue
        (x1, y1, x2, y2), mask = cov
        effect = to_greyscale(gaussian_blur_strong(source[y1:y2, x1:x2]))
        if mask is None:
            frame[y1:y2, x1:x2] = effect
            continue
        m = region.mask
        assert m is not None
        ox, oy = m.x - x1, m.y - y1
        dst = frame[m.y : m.y + m.height, m.x : m.x + m.width]
        src = effect[oy : oy + m.height, ox : ox + m.width]
        dst[mask] = src[mask]
    return frame


def anonymize_regions(
    frame: np.ndarray, boxes: list[Box], padding: float = DEFAULT_PADDING
) -> np.ndarray:
    """Blur then greyscale every box in ``frame`` (in place). Returns ``frame``."""
    return anonymize_frame(frame, [ProtectedRegion(box=b) for b in boxes], padding)
