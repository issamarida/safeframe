"""The SafeFrame Band: a wristband that asks cameras "please don't show me".

Band specification v1
---------------------

* A white strip printed with a repeating sequence of square fiducial markers
  from OpenCV's ArUco ``DICT_4X4_50`` dictionary, in the order
  :data:`BAND_SEQUENCE`. Each marker is a 4x4 bit code inside a one-module
  black border, so a printed marker is 6x6 modules, surrounded by a white
  quiet zone at least one module wide.
* **Every band prints the same sequence.** There is no serial number, so a
  band says only "this person opted out", never who they are, and two
  sightings of a band can't be linked to the same wearer. A band can't be used
  to find or follow a child.
* Fiducials are used, not a learned "bracelet classifier", on purpose: the
  pattern either decodes or it doesn't, so a match is explainable, cheap
  enough for every camera frame, and needs no model weights. Only the four
  reserved ids count, and matching codes are rejected unless they decode with
  few bit errors, which keeps chance matches on clothing and posters rare.
* A band is printable at home (:func:`band_sheet_svg`). A privacy request
  should not cost money.

Detection runs on the full frame, then again on upscaled crops around
wrists located by a pose model. Markers that are too small to decode in the
full frame are often still readable in a zoomed wrist crop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

import cv2
import numpy as np

from safeframe.pipeline.types import Box

SPEC_VERSION = 1
DICTIONARY = cv2.aruco.DICT_4X4_50
# Reserved marker ids, printed in this order around the band.
BAND_SEQUENCE: tuple[int, ...] = (7, 19, 28, 42)
BAND_IDS = frozenset(BAND_SEQUENCE)
# 4x4 data bits + a one-module black border on each side.
MARKER_MODULES = 6

# Below this side length (source pixels) a reported marker is more likely to
# be a chance match on texture than a real, decodable band marker.
MIN_MARKER_SIDE_PX = 7.0
# Wrist crops are upscaled so their longer side is about this many pixels.
ZOOM_TARGET_PX = 480
MAX_ZOOM = 4.0


@dataclass(frozen=True, slots=True)
class BandDetection:
    """One decoded band marker in one frame."""

    marker_id: int
    # Marker corners (x, y) in frame pixels, clockwise from the top-left.
    corners: tuple[tuple[float, float], ...]

    @property
    def center(self) -> tuple[float, float]:
        xs, ys = zip(*self.corners, strict=True)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def side_px(self) -> float:
        pts = np.asarray(self.corners)
        return float(np.mean(np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)))

    @property
    def box(self) -> Box:
        xs, ys = zip(*self.corners, strict=True)
        return Box(min(xs), min(ys), max(xs), max(ys))


def _parameters(min_perimeter_rate: float) -> cv2.aruco.DetectorParameters:
    p = cv2.aruco.DetectorParameters()
    # Candidate count (and run time) explodes below ~1 % of the image size, so
    # the full-frame pass looks for markers of ordinary size and tiny markers
    # are left to the zoomed wrist crops.
    p.minMarkerPerimeterRate = min_perimeter_rate
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 33
    p.adaptiveThreshWinSizeStep = 6
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # Stricter than the default 0.6: fewer corrected bits means fewer chance
    # matches on fabric, text and tiles.
    p.errorCorrectionRate = 0.35
    return p


# Smallest marker side the full-frame pass looks for, and the floor on the
# perimeter rate that keeps it fast on large frames.
FULL_FRAME_MIN_SIDE_PX = 10.0
MIN_PERIMETER_RATE = 0.01


class ArucoBandDetector:
    """Finds SafeFrame Band markers in a BGR frame.

    Stateless and cheap (tens of milliseconds per HD frame), so it can run on
    most frames and be shared between jobs.
    """

    def __init__(self, min_side_px: float = MIN_MARKER_SIDE_PX) -> None:
        self.min_side_px = min_side_px
        self._dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
        self._detectors: dict[float, cv2.aruco.ArucoDetector] = {}

    def _detector_for(self, img: np.ndarray, min_side: float) -> cv2.aruco.ArucoDetector:
        rate = round(max(MIN_PERIMETER_RATE, 4 * min_side / max(img.shape[:2])), 3)
        if rate not in self._detectors:
            self._detectors[rate] = cv2.aruco.ArucoDetector(self._dictionary, _parameters(rate))
        return self._detectors[rate]

    def detect(self, frame: np.ndarray, rois: Sequence[Box] = ()) -> list[BandDetection]:
        """Markers in the full frame, plus any found only in zoomed ``rois``."""
        found = self._detect(frame, 0.0, 0.0, 1.0, max(self.min_side_px, FULL_FRAME_MIN_SIDE_PX))
        h, w = frame.shape[:2]
        for roi in rois:
            r = roi.clip(w, h)
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(np.ceil(r.x2)), int(np.ceil(r.y2))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            scale = min(MAX_ZOOM, ZOOM_TARGET_PX / max(x2 - x1, y2 - y1))
            if scale <= 1.05:
                continue  # the full-frame pass already saw this at full detail
            crop = cv2.resize(
                frame[y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            for det in self._detect(crop, x1, y1, scale, self.min_side_px * scale):
                if not any(_same_marker(det, other) for other in found):
                    found.append(det)
        return found

    def _detect(
        self, img: np.ndarray, ox: float, oy: float, scale: float, min_side: float
    ) -> list[BandDetection]:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids, _ = self._detector_for(grey, min_side).detectMarkers(grey)
        if ids is None:
            return []
        out = []
        for quad, marker_id in zip(corners, ids.ravel().tolist(), strict=True):
            if marker_id not in BAND_IDS:
                continue
            pts = tuple((ox + float(x) / scale, oy + float(y) / scale) for x, y in quad[0])
            det = BandDetection(int(marker_id), pts)
            if det.side_px >= self.min_side_px:
                out.append(det)
        return out


def _same_marker(a: BandDetection, b: BandDetection) -> bool:
    ax, ay = a.center
    bx, by = b.center
    return a.marker_id == b.marker_id and np.hypot(ax - bx, ay - by) < 0.5 * max(
        a.side_px, b.side_px
    )


# --- Printable band -----------------------------------------------------------


def marker_bits(marker_id: int) -> np.ndarray:
    """The 6x6 module grid of a marker: True = black."""
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_MODULES)
    return img < 128


def band_strip(module_px: int, markers: int = 8, quiet_modules: int = 2) -> np.ndarray:
    """A flat band segment as a BGR image (used for tests and demos).

    Markers are separated, and the strip edged, by ``quiet_modules`` white
    modules.
    """
    q = quiet_modules
    height = (MARKER_MODULES + 2 * q) * module_px
    width = (markers * (MARKER_MODULES + q) + q) * module_px
    strip = np.full((height, width), 255, np.uint8)
    for i in range(markers):
        bits = marker_bits(BAND_SEQUENCE[i % len(BAND_SEQUENCE)])
        tile = np.where(bits, 0, 255).astype(np.uint8)
        tile = cv2.resize(tile, None, fx=module_px, fy=module_px, interpolation=cv2.INTER_NEAREST)
        x = (q + i * (MARKER_MODULES + q)) * module_px
        strip[q * module_px : q * module_px + tile.shape[0], x : x + tile.shape[1]] = tile
    return cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)


@dataclass(frozen=True, slots=True)
class BandSize:
    name: str
    label: str
    circumference_mm: float
    width_mm: float


BAND_SIZES = {
    "child": BandSize("child", "Child (wrist up to 15 cm)", 150, 20),
    "adult": BandSize("adult", "Adult (wrist up to 19 cm)", 190, 24),
}
_QUIET_MODULES = 1.5
_OVERLAP_MM = 25  # fastening tab


def band_svg(size: str = "child", x_mm: float = 0, y_mm: float = 0) -> str:
    """One band as an SVG ``<g>`` fragment, in millimetres."""
    spec = BAND_SIZES[size]
    module = spec.width_mm / (MARKER_MODULES + 2 * _QUIET_MODULES)
    pitch = (MARKER_MODULES + _QUIET_MODULES) * module
    length = spec.circumference_mm + _OVERLAP_MM
    count = int((spec.circumference_mm - _QUIET_MODULES * module) // pitch)
    parts = [
        f'<g transform="translate({x_mm:.2f} {y_mm:.2f})">',
        f'<rect width="{length:.2f}" height="{spec.width_mm:.2f}" rx="2" fill="#fff" '
        'stroke="#999" stroke-width="0.2" stroke-dasharray="1 1"/>',
    ]
    for i in range(count):
        bits = marker_bits(BAND_SEQUENCE[i % len(BAND_SEQUENCE)])
        ox = (_QUIET_MODULES + i * (MARKER_MODULES + _QUIET_MODULES)) * module
        oy = _QUIET_MODULES * module
        for r, c in zip(*np.nonzero(bits), strict=True):
            parts.append(
                f'<rect x="{ox + c * module:.3f}" y="{oy + r * module:.3f}" '
                f'width="{module + 0.02:.3f}" height="{module + 0.02:.3f}"/>'
            )
    tab_x = spec.circumference_mm + 2
    parts.append(
        f'<text x="{tab_x:.1f}" y="{spec.width_mm / 2 - 1:.1f}" font-size="3.2" '
        'font-family="sans-serif">SafeFrame</text>'
        f'<text x="{tab_x:.1f}" y="{spec.width_mm / 2 + 3:.1f}" font-size="2.2" '
        'font-family="sans-serif">glue here</text>'
    )
    parts.append("</g>")
    return "".join(parts)


def band_sheet_svg(sizes: Sequence[str] = ("child", "adult"), copies: int = 2) -> str:
    """An A4 sheet of printable bands at true size (print at 100 %, no scaling)."""
    page_w, page_h, margin = 210, 297, 12
    y = margin + 26
    body = []
    for size in sizes:
        spec = BAND_SIZES[size]
        body.append(
            f'<text x="{margin}" y="{y - 3:.1f}" font-size="4" font-family="sans-serif">'
            f"{escape(spec.label)}</text>"
        )
        for _ in range(copies):
            body.append(band_svg(size, margin, y))
            y += spec.width_mm + 8
        y += 6
    header = (
        f'<text x="{margin}" y="{margin + 4}" font-size="7" font-weight="bold" '
        'font-family="sans-serif">SafeFrame Band</text>'
        f'<text x="{margin}" y="{margin + 11}" font-size="3.4" font-family="sans-serif">'
        "Print at 100 % on matte paper or sticker film, cut along the dashed line and wrap "
        "with the markers facing out.</text>"
        f'<text x="{margin}" y="{margin + 16}" font-size="3.4" font-family="sans-serif">'
        "Every band is identical: it carries no name or number and cannot be used to track "
        f"anyone. Spec v{SPEC_VERSION}.</text>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}mm" height="{page_h}mm" '
        f'viewBox="0 0 {page_w} {page_h}">'
        f'<rect width="{page_w}" height="{page_h}" fill="#fff"/>{header}{"".join(body)}</svg>\n'
    )
