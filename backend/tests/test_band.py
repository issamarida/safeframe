"""The real band detector on synthetic band images (no ML, no downloads)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from safeframe.pipeline.band import (
    BAND_IDS,
    BAND_SEQUENCE,
    DICTIONARY,
    ArucoBandDetector,
    band_sheet_svg,
    band_strip,
    marker_bits,
)
from safeframe.pipeline.types import Box


def _scene(module_px: float, angle: float = 0.0, squash: float = 1.0, seed: int = 0):
    """A band strip pasted on textured background, rotated and foreshortened."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (0, 0), 3)
    strip = band_strip(16, markers=4)
    h, w = strip.shape[:2]
    s = module_px / 16
    pts = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) * s
    pts[:, 0] *= squash
    rot = cv2.getRotationMatrix2D((0, 0), angle, 1)[:, :2]
    dst = (pts @ rot.T + [320, 240]).astype(np.float32)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(strip, m, (640, 480), flags=cv2.INTER_AREA)
    cover = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (640, 480)) > 128
    bg[cover] = warped[cover]
    _, buf = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@pytest.mark.parametrize(("angle", "squash"), [(0, 1.0), (25, 0.7), (-40, 0.6), (90, 1.0)])
def test_detects_band_under_rotation_and_foreshortening(angle, squash):
    found = ArucoBandDetector().detect(_scene(4, angle, squash))
    assert found, "band not found"
    assert {d.marker_id for d in found} <= BAND_IDS
    for d in found:
        cx, cy = d.center
        assert abs(cx - 320) < 120 and abs(cy - 240) < 60
        assert d.box.contains(cx, cy) and d.side_px > 10


def test_zoomed_wrist_crop_finds_markers_too_small_for_the_full_frame():
    img = _scene(2.0, angle=10, squash=0.8, seed=3)
    detector = ArucoBandDetector()
    assert detector.detect(img) == []
    found = detector.detect(img, [Box(240, 200, 400, 280)])
    assert found and all(260 < d.center[0] < 380 for d in found)


def test_other_aruco_markers_are_ignored():
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    img = np.full((300, 600), 255, np.uint8)
    other = next(i for i in range(50) if i not in BAND_IDS)
    for x, marker_id in ((40, other), (340, BAND_SEQUENCE[0])):
        img[60:240, x : x + 180] = cv2.aruco.generateImageMarker(dictionary, marker_id, 180)
    found = ArucoBandDetector().detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert [d.marker_id for d in found] == [BAND_SEQUENCE[0]]


def test_plain_texture_is_not_a_band():
    rng = np.random.default_rng(7)
    for _ in range(5):
        noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        assert ArucoBandDetector().detect(noise) == []


def test_every_band_is_identical():
    # No serial numbers: the sequence (and so every band) carries no identity.
    a, b = band_strip(4), band_strip(4)
    assert np.array_equal(a, b)
    assert len(set(BAND_SEQUENCE)) == len(BAND_SEQUENCE)
    assert marker_bits(BAND_SEQUENCE[0]).shape == (6, 6)


def test_printable_sheet_is_valid_svg_at_true_size():
    svg = band_sheet_svg()
    root = ET.fromstring(svg.split("?>", 1)[1])
    assert root.get("width") == "210mm" and root.get("height") == "297mm"
    rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
    assert len(rects) > 200  # marker modules
    assert "cannot be used to track" in svg
