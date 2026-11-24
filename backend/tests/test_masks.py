"""Silhouette masks: wire format, validation, budgets, quality gates, model adapter."""

from __future__ import annotations

import json

import numpy as np
import pytest

from safeframe.pipeline import masks as M
from safeframe.pipeline.detection import observations_from_result
from safeframe.pipeline.masks import Budget, Mask, build_coverage, encode, load_track_masks
from safeframe.pipeline.types import Analysis, Box, Track, VideoInfo

from .conftest import ellipse_raw

VIDEO = VideoInfo(320, 240, 25.0, "25/1", 50, 2.0, False)


def _shape_with_hole_and_island() -> np.ndarray:
    crop = np.zeros((12, 10), bool)
    crop[1:9, 1:7] = True  # body
    crop[4:6, 3:5] = False  # hole
    crop[10, 8] = True  # disconnected pixel
    return crop


# --- encode / decode ----------------------------------------------------------


def test_round_trip_preserves_hole_and_disconnected_pixel():
    crop = _shape_with_hole_and_island()
    mask = encode(crop, 100, 50)
    assert mask is not None
    # Only the occupied crop is stored.
    assert (mask.x, mask.y, mask.width, mask.height) == (101, 51, 8, 10)
    assert sum(mask.runs) == mask.width * mask.height
    full = np.zeros_like(crop)
    full[1:11, 1:9] = mask.decode()
    np.testing.assert_array_equal(full, crop)


def test_crop_starting_in_foreground_has_zero_first_run():
    crop = np.ones((3, 3), bool)
    crop[2, 2] = False
    mask = encode(crop, 0, 0)
    assert mask is not None and mask.runs[0] == 0
    np.testing.assert_array_equal(mask.decode(), crop)


def test_empty_crop_encodes_to_none():
    assert encode(np.zeros((5, 5), bool), 0, 0) is None


def test_track_json_round_trip_with_edge_clipped_mask():
    # A person cut off by the left/top frame edges.
    box = Box(0, 0, 60, 120)
    raw = ellipse_raw(Box(-30, -20, 60, 120), VIDEO.width, VIDEO.height)
    mask = build_coverage(raw, box)
    assert mask is not None and mask.x == 0 and mask.y == 0
    track = Track(track_id=7, boxes={3: box, 4: box}, interpolated={4})
    track.masks = {3: mask}
    analysis = Analysis(video=VIDEO, tracks=[track], settings={})
    loaded = Analysis.from_dict(json.loads(json.dumps(analysis.to_dict())))
    assert loaded.tracks[0].masks == {3: mask}
    assert loaded.tracks[0].mask_for(3) == mask
    assert loaded.tracks[0].mask_for(4) is None


def test_legacy_box_only_json_loads_without_masks():
    legacy = {
        "video": VIDEO.to_dict(),
        "settings": {},
        "tracks": [
            {"track_id": 1, "boxes": [[0, 1, 2, 30, 40, 0], [1, 1, 2, 30, 40, 1]], "samples": []}
        ],
    }
    analysis = Analysis.from_dict(legacy)
    assert analysis.tracks[0].masks == {}
    assert "masks_v1" not in analysis.tracks[0].to_dict()


# --- validation -----------------------------------------------------------------


def _valid_row(frame=0):
    return [frame, 10, 10, 4, 2, [1, 3, 2, 2]]


@pytest.mark.parametrize(
    "row",
    [
        "nope",
        [0, 10, 10, 4, 2],  # too short
        [0, 10.0, 10, 4, 2, [1, 3, 2, 2]],  # float origin
        [True, 10, 10, 4, 2, [1, 3, 2, 2]],  # bool frame
        [-1, 10, 10, 4, 2, [1, 3, 2, 2]],  # negative frame
        [0, -1, 10, 4, 2, [1, 3, 2, 2]],  # outside frame
        [0, 318, 10, 4, 2, [1, 3, 2, 2]],  # past right edge
        [0, 10, 239, 4, 2, [1, 3, 2, 2]],  # past bottom edge
        [0, 10, 10, 0, 2, [0]],  # zero width
        [0, 10, 10, 4, 2, [1, 3, 2, 1]],  # runs sum too small
        [0, 10, 10, 4, 2, [1, 3, 2, 3]],  # runs sum too large
        [0, 10, 10, 4, 2, [1, 0, 5, 2]],  # zero run after the first
        [0, 10, 10, 4, 2, [1, -3, 6, 4]],  # negative run
        [0, 10, 10, 4, 2, [8]],  # background only
        [0, 10, 10, 4, 2, [1.5, 6.5]],  # non-integer runs
        [0, 10, 10, 4, 2, "1,3,2,2"],  # runs not a list
    ],
)
def test_malformed_rows_are_rejected(row):
    assert M.parse_row(row, VIDEO.width, VIDEO.height) is None


def test_too_many_runs_rejected_before_summing():
    runs = [1] * (M.MAX_RUNS_PER_MASK + 2)
    assert M.parse_row([0, 0, 0, 1, len(runs), runs], 10_000, 10_000) is None


def test_invalid_entries_become_missing_not_fatal():
    rows = [
        _valid_row(0),
        [1, 10, 10, 4, 2, [1, 3, 2, 9]],  # bad sum
        _valid_row(2),  # frame 2 is interpolated
        _valid_row(9),  # frame 9 has no box
        _valid_row(3),
        _valid_row(3),  # duplicate: ambiguous, so neither is used
    ]
    out = load_track_masks(rows, {0, 1, 3}, VIDEO.width, VIDEO.height, Budget())
    assert set(out) == {0}
    assert out[0].decode().tolist() == [[False, True, True, True], [False, False, True, True]]


def test_track_and_total_budgets_drop_the_rest(monkeypatch):
    monkeypatch.setattr(M, "MAX_RUNS_PER_TRACK", 10)
    rows = [_valid_row(f) for f in range(5)]  # 4 runs each
    out = load_track_masks(rows, set(range(5)), VIDEO.width, VIDEO.height, Budget())
    assert set(out) == {0, 1}  # 8 runs fit, the third would exceed 10

    budget = Budget(total=6)
    a = load_track_masks(rows, set(range(5)), VIDEO.width, VIDEO.height, budget)
    b = load_track_masks(rows, set(range(5)), VIDEO.width, VIDEO.height, budget)
    assert set(a) == {0} and b == {}


def test_analysis_load_shares_one_budget(monkeypatch):
    monkeypatch.setattr(M, "MAX_RUNS_TOTAL", 4)
    track = {"track_id": 1, "boxes": [[0, 10, 10, 20, 20, 0]], "masks_v1": [_valid_row(0)]}
    data = {"video": VIDEO.to_dict(), "settings": {}, "tracks": [track, {**track, "track_id": 2}]}
    loaded = Analysis.from_dict(data)
    assert [len(t.masks) for t in loaded.tracks] == [1, 0]


# --- coverage construction and quality gates ---------------------------------------


def test_coverage_is_dilated_and_keeps_limbs_outside_the_box():
    box = Box(100, 40, 160, 200)
    raw = ellipse_raw(box, VIDEO.width, VIDEO.height)
    raw[100:104, 160:190] = 1  # an arm reaching past the box edge
    mask = build_coverage(raw, box)
    assert mask is not None
    cov = np.zeros((VIDEO.height, VIDEO.width), bool)
    cov[mask.y : mask.y + mask.height, mask.x : mask.x + mask.width] = mask.decode()
    assert cov[raw > 0].all()  # every model pixel is covered
    assert cov[102, 189 + M.margin_px(box) - 1]  # arm tip plus margin
    assert cov.sum() > (raw > 0).sum()  # margin added
    assert not cov[41, 101]  # background at the box corner stays visible


def test_margin_scales_with_person_and_is_clamped():
    assert M.margin_px(Box(0, 0, 10, 20)) == M.MIN_MARGIN_PX
    assert M.margin_px(Box(0, 0, 100, 300)) == 9
    assert M.margin_px(Box(0, 0, 1000, 3000)) == M.MAX_MARGIN_PX


def test_untrustworthy_masks_are_missing():
    box = Box(100, 40, 160, 200)
    empty = np.zeros((VIDEO.height, VIDEO.width), np.uint8)
    assert build_coverage(empty, box) is None
    tiny = empty.copy()
    tiny[100:104, 120:124] = 1  # 16 px: noise
    assert build_coverage(tiny, box) is None
    hand = empty.copy()
    hand[100:120, 120:140] = 1  # 400 px in a 9600 px box: implausibly small
    assert build_coverage(hand, box) is None
    displaced = ellipse_raw(Box(0, 0, 60, 160), VIDEO.width, VIDEO.height)
    assert build_coverage(displaced, box) is None
    assert build_coverage(ellipse_raw(box, VIDEO.width, VIDEO.height), Box(5, 5, 5, 5)) is None
    assert build_coverage(np.zeros((4, 4, 3)), box) is None


def test_worst_case_fragmentation_is_rejected():
    checker = (np.indices((240, 320)).sum(axis=0) % 2).astype(np.uint8)
    # Dilation would merge a checkerboard, so test the encoder budget directly.
    mask = encode(checker.astype(bool), 0, 0)
    assert mask is not None and mask.run_count > M.MAX_RUNS_PER_MASK
    assert M.parse_row(mask.to_row(0), 320, 240) is None


def test_fragmented_coverage_over_budget_is_missing(monkeypatch):
    box = Box(100, 40, 160, 200)
    raw = ellipse_raw(box, VIDEO.width, VIDEO.height)
    good = build_coverage(raw, box)
    assert good is not None
    monkeypatch.setattr(M, "MAX_RUNS_PER_MASK", good.run_count - 1)
    assert build_coverage(raw, box) is None


# --- model adapter: ID-to-mask alignment ---------------------------------------------


class _T:
    """Minimal stand-in for an Ultralytics tensor."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, ids, conf):
        self.xyxy, self.id, self.conf = _T(xyxy), _T(ids), _T(conf)


class _Masks:
    def __init__(self, data):
        self.data = _T(data)


class _Result:
    def __init__(self, xyxy, ids, conf, masks):
        self.boxes = _Boxes(xyxy, ids, conf)
        self.masks = None if masks is None else _Masks(masks)


def _tracked_rows():
    """Three people; the tracker emits them in a different order than detected."""
    boxes = {5: Box(10, 20, 70, 220), 2: Box(120, 30, 180, 230), 9: Box(230, 10, 300, 200)}
    order = [9, 5, 2]  # tracked row order
    stack = np.stack([ellipse_raw(boxes[t], VIDEO.width, VIDEO.height) for t in order])
    xyxy = [[boxes[t].x1, boxes[t].y1, boxes[t].x2, boxes[t].y2] for t in order]
    return boxes, order, xyxy, stack


def test_adapter_keeps_masks_aligned_with_reordered_tracked_rows():
    boxes, order, xyxy, stack = _tracked_rows()
    result = _Result(xyxy, order, [0.9, 0.8, 0.7], stack)
    obs = observations_from_result(result, (VIDEO.height, VIDEO.width, 3))
    assert [o.track_id for o in obs] == order
    for o in obs:
        assert o.mask is not None
        cx, cy = boxes[o.track_id].center
        # The mask attached to each id is centred on that id's own box.
        assert o.mask.x <= cx <= o.mask.x + o.mask.width
        assert o.mask.y <= cy <= o.mask.y + o.mask.height
        assert o.box == boxes[o.track_id]


def test_adapter_keeps_people_when_masks_are_unusable():
    _boxes, order, xyxy, stack = _tracked_rows()
    # Wrong spatial size (e.g. letterboxed inference resolution): never stretched.
    small = stack[:, :120, :160]
    obs = observations_from_result(
        _Result(xyxy, order, [0.9] * 3, small), (VIDEO.height, VIDEO.width, 3)
    )
    assert [o.track_id for o in obs] == order and all(o.mask is None for o in obs)
    # Row count mismatch.
    obs = observations_from_result(
        _Result(xyxy, order, [0.9] * 3, stack[:2]), (VIDEO.height, VIDEO.width, 3)
    )
    assert len(obs) == 3 and all(o.mask is None for o in obs)
    # Detection-only checkpoint.
    obs = observations_from_result(
        _Result(xyxy, order, [0.9] * 3, None), (VIDEO.height, VIDEO.width, 3)
    )
    assert len(obs) == 3 and all(o.mask is None for o in obs)
    # One row's mask is empty: only that person falls back to the box.
    stack = stack.copy()
    stack[1] = 0
    obs = observations_from_result(
        _Result(xyxy, order, [0.9] * 3, stack), (VIDEO.height, VIDEO.width, 3)
    )
    assert [o.mask is None for o in obs] == [False, True, False]


def test_adapter_without_tracks_returns_nothing():
    r = _Result(np.zeros((0, 4)), [], [], None)
    r.boxes.id = None
    assert observations_from_result(r, (240, 320, 3)) == []


def test_mask_is_hashable_and_comparable():
    a = Mask(0, 0, 2, 1, (0, 2))
    assert a == Mask(0, 0, 2, 1, (0, 2))
    assert len({a, Mask(0, 0, 2, 1, (0, 2))}) == 1
