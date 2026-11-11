import numpy as np

from safeframe.pipeline.anonymize import (
    DEFAULT_PADDING,
    ProtectedRegion,
    anonymize_frame,
    anonymize_regions,
    gaussian_blur_strong,
    pixel_rect,
    to_greyscale,
)
from safeframe.pipeline.faces import assign_faces, crop_face
from safeframe.pipeline.masks import Mask, encode
from safeframe.pipeline.types import Box, FaceDetection

from .conftest import ellipse_mask, person


def _noise(h=200, w=300, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_region_is_blurred_and_greyscale_outside_untouched():
    frame = _noise()
    original = frame.copy()
    anonymize_regions(frame, [Box(50, 40, 150, 160)], padding=0.0)

    region = frame[40:160, 50:150].astype(int)
    assert np.array_equal(region[..., 0], region[..., 1])
    assert np.array_equal(region[..., 1], region[..., 2])
    # High-frequency noise should be almost entirely gone.
    assert region[..., 0].std() < original[40:160, 50:150].std() / 5

    mask = np.ones(frame.shape[:2], bool)
    mask[40:160, 50:150] = False
    assert np.array_equal(frame[mask], original[mask])


def test_padding_grows_region_and_clips_to_frame():
    assert pixel_rect(Box(0, 0, 100, 100), 300, 200, padding=0.1) == (0, 0, 110, 110)
    assert pixel_rect(Box(290, 190, 310, 210), 300, 200, padding=0.0) == (290, 190, 300, 200)
    assert pixel_rect(Box(400, 400, 500, 500), 300, 200) is None


def test_edge_and_tiny_regions_do_not_crash():
    frame = _noise()
    anonymize_regions(frame, [Box(-50, -50, 10, 10), Box(299, 199, 300, 200), Box(5, 5, 5, 5)])


def test_blur_strength_scales_with_region_size():
    big = gaussian_blur_strong(_noise(900, 400, seed=1))
    assert big.std() < 10


def test_face_goes_to_smallest_containing_person():
    adult = person(1, 0, 0, 200, 400)
    child = person(2, 50, 100, 150, 300)  # held by the adult, inside their box
    face = FaceDetection(Box(85, 110, 115, 140), 0.9)
    assert assign_faces([face], [adult, child]) == {2: face}


def test_face_of_child_passing_in_front_is_not_given_to_person_behind():
    # Regression: a smaller box further back used to win on area alone.
    behind = person(3, 300, 90, 370, 300)
    child = person(1, 290, 120, 380, 340)
    face = FaceDetection(Box(318, 122, 352, 156), 0.9)
    assert assign_faces([face], [behind, child]) == {1: face}


def test_ambiguous_face_is_dropped_rather_than_guessed():
    a = person(1, 100, 100, 200, 400)
    b = person(2, 102, 101, 202, 401)  # near-identical boxes
    face = FaceDetection(Box(135, 120, 165, 150), 0.9)
    assert assign_faces([face], [a, b]) == {}


def test_face_in_lower_body_is_ignored():
    p = person(1, 0, 0, 100, 300)
    hand_level = FaceDetection(Box(40, 250, 60, 270), 0.9)
    assert assign_faces([hand_level], [p]) == {}


def test_best_face_per_person():
    p = person(1, 0, 0, 100, 300)
    weak = FaceDetection(Box(10, 10, 30, 30), 0.6)
    strong = FaceDetection(Box(40, 10, 60, 30), 0.95)
    assert assign_faces([weak, strong], [p]) == {1: strong}


def test_crop_face_is_square_with_margin():
    crop = crop_face(_noise(), Box(100, 50, 140, 90), margin=0.25)
    assert crop is not None and crop.shape[:2] == (60, 60)


# --- silhouette compositing ----------------------------------------------------


def _coverage(mask: Mask, h: int, w: int) -> np.ndarray:
    cov = np.zeros((h, w), bool)
    cov[mask.y : mask.y + mask.height, mask.x : mask.x + mask.width] = mask.decode()
    return cov


def _l_shape_mask(box: Box) -> Mask:
    """A non-rectangular, non-convex 'person': an L that leaves the box's
    top-right and bottom-right areas empty."""
    x1, y1, x2, y2 = (int(v) for v in (box.x1, box.y1, box.x2, box.y2))
    crop = np.zeros((y2 - y1, x2 - x1), bool)
    crop[:, : (x2 - x1) // 3] = True  # vertical bar
    crop[-(y2 - y1) // 4 :, :] = True  # foot
    crop[-(y2 - y1) // 4 :, -4:] = False
    mask = encode(crop, x1, y1)
    assert mask is not None
    return mask


def test_silhouette_changes_inside_and_leaves_box_corners_untouched():
    frame = _noise()
    original = frame.copy()
    box = Box(80, 30, 200, 180)
    mask = _l_shape_mask(box)
    anonymize_frame(frame, [ProtectedRegion(box, mask)])
    cov = _coverage(mask, *frame.shape[:2])

    changed = np.any(frame != original, axis=2)
    assert changed[cov].mean() > 0.95  # protected interior changed
    assert not changed[~cov].any()  # nothing outside coverage changed (pre-encoding)
    # The box's top-right corner is background and must stay exactly as it was.
    assert np.array_equal(frame[31, 198], original[31, 198])
    inside = frame[cov].astype(int)
    assert np.array_equal(inside[:, 0], inside[:, 1])  # greyscale


def test_ellipse_mask_at_image_edge():
    frame = _noise()
    original = frame.copy()
    box = Box(-40, 120, 60, 200)
    mask = ellipse_mask(Box(-40, 120, 60, 260), 300, 200)
    anonymize_frame(frame, [ProtectedRegion(box, mask)])
    cov = _coverage(mask, 200, 300)
    changed = np.any(frame != original, axis=2)
    assert changed[cov].mean() > 0.95 and not changed[~cov].any()
    assert cov[199].any() and cov[:, 0].any()  # really touches both edges


def test_overlapping_regions_never_restore_or_double_blur():
    frame = _noise()
    source = frame.copy()
    a = ProtectedRegion(Box(40, 20, 160, 190), ellipse_mask(Box(40, 20, 160, 190), 300, 200))
    b = ProtectedRegion(Box(100, 30, 220, 190), ellipse_mask(Box(100, 30, 220, 190), 300, 200))
    anonymize_frame(frame, [a, b])
    cov_a = _coverage(a.mask, 200, 300)
    cov_b = _coverage(b.mask, 200, 300)
    both = cov_a & cov_b
    assert both.any()
    changed = np.any(frame != source, axis=2)
    assert changed[cov_a | cov_b].mean() > 0.95
    # In the overlap the later region's effect, computed from the untouched
    # source, is what remains: not a blur of an already blurred image.
    x1, y1, x2, y2 = pixel_rect(b.box, 300, 200, DEFAULT_PADDING)
    roi = source[y1:y2, x1:x2]
    effect = to_greyscale(gaussian_blur_strong(roi))
    full = np.zeros_like(frame)
    full[y1:y2, x1:x2] = effect
    assert np.array_equal(frame[both], full[both])


def test_kept_region_overlap_cannot_unblur_and_manual_rect_is_rectangular():
    frame = _noise()
    original = frame.copy()
    person_box = Box(40, 20, 160, 190)
    blur = ProtectedRegion(person_box, ellipse_mask(person_box, 300, 200))
    manual = ProtectedRegion(Box(200, 20, 260, 80), kind="manual")
    # Kept tracks are simply absent from the plan; manual regions are rectangles.
    anonymize_frame(frame, [blur, manual])
    x1, y1, x2, y2 = pixel_rect(manual.box, 300, 200)
    changed = np.any(frame != original, axis=2)
    assert changed[y1:y2, x1:x2].mean() > 0.95
    cov = _coverage(blur.mask, 200, 300)
    assert changed[cov].mean() > 0.95


def test_box_fallback_matches_old_rectangular_behaviour():
    a, b = _noise(), _noise()
    anonymize_regions(a, [Box(50, 40, 150, 160)])
    anonymize_frame(b, [ProtectedRegion(Box(50, 40, 150, 160), None)])
    assert np.array_equal(a, b)


def test_mask_outside_frame_falls_back_to_box():
    frame = _noise()
    original = frame.copy()
    bad = Mask(290, 190, 20, 20, (0, 400))  # extends past a 300x200 frame
    anonymize_frame(frame, [ProtectedRegion(Box(250, 150, 290, 190), bad)])
    x1, y1, x2, y2 = pixel_rect(Box(250, 150, 290, 190), 300, 200)
    changed = np.any(frame != original, axis=2)
    assert changed[y1:y2, x1:x2].mean() > 0.95
