"""Real-model smoke tests (opt-in: ``pytest -m models``).

Downloads YOLO (segmentation + pose), YuNet and the age model, then runs the
full pipeline on sample photos bundled with the ultralytics package. A
SafeFrame Band is pasted onto one person's wrist at a realistic size, found
with the pose model's wrist keypoints. No footage is stored in the repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from .conftest import requires_ffmpeg

pytestmark = [pytest.mark.models, requires_ffmpeg]

# Child band: 20 mm wide = 9 modules (6 marker + 2 x 1.5 quiet).
MODULE_MM = 20 / 9


def _asset(name: str) -> Path:
    ultralytics = pytest.importorskip("ultralytics")
    image = Path(ultralytics.__file__).parent / "assets" / name
    if not image.exists():
        pytest.skip(f"ultralytics sample image {name} not available")
    return image


def _wear_band(img: np.ndarray, person_rank: int, wrist: int = 10) -> np.ndarray:
    """Paste a band across the forearm of the ``person_rank``-th person from the left,
    sized for a ~1.75 m person."""
    from ultralytics import YOLO

    from safeframe.models import yolo_weights_path
    from safeframe.pipeline.band import band_strip

    raw = YOLO(str(yolo_weights_path("yolo11n-pose.pt"))).predict(img, verbose=False)[0]
    boxes = raw.boxes.xyxy.cpu().numpy()
    i = int(np.argsort(boxes[:, 0])[person_rank - 1])
    keypoints = raw.keypoints.xy.cpu().numpy()[i]
    wx, wy = keypoints[wrist]
    ex, ey = keypoints[wrist - 2]  # the elbow on the same arm

    module_px = MODULE_MM * (boxes[i][3] - boxes[i][1]) / 1750
    strip = band_strip(16, markers=3)
    h, w = strip.shape[:2]
    angle = np.degrees(np.arctan2(wy - ey, wx - ex)) + 90
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, module_px / 16)
    cx, cy = wx + (ex - wx) * 0.12, wy + (ey - wy) * 0.12
    m[:, 2] += (cx - w / 2, cy - h / 2)
    out = img.copy()
    size = (img.shape[1], img.shape[0])
    warped = cv2.warpAffine(strip, m, size, flags=cv2.INTER_AREA)
    cover = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, size) > 128
    out[cover] = warped[cover]
    return out


@pytest.fixture(scope="module")
def street_photo(tmp_path_factory) -> Path:
    """bus.jpg at ~14 MP (phone resolution) with a band on the second person."""
    img = cv2.imread(str(_asset("bus.jpg")))
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    path = tmp_path_factory.mktemp("photo") / "street.jpg"
    cv2.imwrite(str(path), _wear_band(img, 2), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def test_band_wearer_in_a_photo_is_the_only_person_blurred(street_photo, tmp_path):
    from safeframe.pipeline.analyze import analyze_photo
    from safeframe.pipeline.context import CaptureContext
    from safeframe.pipeline.factory import ModelBundle
    from safeframe.pipeline.policy import Decision, DecisionPolicy
    from safeframe.pipeline.render import plan_render, render_photo
    from safeframe.pipeline.types import BandVia

    bundle = ModelBundle()
    analysis = analyze_photo(street_photo, bundle.components(photo=True)).analysis
    policy = DecisionPolicy.for_analysis(analysis)
    decisions = {t.track_id: policy.classify(t) for t in analysis.tracks}
    assert decisions.pop(2) is Decision.WEARER, decisions
    assert set(decisions.values()) == {Decision.VISIBLE}
    wearer = next(t for t in analysis.tracks if t.track_id == 2)
    assert wearer.band_hits[0].via is BandVia.WRIST
    assert not any(t.samples for t in analysis.tracks)  # no age estimation needed

    out = tmp_path / "shared.jpg"
    plan = plan_render(analysis, policy)
    assert plan.blurred_tracks == [2] and plan.coverage_stats()["mask"] == 1
    render_photo(street_photo, out, analysis, plan)
    assert cv2.imread(str(out)).shape == cv2.imread(str(street_photo)).shape

    # The same photo taken inside a bank: an adult's band is not honoured.
    guarded = analyze_photo(
        street_photo, bundle.components(photo=True), CaptureContext(guarded=True)
    ).analysis
    policy = DecisionPolicy.for_analysis(guarded)
    wearer = next(t for t in guarded.tracks if t.track_id == 2)
    assert wearer.samples and wearer.p_minor < 0.35
    assert policy.classify(wearer) is Decision.NOT_HONOURED
    assert [t.track_id for t in guarded.tracks if t.samples] == [2]


@pytest.fixture(scope="module")
def wearer_clip(street_photo, tmp_path_factory) -> Path:
    """2 s portrait 4K pan across the street photo (band on the second person).

    4K keeps the band at ~2.5 px per module; at 1080p a full-body person this
    far away is below the decodable size (see docs/TECHNICAL.md, detection range).
    """
    out = tmp_path_factory.mktemp("clip") / "wearer.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-i", str(street_photo),
            "-f", "lavfi", "-i", "sine=frequency=330:duration=2",
            "-vf", "crop=2160:3840:'min(t*100,200)':'min(t*120,240)'",
            "-t", "2", "-r", "25", "-c:v", "libx264", "-crf", "14", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )  # fmt: skip
    return out


def test_pipeline_on_real_models(wearer_clip, tmp_path):
    from safeframe.pipeline.analyze import analyze_video
    from safeframe.pipeline.factory import ModelBundle
    from safeframe.pipeline.policy import Decision, DecisionPolicy
    from safeframe.pipeline.render import plan_render, render_video
    from safeframe.video.io import probe

    analysis = analyze_video(wearer_clip, ModelBundle().components()).analysis
    policy = DecisionPolicy.for_analysis(analysis)
    frames = analysis.video.frame_count

    # People visible throughout keep a single identity for every frame.
    full_length = [t for t in analysis.tracks if len(t.boxes) == frames]
    assert len(full_length) >= 2, [(t.track_id, len(t.boxes)) for t in analysis.tracks]
    wearers = [t for t in analysis.tracks if policy.classify(t) is Decision.WEARER]
    assert len(wearers) == 1 and wearers[0] in full_length
    assert wearers[0].band_confirmed >= 10
    others = [t for t in analysis.tracks if t is not wearers[0]]
    assert all(policy.classify(t) is Decision.VISIBLE for t in others)

    out = tmp_path / "out.mp4"
    plan = plan_render(analysis, policy)
    render_video(wearer_clip, out, analysis, plan)
    src, dst = probe(wearer_clip), probe(out)
    assert (dst.width, dst.height, dst.fps_rational) == (src.width, src.height, src.fps_rational)
    assert dst.has_audio
    assert plan.coverage_stats()["mask"] >= 0.8 * frames

    for track in full_length:
        assert len(track.masks) >= 0.8 * frames, (track.track_id, len(track.masks))
        for f, mask in track.masks.items():
            box = track.boxes[f]
            cx = mask.x + mask.width / 2
            assert box.x1 - 30 <= cx <= box.x2 + 30
            assert mask.decode().any()


def test_box_only_models_still_work(wearer_clip):
    from safeframe.pipeline.analyze import Components, analyze_video
    from safeframe.pipeline.detection import DEFAULT_YOLO_WEIGHTS, YoloPersonTracker
    from safeframe.pipeline.factory import ModelBundle

    assert DEFAULT_YOLO_WEIGHTS.endswith("-seg.pt")
    bundle = ModelBundle()
    parts = bundle.components()
    result = analyze_video(
        wearer_clip,
        Components(
            YoloPersonTracker("yolo11n.pt"),
            parts.band_detector,
            parts.face_detector,
            parts.age_estimator,
            parts.wrist_locator,
        ),
    )
    assert result.analysis.tracks and result.analysis.mask_count == 0
    assert any(t.wears_band for t in result.analysis.tracks)
