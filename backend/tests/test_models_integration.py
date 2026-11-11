"""Real-model smoke test (opt-in: ``pytest -m models``).

Downloads YOLO, YuNet and the age model, builds a short panning clip from a
sample photo bundled with the ultralytics package (two adults), and checks the
full pipeline end to end. No footage is stored in the repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .conftest import requires_ffmpeg

pytestmark = [pytest.mark.models, requires_ffmpeg]


@pytest.fixture(scope="module")
def adults_clip(tmp_path_factory) -> Path:
    ultralytics = pytest.importorskip("ultralytics")
    image = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
    if not image.exists():
        pytest.skip("ultralytics sample image not available")
    out = tmp_path_factory.mktemp("clip") / "adults.mp4"
    # 3 s slow pan across the photo so the tracker has motion to follow.
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:duration=3",
            "-vf",
            "scale=1280:720,crop=1120:630:'min(t*40,160)':'min(t*20,90)',scale=1120:630",
            "-t",
            "3",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
    )
    return out


def test_pipeline_on_real_models(adults_clip, tmp_path):
    from safeframe.pipeline.analyze import analyze_video
    from safeframe.pipeline.factory import ModelBundle
    from safeframe.pipeline.policy import DecisionPolicy
    from safeframe.pipeline.render import plan_render, render_video
    from safeframe.pipeline.types import TrackStatus
    from safeframe.video.io import probe

    bundle = ModelBundle()
    result = analyze_video(
        adults_clip, bundle.new_tracker(), bundle.new_face_detector(), bundle.age_estimator()
    )
    analysis = result.analysis
    policy = DecisionPolicy()
    frames = analysis.video.frame_count

    # The two people visible throughout keep a single identity for every frame
    # (a third, partly visible person may enter as the camera pans).
    full_length = [t for t in analysis.tracks if len(t.boxes) == frames]
    assert len(full_length) == 2, [(t.track_id, len(t.boxes)) for t in analysis.tracks]

    # Everyone in the photo is an adult; nobody may be classified as a minor.
    for track in analysis.tracks:
        assert policy.classify(track) is not TrackStatus.MINOR, track.track_id
    for track in full_length:
        assert len(track.samples) >= policy.min_face_samples
        assert track.p_minor is not None and track.p_minor < 0.2
        assert policy.classify(track) is TrackStatus.ADULT

    out = tmp_path / "out.mp4"
    render_video(adults_clip, out, analysis, plan_render(analysis, policy))
    src, dst = probe(adults_clip), probe(out)
    assert (dst.width, dst.height, dst.fps_rational) == (src.width, src.height, src.fps_rational)
    assert dst.has_audio

    # The default checkpoint segments people: the two full-length tracks carry
    # silhouettes on most frames, each inside (or near) its own box.
    for track in full_length:
        assert len(track.masks) >= 0.8 * frames, (track.track_id, len(track.masks))
        for f, mask in track.masks.items():
            box = track.boxes[f]
            cx = mask.x + mask.width / 2
            assert box.x1 - 30 <= cx <= box.x2 + 30
            assert mask.decode().any()


def test_silhouettes_are_tracked_and_box_only_models_still_work(adults_clip):
    from safeframe.pipeline.analyze import analyze_video
    from safeframe.pipeline.detection import DEFAULT_YOLO_WEIGHTS, YoloPersonTracker
    from safeframe.pipeline.factory import ModelBundle
    from safeframe.video.io import iter_frames

    assert DEFAULT_YOLO_WEIGHTS.endswith("-seg.pt")
    tracker = YoloPersonTracker()
    with_mask = 0

    for index, frame in iter_frames(adults_clip):
        for obs in tracker.update(frame):
            if obs.mask is not None:
                with_mask += 1
                h, w = frame.shape[:2]
                assert obs.mask.x + obs.mask.width <= w and obs.mask.y + obs.mask.height <= h
        if index >= 10:
            break
    assert with_mask >= 10

    # A legacy detection checkpoint gives boxes only (rectangular fallback).
    bundle = ModelBundle()
    result = analyze_video(
        adults_clip,
        YoloPersonTracker("yolo11n.pt"),
        bundle.new_face_detector(),
        bundle.age_estimator(),
    )
    assert result.analysis.tracks and result.analysis.mask_count == 0
