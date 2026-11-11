"""End-to-end analysis + render on synthetic video with scripted detectors."""

import json
import subprocess
import threading

import numpy as np
import pytest

from safeframe.pipeline.analyze import AnalysisSettings, analyze_video
from safeframe.pipeline.interfaces import Cancelled
from safeframe.pipeline.maskstore import MaskStoreReader, MaskStoreWriter
from safeframe.pipeline.policy import DecisionPolicy, Override
from safeframe.pipeline.render import RenderError, plan_render, render_video
from safeframe.pipeline.types import Analysis, Box, ManualRegion, TrackStatus
from safeframe.video.io import iter_frames, probe

from .conftest import (
    HeadFaceDetector,
    ScriptedTracker,
    SizeAgeEstimator,
    ellipse_mask,
    person,
    requires_ffmpeg,
)

pytestmark = requires_ffmpeg

CHILD, ADULT, BACK_TURNED = 1, 2, 3
CHILD_BOX = (20, 40, 100, 220)
ADULT_BOX = (180, 20, 300, 230)
BACK_BOX = (120, 60, 170, 200)
OCCLUDED = range(20, 25)


def _script(frames=50):
    script = {}
    for f in range(frames):
        people = [person(ADULT, *ADULT_BOX)]
        if f not in OCCLUDED:
            people.append(person(CHILD, *CHILD_BOX))
        if 10 <= f <= 30:
            people.append(person(BACK_TURNED, *BACK_BOX))
        script[f] = people
    return script


def _analyze(video, **settings):
    tracker = ScriptedTracker(_script())
    faces = HeadFaceDetector(tracker, face_sizes={CHILD: 34, ADULT: 50})
    ages = SizeAgeEstimator(child_below_px=60)
    return analyze_video(video, tracker, faces, ages, AnalysisSettings(age_every=1, **settings))


def test_analysis_classifies_and_keeps_identity_through_occlusion(make_video):
    result = _analyze(make_video())
    analysis = result.analysis
    tracks = {t.track_id: t for t in analysis.tracks}
    policy = DecisionPolicy()

    assert policy.classify(tracks[CHILD]) is TrackStatus.MINOR
    assert policy.classify(tracks[ADULT]) is TrackStatus.ADULT
    assert policy.classify(tracks[BACK_TURNED]) is TrackStatus.UNCERTAIN

    # Same track id before and after the occlusion, with the gap interpolated.
    child = tracks[CHILD]
    assert all(f in child.boxes for f in OCCLUDED)
    assert set(OCCLUDED) <= child.interpolated

    # Uncertain track is padded by pad_seconds (0.25 s @ 25 fps ≈ 6 frames).
    assert tracks[BACK_TURNED].first_frame == 4
    assert tracks[BACK_TURNED].last_frame == 36

    assert set(result.thumbnails) == {CHILD, ADULT, BACK_TURNED}
    assert analysis.video.frame_count == 50


def test_progress_is_reported(make_video):
    seen = []
    tracker = ScriptedTracker({})
    analyze_video(
        make_video(),
        tracker,
        HeadFaceDetector(tracker, {}),
        SizeAgeEstimator(1),
        progress=seen.append,
    )
    assert seen[-1] == 1.0
    assert seen == sorted(seen)


def test_plan_respects_threshold_overrides_and_manual_regions(make_video):
    analysis = _analyze(make_video()).analysis
    manual = ManualRegion("m1", 5, 7, Box(0, 0, 10, 10))

    plan = plan_render(analysis, DecisionPolicy(), {}, [manual])
    assert sorted(plan.blurred_tracks) == [CHILD, BACK_TURNED]
    assert plan.kept_tracks == [ADULT]
    assert Box(0, 0, 10, 10) in plan.boxes_for(6)
    assert Box(0, 0, 10, 10) not in plan.boxes_for(8)

    # Reviewer keeps the uncertain person and forces the adult to be blurred.
    plan = plan_render(
        analysis, DecisionPolicy(), {BACK_TURNED: Override.KEEP, ADULT: Override.BLUR}
    )
    assert sorted(plan.blurred_tracks) == [CHILD, ADULT]

    # Maximum sensitivity blurs everyone; minimum keeps everyone with evidence.
    assert len(plan_render(analysis, DecisionPolicy(threshold=0.0)).blurred_tracks) == 3
    assert plan_render(analysis, DecisionPolicy(threshold=1.0)).blurred_tracks == [BACK_TURNED]


def _channel_spread(region):
    region = region.astype(int)
    return np.abs(region - region.mean(axis=2, keepdims=True)).mean()


def test_render_blurs_minors_preserves_format_audio_and_strips_metadata(make_video, tmp_path):
    source = make_video(rate="30000/1001", metadata={"location": "+51.5074-000.1278/"})
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()))

    src_info, out_info = probe(source), probe(out)
    assert (out_info.width, out_info.height) == (src_info.width, src_info.height)
    assert out_info.fps_rational == "30000/1001"
    assert out_info.has_audio and out_info.audio_codec == "aac"

    tags = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-print_format", "json", str(out)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["format"].get("tags", {})
    assert "location" not in tags

    src_frames = dict(iter_frames(source))
    out_frames = dict(iter_frames(out))
    assert len(out_frames) == len(src_frames) == 60  # 2 s @ 29.97 fps

    x1, y1, x2, y2 = CHILD_BOX
    ax1, ay1, ax2, ay2 = ADULT_BOX
    for f in (0, 22, 49):  # 22 is mid-occlusion: interpolated, still blurred
        child = out_frames[f][y1 + 10 : y2 - 10, x1 + 10 : x2 - 10]
        assert _channel_spread(child) < 4, f"child not greyscale at frame {f}"
        adult_out = out_frames[f][ay1 + 10 : ay2 - 10, ax1 + 30 : ax2 - 10].astype(int)
        adult_src = src_frames[f][ay1 + 10 : ay2 - 10, ax1 + 30 : ax2 - 10].astype(int)
        assert np.abs(adult_out - adult_src).mean() < 6, f"adult altered at frame {f}"
        assert _channel_spread(adult_out) > 10


def test_non_mp4_audio_is_transcoded(make_video, tmp_path):
    source = make_video(name="in.mkv", audio_codec="pcm_s16le")
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()))
    assert probe(out).audio_codec == "aac"


def test_video_without_audio(make_video, tmp_path):
    source = make_video(audio=False)
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()))
    assert not probe(out).has_audio


def test_cancel_stops_render_and_removes_partial_output(make_video, tmp_path):
    source = make_video()
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()), cancel=cancel)
    assert not out.exists()


# --- silhouettes ------------------------------------------------------------------

SIL_FRAMES = 45  # track observed on 0..44, padded to 50 by pad_seconds
NO_MASK = range(10, 15)  # observed, but the model gave no usable mask
GAP = range(20, 25)  # not observed: interpolated


def _moving_box(f: int) -> Box:
    x = 20 + 2 * f
    return Box(x, 40, x + 80, 220)


def _silhouette_script():
    script = {}
    for f in range(SIL_FRAMES):
        if f in GAP:
            script[f] = []
            continue
        box = _moving_box(f)
        mask = None if f in NO_MASK else ellipse_mask(box, 320, 240)
        script[f] = [person(CHILD, box.x1, box.y1, box.x2, box.y2, mask=mask)]
    return script


def _analyze_silhouettes(video, writer=None):
    tracker = ScriptedTracker(_silhouette_script())
    faces = HeadFaceDetector(tracker, face_sizes={CHILD: 34})
    return analyze_video(
        video,
        tracker,
        faces,
        SizeAgeEstimator(60),
        AnalysisSettings(age_every=1),
        mask_writer=writer,
    ).analysis


def test_mask_frames_and_box_fallback_frames_are_planned(make_video):
    analysis = _analyze_silhouettes(make_video())
    track = analysis.tracks[0]
    observed = set(range(SIL_FRAMES)) - set(GAP)
    assert set(track.masks) == observed - set(NO_MASK)
    plan = plan_render(analysis, DecisionPolicy())
    for f in range(50):
        regions = plan.regions_for(f)
        # Every frame of the (padded, interpolated) track is protected...
        assert len(regions) == 1, f
        # ...by silhouette only where one was observed for that very frame.
        expect_mask = f in observed and f not in NO_MASK
        assert (regions[0].mask is not None) == expect_mask, f
        assert regions[0].box == track.boxes[f]
    stats = plan.coverage_stats()
    assert stats == {"mask": len(observed) - len(NO_MASK), "box": 50 - stats["mask"], "manual": 0}


def _corner(frame, f):
    # Interpolation is linear, so the gap matches _moving_box; padded ends hold the last box.
    b = _moving_box(min(f, SIL_FRAMES - 1))
    x, y = int(b.x1), int(b.y1)
    return frame[y + 2 : y + 8, x + 2 : x + 8]


def _centre(frame, f):
    b = _moving_box(f)
    cx, cy = (int(v) for v in b.center)
    return frame[cy - 20 : cy + 20, cx - 12 : cx + 12]


@pytest.mark.parametrize("storage", ["inline", "store"])
def test_render_silhouette_with_fallback_end_to_end(make_video, tmp_path, storage):
    source = make_video(size="320x240")
    path = tmp_path / "analysis.json"
    observed_masks = SIL_FRAMES - len(GAP) - len(NO_MASK)
    if storage == "store":
        with MaskStoreWriter(tmp_path / "analysis.masks") as writer:
            analysis = _analyze_silhouettes(source, writer)
            analysis.mask_store = writer.close()
        assert analysis.mask_count == 0 and analysis.mask_store["entries"] == observed_masks
    else:
        analysis = _analyze_silhouettes(source)
        assert analysis.mask_count == observed_masks
    # CLI-style persistence: JSON on disk, loaded back for render.
    path.write_text(json.dumps(analysis.to_dict()))
    loaded = Analysis.from_dict(json.loads(path.read_text()))
    out = tmp_path / "out.mp4"
    store = MaskStoreReader.for_analysis(loaded, tmp_path)
    plan = plan_render(loaded, DecisionPolicy())
    stats = render_video(source, out, loaded, plan, masks=store)
    if store is not None:
        assert plan.coverage_stats(store)["mask"] == observed_masks
        store.close()
    assert stats == {"mask": observed_masks, "box": 50 - observed_masks, "manual": 0}
    src, dst = dict(iter_frames(source)), dict(iter_frames(out))

    for f in (3, 30, 40):  # good silhouette frames
        assert _channel_spread(_centre(dst[f], f)) < 4, f"person not blurred at {f}"
        before, after = _corner(src[f], f).astype(int), _corner(dst[f], f).astype(int)
        assert _channel_spread(before) > 10  # the background there is colourful
        assert np.abs(after - before).mean() < 8, f"background at box corner altered at {f}"
    for f in (12, 22, 47):  # missing mask, interpolated gap, padded end
        assert _channel_spread(_corner(dst[f], f)) < 4, f"no box fallback at {f}"
        assert _channel_spread(_centre(dst[f], min(f, SIL_FRAMES - 1))) < 4

    # A legacy (box-only) analysis of the same video renders rectangles.
    legacy = analysis.to_dict()
    for t in legacy["tracks"]:
        t.pop("masks_v1", None)
    legacy_analysis = Analysis.from_dict(legacy)
    assert legacy_analysis.mask_count == 0
    out2 = tmp_path / "legacy.mp4"
    render_video(source, out2, legacy_analysis, plan_render(legacy_analysis, DecisionPolicy()))
    legacy_frames = dict(iter_frames(out2))
    assert _channel_spread(_corner(legacy_frames[3], 3)) < 4


def test_unprotectable_region_fails_render_without_output(make_video, tmp_path):
    source = make_video()
    analysis = _analyze(source).analysis
    child = next(t for t in analysis.tracks if t.track_id == CHILD)
    child.boxes[5] = Box(float("nan"), 0, 10, 10)
    out = tmp_path / "out.mp4"
    with pytest.raises(RenderError, match="frame 5"):
        render_video(source, out, analysis, plan_render(analysis, DecisionPolicy()))
    assert not out.exists()
