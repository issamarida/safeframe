"""End-to-end analysis + render on synthetic video with scripted detectors."""

import json
import subprocess
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from safeframe.pipeline.analyze import AnalysisSettings, analyze_photo, analyze_video
from safeframe.pipeline.context import CaptureContext
from safeframe.pipeline.interfaces import Cancelled
from safeframe.pipeline.policy import Decision, DecisionPolicy, Override
from safeframe.pipeline.render import RenderError, plan_render, render_photo, render_video
from safeframe.pipeline.types import Analysis, BandVia, Box, ManualRegion
from safeframe.video.io import iter_frames, probe

from .conftest import (
    HeadFaceDetector,
    ScriptedBands,
    ScriptedTracker,
    ScriptedWrists,
    SizeAgeEstimator,
    components,
    ellipse_mask,
    person,
    pose,
    requires_ffmpeg,
)

pytestmark = requires_ffmpeg

CHILD, ADULT, BACK_TURNED = 1, 2, 3
CHILD_BOX = (20, 40, 100, 220)
ADULT_BOX = (180, 20, 300, 230)
BACK_BOX = (120, 60, 170, 200)
OCCLUDED = range(20, 25)
# The child's band is only visible for a moment, after the occlusion.
BAND_FRAMES = range(30, 33)
CHILD_WRIST = (35.0, 130.0)
ADULT_WRIST = (190.0, 120.0)


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


def _analyze(video, context=None, band_on=CHILD, **settings):
    tracker = ScriptedTracker(_script())
    faces = HeadFaceDetector(tracker, face_sizes={CHILD: 34, ADULT: 50})
    ages = SizeAgeEstimator(child_below_px=60)
    wrist = CHILD_WRIST if band_on == CHILD else ADULT_WRIST
    bands = ScriptedBands({f: [wrist] for f in BAND_FRAMES})
    wrists = ScriptedWrists(
        {f: [pose(CHILD_BOX, CHILD_WRIST), pose(ADULT_BOX, ADULT_WRIST)] for f in range(50)}
    )
    result = analyze_video(
        video,
        components(tracker, faces, ages, bands, wrists),
        context,
        AnalysisSettings(age_every=1, band_every=1, **settings),
    )
    return SimpleNamespace(analysis=result.analysis, thumbnails=result.thumbnails, ages=ages)


def _policy(analysis, **kw):
    return DecisionPolicy.for_analysis(analysis, **kw)


def test_one_band_sighting_protects_the_whole_track(make_video):
    result = _analyze(make_video())
    analysis = result.analysis
    tracks = {t.track_id: t for t in analysis.tracks}
    policy = _policy(analysis)

    assert policy.classify(tracks[CHILD]) is Decision.WEARER
    assert policy.classify(tracks[ADULT]) is Decision.VISIBLE
    assert policy.classify(tracks[BACK_TURNED]) is Decision.VISIBLE
    child = tracks[CHILD]
    assert child.band_confirmed == len(BAND_FRAMES)
    assert {h.via for h in child.band_hits} == {BandVia.WRIST}

    # Same track id before and after the occlusion, with the gap interpolated.
    assert all(f in child.boxes for f in OCCLUDED)
    assert set(OCCLUDED) <= child.interpolated

    # Outside guarded zones and event mode nobody's age is ever estimated.
    assert result.ages.calls == 0
    assert all(not t.samples for t in analysis.tracks)

    assert set(result.thumbnails) == {CHILD, ADULT, BACK_TURNED}
    assert analysis.video.frame_count == 50
    assert analysis.context == CaptureContext().to_dict()


def test_guarded_zone_honours_children_not_adults(make_video):
    guarded = CaptureContext(guarded=True, guarded_source="registry", zone_kind="bank")

    child = _analyze(make_video(), guarded, band_on=CHILD)
    tracks = {t.track_id: t for t in child.analysis.tracks}
    assert _policy(child.analysis).classify(tracks[CHILD]) is Decision.WEARER

    robber = _analyze(make_video(name="bank.mp4"), guarded, band_on=ADULT)
    tracks = {t.track_id: t for t in robber.analysis.tracks}
    policy = _policy(robber.analysis)
    assert policy.classify(tracks[ADULT]) is Decision.NOT_HONOURED
    assert not policy.should_blur(tracks[ADULT])
    plan = plan_render(robber.analysis, policy)
    assert plan.blurred_tracks == [] and plan.honoured_bands == []

    # Age was estimated for the wearer only: never for bystanders.
    assert tracks[ADULT].samples and not tracks[CHILD].samples
    assert not tracks[BACK_TURNED].samples
    assert robber.ages.crops == len(tracks[ADULT].samples)


def test_event_mode_blurs_likely_minors_and_uncertain_people(make_video):
    analysis = _analyze(make_video(), CaptureContext(protect_minors=True), band_on=ADULT).analysis
    tracks = {t.track_id: t for t in analysis.tracks}
    policy = _policy(analysis)
    assert policy.classify(tracks[ADULT]) is Decision.WEARER  # the band still counts
    assert policy.classify(tracks[CHILD]) is Decision.MINOR
    assert policy.classify(tracks[BACK_TURNED]) is Decision.UNCERTAIN
    assert sorted(plan_render(analysis, policy).blurred_tracks) == [CHILD, ADULT, BACK_TURNED]

    # Uncertain tracks are padded by pad_seconds (0.25 s @ 25 fps ≈ 6 frames).
    assert tracks[BACK_TURNED].first_frame == 4
    assert tracks[BACK_TURNED].last_frame == 36


def test_progress_is_reported(make_video):
    seen = []
    tracker = ScriptedTracker({})
    analyze_video(
        make_video(),
        components(tracker, HeadFaceDetector(tracker, {}), SizeAgeEstimator(1)),
        progress=seen.append,
    )
    assert seen[-1] == 1.0
    assert seen == sorted(seen)


def test_plan_respects_overrides_and_manual_regions(make_video):
    analysis = _analyze(make_video()).analysis
    manual = ManualRegion("m1", 5, 7, Box(0, 0, 10, 10))

    plan = plan_render(analysis, _policy(analysis), {}, [manual])
    assert plan.blurred_tracks == [CHILD]
    assert sorted(plan.kept_tracks) == [ADULT, BACK_TURNED]
    assert plan.honoured_bands == [CHILD]
    assert plan.reasons[CHILD] == "wearer" and plan.reasons[ADULT] == "visible"
    assert Box(0, 0, 10, 10) in plan.boxes_for(6)
    assert Box(0, 0, 10, 10) not in plan.boxes_for(8)

    # Reviewer blurs a bystander and keeps the wearer (with their consent).
    plan = plan_render(
        analysis, _policy(analysis), {CHILD: Override.KEEP, BACK_TURNED: Override.BLUR}
    )
    assert plan.blurred_tracks == [BACK_TURNED]
    assert plan.reasons[CHILD] == "override:keep"
    assert plan.honoured_bands == []

    # Confirming the blur on a wearer still counts as honouring their band.
    plan = plan_render(analysis, _policy(analysis), {CHILD: Override.BLUR})
    assert plan.reasons[CHILD] == "override:blur" and plan.honoured_bands == [CHILD]


def test_orphan_bands_are_reported(make_video):
    tracker = ScriptedTracker(_script())
    bands = ScriptedBands({3: [(310.0, 235.0)]})  # nobody stands there
    analysis = analyze_video(
        make_video(),
        components(tracker, HeadFaceDetector(tracker, {}), SizeAgeEstimator(1), bands),
        settings=AnalysisSettings(band_every=1),
    ).analysis
    assert analysis.orphan_band_frames == [3]
    assert not any(t.wears_band for t in analysis.tracks)
    assert Analysis.from_dict(json.loads(json.dumps(analysis.to_dict()))).orphan_band_frames == [3]


def _channel_spread(region):
    region = region.astype(int)
    return np.abs(region - region.mean(axis=2, keepdims=True)).mean()


def test_render_blurs_wearer_preserves_format_audio_and_strips_metadata(make_video, tmp_path):
    source = make_video(rate="30000/1001", metadata={"location": "+51.5074-000.1278/"})
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, _policy(analysis)))

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
    # 0 is long before the band was seen; 22 is mid-occlusion (interpolated).
    for f in (0, 22, 49):
        child = out_frames[f][y1 + 10 : y2 - 10, x1 + 10 : x2 - 10]
        assert _channel_spread(child) < 4, f"wearer not greyscale at frame {f}"
        adult_out = out_frames[f][ay1 + 10 : ay2 - 10, ax1 + 30 : ax2 - 10].astype(int)
        adult_src = src_frames[f][ay1 + 10 : ay2 - 10, ax1 + 30 : ax2 - 10].astype(int)
        assert np.abs(adult_out - adult_src).mean() < 6, f"adult altered at frame {f}"
        assert _channel_spread(adult_out) > 10


def test_non_mp4_audio_is_transcoded(make_video, tmp_path):
    source = make_video(name="in.mkv", audio_codec="pcm_s16le")
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, _policy(analysis)))
    assert probe(out).audio_codec == "aac"


def test_video_without_audio(make_video, tmp_path):
    source = make_video(audio=False)
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    render_video(source, out, analysis, plan_render(analysis, _policy(analysis)))
    assert not probe(out).has_audio


def test_cancel_stops_render_and_removes_partial_output(make_video, tmp_path):
    source = make_video()
    analysis = _analyze(source).analysis
    out = tmp_path / "out.mp4"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        render_video(source, out, analysis, plan_render(analysis, _policy(analysis)), cancel=cancel)
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


def _analyze_silhouettes(video):
    tracker = ScriptedTracker(_silhouette_script())
    faces = HeadFaceDetector(tracker, face_sizes={CHILD: 34})
    band = {3: [_moving_box(3).center]}  # seen once, inside the silhouette
    return analyze_video(
        video,
        components(tracker, faces, SizeAgeEstimator(60), ScriptedBands(band)),
        settings=AnalysisSettings(band_every=1),
    ).analysis


def test_mask_frames_and_box_fallback_frames_are_planned(make_video):
    analysis = _analyze_silhouettes(make_video())
    track = analysis.tracks[0]
    observed = set(range(SIL_FRAMES)) - set(GAP)
    assert set(track.masks) == observed - set(NO_MASK)
    assert track.band_hits[0].via is BandVia.BODY
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


def test_render_silhouette_with_fallback_end_to_end(make_video, tmp_path):
    source = make_video(size="320x240")
    analysis = _analyze_silhouettes(source)
    # CLI-style persistence: JSON on disk, loaded back for render.
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis.to_dict()))
    loaded = Analysis.from_dict(json.loads(path.read_text()))
    assert loaded.mask_count == analysis.mask_count > 0
    out = tmp_path / "out.mp4"
    render_video(source, out, loaded, plan_render(loaded, DecisionPolicy()))
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
        render_video(source, out, analysis, plan_render(analysis, _policy(analysis)))
    assert not out.exists()


# --- photos ------------------------------------------------------------------------


def _photo(path, gps=None):
    from PIL import Image

    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    img[:8] = 0  # frame index 0 for the scripted fakes
    pil = Image.fromarray(img[:, :, ::-1])
    exif = Image.Exif()
    if gps:
        exif[0x8825] = gps
    exif[0x0110] = "SafePhone 1"  # camera model
    pil.save(path, exif=exif, quality=95)
    return path


def test_photo_is_analysed_rendered_and_stripped(tmp_path):
    from PIL import Image

    src = _photo(tmp_path / "in.jpg", {1: "N", 2: (40.0, 25.0, 6.0), 3: "W", 4: (3.0, 41.0, 40.0)})
    tracker = ScriptedTracker({0: [person(CHILD, *CHILD_BOX), person(ADULT, *ADULT_BOX)]})
    faces = HeadFaceDetector(tracker, {CHILD: 34, ADULT: 50})
    bands = ScriptedBands({0: [CHILD_WRIST]})
    result = analyze_photo(src, components(tracker, faces, SizeAgeEstimator(60), bands))
    analysis = result.analysis
    assert analysis.video.is_photo and analysis.video.frame_count == 1
    tracks = {t.track_id: t for t in analysis.tracks}
    assert tracks[CHILD].band_confirmed == 1
    assert all(len(t.boxes) == 1 for t in analysis.tracks)  # no padding on a photo

    out = tmp_path / "out.jpg"
    render_photo(src, out, analysis, plan_render(analysis, _policy(analysis)))
    with Image.open(out) as img:
        assert not dict(img.getexif()), "metadata leaked into the shared photo"
        pixels = np.asarray(img)[:, :, ::-1].astype(int)
    x1, y1, x2, y2 = CHILD_BOX
    assert _channel_spread(pixels[y1 + 10 : y2 - 10, x1 + 10 : x2 - 10]) < 4
    ax1, ay1, ax2, ay2 = ADULT_BOX
    assert _channel_spread(pixels[ay1 + 10 : ay2 - 10, ax1 + 10 : ax2 - 10]) > 8


def test_photo_in_guarded_zone_needs_only_one_face(tmp_path):
    src = _photo(tmp_path / "in.jpg")
    tracker = ScriptedTracker({0: [person(CHILD, *CHILD_BOX)]})
    faces = HeadFaceDetector(tracker, {CHILD: 34})
    bands = ScriptedBands({0: [CHILD_WRIST]})
    analysis = analyze_photo(
        src,
        components(tracker, faces, SizeAgeEstimator(60), bands),
        CaptureContext(guarded=True, guarded_source="user"),
    ).analysis
    child = analysis.tracks[0]
    assert len(child.samples) == 1
    assert _policy(analysis).classify(child) is Decision.WEARER
    assert DecisionPolicy(guarded=True).classify(child) is Decision.NOT_HONOURED


def test_render_photo_rejects_mismatched_analysis(tmp_path):
    src = _photo(tmp_path / "in.jpg")
    tracker = ScriptedTracker({})
    analysis = analyze_photo(
        src, components(tracker, HeadFaceDetector(tracker, {}), SizeAgeEstimator(1))
    ).analysis
    other = tmp_path / "other.png"
    import cv2

    cv2.imwrite(str(other), np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(RenderError):
        render_photo(other, tmp_path / "o.jpg", analysis, plan_render(analysis, DecisionPolicy()))
