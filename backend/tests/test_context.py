"""Guarded zones and capture-location metadata."""

from __future__ import annotations

import pytest

from safeframe.pipeline.context import (
    CaptureContext,
    Zone,
    ZoneRegistry,
    parse_iso6709,
    parse_location,
    photo_location,
    resolve_context,
    video_location,
)
from safeframe.pipeline.policy import DecisionPolicy
from safeframe.pipeline.types import Analysis, VideoInfo

from .conftest import requires_ffmpeg

BANK = Zone("b1", "Test Bank", "bank", 40.4183, -3.6944, 90)


def test_registry_matches_by_distance():
    reg = ZoneRegistry([BANK])
    assert reg.match(40.4183, -3.6944) is BANK
    assert reg.match(40.4189, -3.6944) is BANK  # ~67 m north
    assert reg.match(40.4200, -3.6944) is None  # ~190 m north
    assert BANK.distance_m(40.4192, -3.6944) == pytest.approx(100, rel=0.02)


def test_registry_rejects_unknown_kinds_and_bad_coordinates():
    with pytest.raises(ValueError):
        ZoneRegistry([Zone("x", "Cafe", "cafe", 0, 0, 10)])
    with pytest.raises(ValueError):
        ZoneRegistry([Zone("x", "Bank", "bank", 95, 0, 10)])


def test_packaged_registry_loads():
    reg = ZoneRegistry.packaged()
    assert reg.zones and reg.match(40.4183, -3.6944).kind == "bank"


def test_resolve_context_never_stores_the_location():
    ctx = resolve_context((40.4183, -3.6944), ZoneRegistry([BANK]))
    assert ctx.guarded and ctx.guarded_source == "registry" and ctx.zone_name == "Test Bank"
    assert "40.41" not in str(ctx.to_dict())
    outside = resolve_context((0.0, 0.0), ZoneRegistry([BANK]))
    assert not outside.guarded and outside.had_location
    declared = resolve_context(None, ZoneRegistry([BANK]), declared_guarded=True)
    assert declared.guarded and declared.guarded_source == "user"


def test_registry_match_cannot_be_turned_off_by_the_uploader():
    ctx = resolve_context((40.4183, -3.6944), ZoneRegistry([BANK]), declared_guarded=False)
    assert ctx.guarded


def test_policy_is_locked_to_the_analysis_context():
    info = VideoInfo(10, 10, 25, "25/1", 1, 0.04, False)
    analysis = Analysis(info, [], {}, CaptureContext(guarded=True).to_dict())
    assert DecisionPolicy.for_analysis(analysis).guarded


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("+40.4183-003.6944+650.000/", (40.4183, -3.6944)),
        ("-33.8688+151.2093/", (-33.8688, 151.2093)),
        ("garbage", None),
        ("+95.0+010.0/", None),
    ],
)
def test_parse_iso6709(text, expected):
    assert parse_iso6709(text) == expected


def test_parse_location():
    assert parse_location("40.5,-3.25") == (40.5, -3.25)
    for bad in ("40.5", "a,b", "100,0"):
        with pytest.raises(ValueError):
            parse_location(bad)


@requires_ffmpeg
def test_video_location_is_read_from_container_metadata(make_video):
    tagged = make_video(name="tagged.mp4", metadata={"location": "+40.4183-003.6944/"})
    assert video_location(tagged) == pytest.approx((40.4183, -3.6944))
    assert video_location(make_video(name="plain.mp4")) is None


def test_photo_location_is_read_from_exif(tmp_path):
    from PIL import Image

    path = tmp_path / "gps.jpg"
    exif = Image.Exif()
    exif[0x8825] = {1: "N", 2: (40.0, 25.0, 6.0), 3: "W", 4: (3.0, 41.0, 40.0)}
    Image.new("RGB", (32, 32)).save(path, exif=exif)
    lat, lon = photo_location(path)
    assert lat == pytest.approx(40.4183, abs=1e-4) and lon == pytest.approx(-3.6944, abs=1e-4)

    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (32, 32)).save(plain)
    assert photo_location(plain) is None
    assert photo_location(tmp_path / "missing.jpg") is None
