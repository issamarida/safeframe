"""HTTP API tests with scripted (model-free) pipeline components."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from safeframe.api.app import create_app
from safeframe.config import Settings
from safeframe.video.io import probe

from .conftest import (
    HeadFaceDetector,
    ScriptedBands,
    ScriptedTracker,
    SizeAgeEstimator,
    components,
    ellipse_mask,
    person,
    requires_ffmpeg,
)

pytestmark = requires_ffmpeg

# Person 1 wears a band, seen once at frame 6 at their wrist.
BAND = {6: [(30.0, 130.0)]}
# A band on person 2 (the "adult"), for guarded-zone tests.
ADULT_BAND = {6: [(240.0, 120.0)]}
_band_script = {"script": BAND}


def _components(photo: bool = False):
    from safeframe.pipeline.types import Box

    child_mask = ellipse_mask(Box(20, 40, 100, 220), 320, 240)
    # The child has a silhouette except on frames 10-14 (box fallback).
    script = {
        f: [
            person(1, 20, 40, 100, 220, mask=None if 10 <= f < 15 else child_mask),
            person(2, 180, 20, 300, 230),
        ]
        for f in range(50)
    }
    if photo:
        script = {0: script[0]}
        bands = {0: next(iter(_band_script["script"].values()))}
    else:
        bands = _band_script["script"]
    tracker = ScriptedTracker(script)
    return components(
        tracker,
        HeadFaceDetector(tracker, {1: 34, 2: 50}),
        SizeAgeEstimator(60),
        ScriptedBands(bands),
    )


@pytest.fixture(autouse=True)
def _reset_band_script():
    _band_script["script"] = BAND
    yield
    _band_script["script"] = BAND


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        warm_up_models=False,
        frontend_dist=tmp_path / "no-frontend",
        sweep_interval_seconds=3600,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, components=_components)) as c:
        yield c


def _wait(client, job_id, *states, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        assert job["state"] not in ("failed", "cancelled"), job
        time.sleep(0.1)
    raise AssertionError(f"job did not reach {states}: {job}")


def _upload(client, video, keep=False, filename="holiday.mp4"):
    r = client.post(
        "/api/jobs",
        params={"filename": filename, "keep_upload": keep},
        content=video.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_config_exposes_notice_and_intended_use(client):
    body = client.get("/api/config").json()
    assert "SafeFrame Band" in body["notice"]
    assert "not a disguise" in body["intended_use"]
    assert 0 < body["default_threshold"] < 0.5
    assert body["sealing_enabled"] is False and body["band_spec_version"] == 1


def test_printable_band_sheet(client):
    r = client.get("/api/band.svg")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml")
    assert b"SafeFrame Band" in r.content


def test_full_flow_deletes_upload_after_render(client, make_video, settings):
    job = _upload(client, make_video(), filename="../../etc/Holiday clip!.mp4")
    assert job["filename"] == "Holiday clip_.mp4"
    assert job["video"]["frame_count"] == 50

    job = _wait(client, job["id"], "review")
    jid = job["id"]
    job_dir = settings.jobs_dir / jid

    analysis = client.get(f"/api/jobs/{jid}/analysis").json()
    assert [t["track_id"] for t in analysis["tracks"]] == [1, 2]
    wearer, other = analysis["tracks"]
    assert wearer["band"]["confirmed"] == 1 and wearer["band"]["first_frame"] == 6
    assert other["band"]["confirmed"] == other["band"]["possible"] == 0
    # Nobody's age is estimated outside guarded zones and event mode.
    assert wearer["p_minor"] is None and wearer["face_samples"] == 0
    assert "samples" not in wearer
    assert analysis["context"]["guarded"] is False

    preview = client.get(f"/api/jobs/{jid}/preview", headers={"Range": "bytes=0-99"})
    assert preview.status_code == 206 and len(preview.content) == 100
    assert preview.headers["cache-control"] == "no-store"
    thumb = client.get(f"/api/jobs/{jid}/tracks/1/thumbnail")
    assert thumb.status_code == 200 and thumb.content[:2] == b"\xff\xd8"

    r = client.post(
        f"/api/jobs/{jid}/render",
        json={
            "threshold": 0.35,
            "overrides": {"2": "blur"},
            "manual_regions": [
                {"region_id": "m1", "start_frame": 0, "end_frame": 10, "box": [0, 0, 40, 40]}
            ],
        },
    )
    assert r.status_code == 202, r.text
    job = _wait(client, jid, "done")
    assert job["output_ready"] and job["upload_deleted"] and not job["sealed_ready"]

    # Everything showing unblurred people is gone.
    assert sorted(p.name for p in job_dir.iterdir()) == ["analysis.json", "job.json", "output.mp4"]
    assert client.get(f"/api/jobs/{jid}/preview").status_code == 410
    assert client.post(f"/api/jobs/{jid}/render", json={}).status_code == 410

    download = client.get(f"/api/jobs/{jid}/download")
    assert download.status_code == 200
    disposition = download.headers["content-disposition"]
    assert disposition.startswith("attachment")
    assert "Holiday%20clip_-safeframe.mp4" in disposition
    out = job_dir / "check.mp4"
    out.write_bytes(download.content)
    assert probe(out).frame_count == 50
    out.unlink()

    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert not job_dir.exists()
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_keep_upload_allows_re_render(client, make_video, settings):
    jid = _upload(client, make_video(), keep=True)["id"]
    _wait(client, jid, "review")
    client.post(f"/api/jobs/{jid}/render", json={"threshold": 0.5})
    job = _wait(client, jid, "done")
    assert not job["upload_deleted"]
    assert (settings.jobs_dir / jid / "upload").exists()
    assert client.post(f"/api/jobs/{jid}/render", json={"threshold": 0.1}).status_code == 202
    _wait(client, jid, "done")


def test_rejects_non_video_and_cleans_up(client, settings):
    r = client.post("/api/jobs", content=b"definitely not a video" * 100)
    assert r.status_code == 415
    assert list(settings.jobs_dir.iterdir()) == []


def test_rejects_empty_and_oversized_uploads(tmp_path, make_video):
    settings = Settings(
        data_dir=tmp_path / "d", warm_up_models=False, max_upload_mb=1, frontend_dist=tmp_path / "x"
    )
    with TestClient(create_app(settings, components=_components)) as client:
        assert client.post("/api/jobs", content=b"").status_code == 400
        r = client.post("/api/jobs", content=b"\0" * (1024**2 + 1))
        assert r.status_code == 413
        assert list(settings.jobs_dir.iterdir()) == []


def test_invalid_ids_and_premature_actions(client, make_video):
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404
    jid = _upload(client, make_video())["id"]
    assert client.get(f"/api/jobs/{jid}/download").status_code == 409
    _wait(client, jid, "review")
    bad = {
        "manual_regions": [
            {"region_id": "x", "start_frame": 9, "end_frame": 1, "box": [0, 0, 1, 1]}
        ]
    }
    assert client.post(f"/api/jobs/{jid}/render", json=bad).status_code == 422
    assert client.post(f"/api/jobs/{jid}/render", json={"threshold": 2}).status_code == 422
    assert (
        client.post(f"/api/jobs/{jid}/render", json={"overrides": {"1": "maybe"}}).status_code
        == 422
    )


def test_events_stream_reports_progress(client, make_video):
    jid = _upload(client, make_video())["id"]
    _wait(client, jid, "review")
    with client.stream("GET", f"/api/jobs/{jid}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert '"state":"review"' in lines[-1]


def test_sweeper_expires_jobs_and_orphans(client, make_video, settings):
    jid = _upload(client, make_video())["id"]
    _wait(client, jid, "review")
    orphan = settings.jobs_dir / ("f" * 32)
    orphan.mkdir()
    manager = client.app.state.jobs
    assert manager.sweep(now=time.time() + settings.job_ttl_hours * 3600 + 1) == 1
    assert list(settings.jobs_dir.iterdir()) == []


def test_restart_recovers_finished_jobs_and_drops_interrupted(settings, make_video):
    with TestClient(create_app(settings, components=_components)) as client:
        jid = _upload(client, make_video(), keep=True)["id"]
        _wait(client, jid, "review")
    interrupted = settings.jobs_dir / ("a" * 32)
    interrupted.mkdir()
    (interrupted / "job.json").write_text(
        '{"id": "' + "a" * 32 + '", "filename": "x.mp4", "keep_upload": false, '
        '"created_at": 0, "updated_at": 0, "state": "analyzing", "stage": "", '
        '"progress": 0.5, "error": null, "upload_deleted": false, "video": null}'
    )
    with TestClient(create_app(settings, components=_components)) as client:
        assert client.get(f"/api/jobs/{jid}").json()["state"] == "review"
        assert not interrupted.exists()


def test_analysis_serves_validated_masks_gzipped(client, make_video, settings):
    import json

    job = _wait(client, _upload(client, make_video(), keep=True)["id"], "review")
    jid = job["id"]
    r = client.get(f"/api/jobs/{jid}/analysis")
    body = r.json()
    child = next(t for t in body["tracks"] if t["track_id"] == 1)
    adult = next(t for t in body["tracks"] if t["track_id"] == 2)
    frames = [row[0] for row in child["masks_v1"]]
    assert frames == [f for f in range(50) if not 10 <= f < 15]
    assert "masks_v1" not in adult  # box-only person
    _frame, x, y, w, h, runs = child["masks_v1"][0]
    assert sum(runs) == w * h and x >= 0 and x + w <= 320 and y + h <= 240

    # Large payloads are compressed when the client accepts it.
    headers = client.get(f"/api/jobs/{jid}/analysis", headers={"Accept-Encoding": "gzip"}).headers
    assert headers.get("content-encoding") == "gzip"
    # A tampered analysis on disk is validated, not trusted: bad rows are dropped.
    path = settings.jobs_dir / jid / "analysis.json"
    data = json.loads(path.read_text())
    data["tracks"][0]["masks_v1"][0][5] = [1, 2, 3]
    path.write_text(json.dumps(data))
    child = next(
        t for t in client.get(f"/api/jobs/{jid}/analysis").json()["tracks"] if t["track_id"] == 1
    )
    assert child["masks_v1"][0][0] == 1

    # Rendering the tampered analysis still protects frame 0 (box fallback).
    r = client.post(f"/api/jobs/{jid}/render", json={})
    assert r.status_code == 202
    _wait(client, jid, "done")


def test_guarded_zone_from_location_is_locked_in(client, make_video):
    _band_script["script"] = ADULT_BAND
    r = client.post(
        "/api/jobs",
        params={"filename": "bank.mp4", "location": "40.4183,-3.6944", "guarded": False},
        content=make_video().read_bytes(),
    )
    assert r.status_code == 201, r.text
    job = _wait(client, r.json()["id"], "review")
    assert job["context"]["guarded"] and job["context"]["zone_kind"] == "bank"
    assert "40.41" not in json.dumps(job)  # the location itself is never stored
    analysis = client.get(f"/api/jobs/{job['id']}/analysis").json()
    adult = next(t for t in analysis["tracks"] if t["track_id"] == 2)
    child = next(t for t in analysis["tracks"] if t["track_id"] == 1)
    assert adult["band"]["confirmed"] == 1 and adult["face_samples"] > 0
    assert adult["p_minor"] < 0.2  # an adult: the band is not honoured here
    assert child["face_samples"] == 0  # no band, so their face was never analysed


def test_declared_guarded_and_bad_location(client, make_video):
    video = make_video().read_bytes()
    r = client.post("/api/jobs", params={"guarded": True}, content=video)
    assert r.json()["context"]["guarded_source"] == "user"
    assert client.post("/api/jobs", params={"location": "north"}, content=video).status_code == 422


def _jpeg(path):
    import cv2
    import numpy as np

    img = np.random.default_rng(2).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    img[:8] = 0
    cv2.imwrite(str(path), img)
    return path


def test_photo_flow(client, settings, tmp_path):
    job = _upload(client, _jpeg(tmp_path / "IMG_0001.jpg"), filename="IMG_0001.jpg")
    assert job["media"] == "photo" and job["video"]["frame_count"] == 1
    job = _wait(client, job["id"], "review")
    jid = job["id"]
    preview = client.get(f"/api/jobs/{jid}/preview")
    assert preview.status_code == 200
    analysis = client.get(f"/api/jobs/{jid}/analysis").json()
    assert analysis["video"]["media"] == "photo"
    assert analysis["tracks"][0]["band"]["confirmed"] == 1
    assert client.post(f"/api/jobs/{jid}/render", json={}).status_code == 202
    job = _wait(client, jid, "done")
    download = client.get(f"/api/jobs/{jid}/download")
    assert download.headers["content-type"] == "image/jpeg"
    assert "IMG_0001-safeframe.jpg" in download.headers["content-disposition"]
    assert download.content[:2] == b"\xff\xd8"


def test_sealed_original_is_written_for_honoured_bands(tmp_path, make_video):
    from safeframe.seal import TrusteeShare, create_escrow, unseal_file

    public, shares = create_escrow(2, 3)
    key = tmp_path / "escrow.pub.json"
    key.write_text(public.to_json())
    settings = Settings(
        data_dir=tmp_path / "d",
        warm_up_models=False,
        frontend_dist=tmp_path / "x",
        escrow_public_key=key,
    )
    video = make_video()
    with TestClient(create_app(settings, components=_components)) as client:
        assert client.get("/api/config").json()["sealing_enabled"]
        jid = _upload(client, video)["id"]
        _wait(client, jid, "review")
        client.post(f"/api/jobs/{jid}/render", json={})
        job = _wait(client, jid, "done")
        assert job["sealed_ready"] and job["upload_deleted"]
        r = client.get(f"/api/jobs/{jid}/sealed")
        assert r.status_code == 200 and r.content.startswith(b"SFSEAL1")
        envelope = tmp_path / "e.sfseal"
        envelope.write_bytes(r.content)
        restored = tmp_path / "restored.mp4"
        unseal_file(envelope, [TrusteeShare.from_json(s.to_json()) for s in shares[1:]], restored)
        assert restored.read_bytes() == video.read_bytes()

        # Keeping the wearer (with their consent) means nothing is hidden: no envelope.
        jid = _upload(client, video, keep=True)["id"]
        _wait(client, jid, "review")
        client.post(f"/api/jobs/{jid}/render", json={"overrides": {"1": "keep"}})
        assert not _wait(client, jid, "done")["sealed_ready"]
        assert client.get(f"/api/jobs/{jid}/sealed").status_code == 404
