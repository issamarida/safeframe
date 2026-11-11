"""On-disk, frame-addressable silhouette store ("sfm1")."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from safeframe.pipeline import maskstore as S
from safeframe.pipeline.masks import Mask, encode
from safeframe.pipeline.maskstore import MaskStoreReader, MaskStoreWriter, resolve
from safeframe.pipeline.types import Analysis, Box, Track, VideoInfo

VIDEO = VideoInfo(320, 240, 25.0, "25/1", 50, 2.0, False)


def _mask(seed: int = 0, x: int = 20, y: int = 30) -> Mask:
    rng = np.random.default_rng(seed)
    crop = rng.random((40, 30)) > 0.4
    crop[0, 0] = True
    m = encode(crop, x, y)
    assert m is not None
    return m


def _analysis(frames=range(10), interpolated=(), track_ids=(1, 2)) -> Analysis:
    tracks = []
    for tid in track_ids:
        t = Track(track_id=tid, boxes={f: Box(10, 20, 80, 200) for f in frames})
        t.interpolated = set(interpolated)
        tracks.append(t)
    return Analysis(video=VIDEO, tracks=tracks, settings={})


def _write(path, entries) -> dict:
    with MaskStoreWriter(path) as w:
        for frame, tid, mask in entries:
            w.add(frame, tid, mask)
        return w.close()


def _reader(path, analysis, descriptor) -> MaskStoreReader:
    analysis.mask_store = descriptor
    reader = MaskStoreReader.for_analysis(analysis, path.parent)
    assert reader is not None
    return reader


def test_round_trip_by_frame_and_range(tmp_path):
    path = tmp_path / "a.masks"
    m1, m2, m3 = _mask(1), _mask(2, x=200), _mask(3)
    desc = _write(path, [(0, 1, m1), (0, 2, m2), (3, 1, m3)])
    assert desc["entries"] == 3 and desc["file"] == "a.masks"
    assert not path.with_name("a.masks.partial").exists()
    with _reader(path, _analysis(), desc) as r:
        assert len(r) == 3
        assert r.masks_for_frame(0) == {1: m1, 2: m2}
        assert r.masks_for_frame(1) == {}
        assert r.masks_for_frame(3) == {1: m3}
        end, rows = r.rows(0, 10)
        assert end == 10
        assert rows == {1: [m1.to_row(0), m3.to_row(3)], 2: [m2.to_row(0)]}
        assert r.count_for([(0, 1), (0, 2), (1, 1), (3, 1)]) == 3


def test_analysis_json_links_store_and_round_trips(tmp_path):
    path = tmp_path / "x.masks"
    desc = _write(path, [(2, 1, _mask())])
    analysis = _analysis()
    analysis.mask_store = desc
    loaded = Analysis.from_dict(json.loads(json.dumps(analysis.to_dict())))
    assert loaded.mask_store == desc
    with MaskStoreReader.for_analysis(loaded, tmp_path) as r:
        assert set(r.masks_for_frame(2)) == {1}


def test_writer_requires_order_and_aborts_cleanly(tmp_path):
    path = tmp_path / "a.masks"
    with pytest.raises(ValueError), MaskStoreWriter(path) as w:
        w.add(5, 1, _mask())
        w.add(5, 1, _mask())  # duplicate key
    assert not path.exists() and not path.with_name("a.masks.partial").exists()
    # Unclosed writer (e.g. cancelled analysis) leaves nothing behind.
    with MaskStoreWriter(path) as w:
        w.add(0, 1, _mask())
    assert list(tmp_path.iterdir()) == []


def test_writer_budgets_are_sticky(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "MAX_ENTRIES", 2)
    desc = _write(tmp_path / "a.masks", [(f, 1, _mask(f)) for f in range(5)])
    assert desc["entries"] == 2
    monkeypatch.setattr(S, "MAX_ENTRIES", 1000)
    monkeypatch.setattr(S, "MAX_MASKS_PER_FRAME", 1)
    desc = _write(tmp_path / "b.masks", [(0, 1, _mask()), (0, 2, _mask()), (1, 2, _mask())])
    assert desc["entries"] == 2  # second mask on frame 0 over the per-frame cap


def test_masks_only_on_observed_frames_and_known_tracks(tmp_path):
    path = tmp_path / "a.masks"
    entries = [(0, 1, _mask()), (1, 1, _mask()), (2, 9, _mask()), (20, 1, _mask())]
    desc = _write(path, entries)
    # Frame 1 is interpolated, track 9 doesn't exist, frame 20 has no box.
    with _reader(path, _analysis(frames=range(10), interpolated={1}), desc) as r:
        assert len(r) == 1 and set(r.masks_for_frame(0)) == {1}


def test_out_of_bounds_entries_are_dropped(tmp_path):
    path = tmp_path / "a.masks"
    desc = _write(path, [(0, 1, _mask(x=300)), (1, 1, _mask())])  # 300 + 30 > 320
    with _reader(path, _analysis(), desc) as r:
        assert r.masks_for_frame(0) == {} and len(r.masks_for_frame(1)) == 1


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda b: b"XXXXXXXX" + b[8:],  # magic
        lambda b: b[:-30],  # truncated
        lambda b: b[:-8] + b"XXXXXXXX",  # footer magic
        lambda b: b + b"\0",  # trailing bytes
    ],
)
def test_corrupt_file_is_ignored_entirely(tmp_path, corrupt):
    path = tmp_path / "a.masks"
    desc = _write(path, [(0, 1, _mask()), (1, 1, _mask())])
    path.write_bytes(corrupt(path.read_bytes()))
    analysis = _analysis()
    analysis.mask_store = desc
    with MaskStoreReader.for_analysis(analysis, tmp_path) as r:
        assert len(r) == 0 and r.masks_for_frame(0) == {}


def _patch_index(path, i, field, value):
    data = bytearray(path.read_bytes())
    index_at, count, _ = S.FOOTER.unpack(data[-S.FOOTER.size :])
    idx = np.frombuffer(bytes(data[index_at : index_at + count * 44]), dtype=S.INDEX_DTYPE).copy()
    idx[i][field] = value
    data[index_at : index_at + count * 44] = idx.tobytes()
    path.write_bytes(bytes(data))


def test_unsorted_or_duplicate_index_rejects_store(tmp_path):
    path = tmp_path / "a.masks"
    desc = _write(path, [(0, 1, _mask()), (1, 1, _mask()), (2, 1, _mask())])
    _patch_index(path, 2, "frame", 1)  # duplicate (1, 1)
    with _reader(path, _analysis(), desc) as r:
        assert len(r) == 0


def test_bad_payload_makes_only_that_mask_missing(tmp_path):
    path = tmp_path / "a.masks"
    good, bad = _mask(1), _mask(2)
    desc = _write(path, [(0, 1, bad), (1, 1, good)])
    _patch_index(path, 0, "runs", bad.run_count - 1)  # varint count mismatch
    with _reader(path, _analysis(), desc) as r:
        assert r.masks_for_frame(0) == {} and r.masks_for_frame(1) == {1: good}
    # Run sum mismatch: overwrite the first payload byte (first run length).
    desc = _write(path, [(0, 1, bad)])
    data = bytearray(path.read_bytes())
    data[S.HEADER.size] = (data[S.HEADER.size] + 1) & 0x7F
    path.write_bytes(bytes(data))
    with _reader(path, _analysis(), desc) as r:
        assert r.masks_for_frame(0) == {}


def test_store_from_another_analysis_is_ignored(tmp_path):
    path = tmp_path / "a.masks"
    desc = _write(path, [(0, 1, _mask())])
    with _reader(path, _analysis(), {**desc, "id": "0" * 32}) as r:
        assert len(r) == 0


@pytest.mark.parametrize("name", ["../a.masks", "sub/a.masks", "C:a.masks", "", ".hidden", 7])
def test_only_sibling_file_names_resolve(tmp_path, name):
    (tmp_path / "a.masks").write_bytes(b"")
    assert resolve({"format": "sfm1", "file": name}, tmp_path) is None
    assert resolve({"format": "sfm1", "file": "a.masks"}, tmp_path) == tmp_path / "a.masks"
    assert resolve({"format": "other", "file": "a.masks"}, tmp_path) is None


def test_range_is_bounded_and_cut_on_frame_boundary(tmp_path, monkeypatch):
    path = tmp_path / "a.masks"
    m = _mask()
    desc = _write(path, [(f, t, m) for f in range(10) for t in (1, 2)])
    with _reader(path, _analysis(), desc) as r:
        monkeypatch.setattr(S, "MAX_RANGE_FRAMES", 4)
        end, rows = r.rows(2, 50)
        assert end == 6 and [row[0] for row in rows[1]] == [2, 3, 4, 5]
        monkeypatch.setattr(S, "MAX_RANGE_FRAMES", 256)
        monkeypatch.setattr(S, "MAX_RANGE_RUNS", 3 * 2 * m.run_count)
        end, rows = r.rows(0, 10)
        assert end == 3 and [row[0] for row in rows[2]] == [0, 1, 2]


def test_varint_round_trip_including_large_and_zero_runs():
    runs = [0, 1, 127, 128, 16383, 16384, 2**28 + 5]
    assert S._decode_runs(S._encode_runs(runs), len(runs)).tolist() == runs
    assert S._decode_runs(b"\x80\x80\x80\x80\x80\x01", 1) is None  # > 5 bytes
    assert S._decode_runs(b"\x81", 1) is None  # unterminated


def test_payload_is_much_smaller_than_inline_json(tmp_path):
    from .conftest import ellipse_mask

    path = tmp_path / "a.masks"
    masks = [ellipse_mask(Box(40 + f, 20, 140 + f, 230), 320, 240) for f in range(20)]
    _write(path, [(f, 1, m) for f, m in enumerate(masks)])
    inline = len(json.dumps([m.to_row(f) for f, m in enumerate(masks)]))
    assert path.stat().st_size < inline / 2
    assert struct.calcsize("<QQ8s") == S.FOOTER.size
