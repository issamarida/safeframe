"""Bounded, frame-addressable on-disk store for silhouette masks ("sfm1").

Inline ``masks_v1`` JSON costs ~4 KB per person-frame, which caps silhouettes
at about a minute of video. This store keeps masks out of the analysis JSON:
analysis streams them to disk, render reads them frame by frame, and the API
serves them to the review UI in bounded frame ranges. Nothing holds every mask
in memory.

Layout (all integers little-endian)::

    header   8s magic "SFMASK01" | 16s store id (uuid4 bytes) | 8 reserved
    payload  per mask: its runs as unsigned LEB128 varints
    index    N x 44-byte records, strictly sorted by (frame, track_id):
             u32 frame | i64 track_id | u64 offset | u32 length
             | u32 x | u32 y | u32 width | u32 height | u32 run_count
    footer   u64 index offset | u64 N | 8s magic "SFMEND01"

Mask geometry and run semantics are exactly those of ``masks_v1`` (see
``masks.py``). The analysis JSON links the store with
``{"format": "sfm1", "file": <basename>, "id": <hex>, "entries": N}``.

Fail-safe rules: the writer stops accepting masks once a budget is reached
(those frames use the padded box). A reader that finds a wrong magic, id,
size or an index that is not strictly sorted (including duplicates) ignores
the whole store. Individual index entries that are out of bounds, not on an
observed box frame, or beyond the per-frame cap are dropped. Payloads are
validated when decoded; a bad payload makes that one mask missing. Every
"missing" mask means box fallback, never an unprotected frame.
"""

from __future__ import annotations

import logging
import os
import re
import struct
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from safeframe.pipeline import masks as M
from safeframe.pipeline.masks import Mask

log = logging.getLogger(__name__)

FORMAT = "sfm1"
MAGIC = b"SFMASK01"
END_MAGIC = b"SFMEND01"
HEADER = struct.Struct("<8s16s8x")
FOOTER = struct.Struct("<QQ8s")
INDEX_DTYPE = np.dtype(
    [
        ("frame", "<u4"),
        ("track", "<i8"),
        ("offset", "<u8"),
        ("length", "<u4"),
        ("x", "<u4"),
        ("y", "<u4"),
        ("w", "<u4"),
        ("h", "<u4"),
        ("runs", "<u4"),
    ]
)
assert INDEX_DTYPE.itemsize == 44

# Budgets. ~1.5 KB per typical person mask, so 1 GiB is several hours of a
# few people; frames past a budget fall back to the padded box.
MAX_STORE_BYTES = 1 << 30
MAX_ENTRIES = 1_000_000
MAX_MASKS_PER_FRAME = 64
# A varint for a run length up to 2**35 - 1 never needs more than 5 bytes.
MAX_VARINT_BYTES = 5
# Bounds for one API range response.
MAX_RANGE_FRAMES = 256
MAX_RANGE_RUNS = 1_000_000

_FILENAME = re.compile(r"^[\w][\w.-]{0,99}$")


def _encode_runs(runs: Iterable[int]) -> bytes:
    out = bytearray()
    for r in runs:
        while True:
            byte = r & 0x7F
            r >>= 7
            if r:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
    return bytes(out)


def _decode_runs(data: bytes, count: int) -> np.ndarray | None:
    """Vectorised LEB128 decode. None unless ``data`` holds exactly ``count`` varints."""
    b = np.frombuffer(data, dtype=np.uint8)
    if b.size == 0:
        return None
    ends = np.flatnonzero((b & 0x80) == 0)
    if ends.size != count or ends[-1] != b.size - 1:
        return None
    starts = np.concatenate(([0], ends[:-1] + 1))
    if np.any(ends - starts >= MAX_VARINT_BYTES):
        return None
    group = np.repeat(np.arange(count), ends - starts + 1)
    pos = np.arange(b.size) - starts[group]
    vals = (b & 0x7F).astype(np.int64) << (7 * pos).astype(np.int64)
    return np.add.reduceat(vals, starts)


class MaskStoreWriter:
    """Streams masks to ``path`` (via a ``.partial`` file renamed on close)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + ".partial")
        self.id = uuid.uuid4().hex
        self._fh = self.partial.open("wb")
        self._fh.write(HEADER.pack(MAGIC, bytes.fromhex(self.id)))
        self._pos = HEADER.size
        self._index: list[tuple] = []
        self._last: tuple[int, int] = (-1, -1)
        self._per_frame = 0
        self.full = False
        self.dropped = 0
        self.closed = False

    @property
    def entries(self) -> int:
        return len(self._index)

    def add(self, frame: int, track_id: int, mask: Mask) -> bool:
        """Append one mask. Calls must be in strictly increasing (frame, track_id)
        order. Returns False (box fallback) when a budget is exhausted."""
        key = (frame, track_id)
        if key <= self._last:
            raise ValueError(f"masks must be added in (frame, track) order, got {key}")
        self._per_frame = self._per_frame + 1 if frame == self._last[0] else 1
        self._last = key
        if self.full or self._per_frame > MAX_MASKS_PER_FRAME:
            self.dropped += 1
            return False
        payload = _encode_runs(mask.runs)
        projected = self._pos + len(payload) + (len(self._index) + 1) * 44 + FOOTER.size
        if len(self._index) >= MAX_ENTRIES or projected > MAX_STORE_BYTES:
            # Sticky: once full, every later mask uses the box, so the result
            # never depends on which later masks happen to be small.
            self.full = True
            self.dropped += 1
            return False
        self._fh.write(payload)
        self._index.append(
            (
                frame,
                track_id,
                self._pos,
                len(payload),
                mask.x,
                mask.y,
                mask.width,
                mask.height,
                mask.run_count,
            )
        )
        self._pos += len(payload)
        return True

    def close(self) -> dict:
        """Write index and footer, publish atomically, return the JSON descriptor."""
        index = np.array(self._index, dtype=INDEX_DTYPE)
        self._fh.write(index.tobytes())
        self._fh.write(FOOTER.pack(self._pos, len(self._index), END_MAGIC))
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self.partial.replace(self.path)
        self.closed = True
        if self.dropped:
            log.warning("mask store full: %d masks fall back to boxes", self.dropped)
        return self.descriptor()

    def descriptor(self) -> dict:
        return {"format": FORMAT, "file": self.path.name, "id": self.id, "entries": self.entries}

    def abort(self) -> None:
        """Discard everything written (failure or cancellation)."""
        if not self._fh.closed:
            self._fh.close()
        self.partial.unlink(missing_ok=True)
        if not self.closed:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> MaskStoreWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None or not self.closed:
            self.abort()


def resolve(descriptor: object, base_dir: Path) -> Path | None:
    """Path of the store an analysis links to, or None if it can't be trusted.

    Only a bare file name next to the analysis is accepted (no paths)."""
    if not isinstance(descriptor, dict) or descriptor.get("format") != FORMAT:
        return None
    name = descriptor.get("file")
    if not isinstance(name, str) or not _FILENAME.match(name) or ".." in name:
        return None
    path = Path(base_dir) / name
    return path if path.is_file() else None


class MaskStoreReader:
    """Validated read access to one store for one analysis.

    ``observed(track_id) -> frames`` gives each track's observed
    (non-interpolated) box frames; masks anywhere else are dropped.
    """

    def __init__(
        self,
        path: Path,
        descriptor: dict,
        frame_width: int,
        frame_height: int,
        frame_count: int,
        observed: Callable[[int], Iterable[int]],
        track_ids: Iterable[int],
    ) -> None:
        self.path = Path(path)
        self.width, self.height = frame_width, frame_height
        self._fh = None
        self.index = np.zeros(0, dtype=INDEX_DTYPE)
        try:
            self._fh = self.path.open("rb")
            self.index = self._load(descriptor, frame_count, observed, set(track_ids))
        except (OSError, ValueError) as exc:
            log.warning("ignoring mask store %s: %s; using boxes", self.path.name, exc)
            self.index = np.zeros(0, dtype=INDEX_DTYPE)

    @classmethod
    def for_analysis(cls, analysis, base_dir: Path) -> MaskStoreReader | None:
        path = resolve(getattr(analysis, "mask_store", None), base_dir)
        if path is None:
            return None
        tracks = {t.track_id: t for t in analysis.tracks}
        v = analysis.video
        return cls(
            path,
            analysis.mask_store,
            v.width,
            v.height,
            v.frame_count,
            lambda tid: set(tracks[tid].boxes) - tracks[tid].interpolated,
            tracks,
        )

    # --- loading ---------------------------------------------------------------
    def _load(self, descriptor, frame_count, observed, track_ids) -> np.ndarray:
        assert self._fh is not None
        size = os.fstat(self._fh.fileno()).st_size
        if size > MAX_STORE_BYTES or size < HEADER.size + FOOTER.size:
            raise ValueError("bad size")
        magic, raw_id = HEADER.unpack(self._fh.read(HEADER.size))
        if magic != MAGIC:
            raise ValueError("bad magic")
        if not isinstance(descriptor, dict) or descriptor.get("id") != raw_id.hex():
            raise ValueError("store does not belong to this analysis")
        self._fh.seek(size - FOOTER.size)
        index_at, count, end = FOOTER.unpack(self._fh.read(FOOTER.size))
        if end != END_MAGIC or count > MAX_ENTRIES:
            raise ValueError("bad footer")
        if index_at < HEADER.size or index_at + count * 44 + FOOTER.size != size:
            raise ValueError("bad index geometry")
        self._fh.seek(index_at)
        idx = np.frombuffer(self._fh.read(count * 44), dtype=INDEX_DTYPE)
        if idx.size != count:
            raise ValueError("truncated index")
        if count == 0:
            return idx
        # Strictly increasing (frame, track): anything else (including a
        # duplicate entry) means the file can't be trusted at all.
        f, t = idx["frame"].astype(np.int64), idx["track"]
        if np.any((f[1:] < f[:-1]) | ((f[1:] == f[:-1]) & (t[1:] <= t[:-1]))):
            raise ValueError("index not strictly sorted")

        w, h = idx["w"].astype(np.int64), idx["h"].astype(np.int64)
        x, y = idx["x"].astype(np.int64), idx["y"].astype(np.int64)
        runs, length = idx["runs"].astype(np.int64), idx["length"].astype(np.int64)
        off = idx["offset"].astype(np.int64)
        ok = (
            (f < frame_count)
            & (w >= 1)
            & (h >= 1)
            & (x + w <= self.width)
            & (y + h <= self.height)
            & (runs >= 2)
            & (runs <= M.MAX_RUNS_PER_MASK)
            & (length >= runs)
            & (length <= runs * MAX_VARINT_BYTES)
            & (off >= HEADER.size)
            & (off + length <= index_at)
        )
        ok &= np.isin(t, np.fromiter(track_ids, dtype=np.int64))
        for tid in np.unique(t[ok]):
            sel = ok & (t == tid)
            frames = np.fromiter(observed(int(tid)), dtype=np.int64)
            ok[sel] = np.isin(f[sel], frames)
        kept = idx[ok]
        # Per-frame cap, in file order.
        if kept.size:
            kf = kept["frame"]
            first = np.searchsorted(kf, kf, side="left")
            kept = kept[np.arange(kf.size) - first < MAX_MASKS_PER_FRAME]
        dropped = count - kept.size
        if dropped:
            log.warning("dropped %d invalid mask store entries; using boxes", dropped)
        return np.array(kept)

    # --- access ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.index.size)

    def has(self, frame: int, track_id: int) -> bool:
        return self._slice(frame, frame + 1, track_id) is not None

    def _slice(self, start: int, end: int, track_id: int | None = None):
        frames = self.index["frame"]
        lo = int(np.searchsorted(frames, start, side="left"))
        hi = int(np.searchsorted(frames, end, side="left"))
        if lo >= hi:
            return None
        part = self.index[lo:hi]
        if track_id is not None:
            part = part[part["track"] == track_id]
        return part if part.size else None

    def _decode(self, rec) -> Mask | None:
        assert self._fh is not None
        self._fh.seek(int(rec["offset"]))
        data = self._fh.read(int(rec["length"]))
        runs = _decode_runs(data, int(rec["runs"]))
        if runs is None:
            return None
        w, h = int(rec["w"]), int(rec["h"])
        if runs[0] < 0 or np.any(runs[1:] <= 0) or int(runs.sum()) != w * h:
            return None
        return Mask(int(rec["x"]), int(rec["y"]), w, h, tuple(int(r) for r in runs))

    def masks_for_frame(self, frame: int) -> dict[int, Mask]:
        """Valid masks on ``frame`` keyed by track id (missing = use the box)."""
        part = self._slice(frame, frame + 1)
        if part is None:
            return {}
        out = {}
        for rec in part:
            mask = self._decode(rec)
            if mask is not None:
                out[int(rec["track"])] = mask
        return out

    def rows(self, start: int, end: int) -> tuple[int, dict[int, list[list]]]:
        """Masks for frames ``[start, end)`` as ``masks_v1`` rows grouped by track.

        The range is clamped to ``MAX_RANGE_FRAMES`` and cut at a frame
        boundary once ``MAX_RANGE_RUNS`` is reached. Returns the exclusive end
        actually covered and the rows.
        """
        end = min(end, start + MAX_RANGE_FRAMES)
        out: dict[int, list[list]] = {}
        part = self._slice(start, end)
        if part is None:
            return end, out
        total = 0
        for frame in np.unique(part["frame"]):
            recs = part[part["frame"] == frame]
            n = int(recs["runs"].sum())
            if total and total + n > MAX_RANGE_RUNS:
                return int(frame), out
            total += n
            for rec in recs:
                mask = self._decode(rec)
                if mask is not None:
                    out.setdefault(int(rec["track"]), []).append(mask.to_row(int(frame)))
        return end, out

    def count_for(self, track_frames: Iterable[tuple[int, int]]) -> int:
        """How many of the given (frame, track) pairs have an index entry."""
        if not self.index.size:
            return 0
        keys = self.index["frame"].astype(np.int64) * (1 << 32) + (self.index["track"] & 0xFFFFFFFF)
        want = np.fromiter(
            (f * (1 << 32) + (t & 0xFFFFFFFF) for f, t in track_frames), dtype=np.int64
        )
        return int(np.isin(want, keys).sum())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> MaskStoreReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
