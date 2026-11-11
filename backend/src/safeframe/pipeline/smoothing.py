"""Temporal smoothing of tracks so blur never flickers off.

Detectors miss people for a frame or two (motion blur, partial occlusion),
and trackers report a track only on frames where it was matched. For a privacy
tool every missed frame is a leak, so we:

1. linearly interpolate boxes across short gaps inside a track, and
2. pad each track a few frames before its first and after its last detection.

Synthetic (interpolated or padded) frames get a box but never a silhouette:
copying a mask to a new position could leave part of the person uncovered, so
those frames are blurred with the padded box instead.
"""

from __future__ import annotations

import itertools

from safeframe.pipeline.types import Track


def fill_gaps(track: Track, max_gap: int) -> None:
    """Interpolate missing frames for gaps of at most ``max_gap`` frames."""
    frames = sorted(track.boxes)
    for a, b in itertools.pairwise(frames):
        gap = b - a - 1
        if gap <= 0 or gap > max_gap:
            continue
        box_a, box_b = track.boxes[a], track.boxes[b]
        for f in range(a + 1, b):
            track.boxes[f] = box_a.lerp(box_b, (f - a) / (b - a))
            track.interpolated.add(f)


def pad_ends(track: Track, pad: int, frame_count: int) -> None:
    """Hold the first/last box for ``pad`` frames before/after the track."""
    if not track.boxes or pad <= 0:
        return
    first, last = track.first_frame, track.last_frame
    for f in range(max(0, first - pad), first):
        track.boxes[f] = track.boxes[first]
        track.interpolated.add(f)
    upper = last + pad if frame_count <= 0 else min(frame_count - 1, last + pad)
    for f in range(last + 1, upper + 1):
        track.boxes[f] = track.boxes[last]
        track.interpolated.add(f)


def smooth_track(track: Track, max_gap: int, pad: int, frame_count: int) -> Track:
    fill_gaps(track, max_gap)
    pad_ends(track, pad, frame_count)
    return track
