"""Render stage: apply blur decisions to every frame and encode the output."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from safeframe.pipeline.anonymize import DEFAULT_PADDING, ProtectedRegion, anonymize_frame
from safeframe.pipeline.interfaces import Cancelled, CancelToken
from safeframe.pipeline.maskstore import MaskStoreReader
from safeframe.pipeline.policy import DecisionPolicy, Override
from safeframe.pipeline.types import Analysis, Box, ManualRegion
from safeframe.video.io import VideoWriter, iter_frames


class RenderError(RuntimeError):
    """A planned region cannot be protected; the frame must not be written."""


@dataclass(slots=True)
class RenderPlan:
    # frame -> regions to protect, in deterministic order (tracks, then manual).
    regions: dict[int, list[ProtectedRegion]] = field(default_factory=dict)
    blurred_tracks: list[int] = field(default_factory=list)
    kept_tracks: list[int] = field(default_factory=list)

    def regions_for(self, frame: int) -> list[ProtectedRegion]:
        return self.regions.get(frame, [])

    def boxes_for(self, frame: int) -> list[Box]:
        """Fallback boxes of every planned region (the pre-silhouette view)."""
        return [r.box for r in self.regions.get(frame, [])]

    def coverage_stats(self, store: MaskStoreReader | None = None) -> dict[str, int]:
        """Track-region frames protected by silhouette vs box fallback.

        With ``store``, a track frame counts as silhouette when the store has a
        valid index entry for it (its payload is checked again at render time).
        """
        if store is not None and len(store):
            pairs = [
                (f, r.track_id)
                for f, regions in self.regions.items()
                for r in regions
                if r.kind == "track" and r.mask is None
            ]
            from_store = store.count_for(pairs)
            stats = self.coverage_stats()
            return {**stats, "mask": stats["mask"] + from_store, "box": stats["box"] - from_store}
        stats = {"mask": 0, "box": 0, "manual": 0}
        for regions in self.regions.values():
            for r in regions:
                if r.kind == "manual":
                    stats["manual"] += 1
                else:
                    stats["mask" if r.mask is not None else "box"] += 1
        return stats


def plan_render(
    analysis: Analysis,
    policy: DecisionPolicy,
    overrides: Mapping[int, Override] | None = None,
    manual_regions: Iterable[ManualRegion] = (),
) -> RenderPlan:
    """Decide what to anonymise on which frames.

    A blurred track contributes its silhouette on frames with a valid mask and
    its padded box everywhere else. Kept tracks contribute nothing, so they can
    never undo protection from another region.
    """
    overrides = overrides or {}
    regions: dict[int, list[ProtectedRegion]] = defaultdict(list)
    plan = RenderPlan()
    for track in analysis.tracks:
        if policy.should_blur(track, overrides.get(track.track_id)):
            plan.blurred_tracks.append(track.track_id)
            for frame in sorted(track.boxes):
                regions[frame].append(
                    ProtectedRegion(
                        box=track.boxes[frame],
                        mask=track.mask_for(frame),
                        kind="track",
                        track_id=track.track_id,
                    )
                )
        else:
            plan.kept_tracks.append(track.track_id)
    last = max(analysis.video.frame_count - 1, 0)
    for region in manual_regions:
        for frame in range(max(0, region.start_frame), min(region.end_frame, last) + 1):
            regions[frame].append(ProtectedRegion(box=region.box, kind="manual"))
    plan.regions = dict(regions)
    return plan


def _with_store_masks(
    regions: list[ProtectedRegion], frame: int, store: MaskStoreReader | None
) -> list[ProtectedRegion]:
    if store is None or not len(store):
        return regions
    masks = store.masks_for_frame(frame)
    if not masks:
        return regions
    return [
        replace(r, mask=masks[r.track_id])
        if r.kind == "track" and r.mask is None and r.track_id in masks
        else r
        for r in regions
    ]


def _check_protectable(frame: int, regions: list[ProtectedRegion], w: int, h: int) -> None:
    for r in regions:
        if r.mask is not None:
            continue
        b = r.box
        coords = (b.x1, b.y1, b.x2, b.y2)
        clipped = b.clip(w, h)
        if not all(map(math.isfinite, coords)) or clipped.width < 1 or clipped.height < 1:
            who = f"track {r.track_id}" if r.track_id is not None else "a manual region"
            raise RenderError(f"frame {frame}: no usable mask or box for {who}")


def render_video(
    source: Path,
    output: Path,
    analysis: Analysis,
    plan: RenderPlan,
    progress: Callable[[float], None] | None = None,
    cancel: CancelToken | None = None,
    padding: float = DEFAULT_PADDING,
    masks: MaskStoreReader | None = None,
) -> dict[str, int]:
    """Write ``output`` with every planned region blurred and greyscaled.

    Track regions without an inline mask take their silhouette from ``masks``
    (the analysis' on-disk store) when it has a valid one for that frame.
    Resolution, exact frame rate and audio are preserved; untouched pixels are
    only re-encoded (visually lossless CRF 18). Raises :class:`RenderError`
    before writing a frame whose planned region has neither a valid mask nor a
    valid fallback box; the writer then deletes the partial output.

    Returns the person-frames actually protected by silhouette and by box.
    """
    info = analysis.video
    total = max(info.frame_count, 1)
    stats = {"mask": 0, "box": 0, "manual": 0}
    with VideoWriter(output, info, audio_source=source) as writer:
        for index, frame in iter_frames(source):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            regions = _with_store_masks(plan.regions_for(index), index, masks)
            if regions:
                _check_protectable(index, regions, frame.shape[1], frame.shape[0])
                anonymize_frame(frame, regions, padding)
                for r in regions:
                    kind = "manual" if r.kind == "manual" else "mask" if r.mask else "box"
                    stats[kind] += 1
            writer.write(frame)
            if progress is not None and index % 10 == 0:
                progress(min(0.999, (index + 1) / total))
    if progress is not None:
        progress(1.0)
    return stats
