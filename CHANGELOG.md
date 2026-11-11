# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- People are blurred to their **silhouette** instead of their bounding box, so
  the background inside the box stays visible. The default person model is now
  `yolo11n-seg.pt` (instance segmentation, same tracker and ReID).
  Detection-only checkpoints such as `yolo11n.pt` still work and blur rectangles.
- Frames without a usable silhouette (interpolated gaps, padded ends, rejected
  or missing masks, over-budget masks, legacy analyses) fall back to the padded
  rectangle, in both the export and the review preview.
- Overlapping regions are each computed from the untouched frame, so they no
  longer blur already-blurred pixels.

### Added
- Optional `masks_v1` field on each track in the analysis JSON (validated,
  budgeted run-length encoding); legacy box-only analyses still load and render.
- Export fails, leaving no output, if a planned region has neither a usable
  mask nor a valid box.
- The CLI and job log report silhouette vs. box-fallback person-frames; the
  review screen labels each blurred person `silhouette` or `box` per frame.
- The analysis endpoint is gzip-compressed when the client accepts it.

## [0.1.0] — 2026-10-13

First release.

### Pipeline
- Person detection with Ultralytics YOLO and identity tracking with a tuned
  BoT-SORT (3 s track buffer, appearance ReID), so identities persist through
  head turns and brief occlusion.
- Face detection with OpenCV YuNet (checksum-pinned) and apparent-age
  estimation with a FairFace-trained ViT; P(under 18) is derived from the
  full age-range distribution.
- Per-person evidence aggregation and a fail-closed decision policy: people
  without enough face evidence are *uncertain* and blurred by default.
- Gap interpolation and end padding so blur never drops out on missed frames.
- Strong, size-proportional Gaussian blur followed by greyscale conversion.
- FFmpeg output at the original resolution and exact frame rate with the
  original audio; location and device metadata stripped.
- `safeframe process | analyze | render | serve` CLI.

### API
- FastAPI service with streamed uploads, server-sent progress events, review
  endpoints, render, download and delete.
- Data retention: source material deleted after export unless kept; idle
  jobs expire; interrupted jobs are removed on restart.

### UI
- Upload with retention choice and an always-visible notice that age
  estimation is probabilistic.
- Review screen with frame-accurate live blur preview, per-person timeline,
  flagged-segment navigation, sensitivity slider, Auto/Blur/Keep overrides,
  manual regions and veiled thumbnails.
- Export screen with download and immediate deletion.

### Fixed
- Faces are attributed to the person whose head position matches best, and
  ambiguous faces are dropped, instead of going to the smallest overlapping
  box (which could credit one person's face to another).
- `lap` is declared in the `[ml]` extra so Ultralytics no longer pip-installs
  it into the environment at runtime.

[0.1.0]: https://github.com/issamarida/meta-support/releases/tag/v0.1.0
