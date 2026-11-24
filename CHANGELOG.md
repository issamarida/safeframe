# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

SafeFrame is now built around the **SafeFrame Band**: people are blurred
because they wear a band that asks not to be shown, not because a model
guessed their age.

### Added
- **Band spec v1** and detector: ArUco `DICT_4X4_50` markers with reserved ids
  7-19-28-42, identical on every band (no serial). Full-frame detection plus
  zoomed passes around wrists found by `yolo11n-pose.pt`. Printable A4 sheet
  via `safeframe band` and `GET /api/band.svg`.
- **Attribution** of each sighting to a person by wrist keypoint, then
  silhouette, then box. Ambiguous sightings protect every candidate
  (`possible`); unattributed sightings are reported for review.
- **Guarded zones**: in banks, ATMs, courts, police stations, prisons,
  airports, border crossings, government buildings, jewellers, pharmacies and
  casinos, a band protects only people with positive evidence of being a
  child. The zone comes from the file's own GPS (EXIF / ISO 6709) matched
  against a registry, or from the uploader; it can only make rules stricter,
  and the location is never stored.
- **Sealed originals**: when a band is honoured, the unblurred original can be
  sealed (X25519 + HKDF-SHA256 + AES-256-GCM, chunked and authenticated) to an
  escrow key Shamir-split k-of-n between trustees. `safeframe trustees init`,
  `safeframe unseal`, `safeframe inspect`, `--seal-to`,
  `SAFEFRAME_ESCROW_PUBLIC_KEY`, `GET /api/jobs/{id}/sealed`; 30-day expiry.
- **Photos** (JPEG, PNG, WebP) in the CLI, API and UI, re-encoded without any
  metadata.
- **Event mode** (`--event-mode`, `protect_minors`): additionally blur anyone
  who may be under 18, band or not (the v0.1 behaviour).
- Review UI: decision badges, band sightings on the timeline, a capture-context
  banner, orphan-sighting warnings, a consent confirmation before keeping a
  wearer, sealed-envelope download, upload options for guarded places and
  event mode.
- [docs/PROTOCOL.md](docs/PROTOCOL.md): the partner protocol (band, detection,
  attribution, decisions, guarded zones, sealing, range, certification).
- `AGENTS.md`: a single context file for contributors and coding agents.

### Changed
- **Breaking:** `analyze_video` takes a `Components` bundle and an optional
  `CaptureContext`; `TrackStatus` is now `AgeStatus`;
  `DecisionPolicy.classify` returns a `Decision`
  (`wearer, possible, not_honoured, minor, uncertain, visible`).
- Analysis is two-pass and data-minimised: age is estimated only for band
  wearers in guarded zones, or for everyone in event mode. Nobody else's face
  is cropped or analysed, and the age model is no longer loaded at start-up.
- `cryptography` and `pillow` are core dependencies.
- The forbidden-files check also rejects `.sfseal` envelopes and trustee
  shares.
- People are blurred to their **silhouette** instead of their bounding box, so
  the background inside the box stays visible. The default person model is now
  `yolo11n-seg.pt` (instance segmentation, same tracker and ReID).
  Detection-only checkpoints such as `yolo11n.pt` still work and blur rectangles.
- Frames without a usable silhouette (interpolated gaps, padded ends, rejected
  or missing masks, over-budget masks, legacy analyses) fall back to the padded
  rectangle, in both the export and the review preview.
- Overlapping regions are each computed from the untouched frame, so they no
  longer blur already-blurred pixels.
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

[Unreleased]: https://github.com/issamarida/safeframe/compare/v0.1.0...develop
[0.1.0]: https://github.com/issamarida/safeframe/releases/tag/v0.1.0
