# Architecture

```
frontend/  React + Vite + TypeScript review UI
backend/   Python package `safeframe`
  src/safeframe/
    pipeline/   detection, bands, attribution, context, age, policy, smoothing, render
    video/      ffprobe / OpenCV decode / FFmpeg encode
    api/        FastAPI app, job manager, schemas
    seal.py     sealed originals and trustee shares
    cli.py      `safeframe process|analyze|render|band|trustees|unseal|inspect|serve`
    models.py   weight download + checksum verification
scripts/   repository hygiene checks
```

## Two-phase pipeline

Processing is split into **analysis** and **render** so that a human can review
between them, and so that changing a threshold or overriding a decision never
needs the models to run again. Photos go through the same pipeline as a
one-frame video (`VideoInfo.media == "photo"`).

```mermaid
sequenceDiagram
    participant UI
    participant API
    participant Worker
    UI->>API: POST /api/jobs (raw body, streamed to disk; guarded, protect_minors, location?)
    API->>API: sniff photo / ffprobe video, read location once → CaptureContext
    API->>Worker: queue analysis
    Worker->>Worker: preview proxy (H.264 ≤720p, CFR; one frame for photos)
    Worker->>Worker: pass 1: YOLO-seg + BoT-SORT, pose wrists, ArUco bands → attribution
    Worker->>Worker: pass 2 (only if needed): YuNet + ViT for the tracks the policy can use
    Worker->>Worker: analysis.json (boxes, masks, bands, context), thumbnails
    UI-->>API: GET /events (SSE progress)
    UI->>API: GET /analysis, /preview, /thumbnail
    Note over UI: Review: overrides, manual regions, sensitivity where age matters.<br/>Live preview uses the same policy as the renderer.
    UI->>API: POST /render {threshold, min_band_hits, overrides, manual_regions}
    API->>Worker: queue render
    Worker->>Worker: blur+greyscale in silhouette or box → FFmpeg / image encoder, no metadata
    Worker->>Worker: seal original if a band was honoured and an escrow key is set
    Worker->>Worker: delete upload, preview, thumbnails (unless kept)
    UI->>API: GET /download, GET /sealed
    UI->>API: DELETE /jobs/{id}
```

### Analysis (`pipeline/analyze.py`)

`analyze_video(path, components, context)` and `analyze_photo(...)` take a
`Components` bundle (`PersonTracker`, `BandDetector`, `FaceDetector`,
`AgeEstimator`, optional `WristLocator`, all `Protocol`s in
`pipeline/interfaces.py`) and a `CaptureContext`.

**Pass 1, every frame:**

1. `PersonTracker.update(frame)` → `PersonObservation(track_id, box, conf, mask)`.
   The default `YoloPersonTracker` runs YOLO instance segmentation
   (`yolo11n-seg.pt`, `classes=[person]`, low confidence 0.1 so BoT-SORT's
   second association stage can keep tracks alive, `retina_masks=True`) with a
   tuned BoT-SORT config: a 3-second track buffer and appearance ReID so
   identities survive occlusion. Photos use untracked prediction (conf ≥ 0.25)
   and number people left to right.
2. Every `band_every` frames (2; every photo): `WristLocator.locate(frame)`
   (YOLO pose, keypoints 9 and 10) gives wrists, `wearers.wrist_rois` turns
   them into square crops (0.22 × body height, ≥ 40 px), and
   `BandDetector.detect(frame, rois)` returns `BandDetection`s from the full
   frame and the zoomed crops (see *Bands* below). `wearers.frame_hits`
   attributes them and adds at most one `BandHit(frame, via, ambiguous,
   side_px)` per track per frame; unattributed markers record the frame in
   `orphan_band_frames`.

**Pass 2, only for tracks the policy can use** (`policy.needs_age`: band
wearers in a guarded zone, everyone in event mode). Frames are re-read every
`age_every` (3) frames; faces from `FaceDetector.detect` are assigned with
`assign_faces` against **everyone** present (so one person's face can't be
credited to another), but only wanted tracks' faces are cropped and passed to
`AgeEstimator.estimate`. Faces with two near-equal candidates are dropped.
Outside guarded zones and event mode, pass 2 doesn't run and no face is ever
cropped.

After the passes, masks are trimmed to the payload budgets, samples are
aggregated (`policy.aggregate`) and each track is smoothed
(`smoothing.smooth_track`): gaps ≤ 1.5 s are linearly interpolated and the
first/last box is held for 0.25 s. Synthetic frames get boxes only.

### Bands (`pipeline/band.py`)

ArUco `DICT_4X4_50`, reserved ids `7, 19, 28, 42` (`BAND_SEQUENCE`). Only those
ids are accepted, with `errorCorrectionRate = 0.35` and sub-pixel corner
refinement. The full-frame pass sets `minMarkerPerimeterRate` from the image
size so markers below ~10 px a side aren't searched for (in a 4K frame a
rate of 0.004 produced ~4 000 candidates and ~800 ms per call; the derived
rate takes ~12 ms); the zoomed wrist passes find the small ones. Detections
below 7 px a side are discarded and duplicates across passes are merged.

`band_strip` renders a strip for tests and `band_sheet_svg` the A4 sheet:
child bands are 150 × 20 mm (module ≈ 2.2 mm), adult 190 × 24 mm. See
[PROTOCOL.md](PROTOCOL.md) for the full specification.

### Attribution (`pipeline/wearers.py`)

Pose people are matched to tracked people by box IoU (≥ 0.4, greedy). For
each marker: a wrist keypoint (confidence ≥ 0.3) within `WRIST_RADIUS` = 0.1 ×
body height wins; wrists within `WRIST_AMBIGUITY` = 0.03 of the best make the
hit ambiguous. Otherwise the person whose silhouette contains the marker
centre (`BODY`), otherwise whose box (+5 %) contains it (`BOX`); several
matches are ambiguous. Per frame a track keeps its best hit (clean over
ambiguous, then wrist > body > box, then the larger marker). Tracks keep the
first 200 hits plus total counts.

### Capture context (`pipeline/context.py`)

`resolve_context(location, registry, declared_guarded, protect_minors)` returns
a `CaptureContext(guarded, guarded_source, zone_name, zone_kind,
protect_minors, had_location)`. `guarded_source` is `registry` when the
location falls in a `Zone` (haversine distance ≤ radius), `user` when only
declared. A declaration can't remove a registry match. Locations come from the
request, or from the file: `photo_location` reads the EXIF GPS IFD with Pillow,
`video_location` reads the ISO 6709 tag (`com.apple.quicktime.location.iso6709`,
`location`) with ffprobe. The location itself is never stored: the job and the
analysis keep only the context. `ZoneRegistry.packaged()` loads the bundled
sample `guarded_zones.json`; `SAFEFRAME_ZONE_REGISTRY` points to a real one.

### Silhouettes (`pipeline/masks.py`, `pipeline/detection.py`)

**Alignment.** Masks come from the *same* tracked result as the boxes; there
is no second inference pass. In ultralytics 8.4.176 the tracker callback
(`trackers/track.py: on_predict_postprocess_end`) re-indexes the whole
`Results` with each surviving track's detection index (`result[idx]`), which
reorders `boxes` and `masks` together, then replaces the boxes with the
tracker's. A spike on three synthetic clips checked every tracked row's mask
against its own detection box: 0 mismatches over 971 rows, 131 of them in
frames the tracker had reordered. `tests/test_masks.py` pins the adapter
contract with reordered fake rows.

**Coordinates.** With `retina_masks=True`, Ultralytics upsamples prototypes and
removes the letterbox itself (`ops.process_mask_native` → `scale_masks`), so
`masks.data` is `(N, H, W)` at the decoded frame size. Any other shape is
discarded (box fallback), never stretched. Ultralytics crops each mask to its
detection box.

**Empty masks.** The segment predictor drops detections whose mask is empty
*unless* native ReID features are being captured. Our tracker config keeps
`with_reid: True, model: auto`, so no person box is lost; the tracker logs a
warning if that ever stops being true.

**Binarisation and coverage.** Foreground = mask logit > 0. The stored mask is
the final coverage: foreground dilated (elliptical kernel) by
`clamp(round(0.03 × longer box side), 3, 24)` px, clipped to the frame, never
clipped to the box.

**Quality gates** (raw mask vs. the frame-clipped box; failure → box fallback):
≥ 24 foreground px; foreground ≥ 12 % of box area; ≥ 90 % of foreground inside
the box grown by 15 %; IoU(mask extent, box) ≥ 0.3; ≤ 8 192 runs. Valid
geometry does not prove the person is fully covered.

**Wire format** (`masks_v1` on each track; `boxes` unchanged):
`[frame, x, y, width, height, runs]`, integer source pixels, half-open crop,
row-major run lengths alternating background/foreground starting with
background (first run may be 0, later runs > 0), summing to `width × height`.
Holes and disconnected parts are preserved.

**Loading** (`Track.from_dict`, mirrored in `frontend/src/review/model.ts`):
rows are validated (shape, integer types, bounds, run count before summing,
exact sum) before anything area-sized is allocated. A row is used only for a
frame with an *observed* box; a frame listed twice is ambiguous and uses the
box. Budgets: 8 192 runs per mask, 1.5 M per track, 4 M per analysis, applied
in file order; once a budget is exceeded every later mask in that scope is
dropped. Invalid entries become missing, never fatal. The API re-serialises
the validated model (gzip when accepted), so the UI receives exactly what the
renderer will use. Masks live inside `analysis.json` and share its job
lifetime, access rules and deletion; they are never logged.

**Payload.** A person mask costs about 1 000 runs (~4 KB of JSON). Measured:
a 15 s, 4-person, 810×900 clip produced 6.5 MB of JSON (0.8 MB gzipped) versus
65 KB box-only. The budgets therefore cap the inline payload at roughly 20 MB,
which corresponds to about 4 000 silhouette person-frames per video; frames
beyond that use the padded box. Moving masks to a bounded, frame-addressable
store would remove that cap.

Tests inject scripted fakes for every component, so the full pipeline,
including FFmpeg encoding, is tested in CI without downloading a single model.

### Decisions (`pipeline/policy.py`)

```
band   = WEARER    if clean sightings >= min_band_hits
         POSSIBLE  if any sighting (clean or ambiguous)
age    = UNCERTAIN if face_samples < min_face_samples (1 for photos)
         MINOR     if P(under 18) >= threshold
         ADULT     otherwise            (computed only when guarded or event mode)

decision = band            if band and not guarded
         = band            if band and guarded and age == MINOR
         = NOT_HONOURED    if band and guarded
         = MINOR           if event mode and age == MINOR
         = UNCERTAIN       if event mode and age == UNCERTAIN and not guarded
         = VISIBLE         otherwise
blur     = override if set, else decision in {WEARER, POSSIBLE, MINOR, UNCERTAIN}
```

`DecisionPolicy.for_analysis(analysis, ...)` locks the capture context in from
the analysis, so a render request can't change whether a capture was guarded.
The frontend re-implements exactly this in `frontend/src/review/model.ts`
(`classify`, `shouldBlur`, `policyFor`), with the same decision matrix in its
unit tests, so the browser preview matches the exported file.

### Render (`pipeline/render.py`, `pipeline/anonymize.py`, `video/io.py`)

`plan_render` turns analysis + policy + overrides + manual regions into a
`frame → [ProtectedRegion(box, mask?)]` map: blurred tracks contribute their
mask on frames that have one and their box otherwise; kept tracks contribute
nothing (so a Keep can never undo another region); manual regions are boxes.
`RenderPlan.boxes_for` still returns the fallback boxes, and
`coverage_stats()` counts silhouette vs. box person-frames (logged per job,
printed by the CLI). The plan records each track's policy decision and the
reason (the decision, or `override:blur` / `override:keep`);
`honoured_bands` lists blurred tracks whose own decision was `wearer` or
`possible` and decides whether the original is sealed.

`render_photo` applies the same regions to the single frame and writes JPEG
(quality 95), PNG or WebP with `cv2.imencode`, which writes no EXIF, GPS or
other metadata.

`render_video` decodes with OpenCV (rotation-aware) and, per frame, computes
each region's effect from an untouched copy of the frame:
`gaussian_blur_strong` (downscale → Gaussian → upscale → Gaussian, sigma
proportional to the ROI) then greyscale, over the padded box grown to contain
the silhouette. The effect is written only where the region's coverage is true
(the mask, or the whole padded box), in plan order. Overlaps therefore never
restore original pixels and never blur already-blurred pixels. If a planned
region has neither a mask nor a finite, non-empty box, `RenderError` is raised
before that frame is written; the FFmpeg writer then aborts and deletes the
output, and a web job returns to review with the error.

### Review preview (`frontend/src/components/VideoStage.tsx`)

The canvas overlay picks the same mask-or-box per frame as the renderer. For a
silhouette it rasterises the mask to preview pixels using separate x/y ratios
and a conservative rule (a preview pixel is covered if any covered source
pixel touches it; no partial alpha), draws the blurred video into an offscreen
canvas, keeps it only where the mask is opaque (`destination-in`), and
composites that onto the review canvas. Decoded rasters are LRU-cached (64).
Without Canvas filter support the same coverage is filled opaque grey. The
overlay sits on top of the unredacted preview; only the export has the blur
baked in.
Frames are piped as raw BGR into FFmpeg:

```
ffmpeg -f rawvideo -pix_fmt bgr24 -s WxH -framerate <exact rational> -i pipe:0
       -i <original> -map 0:v:0 -map 1:a? -c:a copy|aac
       -c:v libx264 -crf 18 -pix_fmt yuv420p -map_metadata -1 -movflags +faststart
```

## Sealed originals (`seal.py`)

`create_escrow(k, n)` generates an X25519 key pair and Shamir-splits the
private key over GF(2^521 − 1) into n `TrusteeShare`s; only the
`EscrowPublicKey` is kept by the server. `seal_file` writes the `SFSEAL1`
envelope (format in [PROTOCOL.md](PROTOCOL.md#6-sealed-originals)) with an
authenticated header (key id, created/expires, original name and size,
SHA-256 of the published copy, reason, capture context). `unseal_file`
combines any k shares, checks the key id, refuses expired envelopes unless
`allow_expired`, and verifies every chunk before writing the output to a
temporary file that is only moved into place when the whole envelope
authenticates.

The job manager seals after a successful render and before deleting the
source, when `SAFEFRAME_ESCROW_PUBLIC_KEY` is set and `honoured_bands` is not
empty. The envelope (`original.sfseal`) follows the job's retention and is
served by `GET /api/jobs/{id}/sealed`.

## Jobs and retention (`api/jobs.py`)

- One job = one directory `data/jobs/<128-bit random id>/`. The id is the only
  handle to a job; it is validated against `^[0-9a-f]{32}$` before touching disk.
- Uploads are streamed from the request body straight into the job directory
  (no multipart temp files in shared `/tmp`), with the size limit enforced while
  streaming.
- A single worker thread runs analysis and renders sequentially (they share the GPU).
- Uploads are sniffed: JPEG, PNG and WebP magic bytes make a photo job; anything
  else must pass ffprobe as a video.
- After a successful render, `upload`, `preview.mp4` and `thumbs/` are deleted
  unless the user opted to keep the upload. A sealed original, when written,
  stays with the job until it expires or is deleted.
- `DELETE /api/jobs/{id}` cancels any running work and removes the directory.
- A sweeper deletes jobs idle for longer than `SAFEFRAME_JOB_TTL_HOURS`
  (default 24) and any directory it does not recognise. On start-up, jobs that
  were interrupted mid-processing are deleted.
- Media responses carry `Cache-Control: no-store`.

SafeFrame is designed to run locally or on a trusted self-hosted machine. It
has no user accounts: anyone who knows a job id can access that job, so do not
expose the server to the internet without putting authentication in front of it.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SAFEFRAME_DATA_DIR` | `data` | Job storage |
| `SAFEFRAME_MAX_UPLOAD_MB` | `4096` | Upload size limit |
| `SAFEFRAME_JOB_TTL_HOURS` | `24` | Delete jobs after this long without activity |
| `SAFEFRAME_DEVICE` | auto | `cpu`, `cuda`, `cuda:1`, `mps` |
| `SAFEFRAME_YOLO_WEIGHTS` | `yolo11n-seg.pt` | Ultralytics segmentation model (silhouettes) or detection model (boxes only); CLI: `--yolo` |
| `SAFEFRAME_POSE_WEIGHTS` | `yolo11n-pose.pt` | Pose model for wrist attribution; empty disables it (silhouette/box attribution only); CLI: `--pose` |
| `SAFEFRAME_ZONE_REGISTRY` | packaged sample | Guarded-zone registry JSON; CLI: `--zones` |
| `SAFEFRAME_ESCROW_PUBLIC_KEY` | unset | `escrow.pub.json` from `safeframe trustees init`; enables sealed originals |
| `SAFEFRAME_SEAL_RETENTION_DAYS` | `30` | Sealed envelopes expire after this long |
| `SAFEFRAME_AGE_MODEL` | `nateraw/vit-age-classifier` | Hugging Face image-classification model with age-range labels |
| `SAFEFRAME_AGE_MODEL_REVISION` | latest | Pin a model revision |
| `SAFEFRAME_MODEL_DIR` | `~/.cache/safeframe/models` | Weight cache |
| `SAFEFRAME_WARM_UP_MODELS` | `true` | Load person, pose and face models at start-up (the age model loads on first use) |
