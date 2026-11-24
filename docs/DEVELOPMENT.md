# Development guide

Practical notes for working on SafeFrame. For *what* the system does and
why, see [ARCHITECTURE.md](ARCHITECTURE.md) and [SAFETY.md](SAFETY.md). For
branching and commit rules, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Repository layout

```
backend/                     uv project (Python 3.13), package `safeframe`
  src/safeframe/
    pipeline/
      types.py               Box, Track, BandHit, AgeSample, VideoInfo, Analysis (+ JSON)
      detection.py           YOLO-seg + BoT-SORT person tracker, mask extraction
      band.py                band spec, ArUco detector, printable band SVG
      wearers.py             band sighting → person (wrist / silhouette / box), YOLO-pose wrists
      context.py             guarded-zone registry, capture location (EXIF / ISO 6709)
      guarded_zones.json     illustrative sample registry
      masks.py               silhouette coverage, masks_v1 format, validation, budgets
      botsort_safeframe.yaml tuned tracker config (3 s buffer, ReID)
      faces.py               YuNet face detector, face → person assignment
      age.py                 ViT apparent-age estimator → P(under 18)
      policy.py              band + age decisions (keep in sync with frontend model.ts)
      smoothing.py           gap interpolation and end padding
      anonymize.py           Gaussian blur + greyscale inside silhouette or box
      analyze.py / render.py the two pipeline phases (photos are one-frame videos)
      factory.py             builds real model components
    video/io.py              ffprobe, OpenCV decode, FFmpeg encode, preview proxy
    api/                     FastAPI app, job manager (lifecycle + retention), schemas
    seal.py                  sealed originals (X25519 + AES-GCM), Shamir trustee shares
    config.py                SAFEFRAME_* settings
    cli.py                   safeframe process | analyze | render | band | trustees | unseal | inspect | serve
  tests/                     fast suite (synthetic video, scripted fakes) + `-m models`
frontend/                    React + Vite + TypeScript + Vitest
  src/api/                   typed client (XHR upload, SSE progress)
  src/review/model.ts        browser mirror of policy.py
  src/components/            Upload, Processing, Review (VideoStage, Timeline, TrackList, ContextBanner), Export
scripts/check_forbidden_files.py   blocks footage, sealed envelopes, trustee shares, weights and secrets
```

## Everyday commands

```bash
make install       # backend: uv sync --all-extras ; frontend: npm ci
make dev           # API on :8000 (reload) + Vite on :5173 (proxies /api)
make serve         # build the UI and serve everything from FastAPI on :8000
make test          # fast backend suite + frontend tests
make test-models   # real-model smoke test (downloads models on first run)
make lint          # ruff check, ruff format --check, tsc
make hygiene       # scan tracked files for footage, weights, secrets
make clean-data    # delete all local job data
```

Model weights are cached outside the repository: `~/.cache/safeframe/models`
(YuNet, SHA-256 pinned, and the YOLO seg and pose weights) and the Hugging Face
cache (age model).

## Key defaults

| Setting | Value | Where |
| --- | --- | --- |
| Band ids | 7, 19, 28, 42 (`DICT_4X4_50`) | `band.BAND_SEQUENCE` |
| Band search | every 2 frames; full frame + wrist crops ≤ 4× | `AnalysisSettings.band_every`, `band.MAX_ZOOM` |
| Smallest marker | 7 px side (10 px in the full-frame pass) | `band.MIN_MARKER_SIDE_PX` |
| Wrist attribution radius / ambiguity | 0.10 / 0.03 × body height | `wearers.WRIST_RADIUS`, `WRIST_AMBIGUITY` |
| Clean sightings for "wearer" | 1 | `policy.DEFAULT_MIN_BAND_HITS` |
| Child threshold, P(under 18) | ≥ 0.35 (guarded zones, event mode) | `policy.DEFAULT_THRESHOLD` |
| Face samples before age counts | 3 (1 for photos) | `policy.DEFAULT_MIN_FACE_SAMPLES` |
| Age sampling | every 3 frames, ≤ 40 samples per person, only where needed | `AnalysisSettings`, `policy.needs_age` |
| Minimum usable face | 32 px | `AnalysisSettings.min_face_px` |
| Gap interpolation / end padding | 1.5 s / 0.25 s | `AnalysisSettings` |
| Region padding (box fallback, manual) | 8 % per side | `anonymize.DEFAULT_PADDING` |
| Person model | `yolo11n-seg.pt` | `models.DEFAULT_YOLO_WEIGHTS` |
| Silhouette margin | 3 % of longer box side, 3–24 px | `masks.margin_px` |
| Mask budgets | 8 192 runs/mask, 1.5 M/track, 4 M/analysis | `masks.MAX_RUNS_*` |
| Job expiry | 24 h idle | `SAFEFRAME_JOB_TTL_HOURS` |
| Sealed envelope expiry | 30 days | `SAFEFRAME_SEAL_RETENTION_DAYS` |

## Keeping the preview honest

The review screen's blur preview must match the exported file. When you
change decision logic in `backend/src/safeframe/pipeline/policy.py`, make the
same change in `frontend/src/review/model.ts` and update both test files
(`tests/test_policy.py`, `src/review/model.test.ts`; both carry the same
decision matrix). `DecisionPolicy.for_analysis` ↔ `policyFor` (context locked
in, one face sample for photos) and `RenderPlan.honoured_bands` ↔
`Summary.honoured` must agree too. The same applies to
region padding (`anonymize.DEFAULT_PADDING` ↔ `PADDING` in `model.ts`) and to
mask validation and budgets (`pipeline/masks.py` ↔ `parseMaskRow`,
`indexTracks` and `MAX_RUNS_*` in `model.ts`), so the preview and the export
choose the same silhouette or box on every frame.

## Testing notes

- Tests never use real footage. `tests/conftest.py` generates clips with
  FFmpeg's `testsrc2` and provides scripted fakes (`ScriptedTracker`,
  `ScriptedBands`, `ScriptedWrists`, `HeadFaceDetector`, `SizeAgeEstimator`)
  and a `components(...)` helper. Use the same fakes to run the API locally
  without models, for example when developing the UI.
- Synthetic clips are **stamped with their frame index** (12 bits in 8×8 px
  blocks along the top row, FFmpeg `geq`), and `frame_index(frame)` reads it
  back. Fakes use it to know which frame they were given, which matters for
  pass 2 (it re-reads only some frames).
- Band detection tests use the real ArUco detector on perspective-warped
  `band_strip` images; no fake is needed.
- The `-m models` tests paste a band at a realistic size onto the wrist of a
  person in the Ultralytics sample photos (`bus.jpg` upscaled to ~14 MP, then
  a 4K pan), so they also need no footage in the repository.
- Starlette's `TestClient` uses `httpx2` (dev dependency).
- Download filenames containing spaces are sent RFC 5987-encoded
  (`filename*=utf-8''…`).

## Dependency notes

- The heavy ML stack lives in the `[ml]` extra; CI installs only the core
  dependencies because the fast suite does not need it.
- `lap` is declared explicitly. Without it, Ultralytics pip-installs it into
  the active environment at runtime.

## Roadmap

- Evaluate wearer recall by distance, angle and lighting with real printed and
  woven bands on a consented dataset kept outside the repository (see
  [SAFETY.md](SAFETY.md#how-to-evaluate-a-change)).
- A partner SDK (Swift: Vision body pose + person instance masks + CryptoKit
  sealing) and a public conformance suite built from the reference tests.
- Signed registry distribution and venue beacons ([PROTOCOL.md](PROTOCOL.md#5-guarded-zones)).
- A frame-addressable mask store (the unmerged `feature/silhouette-blur` mask-store
  commits) to lift the inline silhouette budget.
- Solid-fill anonymisation option for high-risk contexts.
- Keyframed (moving) manual regions.
- A tool to split a track when the tracker swaps identities in crowds.
- Exact timing for variable-frame-rate input.
- Authentication and a container image for non-local deployment.
