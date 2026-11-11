# Development guide

Practical notes for working on SafeFrame. For *what* the system does and
why, see [ARCHITECTURE.md](ARCHITECTURE.md) and [SAFETY.md](SAFETY.md). For
branching and commit rules, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Repository layout

```
backend/                     uv project (Python 3.13), package `safeframe`
  src/safeframe/
    pipeline/
      types.py               Box, Track, AgeSample, VideoInfo, Analysis (+ JSON)
      detection.py           YOLO + BoT-SORT person tracker
      botsort_safeframe.yaml tuned tracker config (3 s buffer, ReID)
      faces.py               YuNet face detector, face → person assignment
      age.py                 ViT apparent-age estimator → P(under 18)
      policy.py              decision policy (keep in sync with frontend model.ts)
      smoothing.py           gap interpolation and end padding
      anonymize.py           Gaussian blur + greyscale
      analyze.py / render.py the two pipeline phases
      factory.py             builds real model components
    video/io.py              ffprobe, OpenCV decode, FFmpeg encode, preview proxy
    api/                     FastAPI app, job manager (lifecycle + retention), schemas
    config.py                SAFEFRAME_* settings
    cli.py                   safeframe process | analyze | render | serve
  tests/                     fast suite (synthetic video, scripted fakes) + `-m models`
frontend/                    React + Vite + TypeScript + Vitest
  src/api/                   typed client (XHR upload, SSE progress)
  src/review/model.ts        browser mirror of policy.py
  src/components/            Upload, Processing, Review (VideoStage, Timeline, TrackList), Export
scripts/check_forbidden_files.py   blocks footage, weights and secrets
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
(YuNet, SHA-256 pinned, and YOLO) and the Hugging Face cache (age model).

## Key defaults

| Setting | Value | Where |
| --- | --- | --- |
| Minor threshold, P(under 18) | ≥ 0.35 | `policy.DEFAULT_THRESHOLD` |
| Face samples before "adult" is possible | 3 | `policy.DEFAULT_MIN_FACE_SAMPLES` |
| Age sampling | every 3 frames, ≤ 40 samples per person | `AnalysisSettings` |
| Minimum usable face | 32 px | `AnalysisSettings.min_face_px` |
| Gap interpolation / end padding | 1.5 s / 0.25 s | `AnalysisSettings` |
| Region padding | 8 % per side | `anonymize.DEFAULT_PADDING` |
| Job expiry | 24 h idle | `SAFEFRAME_JOB_TTL_HOURS` |

## Keeping the preview honest

The review screen's blur preview must match the exported file. When you
change decision logic in `backend/src/safeframe/pipeline/policy.py`, make the
same change in `frontend/src/review/model.ts` and update both test files
(`tests/test_policy.py`, `src/review/model.test.ts`). The same applies to
region padding (`anonymize.DEFAULT_PADDING` ↔ `PADDING` in `VideoStage.tsx`).

## Testing notes

- Tests never use real footage. `tests/conftest.py` generates clips with
  FFmpeg's `testsrc2` and provides scripted fakes (`ScriptedTracker`,
  `HeadFaceDetector`, `SizeAgeEstimator`). Use the same fakes to run the API
  locally without models, for example when developing the UI.
- The `-m models` smoke test builds a clip from a sample photo shipped inside
  the Ultralytics package, so it also needs no footage in the repository.
- Starlette's `TestClient` uses `httpx2` (dev dependency).
- Download filenames containing spaces are sent RFC 5987-encoded
  (`filename*=utf-8''…`).

## Dependency notes

- The heavy ML stack lives in the `[ml]` extra; CI installs only the core
  dependencies because the fast suite does not need it.
- `lap` is declared explicitly. Without it, Ultralytics pip-installs it into
  the active environment at runtime.

## Roadmap

- Evaluate minor recall on a consented, labelled dataset kept outside the
  repository (see [SAFETY.md](SAFETY.md#how-to-evaluate-a-change)).
- Solid-fill anonymisation option for high-risk contexts.
- Keyframed (moving) manual regions.
- A tool to split a track when the tracker swaps identities in crowds.
- Exact timing for variable-frame-rate input.
- Authentication and a container image for non-local deployment.
