# SafeFrame — session handoff notes

Shared handoff notes for collaborators picking up the project. Last updated 2026-10-13.
See also [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Status

- **v0.1.0 released and pushed** to https://github.com/issamarida/meta-support
  (public). CI is green on `main` (0c5d554) and `develop` (87bbec9).
- All three stages are done: pipeline → API → UI.
- `develop` is ahead of `main` by docs only (docs/DEVELOPMENT.md, this file);
  they ship in the next release.

## Git conventions (user requirements)

- **Every commit is authored solely by `issamarida <issamaarida@gmail.com>`.
  NO `Co-Authored-By` / Claude trailers, ever.**
- Git Flow: `main` (releases + tags), `develop` (integration), branches
  `feature/*`, `fix/*`, `chore/*`, `release/vX.Y.Z`; always merge with `--no-ff`.
- Conventional Commits (`feat(pipeline): …`, `fix(api): …`, `docs: …`), granular,
  with explanatory bodies.
- Branches on origin: main, develop, chore/repo-foundation,
  feature/detection-tracking-pipeline, feature/processing-api,
  feature/review-ui, fix/face-to-person-attribution, release/v0.1.0; tag v0.1.0.
- Push works over HTTPS with stored credentials. `gh` is **not** logged in, so
  there are no PRs or branch protection via CLI. CI status can be read without
  auth: `curl -s "https://api.github.com/repos/issamarida/meta-support/actions/runs?branch=main&per_page=1"`.
- Never commit footage, model weights or secrets: `scripts/check_forbidden_files.py --all`.

## Layout

```
backend/   uv project, Python 3.13 (.python-version), package `safeframe`
  src/safeframe/pipeline/  types, detection (YOLO+BoT-SORT, botsort_safeframe.yaml),
                           faces (YuNet + assign_faces), age (ViT FairFace),
                           policy, smoothing, anonymize, analyze, render, factory
  src/safeframe/video/io.py  ffprobe / OpenCV decode / FFmpeg encode + preview
  src/safeframe/api/         app.py (routes, SSE), jobs.py (lifecycle/retention), schemas.py
  src/safeframe/config.py    SAFEFRAME_* env settings
  src/safeframe/cli.py       safeframe process|analyze|render|serve
  tests/                     47 fast tests (synthetic video, scripted fakes) + 1 `-m models` test
frontend/  React 19 + Vite 8 + TS 7 + Vitest 5
  src/review/model.ts  mirrors backend policy.py — keep both in sync
  src/components/      Upload, Processing, Review (VideoStage, Timeline, TrackList), Export
docs/ARCHITECTURE.md, docs/SAFETY.md, CHANGELOG.md, CONTRIBUTING.md, SECURITY.md, Makefile
```

## Run / test

```bash
make install            # cd backend && uv sync --all-extras ; cd frontend && npm ci
make dev                # API :8000 + Vite :5173
make serve              # build UI, serve everything from FastAPI on :8000
make test               # backend fast suite + frontend
make test-models        # real-model smoke test (~10 s on the RTX 4070)
make lint               # ruff check + ruff format --check + tsc
```

- `backend/.venv` has the full ML stack installed (torch 2.14.1+cu130, CUDA works,
  ultralytics 8.4.176, transformers 5.19, OpenCV 5.0).
- Model caches (outside the repo): `~/.cache/safeframe/models` (YuNet,
  SHA-256-pinned, plus yolo11n.pt) and the Hugging Face cache (`nateraw/vit-age-classifier`).
- Key defaults: threshold P(<18) ≥ 0.35; min 3 face samples, otherwise UNCERTAIN
  (blurred); age sampled every 3 frames, up to 40 samples per track; min face 32 px;
  gap interpolation 1.5 s; end padding 0.25 s; region padding 8 %; job TTL 24 h.

## Verified this session

- Browser e2e (Playwright, headless Chromium from `~/.cache/ms-playwright/chromium-1243`)
  with scripted detectors and with real models: upload → review → override →
  manual region → export → download → delete. Exported pixels were checked: the
  child was blurred through the occlusion and the adult was untouched.
  The harness lived in the session scratchpad (now gone); rebuild from
  `tests/conftest.py` fakes if needed.
- Real models on the Ultralytics `zidane.jpg` pan clip: 3 adults classified as
  adult, stable IDs for all 75 frames.

## Bugs fixed

- `assign_faces` used "smallest box wins", which misattributed faces when people
  overlapped. It now uses the nearest expected head position, and drops ambiguous
  matches (margin 0.05).
- Ultralytics auto-pip-installed `lap` at runtime, so `lap` is now declared in the `[ml]` extra.

## Open to-dos / ideas

- User: enable branch protection (main/develop, require CI); create a GitHub
  Release from tag v0.1.0; add UI screenshots to the README; `pre-commit install`.
- No evaluation on real minors yet (no footage available). See the
  "How to evaluate" section in docs/SAFETY.md.
- Ideas: solid-fill anonymisation option; keyframed (moving) manual regions;
  VFR-exact timing; auth for non-local deployment; Dockerfile (Docker isn't
  installed on this machine); per-track "split identity" tool for ID switches.
- Ultralytics YOLO is AGPL-3.0 (noted in the README).

## Environment gotchas

- The user's shell is zsh. **`c` is an alias**, so don't define a shell function named `c`.
- `pkill -f <pattern>` kills its own shell when the pattern appears in the command
  line. Kill by port instead: `ss -ltnp | grep :PORT`.
- `uv add` hung on a lock while another `uv sync` ran; editing pyproject then
  `uv lock` worked.
- `npm init -y --prefix X` also modified `frontend/package.json`. Avoid it.
- Starlette's TestClient wants `httpx2` (dev dependency) rather than `httpx`.
- Starlette encodes download filenames with spaces as `filename*=utf-8''…`.
