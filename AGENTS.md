# AGENTS.md: SafeFrame context

Everything a contributor or coding agent needs to work on SafeFrame. Read this
first; the linked docs go deeper. Keep it current when something here changes.

## What SafeFrame is

**The SafeFrame Band** is a wristband (printed or woven ArUco markers) that a
child, or anyone, wears to say *"please don't show me"*. Partner cameras (for
example the iPhone camera), recording apps and social platforms run software
that blurs **everyone wearing a band, and only them**, before a photo or video
is saved or shared. In ML terms: a detector finds band markers, attributes each
one to the person wearing it, and the pipeline blurs those people for every
frame they appear in.

This repository is the **reference implementation**:

- `backend/`: Python package `safeframe`. The pipeline (people, bands,
  attribution, context, age where needed, policy, render), sealed originals,
  CLI and a FastAPI app.
- `frontend/`: React review UI. A person checks every decision before export.
- [docs/PROTOCOL.md](docs/PROTOCOL.md): what partners must implement.

An IE Challenge project by the SafeFrame Crew: Javier Cruz Villarreal
(coordinator), Issam Arida (research), Christoph Rintz (backend/AI), Kiril
Petrovski (frontend), Xander Nainoa Chen (quality/safety), Jamie Blewett
(presentation). v0.1 blurred people *estimated* to be under 18; that is now
**event mode**.

## The abuse problem and how it's solved

*"Couldn't a robber wear this into a bank?"* No. A band is a privacy request to
people sharing images, never a disguise:

1. **Security cameras never honour it.** Certification forbids it on CCTV,
   body cameras and evidence devices (PROTOCOL §8).
2. **Guarded zones** (bank, atm, jeweller, pharmacy, casino, court, police,
   prison, airport, border, government): a band is honoured only with
   **positive evidence of a child** (≥ 3 face observations, 1 in a photo, and
   P(under 18) ≥ 0.35). Adult face, or no face, means `not_honoured`. Each person
   is judged on their own face, so an adult can't piggyback on a child's band.
3. **Context only gets stricter.** Location comes from the file's EXIF GPS /
   ISO 6709 tag (or the request), matched against a registry, and is never
   stored. The user can declare "guarded" but can't un-guard a registry match.
   Partners also use signed venue beacons, and treat everything as guarded
   when their registry is stale.
4. **Sealed originals.** When a band *is* honoured, the original can be sealed
   (X25519 + HKDF + AES-256-GCM, Shamir k-of-n trustees, 30-day expiry), so a
   court can recover evidence but nobody can open it alone.
5. **No tracking.** Every band is identical (no serial); sightings are never
   logged or transmitted; no faces or identities are stored.

Threat model: [docs/SAFETY.md](docs/SAFETY.md).

## Decision policy (memorise this)

| Band | Guarded? | Age evidence | Decision | Blurred |
| --- | --- | --- | --- | --- |
| clean | no | not computed | `wearer` | ✅ |
| ambiguous only (or < `min_band_hits` clean) | no | not computed | `possible` | ✅ |
| any | yes | likely minor | `wearer` / `possible` | ✅ |
| any | yes | adult or no face | `not_honoured` | ❌ |
| none | event mode, not guarded | minor / uncertain | `minor` / `uncertain` | ✅ |
| none | event mode, guarded | minor | `minor` | ✅ |
| none | otherwise | — | `visible` | ❌ |

- One attributed sighting protects the whole track (interpolation + padding).
- Age is estimated **only** where it can change a decision (`policy.needs_age`):
  band wearers in guarded zones, everyone in event mode. Nobody else's face is
  ever cropped.
- Reviewer overrides (`blur` / `keep`) win. Keeping a wearer needs a consent
  confirmation in the UI. `RenderPlan.honoured_bands` uses the policy's own
  decision, so a confirmed blur still seals.
- `backend/src/safeframe/pipeline/policy.py` and
  `frontend/src/review/model.ts` must agree exactly; both test files carry the
  same decision matrix.

## Repository layout

```
backend/                      uv project (Python 3.13), package `safeframe`
  src/safeframe/pipeline/
    band.py                   band spec v1, ArucoBandDetector, band_strip, band_sheet_svg
    wearers.py                sighting → person (wrist / silhouette / box), YoloWristLocator
    context.py                Zone, ZoneRegistry, CaptureContext, resolve_context, EXIF/ffprobe location
    guarded_zones.json        illustrative sample registry (Banco de España, Bank of England, ...)
    detection.py              YOLO-seg + BoT-SORT tracker (untracked mode for photos), masks
    masks.py                  silhouette coverage, masks_v1 wire format, budgets
    faces.py / age.py         YuNet faces + assign_faces; ViT FairFace age → P(under 18)
    policy.py                 Decision, DecisionPolicy(.for_analysis), needs_age, aggregation
    analyze.py                two passes: people + bands, then age only where needed
    render.py / anonymize.py  plan_render, render_video, render_photo; blur + greyscale
    smoothing.py              gap interpolation, end padding
    factory.py                ModelBundle: lazy real components
    types.py                  Box, Track (band hits), BandHit, VideoInfo (media), Analysis (context)
  src/safeframe/seal.py       SFSEAL1 envelopes, Shamir shares, unseal
  src/safeframe/video/io.py   ffprobe / OpenCV decode / FFmpeg encode + preview
  src/safeframe/api/          app.py (routes, SSE), jobs.py (lifecycle, sealing, retention), schemas.py
  src/safeframe/cli.py        process | analyze | render | band | trustees init | unseal | inspect | serve
  tests/                      fast suite (synthetic video, scripted fakes) + `-m models`
frontend/                     React 19 + Vite 8 + TS 7 + Vitest 5
  src/review/model.ts         mirror of policy.py (classify, shouldBlur, policyFor, summarise)
  src/components/             Upload, Processing, Review (VideoStage, Timeline, TrackList, ContextBanner), Export
docs/                         TECHNICAL, ARCHITECTURE, PROTOCOL, SAFETY, DEVELOPMENT
scripts/check_forbidden_files.py   blocks footage, .sfseal, trustee shares, weights, secrets
```

## Commands

```bash
make install       # backend: uv sync --all-extras ; frontend: npm ci
make dev           # API :8000 (reload) + Vite :5173
make serve         # build UI, serve everything from FastAPI on :8000
make test          # backend fast suite + frontend
make test-models   # real models; bands pasted onto ultralytics sample photos (~1 min)
make lint          # ruff check + ruff format --check + tsc
make hygiene       # forbidden-files scan of tracked files

cd backend
uv run safeframe band -o band.svg
uv run safeframe process in.jpg -o out.jpg [--location LAT,LON] [--guarded] [--event-mode] [--seal-to escrow.pub.json]
uv run safeframe trustees init -k 3 -n 5 --out keys/
uv run safeframe unseal x.sfseal --share t1.json t2.json t3.json -o original.jpg
```

Model caches live outside the repo: `~/.cache/safeframe/models` (YuNet
SHA-256-pinned, `yolo11n-seg.pt`, `yolo11n-pose.pt`, `yolo11n.pt`) and the
Hugging Face cache (`nateraw/vit-age-classifier`). `backend/.venv` on the main
dev machine has the full ML stack with CUDA.

## Rules (non-negotiable)

**Git**

- Every commit is authored solely by `issamarida <issamaarida@gmail.com>`.
  **No `Co-Authored-By` (Claude, Cursor or any other) trailers, ever.**
- Git Flow: `main` (releases + tags), `develop` (integration), `feature/*`,
  `fix/*`, `chore/*`, `docs/*`, `release/vX.Y.Z`, `hotfix/vX.Y.Z`. Merge with
  `--no-ff` (GitHub "merge commit"). Every merge into `develop` or `main` goes
  through a PR with green CI. Release branches merge into `main` and back into
  `develop`. Tags `vX.Y.Z` on `main`.
- Conventional Commits (`feat(band): …`, `fix(api): …`, `docs: …`), granular,
  with bodies explaining why. Version bumps happen on the release branch.
- The repository is `github.com/issamarida/safeframe` (it was previously
  `issamarida/meta-support`; old links may still say so).

**Data**

- Never commit footage (any video/audio), sealed envelopes (`.sfseal`),
  trustee shares, model weights or secrets. `escrow.pub.json` is public and
  fine. CI and pre-commit run `scripts/check_forbidden_files.py`.
- Never paste footage, frames or faces into issues or PRs; describe the scene.
- Tests synthesise their own media; real-model tests use the sample photos
  inside the ultralytics package.

**Safety**

- A change that affects what gets blurred must explain how it trades **missed
  wearers** (privacy failure, can't be undone) against **over-blurring**
  (usability cost, undoable in review). Err toward blurring, except where
  that would make the band a disguise (guarded zones).
- A change that touches guarded zones, sealing, logging or storage must explain
  whether the band could become a disguise or a tracking beacon.
- Never log or persist capture locations, band sightings beyond the job, faces,
  or age estimates beyond the job.

## Conventions

- Components are `Protocol`s (`pipeline/interfaces.py`); tests inject fakes from
  `tests/conftest.py`: `ScriptedTracker`, `ScriptedBands({frame: [(x, y)]})`,
  `ScriptedWrists`, `HeadFaceDetector`, `SizeAgeEstimator`, and
  `components(tracker, faces, ages, bands=None, wrists=None)`.
- Synthetic clips are stamped with their frame index (12 bits as 8×8 blocks on
  the top row via FFmpeg `geq`); `frame_index(frame)` decodes it. Pass 2 only
  re-reads some frames, so fakes must not count calls to know the frame.
- Band detection is tested with the real ArUco detector on warped
  `band_strip` images.
- Photos are one-frame videos (`VideoInfo.photo(w, h)`, `media="photo"`, fps
  1); `DecisionPolicy.for_analysis` uses 1 face sample for photos.
- Match the surrounding code's style and comment density; comments state
  constraints, not history.

## Environment gotchas

- The shell is zsh, and **`c` is an alias**: don't define a function `c`.
- `pkill -f <pattern>` kills its own shell when the pattern is on its command
  line. Kill by PID or port (`ss -ltnp | grep :PORT`).
- `uv add` can hang on a lock while another `uv sync` runs: edit
  `pyproject.toml`, then `uv lock`.
- Starlette's `TestClient` needs `httpx2` (dev dependency), not `httpx`.
- Starlette encodes download filenames with spaces as `filename*=utf-8''…`.
- `npm init -y --prefix X` also modifies `frontend/package.json`; avoid it.
- **Some agent shells append a `Co-authored-by:` trailer to every `git commit`.**
  Check with `git log -1 --format=%B`. To commit without it, use plumbing:
  `git commit-tree $(git write-tree) -p HEAD -F msg.txt` then
  `git update-ref HEAD <sha>`. In zsh, don't follow a heredoc line with `&& \`.
- ArUco with a tiny `minMarkerPerimeterRate` on a 4K frame produces thousands of
  candidates (~800 ms/frame). Keep the derived rate (floor 0.01); small markers
  are the zoomed wrist pass's job.
- At 1080p, full-body people are usually below the 2 px/module decode limit;
  real-model tests use ~14 MP photos and 4K video for that reason.
- `gh` is logged in as `issamarida`. CI: `.github/workflows/ci.yml`
  (hygiene, backend ruff + pytest, frontend typecheck + test + build).

## Status (2026-11-24)

- **v0.1.0** (tag on `main`): age-based blurring, silhouette blur.
- **v0.2.0** (tag on `main`, the band rework): band, attribution, guarded zones, sealed originals,
  photos, event mode, review UI, protocol and docs. Backend 165 fast tests +
  3 real-model tests; frontend 57 tests.
- Verified with real models: on `bus.jpg` at ~14 MP with a band composited at a
  realistic size on one person's wrist, only that person is blurred, to their
  silhouette. At the Banco de España location the same adult's band is
  `not_honoured` (P(<18) = 0.00) and nothing is sealed. A standard run seals
  the original; 2 of 3 required shares are refused, 3 restore it byte for byte.
  In a 4K pan the wearer is blurred in every frame and no one else is.
- Unmerged on origin: `feature/silhouette-blur` holds two mask-store commits
  (`310e649`, `a704b01`) that never landed; they would lift the inline mask
  budget.

## Next steps

- Real-world evaluation: printed and woven bands at distances, angles and
  lighting; false-sighting rate; guarded-zone honour rates by apparent age
  (see SAFETY.md "How to evaluate a change"). Keep data outside the repo.
- Partner SDK sketch in Swift (Vision body pose + person instance masks +
  CryptoKit sealing) and a public conformance suite from the reference tests.
- Signed registry distribution, venue beacons, registry staleness handling.
- Solid-fill anonymisation; keyframed manual regions; track-split tool;
  VFR-exact timing; auth + container image for non-local deployment.
- Repo admin: branch protection on `main`/`develop` requiring CI;
  `pre-commit install`; UI screenshots in the README.
