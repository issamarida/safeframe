# Safety case

This document explains how SafeFrame can fail, what each failure costs, and
which design choices push failures toward the cheaper side.

## What "failure" means here

| Failure | Who is harmed | Cost | Recoverable? |
| --- | --- | --- | --- |
| **Missed minor** — a child appears unblurred | The child | Privacy harm once shared; cannot be undone | No |
| **Over-blurring** — an adult is blurred | The uploader | Aesthetic/usability cost | Yes, in review |
| **Retained footage** — uploads linger on the server | Everyone filmed | Exposure if the server is compromised | Partly |
| **Misuse** — output treated as an age verdict | People in the video | Profiling, discrimination | No |

The asymmetry is the core of the design: the system should trade many
over-blurs for each missed minor, and must keep the human in the loop.

## Failure modes and mitigations

### 1. The age model is wrong

Apparent-age models are least accurate exactly where it matters most — around
the 18-year boundary — and their error rates differ across demographic groups,
lighting conditions and image quality.

- The decision threshold defaults to **P(under 18) ≥ 0.35**, not 0.5.
- The model's distribution, not its top label, is used: probability mass in a
  bucket straddling 18 (`10-19`) counts proportionally toward "minor".
- Evidence is **aggregated over up to 40 observations** per person, weighted by
  face size and detector confidence, so one bad frame cannot flip a decision.
- The review UI shows P(under 18), estimated age and the number of face
  observations behind each decision, so reviewers can judge how much to trust it.
- The age model is pluggable (`SAFEFRAME_AGE_MODEL`). Swapping in a better or
  better-audited model needs no code changes as long as it emits age-range labels.

### 2. There is no usable face

Back-of-head shots, profiles, sunglasses, distance, motion blur.

- People need at least **3 usable face observations** before they can be called
  adults. Without them they are **uncertain, and uncertain is blurred**.
- Faces smaller than 32 px are ignored rather than guessed at.

### 3. The person is not detected, or the tracker loses them

- Detection runs at a low confidence (0.1) and BoT-SORT keeps lost tracks alive
  for ~3 s with appearance re-identification, so identity survives occlusion.
- Gaps up to 1.5 s are interpolated, and every track is padded by 0.25 s, so
  the blur does not drop out on a missed frame.
- The reviewer can add **manual regions** for anyone the detector missed, and
  step frame-by-frame through **flagged segments**.

### 4. The blur is too weak

- Blur strength is proportional to region size (detail finer than ~1/24 of the
  region is destroyed), and the region is padded by 8 %.
- Blurred regions are then converted to greyscale, removing skin-tone and
  clothing-colour cues.
- With silhouette blur only the pixels inside a person's outline (plus a
  margin of 3–24 px scaled to their size) are blurred, so less of the frame is
  obscured than with rectangles. See section 6.
- Note: blur is not encryption. For high-risk contexts, a solid fill is safer;
  this is a planned option.

### 5. Footage is retained

- Original uploads, review previews and per-person thumbnails are deleted as
  soon as the export is ready, unless the user explicitly opts to keep the upload.
- Every job expires after 24 h without activity; interrupted jobs are deleted
  at start-up; unknown directories are swept.
- Uploads stream straight into the job directory (no shared temp files).
- Location and device metadata are stripped from the exported MP4.
- Thumbnails in the review UI are veiled (blurred) until the reviewer chooses
  to reveal them.
- The repository refuses footage, weights and secrets (pre-commit + CI).

### 6. The silhouette misses part of a person

A segmentation mask is a model output: it can omit a raised hand, hair, a bag
or a limb partly hidden behind someone else, and the outline itself still
shows the person's shape.

- The mask is grown by a margin before use and is never clipped further to the
  tracker's box.
- Masks that are empty, tiny, implausibly small for the box, far from the box,
  over-fragmented or malformed are discarded, and that frame falls back to the
  padded rectangle. So do interpolated gaps, padded track ends, frames beyond
  the payload budget and any frame without a mask: a silhouette is never guessed.
- If a planned region has neither a usable mask nor a valid box, the export
  fails instead of writing that frame unprotected.
- The review preview applies the same mask-or-box choice and labels each
  blurred person `silhouette` or `box` per frame, so thin coverage can be seen
  and covered with a manual region before export.
- These checks catch broken masks, not incomplete ones: a well-formed mask can
  still miss a limb. Evaluations should report silhouette-backed vs. fallback
  frames and any uncovered person pixels. Rectangle-only behaviour remains
  available with `--yolo yolo11n.pt` / `SAFEFRAME_YOLO_WEIGHTS=yolo11n.pt`.
- The segmentation model is a different network from the old detector. On
  synthetic clips it found slightly fewer person-frames at the frame edge
  (e.g. 150 vs 160 on one clip, missing a partly visible third person), so
  minor recall must be re-measured on the labelled evaluation set.
- Masks are derived personal data. They live only in the job's
  `analysis.json`, under the same retention and deletion rules, and are never
  logged.

### 7. Misuse

- The README, the UI notice and the API's `/config` all state the intended
  use and that estimates are probabilistic.
- SafeFrame stores no identities, embeddings or age estimates beyond the life
  of a job, and offers no export of per-person age data from the UI.

## How to evaluate a change

Any change that affects blurring decisions should report, on a labelled
evaluation set kept **outside** this repository:

- **Minor recall** (frames where a minor is visible and blurred / frames where a
  minor is visible) — the primary metric; must not regress.
- **Adult pass-through rate** — secondary; how often adults are left untouched.
- Both metrics broken down by apparent age band (especially 13–20), face
  visibility and, where consented annotations exist, demographic group.
