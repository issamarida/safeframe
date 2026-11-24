# Safety case

This document explains how SafeFrame can fail, what each failure costs, who
might try to abuse it, and which design choices push failures toward the
cheaper side.

## What "failure" means here

| Failure | Who is harmed | Cost | Recoverable? |
| --- | --- | --- | --- |
| **Missed wearer**: someone wearing a band appears unblurred | The wearer | Privacy harm once shared; cannot be undone | No |
| **Over-blurring**: a bystander is blurred | The uploader | Aesthetic/usability cost | Yes, in review |
| **Band as disguise**: a band hides someone where being identifiable matters | Victims of a crime, the public | Lost evidence, eroded trust in the band | Partly (sealed originals) |
| **Band as beacon**: the band is used to find or follow a child | The wearer | Physical safety | No |
| **Retained footage**: uploads, originals or envelopes linger | Everyone filmed | Exposure if the server is compromised | Partly |
| **Misuse of age data**: output treated as an age verdict | People in the image | Profiling, discrimination | No |

The asymmetry between the first two rows is the core of the design: the
system trades many over-blurs for each missed wearer, and keeps a human in the
loop. Rows three and four are why the band is not "blur anyone with a band,
everywhere, forever".

## Threat model: who might abuse the band?

| Adversary | Goal | What stops them |
| --- | --- | --- |
| **Robber / fraudster** | Not be identifiable in a bank, at an ATM, in a jeweller's | Security cameras never honour the band (§8 of [PROTOCOL.md](PROTOCOL.md)). In guarded zones the band is honoured only with positive evidence of a child: an adult face is `not_honoured`, and **no face is no protection** (a masked robber is `not_honoured`). Each person is judged on their own face, so standing next to a child wearing a band doesn't help. Whenever a band *is* honoured, the original can be sealed for a trustee quorum. |
| **Robber who spoofs the context** | Make a bank look like a park | The user can only make a capture *more* guarded; a registry match can't be overridden (`resolve_context`). The location is read from the file's own metadata. Partner devices also use signed venue beacons, and treat everything as guarded when their registry is stale. |
| **Someone hiding another person** | Stick a band on someone to remove them from evidence | Same as above in guarded places; elsewhere the original is sealed, so a court can still recover it. |
| **Stalker** | Use bands to find a child | Every band is identical: no serial, no name, nothing linkable between sightings. Partners must not log, count or transmit sightings. SafeFrame stores no faces or identities. |
| **Prankster / vandal** | Blur everyone by hanging bands on a wall | A marker must be attributed to a person (wrist, silhouette or box). A band on a wall protects nobody (an *orphan*, flagged in review); one inside someone's box blurs that person, which the reviewer can undo. |
| **Curious operator** | Open sealed originals | The escrow private key never exists in one place: it's split k-of-n between independent trustees. The server only has the public key. Envelopes expire (30 days by default) and are refused afterwards. |
| **Colluding trustees** | Open envelopes without due process | Needs k independent trustees to collude. Every envelope records its reason and context in an authenticated header, so unsealing is auditable. Choose trustees from different institutions. |

## Failure modes and mitigations

### 1. The band isn't seen

Too far away, facing away, motion-blurred, covered by a sleeve.

- One sighting anywhere in the clip protects the whole track, including frames
  where the band is hidden.
- Detection runs on the full frame and again on upscaled crops around every
  wrist found by a pose model, at up to 4× zoom.
- The detector accepts markers down to 7 px a side and relaxes error
  correction (`errorCorrectionRate = 0.35`). Measured: 2 px per module decodes
  20/20 synthetic warped bands, 1.5 px 5/20 (see the range table in
  [PROTOCOL.md](PROTOCOL.md#7-detection-range)).
- Band sightings that fit nobody are reported, and the review screen links to
  those frames so a missed wearer can be covered by hand.
- What remains: a band never visible to the camera can't be honoured. The UI
  notice and the CLI say so before every run.

### 2. The band is matched to the wrong person

- Attribution prefers a wrist keypoint within 0.1 × body height, then the
  silhouette, then the box.
- When two people fit about equally well, both are blurred (`possible`): the
  failure we accept is over-blurring.
- In guarded zones every candidate still needs their own positive age
  evidence, so ambiguity can't be used to extend a child's protection to an
  adult.

### 3. The person isn't detected, or the tracker loses them

- Detection runs at a low confidence (0.1) and BoT-SORT keeps lost tracks alive
  for ~3 s with appearance re-identification, so identity survives occlusion.
- Gaps up to 1.5 s are interpolated, and every track is padded by 0.25 s, so
  the blur doesn't drop out on a missed frame.
- The reviewer can add manual regions and step through flagged segments.

### 4. Age estimation is wrong (guarded zones and event mode only)

Apparent-age models are least accurate exactly around 18, and their error
rates differ across demographic groups, lighting and image quality.

- Age is only estimated where it can change a decision: for band wearers in
  guarded zones and for everyone in event mode. Everyone else's face is never
  cropped or analysed.
- The threshold defaults to **P(under 18) ≥ 0.35**, not 0.5, and the full
  bucket distribution is used (`10-19` counts 8/10 toward "minor").
- Evidence is aggregated over up to 40 observations per person, weighted by
  face size and detector confidence; at least 3 observations are needed in
  video (1 in a photo).
- In guarded zones, a child the model mistakes for an adult is **not
  protected**. This is the deliberate exception to "when unsure, blur it": it is
  what stops the band being a disguise. In event mode outside guarded zones,
  uncertain people are still blurred.

### 5. The blur is too weak

- Blur strength is proportional to region size, and regions are padded by 8 %.
- Blurred regions are converted to greyscale, removing skin-tone and clothing
  cues.
- Silhouettes are grown by 3–24 px; malformed or implausible masks fall back
  to the padded box, and a frame with neither is never written (the export
  fails). Rectangle-only behaviour remains available with `--yolo yolo11n.pt`.
- Blur is not encryption. For high-risk contexts a solid fill is safer; this
  is a planned option.

### 6. Footage is retained

- Original uploads, previews and thumbnails are deleted as soon as the export
  is ready, unless the user explicitly opts to keep the upload.
- A sealed envelope is the only copy of an original that outlives the job, and
  it is unreadable without a trustee quorum. It expires after
  `SAFEFRAME_SEAL_RETENTION_DAYS` (30) and `unseal` refuses expired envelopes.
- Every job expires after 24 h without activity; interrupted jobs are deleted
  at start-up; unknown directories are swept.
- Outputs carry no metadata at all (photos are re-encoded; MP4s use
  `-map_metadata -1`). The capture location is used once and never stored.
- The repository refuses footage, sealed envelopes, trustee shares, weights
  and secrets (pre-commit + CI).

### 7. Misuse

- The README, the UI notice and `/api/config` state the intended use and that
  the band is a privacy request, not a disguise.
- SafeFrame stores no identities, embeddings, sightings or age estimates beyond
  the life of a job, and offers no export of per-person age data.

## How to evaluate a change

Any change that affects blurring decisions should report, on a labelled
evaluation set kept **outside** this repository:

- **Wearer recall** (frames where a band wearer is visible and blurred / frames
  where a band wearer is visible): the primary metric; must not regress.
  Report it by pixels per module (or distance), band angle, motion blur and
  partial occlusion.
- **Attribution accuracy**: sightings given to the right person, ambiguous, or
  wrong.
- **False sightings** per 1 000 frames without any band.
- **Bystander pass-through rate**: how often people without bands are left
  untouched.
- **Guarded zones**: share of adult wearers `not_honoured` (should be ~100 %)
  and share of child wearers still protected (the age model's minor recall),
  by apparent age band (especially 13–20) and, where consented annotations
  exist, demographic group.
