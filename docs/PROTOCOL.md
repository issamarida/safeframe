# SafeFrame Band protocol (draft v1)

This document is for partners: camera makers, recording apps and platforms
that want to honour the SafeFrame Band. It specifies the band, what a
conforming implementation must do with it, and what it must never do. The
Python package in this repository is the reference implementation; where this
document and the code disagree, file an issue.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## 1. The band

| Property | Value |
| --- | --- |
| Marker family | OpenCV ArUco `DICT_4X4_50` (4×4 data bits, one-module black border: 6×6 modules) |
| Reserved ids | `7, 19, 28, 42`, printed in that order, repeating (`band.BAND_SEQUENCE`) |
| Quiet zone | ≥ 1.5 modules of white around and between markers |
| Child size | 150 mm circumference, 20 mm wide (module ≈ 2.2 mm) |
| Adult size | 190 mm circumference, 24 mm wide (module ≈ 2.7 mm) |
| Surface | Matte. Woven, printed or sticker; markers facing out |
| Spec version | 1 (`band.SPEC_VERSION`) |

**Every band is identical.** A band MUST NOT carry a serial number, name,
colour code or any other per-band or per-wearer data, and implementations
MUST NOT attempt to distinguish one band from another. A band means only
"this person asks not to be shown".

A printable sheet is served at `GET /api/band.svg` and by `safeframe band`.

## 2. Detection

A conforming implementation:

- MUST look for bands on frames at **full sensor or full output resolution**,
  before downscaling for preview or encoding. Range is limited by pixels per
  module (§7), so resolution is the most important factor.
- MUST examine at least every second frame of video, and every photo.
- SHOULD run a second, zoomed pass on crops around detected wrists (2D body
  pose; on iOS `VNDetectHumanBodyPoseRequest`, on Android ML Kit Pose
  Detection). The reference upsamples a square crop of 0.22 × body height to
  about 480 px, at most 4×.
- MUST only accept the four reserved ids, and SHOULD reject markers whose side
  is below 7 px or which decode with more than a small number of bit errors
  (reference: `errorCorrectionRate = 0.35`).

## 3. Attribution

A detected marker is attributed to a person by the strongest available
evidence:

1. **Wrist**: a wrist keypoint within 0.10 × the person's body height of the
   marker centre.
2. **Body**: the marker centre is inside the person's segmentation mask (on iOS
   `VNGeneratePersonInstanceMaskRequest`).
3. **Box**: the marker centre is inside the person's bounding box grown by 5 %.

If two or more people match at the same level about equally well (wrists
within 0.03 × height of the best), the sighting is **ambiguous** and MUST be
attributed to all of them. A marker that matches nobody is an *orphan*: it
protects nobody, and a review UI SHOULD warn about it.

One attributed sighting protects the person for **their whole track**: every
frame in which they are tracked, including frames in which the band is hidden.
Video implementations MUST track people by body (not face) and SHOULD
interpolate short gaps (reference: ≤ 1.5 s) and pad track ends (0.25 s).

## 4. Decisions

Each person gets exactly one decision. "Blurred" decisions MUST be applied to
the whole silhouette, or the padded box where no reliable silhouette exists.

| Band | Context | Age evidence | Decision | Blurred |
| --- | --- | --- | --- | --- |
| clean sighting | standard | not computed | `wearer` | yes |
| ambiguous only | standard | not computed | `possible` | yes |
| clean or ambiguous | guarded | likely under 18 | `wearer` / `possible` | yes |
| clean or ambiguous | guarded | adult, or no usable face | `not_honoured` | no |
| none | event mode, not guarded | likely under 18 / not enough evidence | `minor` / `uncertain` | yes |
| none | event mode, guarded | likely under 18 | `minor` | yes |
| none | otherwise | — | `visible` | no |

"Likely under 18" means at least 3 usable face observations (1 for a photo)
with aggregated P(under 18) ≥ 0.35. The full truth table is
`backend/tests/test_policy.py::test_decision_matrix`, mirrored in
`frontend/src/review/model.test.ts`.

**Data minimisation.** Outside guarded zones and event mode, implementations
MUST NOT run age estimation, or any face analysis, on anyone. In a guarded
zone, only band wearers' faces MAY be analysed.

## 5. Guarded zones

A capture is guarded if any of these is true; none of them can make a capture
*less* guarded:

1. **Registry.** The capture position (GPS, or the platform's best location
   fix) lies inside a zone of the SafeFrame guarded-zone registry. Kinds:
   `bank, atm, jeweller, pharmacy, casino, court, police, prison, airport,
   border, government`.
2. **Venue beacon.** The device receives a valid SafeFrame venue beacon (BLE
   advertisement carrying the registry key id, the zone id and a rotating
   signature over the current 10-minute window). Beacons cover indoor places
   where GPS is imprecise.
3. **User declaration.** The person capturing or uploading marks it as taken in
   a guarded place.

Registry format (version 1; the reference loads it with
`context.ZoneRegistry.from_json`):

```json
{
  "version": 1,
  "zones": [
    {"id": "es-bde-madrid", "name": "Banco de España (Madrid)", "kind": "bank",
     "lat": 40.4183, "lon": -3.6944, "radius_m": 90}
  ]
}
```

Production registries MUST be signed (Ed25519 over the canonical JSON) by the
registry operator and verified before use. Devices SHOULD refresh at least
weekly. If the cached registry is older than 30 days, implementations MUST
treat every capture as guarded until it is refreshed: children wearing a band
stay protected, and a stale device can't be used as a disguise.

The capture location MUST be used only to resolve the context, MUST NOT be
stored with the decision, and MUST be stripped from the shared output.

## 6. Sealed originals

When a band is honoured, the implementation SHOULD seal the unblurred original
rather than discard it, so that evidence of a crime remains recoverable by due
process and by nobody else.

Envelope format `SFSEAL1` (reference: `backend/src/safeframe/seal.py`):

```
"SFSEAL1\n" | u32 header_len | header JSON
            | 12-byte wrap nonce | u32 wrapped_len | wrapped content key
            | (u32 chunk_len | AES-GCM chunk)...
```

All integers are big-endian.

- Key agreement: ephemeral X25519 with the escrow public key; KEK =
  HKDF-SHA256(shared secret, no salt, info = `"safeframe-seal-v1|"` ‖
  ephemeral public ‖ escrow public), 32 bytes. On iOS:
  `Curve25519.KeyAgreement` + `HKDF<SHA256>` + `AES.GCM` (CryptoKit).
- A random 256-bit content key is wrapped with AES-256-GCM under the KEK, with
  the header JSON bytes as associated data.
- Content is AES-256-GCM in 1 MiB chunks; nonce = 4 zero bytes ‖ u64 chunk
  index; associated data = SHA-256(header) ‖ u64 index ‖ final flag (one byte).
  Reordering, truncation, extension and header edits all fail.
- The header records key id, creation and expiry time (default 30 days), the
  original's name and size, the SHA-256 of the published copy, the reason and
  the capture context. It never records a location or any identity.

The escrow private key exists only as **Shamir shares** (k of n, over the prime
field 2^521 − 1, `safeframe-shamir-p521-v1`) held by independent trustees.
`safeframe trustees init -k 3 -n 5` runs the ceremony; `safeframe unseal`
opens an envelope with any k shares and refuses expired envelopes. Partners
MUST NOT hold more than k − 1 shares, and envelopes MUST be deleted at expiry.

## 7. Detection range

The reference decodes reliably at ≥ 2 px per module (20/20 synthetic warped
bands; 5/20 at 1.5 px). With a typical 26 mm-equivalent phone lens:

| Capture | Child band | Adult band |
| --- | --- | --- |
| 12 MP photo / 4K video | ~3 m | ~3.5 m |
| 1080p video | ~1.5 m | ~1.8 m |
| 2× telephoto, 12 MP | ~6 m | ~7 m |

Implementations SHOULD publish their measured range under the conformance
suite and SHOULD prefer reading markers from the highest-resolution stream.

## 8. What partners must never do

- MUST NOT implement the band on security, surveillance or evidence devices:
  CCTV, access control, police body cameras, dashcams or doorbell cameras used
  for security. The band is a request to people sharing images, not a way to
  evade security.
- MUST NOT log, count, transmit or aggregate band sightings, their positions,
  or the identities, faces or locations of wearers. Telemetry MAY report only
  anonymous, aggregate counts of "a band was honoured" per release.
- MUST NOT store face crops or age estimates beyond the processing of one
  capture.
- MUST NOT offer a setting that disables guarded zones or that lets a user
  un-guard a registry zone.
- MUST NOT present a band decision as a statement about anyone's age or
  identity.

## 9. Conformance

A partner implementation is conformant when it passes the decision matrix
(§4), the attribution fixtures (`backend/tests/test_wearers.py`), the guarded
context rules (`backend/tests/test_context.py`), the envelope round-trip and
tamper tests (`backend/tests/test_seal.py`) and publishes its measured range
(§7). The reference test suite is the starting point for the public
conformance suite.

## Changes

- **v1 (draft)**: first version.
