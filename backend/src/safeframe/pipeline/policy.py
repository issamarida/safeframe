"""Turning band and age evidence into blur decisions.

SafeFrame protects the people who asked for it: anyone seen wearing a
SafeFrame Band is blurred for every frame of their track, including frames
where the band itself is hidden. Everyone else is left untouched.

The policy is asymmetric in two directions at once:

* **Toward the wearer.** A single attributed sighting is enough, and a
  sighting that fits several people equally well protects all of them
  (``POSSIBLE``). Missing a wearer is a privacy failure; blurring a bystander
  is a usability cost the reviewer can undo.
* **Against abuse.** In a guarded zone (bank, ATM, court...) a band only
  protects someone with positive evidence of being a child: enough usable
  face observations *and* P(under 18) at or above the threshold. No face, no
  protection. A band is a privacy request, not a disguise.

Event mode (``protect_minors``) additionally protects everyone who may be
under 18, band or not, for schools and clubs publishing event footage. It
keeps the original fail-closed rule (uncertain is blurred) except in guarded
zones, where again only positive evidence protects.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from safeframe.pipeline.types import AgeSample, AgeStatus, Analysis, Track

ADULT_AGE = 18
DEFAULT_THRESHOLD = 0.35
DEFAULT_MIN_FACE_SAMPLES = 3
DEFAULT_MIN_BAND_HITS = 1


class Override(StrEnum):
    BLUR = "blur"
    KEEP = "keep"


class Decision(StrEnum):
    WEARER = "wearer"  # band attributed to this person: blurred
    POSSIBLE = "possible"  # band fitted this person and someone else equally: blurred
    NOT_HONOURED = "not_honoured"  # band seen in a guarded zone, no evidence of a child
    MINOR = "minor"  # event mode: looks under 18: blurred
    UNCERTAIN = "uncertain"  # event mode: not enough face evidence: blurred
    VISIBLE = "visible"  # no band (and not protected by event mode)


BLURRED_DECISIONS = frozenset(
    {Decision.WEARER, Decision.POSSIBLE, Decision.MINOR, Decision.UNCERTAIN}
)


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    # A track is MINOR when its aggregated P(under 18) >= threshold.
    threshold: float = DEFAULT_THRESHOLD
    min_face_samples: int = DEFAULT_MIN_FACE_SAMPLES
    # Unambiguous band sightings needed to call someone a WEARER.
    min_band_hits: int = DEFAULT_MIN_BAND_HITS
    guarded: bool = False
    protect_minors: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if self.min_face_samples < 1:
            raise ValueError("min_face_samples must be >= 1")
        if self.min_band_hits < 1:
            raise ValueError("min_band_hits must be >= 1")

    def age_status(self, track: Track) -> AgeStatus:
        if len(track.samples) < self.min_face_samples or track.p_minor is None:
            return AgeStatus.UNCERTAIN
        return AgeStatus.MINOR if track.p_minor >= self.threshold else AgeStatus.ADULT

    def band_decision(self, track: Track) -> Decision | None:
        if track.band_confirmed >= self.min_band_hits:
            return Decision.WEARER
        if track.wears_band:
            # Too few clean sightings, or only ambiguous ones: still protected.
            return Decision.POSSIBLE
        return None

    def classify(self, track: Track) -> Decision:
        band = self.band_decision(track)
        if band is not None and not self.guarded:
            return band
        age = self.age_status(track) if (self.guarded or self.protect_minors) else None
        if band is not None:
            return band if age is AgeStatus.MINOR else Decision.NOT_HONOURED
        if self.protect_minors:
            if age is AgeStatus.MINOR:
                return Decision.MINOR
            if age is AgeStatus.UNCERTAIN and not self.guarded:
                return Decision.UNCERTAIN
        return Decision.VISIBLE

    def should_blur(self, track: Track, override: Override | None = None) -> bool:
        if override is Override.BLUR:
            return True
        if override is Override.KEEP:
            return False
        return self.classify(track) in BLURRED_DECISIONS

    @classmethod
    def for_analysis(cls, analysis: Analysis, **kwargs) -> DecisionPolicy:
        """The policy for an analysis, with its capture context locked in.

        A photo has one frame, so one usable face is all the age evidence it
        can ever have.
        """
        if analysis.video.is_photo:
            kwargs["min_face_samples"] = 1
        return cls(
            guarded=bool(analysis.context.get("guarded", False)),
            protect_minors=bool(analysis.context.get("protect_minors", False)),
            **kwargs,
        )


def needs_age(track: Track, guarded: bool, protect_minors: bool) -> bool:
    """Whether the policy can ever look at this track's age.

    Age is estimated only where it can change a decision: for band wearers in
    guarded zones, and for everyone in event mode. Nobody else's face is
    analysed at all.
    """
    return protect_minors or (guarded and track.wears_band)


# --- Age-bucket arithmetic -------------------------------------------------

_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
_OPEN_UPPER = re.compile(r"(?:more than|over|>|\+)\s*(\d+)|(\d+)\s*\+")


def parse_age_bucket(label: str) -> tuple[int, int]:
    """Parse labels like ``"10-19"``, ``"(15, 20)"``, ``"more than 70"``, ``"70+"``."""
    text = label.strip().lower().replace("(", "").replace(")", "").replace(",", "-")
    if m := _RANGE.search(text):
        lo, hi = int(m.group(1)), int(m.group(2))
        return min(lo, hi), max(lo, hi)
    if m := _OPEN_UPPER.search(text):
        lo = int(m.group(1) or m.group(2))
        return lo, max(lo, 100)
    raise ValueError(f"unrecognised age bucket label: {label!r}")


def minor_fraction(lo: int, hi: int) -> float:
    """Fraction of integer ages in [lo, hi] that are under 18 (uniform prior)."""
    total = hi - lo + 1
    minors = max(0, min(hi, ADULT_AGE - 1) - lo + 1)
    return minors / total


def bucket_stats(labels: Sequence[str]) -> tuple[list[float], list[float]]:
    """(minor fraction, midpoint age) for each bucket label."""
    fractions, mids = [], []
    for label in labels:
        lo, hi = parse_age_bucket(label)
        fractions.append(minor_fraction(lo, hi))
        mids.append((lo + hi) / 2)
    return fractions, mids


def aggregate(samples: Iterable[AgeSample]) -> tuple[float | None, float | None]:
    """Weighted mean of per-sample P(minor) and expected age."""
    total_w = p_sum = age_sum = 0.0
    for s in samples:
        total_w += s.weight
        p_sum += s.weight * s.p_minor
        age_sum += s.weight * s.expected_age
    if total_w <= 0:
        return None, None
    return p_sum / total_w, age_sum / total_w
