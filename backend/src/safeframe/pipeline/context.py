"""Where a photo or video was captured, and which protection rules apply there.

A SafeFrame Band is a *privacy request*, not an invisibility cloak. In
**guarded zones** (banks, ATMs, jewellers, courts, police stations, airports,
border crossings and similar places where being identifiable on camera is a
matter of public safety) a band is honoured only for people who positively
look like children. An adult wearing a band into a bank is photographed
normally.

The capture location is read once from the file's own metadata (EXIF GPS for
photos, the QuickTime/MP4 location tag for videos), matched against the zone
registry and then discarded: only the matched zone's name and kind are kept,
and output files never carry location metadata.
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

log = logging.getLogger(__name__)

# Kinds of place a registry may mark as guarded. Bands from apparent adults
# are not honoured inside any of them.
GUARDED_KINDS = frozenset(
    {
        "bank",
        "atm",
        "jeweller",
        "pharmacy",
        "casino",
        "court",
        "police",
        "prison",
        "airport",
        "border",
        "government",
    }
)
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True, slots=True)
class Zone:
    id: str
    name: str
    kind: str
    lat: float
    lon: float
    radius_m: float

    def distance_m(self, lat: float, lon: float) -> float:
        p1, p2 = math.radians(self.lat), math.radians(lat)
        dp, dl = p2 - p1, math.radians(lon - self.lon)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


class ZoneRegistry:
    """A list of guarded zones (circles), matched by great-circle distance.

    Partner devices would ship a signed, regularly updated registry built from
    public map data and venue self-registration (see docs/PROTOCOL.md). The
    packaged registry is a small illustrative sample.
    """

    def __init__(self, zones: list[Zone]) -> None:
        for z in zones:
            if z.kind not in GUARDED_KINDS:
                raise ValueError(f"zone {z.id!r}: unknown kind {z.kind!r}")
            if not (-90 <= z.lat <= 90 and -180 <= z.lon <= 180 and z.radius_m > 0):
                raise ValueError(f"zone {z.id!r}: invalid coordinates or radius")
        self.zones = zones

    @classmethod
    def from_json(cls, data: dict) -> ZoneRegistry:
        if data.get("version") != 1:
            raise ValueError("unsupported zone registry version")
        return cls([Zone(**z) for z in data["zones"]])

    @classmethod
    def from_file(cls, path: Path) -> ZoneRegistry:
        return cls.from_json(json.loads(Path(path).read_text()))

    @classmethod
    def packaged(cls) -> ZoneRegistry:
        text = (resources.files("safeframe.pipeline") / "guarded_zones.json").read_text()
        return cls.from_json(json.loads(text))

    def match(self, lat: float, lon: float) -> Zone | None:
        """The nearest zone containing the point, if any."""
        inside = [(z.distance_m(lat, lon), z) for z in self.zones]
        inside = [(d, z) for d, z in inside if d <= z.radius_m]
        return min(inside, key=lambda c: c[0])[1] if inside else None


@dataclass(frozen=True, slots=True)
class CaptureContext:
    """The rules for one analysis. Fixed at analysis time and stored with it.

    ``guarded`` can be set by the registry or by the person uploading, but a
    reviewer can never turn it off later: it only ever removes protection
    from apparent adults, which is exactly what someone hiding behind a band
    would want to undo.
    """

    guarded: bool = False
    # "registry" (location matched a zone), "user" (declared at upload), or "none".
    guarded_source: str = "none"
    zone_name: str | None = None
    zone_kind: str | None = None
    # Event mode: also protect everyone who looks under 18, band or not.
    protect_minors: bool = False
    # Whether the file carried a capture location at all (the location itself
    # is never stored).
    had_location: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> CaptureContext:
        data = data or {}
        return cls(**{k: data[k] for k in cls.__slots__ if k in data})

    @property
    def needs_age_for_wearers(self) -> bool:
        return self.guarded

    @property
    def needs_age_for_everyone(self) -> bool:
        return self.protect_minors


def resolve_context(
    location: tuple[float, float] | None,
    registry: ZoneRegistry | None,
    declared_guarded: bool = False,
    protect_minors: bool = False,
) -> CaptureContext:
    zone = registry.match(*location) if registry is not None and location is not None else None
    if zone is not None:
        return CaptureContext(
            guarded=True,
            guarded_source="registry",
            zone_name=zone.name,
            zone_kind=zone.kind,
            protect_minors=protect_minors,
            had_location=True,
        )
    return CaptureContext(
        guarded=declared_guarded,
        guarded_source="user" if declared_guarded else "none",
        protect_minors=protect_minors,
        had_location=location is not None,
    )


# --- Reading the capture location -------------------------------------------

# ISO 6709 as written by phones: "+40.4183-003.6944+650.000/" (lat, lon, alt).
_ISO6709 = re.compile(r"^\s*([+-]\d{1,2}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)")
_LOCATION_TAGS = ("com.apple.quicktime.location.iso6709", "location", "location-eng")


def parse_iso6709(text: str) -> tuple[float, float] | None:
    m = _ISO6709.match(text)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def video_location(path: Path) -> tuple[float, float] | None:
    """Capture location from an MP4/MOV container's metadata, if present."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    tags = {
        k.lower(): v
        for k, v in json.loads(result.stdout or "{}").get("format", {}).get("tags", {}).items()
    }
    for key in _LOCATION_TAGS:
        if key in tags and (loc := parse_iso6709(str(tags[key]))):
            return loc
    return None


_GPS_IFD = 0x8825


def _dms(value, ref: str) -> float:
    d, m, s = (float(v) for v in value)
    deg = d + m / 60 + s / 3600
    return -deg if ref.upper() in {"S", "W"} else deg


def photo_location(path: Path) -> tuple[float, float] | None:
    """Capture location from a photo's EXIF GPS block, if present."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            gps = img.getexif().get_ifd(_GPS_IFD)
    except Exception:
        log.debug("could not read EXIF from %s", path.name, exc_info=True)
        return None
    try:
        lat = _dms(gps[2], gps.get(1, "N"))
        lon = _dms(gps[4], gps.get(3, "E"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def parse_location(text: str) -> tuple[float, float]:
    """``"lat,lon"`` from the command line or API."""
    try:
        lat_s, lon_s = text.split(",")
        lat, lon = float(lat_s), float(lon_s)
    except ValueError as exc:
        raise ValueError("location must be 'latitude,longitude'") from exc
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("location out of range")
    return lat, lon
