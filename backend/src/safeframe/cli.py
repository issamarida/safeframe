"""Command-line interface.

safeframe process photo.jpg -o shared.jpg              # analyse + render a photo or video
safeframe analyze clip.mp4 -o analysis.json            # analysis only, for review
safeframe render clip.mp4 analysis.json -o shared.mp4  # render a (reviewed) analysis
safeframe band -o band.svg                             # printable SafeFrame Bands (A4)
safeframe trustees init --threshold 3 --trustees 5 --out keys/
safeframe unseal shared.mp4.sfseal --share keys/trustee-1.json ... -o original.mp4
safeframe serve                                        # web app
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from safeframe.models import DEFAULT_YOLO_WEIGHTS
from safeframe.pipeline.policy import (
    DEFAULT_MIN_BAND_HITS,
    DEFAULT_MIN_FACE_SAMPLES,
    DEFAULT_THRESHOLD,
    DecisionPolicy,
)

NOTICE = (
    "NOTE: SafeFrame blurs people wearing a SafeFrame Band. A band it cannot see\n"
    "cannot protect anyone, so review the output before sharing it."
)
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _bar(label: str):
    def update(fraction: float) -> None:
        filled = int(fraction * 30)
        sys.stderr.write(f"\r{label:<10} [{'#' * filled}{'.' * (30 - filled)}] {fraction:6.1%}")
        if fraction >= 1.0:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return update


def is_photo(path: Path) -> bool:
    return path.suffix.lower() in PHOTO_SUFFIXES


def _policy(args, analysis) -> DecisionPolicy:
    return DecisionPolicy.for_analysis(
        analysis,
        threshold=args.threshold,
        min_face_samples=args.min_face_samples,
        min_band_hits=args.min_band_hits,
    )


def _describe_context(context: dict) -> str:
    parts = []
    if context.get("guarded"):
        where = context.get("zone_name") or "a place declared guarded"
        parts.append(f"GUARDED ZONE ({where}): bands protect only people who look like children")
    if context.get("protect_minors"):
        parts.append("EVENT MODE: everyone who may be under 18 is also blurred")
    return "; ".join(parts) or "standard: every band is honoured"


def _summarise(analysis, policy) -> None:
    print(f"context: {_describe_context(analysis.context)}")
    for t in analysis.tracks:
        band = (
            f"band {t.band_confirmed} clean / {t.band_possible} ambiguous"
            if t.wears_band
            else "no band"
        )
        age = "" if t.p_minor is None else f"  P(<18)={t.p_minor:4.2f} ({len(t.samples)} faces)"
        print(
            f"  person {t.track_id:>4}  frames {t.first_frame:>6}-{t.last_frame:<6} "
            f"{band:<28}{age}  -> {policy.classify(t).value}"
        )
    if analysis.orphan_band_frames:
        frames = analysis.orphan_band_frames
        print(
            f"  WARNING: a band was seen on {len(frames)} frame(s) (first: {frames[0]}) but could "
            "not be matched to anyone. Check those frames and add a manual region if needed."
        )


def _coverage(plan) -> None:
    s = plan.coverage_stats()
    print(
        f"protected person-frames: {s['mask']} by silhouette, {s['box']} by box fallback"
        + (f"; {s['manual']} manual region-frames" if s["manual"] else "")
    )


def _context(args, path: Path):
    from safeframe.pipeline.context import (
        ZoneRegistry,
        parse_location,
        photo_location,
        resolve_context,
        video_location,
    )

    if args.location:
        location = parse_location(args.location)
    else:
        location = photo_location(path) if is_photo(path) else video_location(path)
    registry = ZoneRegistry.from_file(args.zones) if args.zones else ZoneRegistry.packaged()
    return resolve_context(location, registry, args.guarded, args.event_mode)


def _analyze(path: Path, args):
    from safeframe.pipeline.analyze import analyze_photo, analyze_video
    from safeframe.pipeline.factory import ModelBundle, ModelConfig

    bundle = ModelBundle(
        ModelConfig(yolo_weights=args.yolo, pose_weights=args.pose, device=args.device)
    )
    context = _context(args, path)
    run = analyze_photo if is_photo(path) else analyze_video
    return run(
        path, bundle.components(photo=is_photo(path)), context, progress=_bar("analysing")
    ).analysis


def _render(source: Path, output: Path, analysis, plan) -> None:
    from safeframe.pipeline.render import render_photo, render_video

    if analysis.video.is_photo:
        if not is_photo(output):
            raise SystemExit("a photo must be written as .jpg, .png or .webp")
        render_photo(source, output, analysis, plan)
    else:
        render_video(source, output, analysis, plan, progress=_bar("rendering"))


def _seal(args, source: Path, output: Path, analysis, plan) -> None:
    if not args.seal_to:
        return
    if not plan.honoured_bands:
        print("nothing was hidden because of a band, so no sealed original was written")
        return
    from safeframe.seal import EscrowPublicKey, seal_file

    envelope = output.with_name(output.name + ".sfseal")
    header = seal_file(
        source,
        envelope,
        EscrowPublicKey.load(args.seal_to),
        reason=f"SafeFrame band honoured for person(s) {plan.honoured_bands}",
        published=output,
        context=analysis.context,
        retention_days=args.retention_days,
    )
    print(
        f"sealed original -> {envelope}  (escrow key {header.key_id}, expires "
        f"{time.strftime('%Y-%m-%d', time.localtime(header.expires_at))})"
    )


def cmd_process(args) -> int:
    from safeframe.pipeline.render import plan_render

    analysis = _analyze(args.input, args)
    policy = _policy(args, analysis)
    _summarise(analysis, policy)
    plan = plan_render(analysis, policy)
    _render(args.input, args.output, analysis, plan)
    print(f"wrote {args.output}  (blurred: {plan.blurred_tracks or 'nobody'})")
    _coverage(plan)
    _seal(args, args.input, args.output, analysis, plan)
    return 0


def cmd_analyze(args) -> int:
    analysis = _analyze(args.input, args)
    _summarise(analysis, _policy(args, analysis))
    args.output.write_text(json.dumps(analysis.to_dict()))
    print(f"wrote {args.output}")
    return 0


def cmd_render(args) -> int:
    from safeframe.pipeline.render import plan_render
    from safeframe.pipeline.types import Analysis

    analysis = Analysis.from_dict(json.loads(args.analysis.read_text()))
    plan = plan_render(analysis, _policy(args, analysis))
    _render(args.input, args.output, analysis, plan)
    print(f"wrote {args.output}")
    _coverage(plan)
    _seal(args, args.input, args.output, analysis, plan)
    return 0


def cmd_band(args) -> int:
    from safeframe.pipeline.band import band_sheet_svg

    sizes = ("child", "adult") if args.size == "both" else (args.size,)
    args.output.write_text(band_sheet_svg(sizes))
    print(f"wrote {args.output}: print at 100 % scale (no 'fit to page')")
    return 0


def cmd_trustees_init(args) -> int:
    from safeframe.seal import create_escrow

    args.out.mkdir(parents=True, exist_ok=True)
    public, shares = create_escrow(args.threshold, args.trustees)
    (args.out / "escrow.pub.json").write_text(public.to_json())
    for share in shares:
        path = args.out / f"trustee-{share.x}.json"
        path.write_text(share.to_json())
        path.chmod(0o600)
    print(
        f"escrow key {public.key_id}: {args.threshold} of {args.trustees} trustees needed to "
        f"unseal.\n  public key  -> {args.out / 'escrow.pub.json'} (give to cameras)\n"
        f"  shares      -> {args.out}/trustee-N.json (give ONE to each trustee, then delete "
        "them here; never commit them)"
    )
    return 0


def cmd_unseal(args) -> int:
    from safeframe.seal import TrusteeShare, unseal_file

    shares = [TrusteeShare.load(p) for p in args.share]
    header = unseal_file(args.envelope, shares, args.output, allow_expired=args.allow_expired)
    print(f"unsealed {header.original_name} -> {args.output}  (reason: {header.reason})")
    return 0


def cmd_inspect(args) -> int:
    from safeframe.seal import read_header

    header = read_header(args.envelope)
    print(json.dumps({k: getattr(header, k) for k in header.__slots__}, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("safeframe.api.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safeframe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def decisions(p):
        p.add_argument(
            "--threshold",
            type=float,
            default=DEFAULT_THRESHOLD,
            help="P(under 18) at or above which someone counts as a child, in guarded zones "
            f"and event mode (default {DEFAULT_THRESHOLD})",
        )
        p.add_argument(
            "--min-face-samples",
            type=int,
            default=DEFAULT_MIN_FACE_SAMPLES,
            help="face observations needed before age evidence counts",
        )
        p.add_argument(
            "--min-band-hits",
            type=int,
            default=DEFAULT_MIN_BAND_HITS,
            help="clean band sightings needed to call someone a wearer (fewer = 'possible', "
            "which is still blurred)",
        )

    def sealing(p):
        p.add_argument(
            "--seal-to",
            type=Path,
            metavar="ESCROW_PUB",
            help="write a sealed original (OUTPUT.sfseal) openable only by the trustee quorum",
        )
        p.add_argument("--retention-days", type=float, default=30.0)

    def analysis(p):
        p.add_argument(
            "--yolo",
            default=DEFAULT_YOLO_WEIGHTS,
            help=f"person model (default {DEFAULT_YOLO_WEIGHTS}; yolo11n.pt blurs rectangles)",
        )
        p.add_argument(
            "--pose",
            default="yolo11n-pose.pt",
            help="pose model for wrist attribution ('' to attribute by silhouette/box only)",
        )
        p.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps (default: auto)")
        p.add_argument(
            "--location",
            help="capture location 'lat,lon' (default: read from the file's metadata)",
        )
        p.add_argument("--zones", type=Path, help="guarded-zone registry JSON")
        p.add_argument(
            "--guarded",
            action="store_true",
            help="treat as captured in a guarded place (bank, court...) even without a location",
        )
        p.add_argument(
            "--event-mode",
            action="store_true",
            help="also blur everyone who may be under 18, band or not",
        )

    p = sub.add_parser("process", help="analyse and render a photo or video in one step")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    analysis(p)
    decisions(p)
    sealing(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("analyze", help="find people and bands; write the analysis JSON")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    analysis(p)
    decisions(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("render", help="render a photo or video from an analysis JSON")
    p.add_argument("input", type=Path)
    p.add_argument("analysis", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    decisions(p)
    sealing(p)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("band", help="write a printable A4 sheet of SafeFrame Bands (SVG)")
    p.add_argument("-o", "--output", type=Path, default=Path("safeframe-band.svg"))
    p.add_argument("--size", choices=["child", "adult", "both"], default="both")
    p.set_defaults(func=cmd_band)

    p = sub.add_parser("trustees", help="evidence-escrow key ceremony")
    tsub = p.add_subparsers(dest="trustees_command", required=True)
    t = tsub.add_parser("init", help="create an escrow key split k-of-n between trustees")
    t.add_argument("--threshold", "-k", type=int, default=3)
    t.add_argument("--trustees", "-n", type=int, default=5)
    t.add_argument("--out", type=Path, required=True)
    t.set_defaults(func=cmd_trustees_init)

    p = sub.add_parser("unseal", help="open a sealed original with a quorum of trustee shares")
    p.add_argument("envelope", type=Path)
    p.add_argument("--share", type=Path, action="append", required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--allow-expired", action="store_true")
    p.set_defaults(func=cmd_unseal)

    p = sub.add_parser("inspect", help="show a sealed envelope's public header")
    p.add_argument("envelope", type=Path)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command in {"process", "analyze", "render"}:
        print(NOTICE, file=sys.stderr)
    from safeframe.seal import SealError
    from safeframe.video.io import VideoError

    try:
        return args.func(args)
    except (SealError, VideoError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
