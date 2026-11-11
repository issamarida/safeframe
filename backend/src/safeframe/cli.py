"""Command-line interface.

safeframe process input.mp4 -o blurred.mp4            # analyse + render in one go
safeframe analyze input.mp4 -o analysis.json          # analysis only, for review
safeframe render input.mp4 analysis.json -o out.mp4  # render a (reviewed) analysis
safeframe serve                                       # web app
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from safeframe.models import DEFAULT_YOLO_WEIGHTS
from safeframe.pipeline.policy import DEFAULT_MIN_FACE_SAMPLES, DEFAULT_THRESHOLD, DecisionPolicy

NOTICE = (
    "NOTE: age estimation is probabilistic and can be wrong. People SafeFrame is\n"
    "unsure about are blurred by default. Review the output before sharing it."
)


def _bar(label: str):
    def update(fraction: float) -> None:
        filled = int(fraction * 30)
        sys.stderr.write(f"\r{label:<10} [{'#' * filled}{'.' * (30 - filled)}] {fraction:6.1%}")
        if fraction >= 1.0:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return update


def _policy(args) -> DecisionPolicy:
    return DecisionPolicy(threshold=args.threshold, min_face_samples=args.min_face_samples)


def _summarise(analysis, policy) -> None:
    for t in analysis.tracks:
        status = policy.classify(t)
        p = "  n/a" if t.p_minor is None else f"{t.p_minor:5.2f}"
        print(
            f"  track {t.track_id:>4}  frames {t.first_frame:>6}-{t.last_frame:<6} "
            f"P(<18)={p}  face samples={len(t.samples):>3}  -> {status.value}"
        )


def _coverage(plan) -> None:
    s = plan.coverage_stats()
    print(
        f"protected person-frames: {s['mask']} by silhouette, {s['box']} by box fallback"
        + (f"; {s['manual']} manual region-frames" if s["manual"] else "")
    )


def _analyze(path: Path, args):
    from safeframe.pipeline.analyze import analyze_video
    from safeframe.pipeline.factory import ModelBundle, ModelConfig

    bundle = ModelBundle(ModelConfig(yolo_weights=args.yolo, device=args.device))
    return analyze_video(
        path,
        bundle.new_tracker(),
        bundle.new_face_detector(),
        bundle.age_estimator(),
        progress=_bar("analysing"),
    ).analysis


def cmd_process(args) -> int:
    from safeframe.pipeline.render import plan_render, render_video

    analysis = _analyze(args.input, args)
    policy = _policy(args)
    _summarise(analysis, policy)
    plan = plan_render(analysis, policy)
    render_video(args.input, args.output, analysis, plan, progress=_bar("rendering"))
    print(f"wrote {args.output}  (blurred tracks: {plan.blurred_tracks or 'none'})")
    _coverage(plan)
    return 0


def cmd_analyze(args) -> int:
    analysis = _analyze(args.input, args)
    _summarise(analysis, _policy(args))
    args.output.write_text(json.dumps(analysis.to_dict()))
    print(f"wrote {args.output}")
    return 0


def cmd_render(args) -> int:
    from safeframe.pipeline.render import plan_render, render_video
    from safeframe.pipeline.types import Analysis

    analysis = Analysis.from_dict(json.loads(args.analysis.read_text()))
    plan = plan_render(analysis, _policy(args))
    render_video(args.input, args.output, analysis, plan, progress=_bar("rendering"))
    print(f"wrote {args.output}")
    _coverage(plan)
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

    def common(p, with_models=True):
        p.add_argument(
            "--threshold",
            type=float,
            default=DEFAULT_THRESHOLD,
            help="P(under 18) at or above which a person is blurred "
            f"(default {DEFAULT_THRESHOLD}; lower = more blurring)",
        )
        p.add_argument(
            "--min-face-samples",
            type=int,
            default=DEFAULT_MIN_FACE_SAMPLES,
            help="face observations needed before a person can be judged an adult",
        )
        if with_models:
            p.add_argument(
                "--yolo",
                default=DEFAULT_YOLO_WEIGHTS,
                help=f"YOLO weights name or path (default {DEFAULT_YOLO_WEIGHTS}; a "
                "detection-only model such as yolo11n.pt blurs rectangles instead of silhouettes)",
            )
            p.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps (default: auto)")

    p = sub.add_parser("process", help="analyse and render in one step")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    common(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("analyze", help="detect, track and age-estimate; write JSON")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    common(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("render", help="render a video from an analysis JSON")
    p.add_argument("input", type=Path)
    p.add_argument("analysis", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    common(p, with_models=False)
    p.set_defaults(func=cmd_render)

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
    if args.command != "serve":
        print(NOTICE, file=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
