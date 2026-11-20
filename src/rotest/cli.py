"""Command-line interface: `rotest compare MOVING REFERENCE` and `rotest ui`."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from rotest.io import DEFAULT_WORK_SIZE, prepare
from rotest.register import RegistrationError, default_threads, register, set_threads
from rotest.report import write_report
from rotest.uncertainty import angle_ci95, angle_uncertainty, spatial_check


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rotest",
        description="Estimate rotation, scale and shift between two top-down images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cmp = sub.add_parser(
        "compare",
        help="register MOVING onto REFERENCE and write results, overlay and summary",
        description=(
            "Estimate the similarity transform that maps MOVING onto REFERENCE. "
            "Angle is clockwise-positive, in degrees. Writes results.json, overlay.png, "
            "checkerboard.png and summary.md into --out."
        ),
    )
    cmp.add_argument("moving", help="path of the moving image")
    cmp.add_argument("reference", help="path of the reference image")
    cmp.add_argument("--mask-moving", help="mask for the moving image (white = use, black = ignore)")
    cmp.add_argument(
        "--mask-reference", help="mask for the reference image (white = use, black = ignore)"
    )
    cmp.add_argument(
        "--work-size",
        type=int,
        default=DEFAULT_WORK_SIZE,
        help="long side in pixels used for matching (default: %(default)s)",
    )
    cmp.add_argument(
        "--threads",
        type=_positive_int,
        default=default_threads(),
        help="OpenCV threads to use (default: %(default)s, half of the logical cores)",
    )
    cmp.add_argument("--out", default="results", help="output directory (default: %(default)s)")
    ui = sub.add_parser(
        "ui",
        help="launch the optional local web interface (needs the 'ui' extra)",
        description="Launch the local Streamlit interface, bound to localhost only.",
    )
    ui.add_argument("--port", type=_positive_int, default=8501, help="port (default: %(default)s)")
    return parser


def _run_ui(port: int) -> int:
    """Launch Streamlit with forced flags: local only, no telemetry, large uploads."""
    if importlib.util.find_spec("streamlit") is None:
        print(
            "error: the UI needs the optional 'ui' extra; install it with: uv sync --extra ui",
            file=sys.stderr,
        )
        return 1
    app = Path(__file__).with_name("ui.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
        "--server.maxUploadSize", "2048",
        "--server.showEmailPrompt", "false",
        "--server.port", str(port),
    ]  # fmt: skip
    return subprocess.run(cmd, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code (0 ok, 1 failure, 2 bad usage)."""
    args = build_parser().parse_args(argv)
    if args.command == "ui":
        return _run_ui(args.port)
    try:
        set_threads(args.threads)
        moving = prepare(args.moving, args.mask_moving, args.work_size)
        reference = prepare(args.reference, args.mask_reference, args.work_size)
        result = register(moving, reference)
        ci = angle_ci95(result)
        spatial = spatial_check(result)
        warnings = write_report(args.out, moving, reference, result, ci, spatial)
    except RegistrationError as exc:
        print(f"error: registration failed, not enough reliable matches: {exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"angle: {result.angle_cw_deg:.3f} deg clockwise "
        f"(95% CI {ci[0]:.3f} to {ci[1]:.3f}), scale: {result.scale:.4f}"
    )
    dev = "n/a" if spatial.angle_dev_deg is None else f"{spatial.angle_dev_deg:.4f}"
    print(
        f"angle uncertainty: +-{angle_uncertainty(ci, spatial):.3f} deg "
        f"(bootstrap {(ci[1] - ci[0]) / 2:.4f} + spatial {dev})"
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"outputs written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
