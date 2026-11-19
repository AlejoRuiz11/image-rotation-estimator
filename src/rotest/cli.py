"""Command-line interface: `rotest compare MOVING REFERENCE`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rotest.io import DEFAULT_WORK_SIZE, prepare
from rotest.register import RegistrationError, register
from rotest.report import write_report
from rotest.uncertainty import angle_ci95


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
    cmp.add_argument("--out", default="results", help="output directory (default: %(default)s)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code (0 ok, 1 failure, 2 bad usage)."""
    args = build_parser().parse_args(argv)
    try:
        moving = prepare(args.moving, args.mask_moving, args.work_size)
        reference = prepare(args.reference, args.mask_reference, args.work_size)
        result = register(moving, reference)
        ci = angle_ci95(result)
        warnings = write_report(args.out, moving, reference, result, ci)
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
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"outputs written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
