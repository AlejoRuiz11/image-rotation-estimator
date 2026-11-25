# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fabian Alejandro Ruiz Osorio (AlejoRuiz11)
"""Write registration outputs: results.json, overlay.png, checkerboard.png, summary.md."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from rotest.io import Prepared
from rotest.register import Result
from rotest.uncertainty import SpatialCheck, angle_uncertainty

SCHEMA_VERSION = 2
SPATIAL_DEV_WARN_DEG = 0.05
MIN_INLIERS_WARN = 30
MAX_RESIDUAL_WARN_PX = 3.0
SCALE_RANGE = (0.5, 2.0)
TILE_PX = 64


def quality_warnings(result: Result, spatial: SpatialCheck) -> list[str]:
    """Quality warnings (not errors); the result is still reported."""
    warnings: list[str] = []
    if result.inliers < MIN_INLIERS_WARN:
        warnings.append(f"low inlier count: {result.inliers} < {MIN_INLIERS_WARN}")
    if result.residual_rms_px > MAX_RESIDUAL_WARN_PX:
        warnings.append(
            f"high residual RMS: {result.residual_rms_px:.2f} px > {MAX_RESIDUAL_WARN_PX:g} px"
        )
    if not SCALE_RANGE[0] <= result.scale <= SCALE_RANGE[1]:
        warnings.append(
            f"scale {result.scale:.3f} outside [{SCALE_RANGE[0]}, {SCALE_RANGE[1]}]"
        )
    dev = spatial.angle_dev_deg
    if dev is None:
        warnings.append(
            "spatial check skipped: fewer than 2 regions with >= 30 inliers; "
            "systematic error not estimated"
        )
    elif dev > SPATIAL_DEV_WARN_DEG:
        warnings.append(
            f"spatial inconsistency: region angles deviate up to {dev:.3f} deg beyond their "
            f"noise (> {SPATIAL_DEV_WARN_DEG:g} deg); possible non-rigid distortion"
        )
    return warnings


def warp_moving(
    moving: Prepared, reference: Prepared, result: Result
) -> NDArray[np.uint8]:
    """Moving working image warped onto the reference working grid."""
    ref_h, ref_w = reference.image.shape[:2]
    return cv2.warpAffine(moving.image, result.matrix_work, (ref_w, ref_h))


def _checkerboard(
    a: NDArray[np.uint8], b: NDArray[np.uint8], tile: int = TILE_PX
) -> NDArray[np.uint8]:
    rows, cols = np.indices(a.shape[:2])
    use_a = ((rows // tile + cols // tile) % 2) == 0
    return np.where(use_a, a, b).astype(np.uint8)


def _fmt(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _spatial_lines(ci: tuple[float, float], spatial: SpatialCheck) -> list[str]:
    lines = [
        "## Spatial consistency",
        "",
        "The inliers are split into four quadrants of the moving image and each is fitted",
        "separately; disagreement beyond each quadrant's own noise suggests systematic error.",
        "",
        "| Region | Inliers | Angle (deg) | 95% interval (deg) | Scale |",
        "|---|---:|---:|---|---:|",
    ]
    for r in spatial.regions:
        interval = (
            "n/a" if r.angle_ci95 is None else f"[{r.angle_ci95[0]:.3f}, {r.angle_ci95[1]:.3f}]"
        )
        lines.append(
            f"| {r.name} | {r.inliers} | {_fmt(r.angle_cw_deg, 3)} | {interval} | "
            f"{_fmt(r.scale, 4)} |"
        )
    lines += [
        "",
        f"- Spatial deviation beyond noise (deg): {_fmt(spatial.angle_dev_deg, 3)}",
        (
            f"- Total angle uncertainty (deg): +-{angle_uncertainty(ci, spatial):.3f}"
            " (bootstrap half-width + spatial deviation)"
        ),
        "",
    ]
    return lines


def _summary(
    result: Result, ci: tuple[float, float], spatial: SpatialCheck, warnings: list[str]
) -> str:
    tx, ty = result.translation_px
    lines = [
        "# Registration summary",
        "",
        "## Result",
        "",
        f"- Angle (clockwise, deg): {result.angle_cw_deg:.3f}",
        f"- 95% interval (deg): [{ci[0]:.3f}, {ci[1]:.3f}]",
        f"- Scale: {result.scale:.4f}",
        f"- Translation (px, original reference): [{tx:.2f}, {ty:.2f}]",
        f"- Inliers / matches: {result.inliers} / {result.matches}",
        f"- Residual RMS (working px): {result.residual_rms_px:.2f}",
        "",
        *_spatial_lines(ci, spatial),
        "## Warnings",
        "",
    ]
    lines += [f"- {w}" for w in warnings] if warnings else ["None."]
    lines += [
        "",
        "## Assumptions",
        "",
        "- Both images are top-down views related by a similarity (rotation, uniform scale, shift).",
        "- The angle is the clockwise rotation, as seen on screen, applied to the moving image to",
        "  align it with the reference. With a north-up reference it is the heading of the moving",
        "  image's up direction.",
        "- The interval is a bootstrap over inliers; it covers matching noise, not systematic",
        "  error. The spatial check estimates systematic error from disagreement between image",
        "  quadrants; the total uncertainty adds both.",
        "- overlay.png and checkerboard.png are grayscale at the reference working resolution.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    out_dir: str | Path,
    moving: Prepared,
    reference: Prepared,
    result: Result,
    angle_ci95: tuple[float, float],
    spatial: SpatialCheck,
) -> list[str]:
    """Write all outputs into `out_dir` (created if missing); return the warnings."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings = quality_warnings(result, spatial)

    data = {
        "schema_version": SCHEMA_VERSION,
        "angle_cw_deg": result.angle_cw_deg,
        "angle_ci95": [angle_ci95[0], angle_ci95[1]],
        "scale": result.scale,
        "translation_px": list(result.translation_px),
        "inliers": result.inliers,
        "matches": result.matches,
        "residual_rms_px": result.residual_rms_px,
        "warnings": warnings,
        "angle_uncertainty_deg": angle_uncertainty(angle_ci95, spatial),
        "spatial_check": {
            "angle_dev_deg": spatial.angle_dev_deg,
            "regions": [
                {
                    "name": r.name,
                    "inliers": r.inliers,
                    "angle_cw_deg": r.angle_cw_deg,
                    "angle_ci95": None if r.angle_ci95 is None else list(r.angle_ci95),
                    "scale": r.scale,
                }
                for r in spatial.regions
            ],
        },
    }
    (out / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    warped = warp_moving(moving, reference, result)
    overlay = cv2.addWeighted(reference.image, 0.5, warped, 0.5, 0)
    for name, img in (
        ("overlay.png", overlay),
        ("checkerboard.png", _checkerboard(reference.image, warped)),
    ):
        if not cv2.imwrite(str(out / name), img):
            raise OSError(f"could not write {out / name}")

    (out / "summary.md").write_text(_summary(result, angle_ci95, spatial, warnings), encoding="utf-8")
    return warnings
