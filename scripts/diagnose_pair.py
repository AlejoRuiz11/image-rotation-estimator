# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fabian Alejandro Ruiz Osorio (AlejoRuiz11)
"""Diagnose why the angle of a pair moves with the matcher or the image zone.

Usage: uv run python scripts/diagnose_pair.py MOVING REFERENCE --out DIR [--mask-moving M]
       [--mask-reference M] [--work-size N] [--threads N]

Runs the production `register` with three matchers (BFMatcher, the FLANN of the package and a
more precise FLANN), and for each one fits the angle on quadrants and halves of the moving image.
It then compares which point pairs each matcher keeps. Writes `diagnosis.json` and `diagnosis.md`
into DIR, which must be OUTSIDE the repository. The outputs never contain input paths or images;
the "Shareable summary" part holds only angles and angle differences.

`bf` can take minutes with large images (it is brute force).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import rotest.register as register_module
from rotest.geometry import from_matrix
from rotest.io import DEFAULT_WORK_SIZE, prepare
from rotest.register import (
    FLANN_SEED,
    RATIO,
    RegistrationError,
    Result,
    default_threads,
    fit_similarity,
    register,
    set_threads,
)
from rotest.uncertainty import (
    MIN_REGION_INLIERS,
    REGION_NAMES,
    angle_ci95,
    angle_uncertainty,
    spatial_check,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BF_DESCRIPTOR_LIMIT = 262_143  # BFMatcher knnMatch refuses larger reference sets
PRECISE_TREES = 8
PRECISE_CHECKS = 1024
_FLANN_INDEX_KDTREE = 1
VARIANTS = ("bf", "flann", "flann_precise")
HALF_NAMES = ("left", "right", "top", "bottom")
COORD_DECIMALS = 3  # 1e-3 px: enough to identify the same keypoint pair across runs


class _Skipped(Exception):
    """Raised by a matcher that cannot run on these inputs."""


def _ratio_test(knn) -> list[cv2.DMatch]:
    return [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO * p[1].distance]


def _match_bf(des_m, des_r) -> list[cv2.DMatch]:
    if len(des_r) > BF_DESCRIPTOR_LIMIT:
        raise _Skipped("BFMatcher limit")
    return _ratio_test(cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_m, des_r, k=2))


def _match_flann_precise(des_m, des_r) -> list[cv2.DMatch]:
    cv2.setRNGSeed(FLANN_SEED)
    matcher = cv2.FlannBasedMatcher(
        {"algorithm": _FLANN_INDEX_KDTREE, "trees": PRECISE_TREES}, {"checks": PRECISE_CHECKS}
    )
    return _ratio_test(matcher.knnMatch(des_m, des_r, k=2))


def _wrap(deg: float) -> float:
    """Wrap an angle difference to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def _fit(src: np.ndarray, dst: np.ndarray, s_w: float, result: Result) -> dict | None:
    """Similarity fit of a point subset: angle and scale in original pixels (None if too few)."""
    if len(src) < MIN_REGION_INLIERS:
        return None
    try:
        angle, scale_w, _, _ = from_matrix(fit_similarity(src, dst))
    except RegistrationError:
        return None
    return {"inliers": len(src), "angle_cw_deg": angle, "scale": scale_w * result.scale / s_w}


def _subsets(result: Result) -> dict[str, dict[str, np.ndarray]]:
    """Quadrants and halves of the moving image, split at the median of the inlier coordinates."""
    src = result.src_pts
    right = src[:, 0] >= np.median(src[:, 0])
    bottom = src[:, 1] >= np.median(src[:, 1])
    quads = (~right & ~bottom, right & ~bottom, ~right & bottom, right & bottom)
    halves = (~right, right, ~bottom, bottom)
    return {
        "quadrants": dict(zip(REGION_NAMES, quads, strict=True)),
        "halves": dict(zip(HALF_NAMES, halves, strict=True)),
    }


def _run_variant(name: str, matcher, moving, reference) -> tuple[dict, Result | None]:
    """Run the production `register` with `matcher` swapped in; restore it afterwards."""
    original = register_module._match
    if matcher is not None:
        register_module._match = matcher
    start = time.perf_counter()
    try:
        result = register(moving, reference)
    except _Skipped as exc:
        return {"status": f"skipped: {exc}"}, None
    except RegistrationError as exc:
        return {"status": f"failed: {exc}"}, None
    finally:
        register_module._match = original
    seconds = time.perf_counter() - start

    ci = angle_ci95(result)
    spatial = spatial_check(result)
    s_w = from_matrix(result.matrix_work)[1]
    data: dict = {
        "status": "ok",
        "angle_cw_deg": result.angle_cw_deg,
        "angle_ci95": list(ci),
        "ci95_halfwidth": (ci[1] - ci[0]) / 2,
        "angle_dev_deg": spatial.angle_dev_deg,
        "angle_uncertainty": angle_uncertainty(ci, spatial),
        "scale": result.scale,
        "matches": result.matches,
        "inliers": result.inliers,
        "residual_rms_px": result.residual_rms_px,
        "seconds": seconds,
        "spatial_check": {
            r.name: {"inliers": r.inliers, "angle_cw_deg": r.angle_cw_deg, "scale": r.scale}
            for r in spatial.regions
        },
    }
    groups = _subsets(result)
    data["halves"] = {
        key: _fit(result.src_pts[m], result.dst_pts[m], s_w, result)
        for key, m in groups["halves"].items()
    }
    for container in (data["spatial_check"], data["halves"]):
        for fit in container.values():
            if fit is not None:
                angle = fit["angle_cw_deg"]
                fit["delta_deg"] = None if angle is None else _wrap(angle - result.angle_cw_deg)
    return data, result


def _pair_keys(result: Result) -> dict[tuple[float, ...], int]:
    pts = np.round(np.hstack([result.src_pts, result.dst_pts]), COORD_DECIMALS)
    return {tuple(row): i for i, row in enumerate(pts)}


def _compare_pairs(a: Result, b: Result) -> dict:
    """Which inlier pairs two matchers share, and the angle fitted on common / exclusive sets."""
    ka, kb = _pair_keys(a), _pair_keys(b)
    common = ka.keys() & kb.keys()
    sets = {
        "common": (a, [ka[k] for k in common]),
        "only_a": (a, [ka[k] for k in ka.keys() - kb.keys()]),
        "only_b": (b, [kb[k] for k in kb.keys() - ka.keys()]),
    }
    out: dict = {}
    for label, (res, idx) in sets.items():
        entry: dict = {"pairs": len(idx), "angle_cw_deg": None}
        if len(idx) >= MIN_REGION_INLIERS:
            try:
                entry["angle_cw_deg"] = from_matrix(
                    fit_similarity(res.src_pts[idx], res.dst_pts[idx])
                )[0]
            except RegistrationError:
                pass
        out[label] = entry
    return out


def _fmt(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _render_markdown(variants: dict, comparisons: dict, base: str | None) -> str:
    ok = {k: v for k, v in variants.items() if v["status"] == "ok"}
    lines = ["# Pair diagnosis", "", "## Shareable summary (angles and differences only)", ""]
    for name, status in variants.items():
        if status["status"] != "ok":
            lines.append(f"- {name}: {status['status']}")
    for name, v in ok.items():
        lines += [
            "",
            f"### {name}",
            f"- angle: {_fmt(v['angle_cw_deg'])} deg cw, CI95 half-width {_fmt(v['ci95_halfwidth'])}",
            (
                f"- angle_dev_deg: {_fmt(v['angle_dev_deg'])}, "
                f"angle_uncertainty: +-{_fmt(v['angle_uncertainty'])} deg"
            ),
            "- delta vs global (quadrants): "
            + ", ".join(f"{k} {_fmt(f['delta_deg'], signed=True)}" for k, f in v["spatial_check"].items()),
            "- delta vs global (halves): "
            + ", ".join(
                f"{k} {_fmt(f and f['delta_deg'], signed=True)}" for k, f in v["halves"].items()
            ),
        ]
    if base in ok:
        lines += ["", f"### Angle difference between variants (relative to {base})"]
        for name, v in ok.items():
            if name != base:
                diff = _wrap(v["angle_cw_deg"] - ok[base]["angle_cw_deg"])
                lines.append(f"- {name} - {base}: {_fmt(diff, signed=True)} deg")
    for name, comp in comparisons.items():
        lines += ["", f"### Pair analysis: {base} (A) vs {name} (B), angle fitted per set"]
        for label in ("common", "only_a", "only_b"):
            lines.append(f"- {label}: angle {_fmt(comp[label]['angle_cw_deg'])} deg")
    lines += ["", "## Local details (do not paste into the repo)", ""]
    for name, v in ok.items():
        lines.append(
            f"- {name}: matches {v['matches']}, inliers {v['inliers']}, scale {v['scale']:.5f}, "
            f"residual RMS {v['residual_rms_px']:.3f} px, time {v['seconds']:.1f} s"
        )
        for group in ("spatial_check", "halves"):
            for key, f in v[group].items():
                if f and f["scale"] is not None:
                    lines.append(f"  - {key}: inliers {f['inliers']}, scale {f['scale']:.5f}")
                else:
                    lines.append(f"  - {key}: not fitted (fewer than {MIN_REGION_INLIERS} inliers)")
    for name, comp in comparisons.items():
        lines.append(
            f"- pairs {base} vs {name}: "
            + ", ".join(f"{label} {comp[label]['pairs']}" for label in ("common", "only_a", "only_b"))
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("moving")
    parser.add_argument("reference")
    parser.add_argument("--out", required=True, help="output directory, must be outside the repo")
    parser.add_argument("--mask-moving")
    parser.add_argument("--mask-reference")
    parser.add_argument("--work-size", type=int, default=DEFAULT_WORK_SIZE)
    parser.add_argument("--threads", type=int, default=default_threads())
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        print("error: --out must be outside the repository", file=sys.stderr)
        return 2

    try:
        set_threads(args.threads)
        moving = prepare(args.moving, args.mask_moving, args.work_size)
        reference = prepare(args.reference, args.mask_reference, args.work_size)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    matchers = {"bf": _match_bf, "flann": None, "flann_precise": _match_flann_precise}
    print("note: the bf variant is brute force and can take minutes with large images", flush=True)
    variants: dict[str, dict] = {}
    results: dict[str, Result] = {}
    for name in VARIANTS:
        print(f"running {name}...", flush=True)
        data, result = _run_variant(name, matchers[name], moving, reference)
        variants[name] = data
        if result is not None:
            results[name] = result

    base = "bf" if "bf" in results else ("flann_precise" if "flann_precise" in results else None)
    comparisons = {
        name: _compare_pairs(results[base], results[name])
        for name in results
        if base is not None and name != base
    }
    if not results:
        print("error: no variant produced a result", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    payload = {"base_variant": base, "variants": variants, "pair_comparisons": comparisons}
    (out / "diagnosis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = _render_markdown(variants, comparisons, base)
    (out / "diagnosis.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
