"""Bootstrap confidence interval for the angle and a spatial-consistency check."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rotest.geometry import from_matrix
from rotest.register import Progress, RegistrationError, Result, fit_similarity

N_BOOT = 1000
SEED = 0
MIN_REGION_INLIERS = 30
REGION_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass(frozen=True)
class Region:
    """Fit of one quadrant of the moving image; the optional fields are None if not fitted."""

    name: str
    inliers: int
    angle_cw_deg: float | None
    angle_ci95: tuple[float, float] | None
    scale: float | None


@dataclass(frozen=True)
class SpatialCheck:
    """Per-quadrant fits and the estimated systematic angle error (None if < 2 fitted regions)."""

    regions: tuple[Region, ...]
    angle_dev_deg: float | None


def _wrap(deg: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to (-180, 180]."""
    return 180.0 - np.mod(180.0 - np.asarray(deg), 360.0)


def angle_ci95(
    result: Result, n_boot: int = N_BOOT, seed: int = SEED, progress: Progress | None = None
) -> tuple[float, float]:
    """95% bootstrap interval `(low, high)` for `result.angle_cw_deg`.

    Resamples the inlier pairs with replacement and refits by least squares. Angles are taken
    as differences from the point estimate, wrapped to (-180, 180], so intervals straddling
    +-180 deg do not break the percentiles. Endpoints are NOT wrapped: `low <= high` always, and
    near +-180 deg they may fall outside (-180, 180]. Degenerate resamples are
    discarded and redrawn. `progress` (optional) receives ("bootstrap", fraction) about every 5%
    of the resamples.
    """
    n = len(result.src_pts)
    if n < 3:
        raise RegistrationError("not enough inliers for a bootstrap")
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    step = max(1, n_boot // 20)
    if progress:
        progress("bootstrap", 0.0)
    attempts = 0
    while len(diffs) < n_boot:
        attempts += 1
        if attempts > 10 * n_boot:
            raise RegistrationError("bootstrap resamples are degenerate")
        idx = rng.integers(0, n, size=n)
        try:
            m = fit_similarity(result.src_pts[idx], result.dst_pts[idx])
        except RegistrationError:
            continue
        diffs.append(float(_wrap(from_matrix(m)[0] - result.angle_cw_deg)))
        if progress and len(diffs) % step == 0 and len(diffs) < n_boot:
            progress("bootstrap", len(diffs) / n_boot)
    if progress:
        progress("bootstrap", 1.0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (float(result.angle_cw_deg + lo), float(result.angle_cw_deg + hi))


def _fit_region(
    result: Result, name: str, mask: np.ndarray, s_w: float, n_boot: int, seed: int
) -> Region:
    """Fit one quadrant; too few inliers or a degenerate fit leaves it unfitted."""
    n = int(mask.sum())
    if n < MIN_REGION_INLIERS:
        return Region(name, n, None, None, None)
    src, dst = result.src_pts[mask], result.dst_pts[mask]
    try:
        angle, scale_w, _, _ = from_matrix(fit_similarity(src, dst))
        sub = replace(result, angle_cw_deg=angle, src_pts=src, dst_pts=dst, inliers=n)
        ci = angle_ci95(sub, n_boot, seed)
    except RegistrationError:
        return Region(name, n, None, None, None)
    return Region(name, n, angle, ci, scale_w * result.scale / s_w)


def spatial_check(
    result: Result, n_boot: int = N_BOOT, seed: int = SEED, progress: Progress | None = None
) -> SpatialCheck:
    """Fit each quadrant of the moving image separately and compare the angles.

    Only the existing inliers are reused (no re-detection). Quadrants are split at the median of
    the inlier coordinates, so they hold similar counts even if inliers cluster. The systematic
    error estimate is `max_i max(0, |theta_i - theta| - h_i)`, with `h_i` the bootstrap half-width
    of region i: subtracting `h_i` removes the disagreement that region noise alone explains, so
    small regions of a rigid pair do not raise false alarms. The max (not a mean) is used because
    one distorted zone is enough to bias the global fit.
    """
    if progress:
        progress("spatial", 0.0)
    src = result.src_pts
    mx, my = np.median(src[:, 0]), np.median(src[:, 1])
    right, bottom = src[:, 0] >= mx, src[:, 1] >= my
    masks = (~right & ~bottom, right & ~bottom, ~right & bottom, right & bottom)
    s_w = from_matrix(result.matrix_work)[1]
    regions = tuple(
        _fit_region(result, name, mask, s_w, n_boot, seed)
        for name, mask in zip(REGION_NAMES, masks, strict=True)
    )
    excess = [
        max(0.0, abs(float(_wrap(r.angle_cw_deg - result.angle_cw_deg))) - (r.angle_ci95[1] - r.angle_ci95[0]) / 2)
        for r in regions
        if r.angle_cw_deg is not None and r.angle_ci95 is not None
    ]
    if progress:
        progress("spatial", 1.0)
    return SpatialCheck(regions, max(excess) if len(excess) >= 2 else None)


def angle_uncertainty(ci: tuple[float, float], spatial: SpatialCheck) -> float:
    """Total angle uncertainty: bootstrap half-width plus the spatial deviation.

    A linear sum assumes no independence between the two sources, so it is a conservative bound.
    """
    return (ci[1] - ci[0]) / 2 + (spatial.angle_dev_deg or 0.0)
