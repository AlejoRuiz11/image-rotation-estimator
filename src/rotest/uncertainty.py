"""Bootstrap confidence interval for the estimated angle."""

from __future__ import annotations

import numpy as np

from rotest.geometry import from_matrix
from rotest.register import RegistrationError, Result, fit_similarity

N_BOOT = 1000
SEED = 0


def _wrap(deg: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to (-180, 180]."""
    return 180.0 - np.mod(180.0 - np.asarray(deg), 360.0)


def angle_ci95(
    result: Result, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float]:
    """95% bootstrap interval `(low, high)` for `result.angle_cw_deg`.

    Resamples the inlier pairs with replacement and refits by least squares. Angles are taken
    as differences from the point estimate, wrapped to (-180, 180], so intervals straddling
    +-180 deg do not break the percentiles. Endpoints are NOT wrapped: `low <= high` always, and
    near +-180 deg they may fall outside (-180, 180]. Degenerate resamples are
    discarded and redrawn.
    """
    n = len(result.src_pts)
    if n < 3:
        raise RegistrationError("not enough inliers for a bootstrap")
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
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
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (float(result.angle_cw_deg + lo), float(result.angle_cw_deg + hi))
