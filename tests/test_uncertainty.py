"""Bootstrap interval tests on synthetic point sets (no images, fixed seeds)."""

import cv2
import numpy as np
import pytest

from rotest.geometry import from_matrix, to_matrix
from rotest.register import Result, fit_similarity, register
from rotest.report import SPATIAL_DEV_WARN_DEG, quality_warnings
from rotest.uncertainty import (
    SpatialCheck,
    angle_ci95,
    angle_uncertainty,
    bootstrap_angles,
    ci95_from_angles,
    spatial_check,
)


def _make_result(true_angle: float, noise: float, seed: int = 1, n: int = 200) -> Result:
    rng = np.random.default_rng(seed)
    src = rng.uniform(0, 1000, size=(n, 2))
    m_true = to_matrix(true_angle, 1.1, 30.0, -20.0)
    dst = src @ m_true[:, :2].T + m_true[:, 2] + rng.normal(0, noise, size=(n, 2))
    m = fit_similarity(src, dst)
    angle, scale, tx, ty = from_matrix(m)
    return Result(angle, scale, (tx, ty), n, n, noise, m, src, dst)


def _width(ci: tuple[float, float]) -> float:
    return ci[1] - ci[0]


def test_ci_contains_true_angle() -> None:
    for angle in (-120.0, -3.0, 0.0, 17.0, 90.0):
        lo, hi = angle_ci95(_make_result(angle, noise=1.0))
        assert lo <= angle <= hi


def test_ci_widens_with_noise() -> None:
    widths = [_width(angle_ci95(_make_result(10.0, noise=s))) for s in (0.5, 2.0, 8.0)]
    assert widths[0] < widths[1] < widths[2]


def test_ci_across_180_is_unwrapped() -> None:
    # Truth at the seam: a naive percentile over raw angles would give a ~360 deg wide interval.
    res = _make_result(179.5, noise=30.0)
    lo, hi = angle_ci95(res)
    assert lo <= res.angle_cw_deg <= hi
    assert hi > 180.0
    assert hi - lo < 10.0


def test_ci_is_deterministic() -> None:
    res = _make_result(5.0, noise=1.0)
    assert angle_ci95(res) == angle_ci95(res)


def test_spatial_check_regions() -> None:
    from synthetic import make_pair

    mov, ref, _ = make_pair(10.0, 1.25)
    res = register(mov, ref)
    spatial = spatial_check(res)
    assert [r.name for r in spatial.regions] == [
        "top_left", "top_right", "bottom_left", "bottom_right",
    ]
    assert sum(r.inliers for r in spatial.regions) == res.inliers
    for r in spatial.regions:
        assert r.angle_cw_deg is not None and r.angle_ci95 is not None
        assert r.angle_ci95[0] <= r.angle_cw_deg <= r.angle_ci95[1]
        assert r.scale == pytest.approx(1.25, rel=0.01)
    assert spatial.angle_dev_deg is not None


def test_spatial_check_skipped_with_few_inliers() -> None:
    res = _make_result(10.0, noise=1.0, n=40)
    spatial = spatial_check(res, n_boot=50)
    assert spatial.angle_dev_deg is None
    assert all(r.angle_cw_deg is None for r in spatial.regions)
    assert sum(r.inliers for r in spatial.regions) == 40


def test_spatial_check_degenerate_region() -> None:
    # Four well separated clusters of 50 points, one per quadrant; the top-left one collapses
    # to a single point, so its fit is degenerate.
    rng = np.random.default_rng(1)
    centers = [(-100.0, -100.0), (100.0, -100.0), (-100.0, 100.0), (100.0, 100.0)]
    src = np.vstack([rng.normal(c, 10.0, size=(50, 2)) for c in centers])
    src[:50] = (-100.0, -100.0)  # exact in float, so the centred points are exactly 0
    m = to_matrix(10.0, 1.0, 5.0, 5.0)
    dst = src @ m[:, :2].T + m[:, 2]
    res = Result(10.0, 1.0, (5.0, 5.0), 200, 200, 0.0, m, src, dst)
    spatial = spatial_check(res, n_boot=50)
    first = spatial.regions[0]
    assert first.inliers == 50
    assert first.angle_cw_deg is None and first.angle_ci95 is None and first.scale is None
    assert all(r.angle_cw_deg is not None for r in spatial.regions[1:])


def test_angle_uncertainty() -> None:
    regions = spatial_check(_make_result(10.0, noise=1.0), n_boot=20).regions
    assert angle_uncertainty((1.0, 1.4), SpatialCheck(regions, 0.1)) == pytest.approx(0.3)
    assert angle_uncertainty((1.0, 1.4), SpatialCheck(regions, None)) == pytest.approx(0.2)


def _spatial_pair(mov, ref):
    res = register(mov, ref)
    ci = angle_ci95(res)
    spatial = spatial_check(res)
    return res, ci, spatial, quality_warnings(res, spatial)


def test_spatial_check_quiet_on_rigid_pairs() -> None:
    from synthetic import make_pair, prepare, texture
    from test_accuracy import CASES

    pairs = [make_pair(a, s, n, c, seed=0)[:2] for a, s, n, c in CASES]
    # Same pair as the sign test: reference rotated 25 deg clockwise to get the moving image.
    ref_img = texture()
    c = (ref_img.shape[0] - 1) / 2
    rot = to_matrix(25.0, 1.0, 0.0, 0.0)
    tx = c - (rot[0, 0] * c + rot[0, 1] * c)
    ty = c - (rot[1, 0] * c + rot[1, 1] * c)
    mov_img = cv2.warpAffine(ref_img, to_matrix(25.0, 1.0, tx, ty), ref_img.shape[::-1])
    pairs.append((prepare(mov_img), prepare(ref_img)))
    for mov, ref in pairs:
        _, _, spatial, warnings = _spatial_pair(mov, ref)
        assert spatial.angle_dev_deg is None or spatial.angle_dev_deg < SPATIAL_DEV_WARN_DEG
        assert not any(w.startswith("spatial inconsistency") for w in warnings)


def test_spatial_check_detects_local_twist() -> None:
    from synthetic import make_warped_pair

    mov, ref, truth = make_warped_pair(10.0, 1.0)
    res, ci, spatial, warnings = _spatial_pair(mov, ref)
    assert truth == 10.0
    assert spatial.angle_dev_deg is not None and spatial.angle_dev_deg > 0.1
    worst = max(
        (r for r in spatial.regions if r.angle_cw_deg is not None),
        key=lambda r: abs(r.angle_cw_deg - res.angle_cw_deg),
    )
    assert worst.name == "top_left"
    assert any(w.startswith("spatial inconsistency") for w in warnings)
    error = abs(res.angle_cw_deg - truth)
    assert angle_uncertainty(ci, spatial) >= error
    assert (ci[1] - ci[0]) / 2 < error  # the bootstrap alone misses the systematic error


def test_spatial_progress_events() -> None:
    events: list[tuple[str, float]] = []
    spatial_check(
        _make_result(10.0, noise=1.0), n_boot=20, progress=lambda s, f: events.append((s, f))
    )
    assert events == [("spatial", 0.0), ("spatial", 1.0)]


def _reference_ci95(res: Result, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """The previous algorithm, kept here to prove `angle_ci95` did not change."""
    rng = np.random.default_rng(seed)
    n = len(res.src_pts)
    diffs: list[float] = []
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, size=n)
        m = fit_similarity(res.src_pts[idx], res.dst_pts[idx])
        d = from_matrix(m)[0] - res.angle_cw_deg
        diffs.append(180.0 - np.mod(180.0 - d, 360.0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (float(res.angle_cw_deg + lo), float(res.angle_cw_deg + hi))


def test_angle_ci95_unchanged_by_refactor() -> None:
    for angle in (-179.0, 5.0, 178.0):
        res = _make_result(angle, noise=1.0)
        assert angle_ci95(res) == pytest.approx(_reference_ci95(res), abs=1e-9)


def test_bootstrap_angles_absolute_and_deterministic() -> None:
    res = _make_result(179.0, noise=2.0)
    angles = bootstrap_angles(res, n_boot=200)
    assert angles.shape == (200,)
    assert np.array_equal(angles, bootstrap_angles(res, n_boot=200))
    assert abs(float(np.median(angles)) - res.angle_cw_deg) < 1.0  # not wrapped: near 179
    assert ci95_from_angles(angles) == angle_ci95(res, n_boot=200)


def test_ci95_from_angles_percentiles() -> None:
    lo, hi = ci95_from_angles(np.arange(101.0))
    assert (lo, hi) == pytest.approx((2.5, 97.5))
