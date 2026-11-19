"""Bootstrap interval tests on synthetic point sets (no images, fixed seeds)."""

import numpy as np

from rotest.geometry import from_matrix, to_matrix
from rotest.register import Result, fit_similarity
from rotest.uncertainty import angle_ci95


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
