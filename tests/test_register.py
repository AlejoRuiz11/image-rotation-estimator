"""Registration tests on synthetic images (fixed seed, no network)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rotest.geometry import to_matrix
from rotest.io import Prepared, apply_clahe
from rotest.register import RegistrationError, _match, fit_similarity, register


def _texture(size: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    for _ in range(250):
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.circle(img, c, int(rng.integers(3, 18)), int(rng.integers(40, 255)), -1)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def _prep(img: np.ndarray, factor: float = 1.0) -> Prepared:
    h, w = img.shape
    return Prepared(apply_clahe(img), None, factor, (round(w / factor), round(h / factor)))


def _pair(angle: float, scale: float, tx: float, ty: float):
    """Reference = moving warped by the known transform (moving -> reference)."""
    moving = _texture()
    m = to_matrix(angle, scale, tx, ty)
    ref = cv2.warpAffine(moving, m, (600, 600))
    return _prep(moving), _prep(ref)


@pytest.mark.parametrize(
    "angle,scale,tx,ty",
    [(0.0, 1.0, 0.0, 0.0), (17.0, 1.1, 30.0, -20.0), (-40.0, 0.9, 100.0, 50.0)],
)
def test_recovers_known_transform(angle, scale, tx, ty):
    mov, ref = _pair(angle, scale, tx, ty)
    r = register(mov, ref)
    assert abs(r.angle_cw_deg - angle) < 0.05
    assert r.scale == pytest.approx(scale, rel=1e-3)
    assert r.translation_px == pytest.approx((tx, ty), abs=0.5)
    assert r.inliers <= r.matches
    assert r.residual_rms_px < 1.0


def test_sign_convention_clockwise_positive():
    mov, ref = _pair(25.0, 1.0, 0.0, 0.0)
    assert register(mov, ref).angle_cw_deg > 0


def test_original_pixel_conversion_with_scale_factors():
    # Original moving is 2x the working one, reference is 4x: f_m=0.5, f_r=0.25.
    moving = _texture()
    ref = cv2.warpAffine(moving, to_matrix(10.0, 1.2, 40.0, 20.0), (600, 600))
    mov_p, ref_p = _prep(moving, 0.5), _prep(ref, 0.25)
    r = register(mov_p, ref_p)
    assert r.angle_cw_deg == pytest.approx(10.0, abs=0.05)
    assert r.scale == pytest.approx(1.2 * 0.5 / 0.25, rel=1e-3)
    assert r.translation_px == pytest.approx((40.0 / 0.25, 20.0 / 0.25), abs=2.0)
    assert r.matrix_work[0, 2] == pytest.approx(40.0, abs=0.5)


def test_fit_similarity_exact():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 100, (20, 2))
    m = to_matrix(33.0, 1.3, 5.0, -7.0)
    dst = src @ m[:, :2].T + m[:, 2]
    assert fit_similarity(src, dst) == pytest.approx(m, abs=1e-9)


def test_blank_images_raise():
    blank = _prep(np.zeros((200, 200), np.uint8))
    with pytest.raises(RegistrationError):
        register(blank, blank)


def test_progress_events_order_and_identical_result():
    from rotest.uncertainty import angle_ci95

    moving, ref = _pair(12.0, 1.0, 10.0, 5.0)
    events: list[tuple[str, float]] = []
    result = register(moving, ref, progress=lambda s, f: events.append((s, f)))
    ci = angle_ci95(result, n_boot=200, progress=lambda s, f: events.append((s, f)))

    stages = ["detect_moving", "detect_reference", "match", "ransac", "refine", "bootstrap"]
    seen = list(dict.fromkeys(s for s, _ in events))
    assert seen == stages
    for stage in stages:
        fractions = [f for s, f in events if s == stage]
        assert fractions[0] == 0.0 and fractions[-1] == 1.0
        assert fractions == sorted(fractions)
        assert all(0.0 <= f <= 1.0 for f in fractions)
    assert len([f for s, f in events if s == "bootstrap"]) > 2

    plain = register(moving, ref)
    assert plain.angle_cw_deg == result.angle_cw_deg
    assert angle_ci95(plain, n_boot=200) == ci


def test_register_deterministic_in_process():
    mov, ref = _pair(17.0, 1.1, 30.0, -20.0)
    a = register(mov, ref)
    cv2.setRNGSeed(123)
    cv2.randu(np.empty((8, 1), np.float32), 0, 1)
    b = register(mov, ref)
    assert (a.matches, a.inliers, a.angle_cw_deg) == (b.matches, b.inliers, b.angle_cw_deg)
    assert np.array_equal(a.matrix_work, b.matrix_work)
    assert np.array_equal(a.src_pts, b.src_pts)


def test_match_beyond_bfmatcher_limit():
    rng = np.random.default_rng(0)
    des_m = rng.random((300, 128), dtype=np.float32)
    des_r = rng.random((270_000, 128), dtype=np.float32)  # > 262 143, the BFMatcher limit
    assert isinstance(_match(des_m, des_r), list)


def test_match_ratio_rule():
    rng = np.random.default_rng(1)
    des_r = (rng.random((20, 128)) * 100).astype(np.float32)
    des_m = des_r[:1] + np.float32(0.01)  # almost identical to reference 0, far from the rest
    good = _match(des_m, des_r)
    assert [(m.queryIdx, m.trainIdx) for m in good] == [(0, 0)]

    # two equidistant candidates: ratio is 1.0, so the pair is rejected
    des_r2 = np.zeros((20, 128), np.float32)
    des_r2[1:] = 100.0
    des_r2[0, 0] = 1.0
    des_r2[1] = 0.0
    des_r2[1, 0] = -1.0
    des_m2 = np.zeros((1, 128), np.float32)
    assert _match(des_m2, des_r2) == []
