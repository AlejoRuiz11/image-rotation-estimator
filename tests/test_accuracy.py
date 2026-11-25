# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fabian Alejandro Ruiz Osorio (AlejoRuiz11)
"""Accuracy grid on synthetic pairs: rotation, scale, noise and partial overlap."""

from __future__ import annotations

import cv2
import pytest

from rotest.geometry import to_matrix
from rotest.register import register
from rotest.uncertainty import angle_ci95
from synthetic import make_pair, prepare, texture

ANGLE_TOL_DEG = 0.2
SCALE_TOL_REL = 0.01
TRANS_TOL_PX = 2.0

# (angle, scale, noise sigma, crop px)
CASES = [
    (-45.0, 1.0, 0.0, 0),
    (-20.0, 0.8, 0.0, 0),
    (0.0, 1.0, 0.0, 0),
    (15.0, 1.25, 0.0, 0),
    (45.0, 0.9, 0.0, 0),
    (30.0, 1.0, 5.0, 0),
    (-30.0, 1.1, 5.0, 0),
    (10.0, 1.0, 0.0, 120),
    (-35.0, 1.2, 3.0, 100),
]


@pytest.mark.parametrize("angle,scale,noise,crop", CASES)
def test_grid_accuracy(angle, scale, noise, crop):
    mov, ref, (a, s, tx, ty) = make_pair(angle, scale, noise, crop)
    r = register(mov, ref)
    assert abs(r.angle_cw_deg - a) < ANGLE_TOL_DEG
    assert r.scale == pytest.approx(s, rel=SCALE_TOL_REL)
    assert r.translation_px == pytest.approx((tx, ty), abs=TRANS_TOL_PX)


def test_sign_moving_is_reference_rotated_clockwise():
    # Moving = reference rotated +25 deg clockwise, so moving -> reference is -25 deg.
    ref_img = texture()
    c = (ref_img.shape[0] - 1) / 2
    rot = to_matrix(25.0, 1.0, 0.0, 0.0)
    tx = c - (rot[0, 0] * c + rot[0, 1] * c)
    ty = c - (rot[1, 0] * c + rot[1, 1] * c)
    mov_img = cv2.warpAffine(ref_img, to_matrix(25.0, 1.0, tx, ty), ref_img.shape[::-1])
    r = register(prepare(mov_img), prepare(ref_img))
    assert r.angle_cw_deg == pytest.approx(-25.0, abs=ANGLE_TOL_DEG)


def test_ci_contains_truth_and_is_tight():
    mov, ref, (a, _, _, _) = make_pair(20.0, 1.0, 3.0)
    r = register(mov, ref)
    low, high = angle_ci95(r, n_boot=200, seed=0)
    assert low <= high
    assert low - 0.1 <= a <= high + 0.1
    assert high - low < 1.0
