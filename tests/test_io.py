"""Tests for rotest.io using only synthetic images written to tmp_path."""

import cv2
import numpy as np
import pytest

from rotest.io import apply_clahe, compute_scale_factor, prepare


def _write(path, img):
    assert cv2.imwrite(str(path), img)


def test_scale_factor_and_no_upscale():
    assert compute_scale_factor((12000, 10000), 4000) == pytest.approx(1 / 3)
    assert compute_scale_factor((500, 300), 4000) == 1.0


def test_small_image_unchanged_size(tmp_path):
    rng = np.random.default_rng(0)
    _write(tmp_path / "a.png", rng.integers(0, 256, (120, 200, 3), dtype=np.uint8))
    p = prepare(tmp_path / "a.png")
    assert p.image.shape == (120, 200)
    assert p.scale_factor == 1.0
    assert p.original_size == (200, 120)
    assert p.mask is None


def test_huge_image_is_reduced_with_correct_factor(tmp_path):
    # 10 000 x 12 000 (w x h) gradient; compresses well so the file stays small.
    h, w = 12000, 10000
    row = (np.arange(w) % 256).astype(np.uint8)
    img = np.tile(row, (h, 1))
    _write(tmp_path / "big.png", img)
    del img
    p = prepare(tmp_path / "big.png", work_size=4000)
    assert p.original_size == (w, h)
    assert p.scale_factor == pytest.approx(4000 / 12000)
    assert p.image.shape == (4000, round(w * 4000 / 12000))
    assert p.image.dtype == np.uint8


def test_mask_white_use_black_ignore_and_resized(tmp_path):
    img = np.full((400, 200), 128, np.uint8)
    mask = np.zeros((400, 200), np.uint8)
    mask[:, 100:] = 255
    _write(tmp_path / "i.png", img)
    _write(tmp_path / "m.png", mask)
    p = prepare(tmp_path / "i.png", tmp_path / "m.png", work_size=100)
    assert p.mask is not None and p.mask.shape == p.image.shape == (100, 50)
    assert set(np.unique(p.mask)) <= {0, 255}
    assert p.mask[:, :20].max() == 0 and p.mask[:, 30:].min() == 255


def test_mask_shape_mismatch_raises(tmp_path):
    _write(tmp_path / "i.png", np.zeros((50, 50), np.uint8))
    _write(tmp_path / "m.png", np.zeros((40, 50), np.uint8))
    with pytest.raises(ValueError, match="mask shape"):
        prepare(tmp_path / "i.png", tmp_path / "m.png")


def test_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        prepare(tmp_path / "nope.png")


def test_clahe_increases_contrast():
    rng = np.random.default_rng(0)
    low = rng.integers(100, 120, (128, 128), dtype=np.uint8)
    assert apply_clahe(low).std() > low.std()
