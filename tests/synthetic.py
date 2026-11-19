"""Synthetic image-pair generator shared by the accuracy tests (no files, fixed seeds)."""

from __future__ import annotations

import cv2
import numpy as np

from rotest.geometry import to_matrix
from rotest.io import Prepared, apply_clahe


def texture(size: int = 500, seed: int = 0) -> np.ndarray:
    """Blobby texture with plenty of distinct SIFT keypoints."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 90, np.uint8)
    for _ in range(220):
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.circle(img, c, int(rng.integers(3, 18)), int(rng.integers(30, 255)), -1)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def prepare(img: np.ndarray) -> Prepared:
    h, w = img.shape
    return Prepared(apply_clahe(img), None, 1.0, (w, h))


def make_pair(
    angle_cw_deg: float,
    scale: float,
    noise_sigma: float = 0.0,
    crop: int = 0,
    seed: int = 0,
) -> tuple[Prepared, Prepared, tuple[float, float, float, float]]:
    """Return (moving, reference, truth) with truth = (angle, scale, tx, ty) moving -> reference.

    The reference is the moving image warped about its centre by the known transform. `crop`
    removes that many pixels from the left/top of the reference (partial overlap, which also
    shifts the translation). Noise is Gaussian, applied to both images.
    """
    moving = texture(seed=seed)
    size = moving.shape[0]
    c = (size - 1) / 2
    rot = to_matrix(angle_cw_deg, scale, 0.0, 0.0)
    tx = c - (rot[0, 0] * c + rot[0, 1] * c) - crop
    ty = c - (rot[1, 0] * c + rot[1, 1] * c) - crop
    ref = cv2.warpAffine(moving, to_matrix(angle_cw_deg, scale, tx, ty), (size, size))
    ref = ref[: size - crop, : size - crop] if crop else ref
    if noise_sigma:
        rng = np.random.default_rng(seed + 1)
        moving = _noisy(moving, noise_sigma, rng)
        ref = _noisy(ref, noise_sigma, rng)
    return prepare(moving), prepare(np.ascontiguousarray(ref)), (angle_cw_deg, scale, tx, ty)


def _noisy(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float64) + rng.normal(0.0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)
