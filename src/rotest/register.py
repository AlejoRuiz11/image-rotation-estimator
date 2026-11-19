"""Feature-based similarity registration (SIFT + ratio test + RANSAC + least squares)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from rotest.geometry import from_matrix
from rotest.io import Prepared

RATIO = 0.75
MIN_INLIERS = 3  # a similarity has 4 DOF; 2 point pairs are enough, 3 gives a sanity margin


class RegistrationError(ValueError):
    """Raised when there are not enough matches to estimate a transform."""


@dataclass(frozen=True)
class Result:
    """Registration output, reported in original pixels except where noted.

    `matrix_work` maps moving working pixels to reference working pixels (used to warp for
    visuals). `src_pts` / `dst_pts` are the inlier pairs in working pixels (used by the
    bootstrap). `residual_rms_px` is in reference working pixels.
    """

    angle_cw_deg: float
    scale: float
    translation_px: tuple[float, float]
    inliers: int
    matches: int
    residual_rms_px: float
    matrix_work: NDArray[np.float64]
    src_pts: NDArray[np.float64]
    dst_pts: NDArray[np.float64]


def fit_similarity(
    src: NDArray[np.float64], dst: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Least-squares similarity (Umeyama) mapping `src` points to `dst`, as a 2x3 matrix."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    s = src - src_c
    d = dst - dst_c
    denom = float((s**2).sum())
    if denom == 0.0:
        raise RegistrationError("degenerate point set (all source points coincide)")
    # Complex form: dst = z * src, with z = s*(cos t + i sin t) (y-down matrix [[a,-b],[b,a]]).
    sz = s[:, 0] + 1j * s[:, 1]
    dz = d[:, 0] + 1j * d[:, 1]
    z = (np.conj(sz) * dz).sum() / denom
    a, b = z.real, z.imag
    t = dst_c - np.array([[a, -b], [b, a]]) @ src_c
    return np.array([[a, -b, t[0]], [b, a, t[1]]], dtype=np.float64)


def _detect(img: NDArray[np.uint8], mask: NDArray[np.uint8] | None):
    return cv2.SIFT_create().detectAndCompute(img, mask)


def register(moving: Prepared, reference: Prepared) -> Result:
    """Estimate the similarity moving -> reference and convert it to original pixels."""
    kp_m, des_m = _detect(moving.image, moving.mask)
    kp_r, des_r = _detect(reference.image, reference.mask)
    if des_m is None or des_r is None or len(kp_m) < 2 or len(kp_r) < 2:
        raise RegistrationError("not enough features detected")

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_m, des_r, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        raise RegistrationError(f"only {len(good)} matches after ratio test")

    src = np.float32([kp_m[m.queryIdx].pt for m in good])
    dst = np.float32([kp_r[m.trainIdx].pt for m in good])
    # Moving points are the source, so the matrix maps moving -> reference.
    _, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if inlier_mask is None:
        raise RegistrationError("RANSAC found no similarity")
    keep = inlier_mask.ravel().astype(bool)
    if keep.sum() < MIN_INLIERS:
        raise RegistrationError(f"only {int(keep.sum())} inliers")

    src_in = src[keep].astype(np.float64)
    dst_in = dst[keep].astype(np.float64)
    matrix = fit_similarity(src_in, dst_in)
    proj = src_in @ matrix[:, :2].T + matrix[:, 2]
    rms = float(np.sqrt(((proj - dst_in) ** 2).sum(axis=1).mean()))

    angle, s_w, tx_w, ty_w = from_matrix(matrix)
    f_m, f_r = moving.scale_factor, reference.scale_factor
    return Result(
        angle_cw_deg=angle,
        scale=s_w * f_m / f_r,
        translation_px=(tx_w / f_r, ty_w / f_r),
        inliers=int(keep.sum()),
        matches=len(good),
        residual_rms_px=rms,
        matrix_work=matrix,
        src_pts=src_in,
        dst_pts=dst_in,
    )
