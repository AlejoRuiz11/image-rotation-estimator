"""Similarity transform <-> 2x3 matrix, following the angle convention in the README.

Image coordinates: origin top-left, x right, y down. A positive angle is a clockwise rotation as
seen on screen. Because y points down, that is the usual matrix [[c, -s], [s, c]].
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def normalize_angle(angle_deg: float) -> float:
    """Wrap an angle in degrees to the range (-180, 180]."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def to_matrix(angle_cw_deg: float, scale: float, tx: float, ty: float) -> NDArray[np.float64]:
    """Build the 2x3 matrix mapping moving-image points to reference-image points.

    The point is rotated clockwise by ``angle_cw_deg`` about the origin, scaled, then
    translated by ``(tx, ty)``.
    """
    theta = math.radians(angle_cw_deg)
    a = scale * math.cos(theta)
    b = scale * math.sin(theta)
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def from_matrix(matrix: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Inverse of :func:`to_matrix`: return ``(angle_cw_deg, scale, tx, ty)``."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3):
        raise ValueError(f"expected a 2x3 matrix, got shape {m.shape}")
    a, b = m[0, 0], m[1, 0]
    scale = math.hypot(a, b)
    angle = normalize_angle(math.degrees(math.atan2(b, a)))
    return angle, scale, float(m[0, 2]), float(m[1, 2])
