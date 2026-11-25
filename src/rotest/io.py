# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fabian Alejandro Ruiz Osorio (AlejoRuiz11)
"""Image loading and preparation: grayscale, optional mask, CLAHE, downscaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

DEFAULT_WORK_SIZE = 4000


@dataclass(frozen=True)
class Prepared:
    """An image ready for registration.

    `scale_factor` is working pixels per original pixel (<= 1). To report a
    translation in original pixels, divide it by this factor; angle is unaffected.
    """

    image: NDArray[np.uint8]  # grayscale + CLAHE, long side <= work_size
    mask: NDArray[np.uint8] | None  # 255 = use, 0 = ignore, same shape as image
    scale_factor: float
    original_size: tuple[int, int]  # (width, height)


def load_gray(path: str | Path) -> NDArray[np.uint8]:
    """Read an image straight to grayscale (avoids a 3-channel copy of huge files)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def compute_scale_factor(shape: tuple[int, int], work_size: int) -> float:
    """Factor that brings the long side to `work_size`; never upscales."""
    if work_size <= 0:
        raise ValueError("work_size must be positive")
    return min(1.0, work_size / max(shape[:2]))


def resize_by_factor(
    img: NDArray[np.uint8], factor: float, *, is_mask: bool = False
) -> NDArray[np.uint8]:
    """Resize by `factor`; INTER_AREA for images, nearest neighbour for masks."""
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, size, interpolation=interp)


def apply_clahe(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Local contrast equalisation so features survive lighting differences."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def prepare(
    path: str | Path,
    mask_path: str | Path | None = None,
    work_size: int = DEFAULT_WORK_SIZE,
) -> Prepared:
    """Load, optionally mask, downscale to `work_size` and apply CLAHE.

    Mask convention: white (>127) = use, black = ignore.
    """
    gray = load_gray(path)
    h, w = gray.shape
    factor = compute_scale_factor((h, w), work_size)
    image = apply_clahe(resize_by_factor(gray, factor))

    mask: NDArray[np.uint8] | None = None
    if mask_path is not None:
        raw = load_gray(mask_path)
        if raw.shape != gray.shape:
            raise ValueError(
                f"mask shape {raw.shape} does not match image shape {gray.shape}"
            )
        small = resize_by_factor(raw, factor, is_mask=True)
        mask = np.where(small > 127, 255, 0).astype(np.uint8)

    return Prepared(image=image, mask=mask, scale_factor=factor, original_size=(w, h))
