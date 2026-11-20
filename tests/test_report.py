"""Report output tests on synthetic data, writing to tmp_path (no network)."""

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from rotest.geometry import to_matrix
from rotest.io import Prepared
from rotest.register import Result
from rotest.report import quality_warnings, write_report
from rotest.uncertainty import Region, SpatialCheck, spatial_check


def _prepared(seed: int, shape: tuple[int, int] = (200, 300)) -> Prepared:
    img = np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)
    return Prepared(img, None, 1.0, (shape[1], shape[0]))


def _result(**overrides: object) -> Result:
    xs, ys = np.meshgrid(np.arange(24) * 10.0, np.arange(25) * 10.0)
    src = np.column_stack([xs.ravel(), ys.ravel()])
    m = to_matrix(6.21, 1.004, 12.5, -3.1)
    dst = src @ m[:, :2].T + m[:, 2]
    base = Result(
        angle_cw_deg=6.21,
        scale=1.004,
        translation_px=(12.5, -3.1),
        inliers=612,
        matches=905,
        residual_rms_px=1.8,
        matrix_work=m,
        src_pts=src,
        dst_pts=dst,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _spatial(dev: float | None = 0.004) -> SpatialCheck:
    names = ("top_left", "top_right", "bottom_left", "bottom_right")
    regions = tuple(
        Region(n, 150, 6.2 + i / 100, (6.1 + i / 100, 6.3 + i / 100), 1.004)
        for i, n in enumerate(names)
    )
    return SpatialCheck(regions, dev)


def test_all_outputs_and_schema(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "out"
    moving, reference = _prepared(1), _prepared(2)
    spatial = _spatial()
    warnings = write_report(out, moving, reference, _result(), (6.12, 6.30), spatial)
    assert warnings == []
    for name in ("results.json", "overlay.png", "checkerboard.png", "summary.md"):
        assert (out / name).is_file()

    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert list(data) == [
        "schema_version", "angle_cw_deg", "angle_ci95", "scale", "translation_px", "inliers",
        "matches", "residual_rms_px", "warnings", "angle_uncertainty_deg", "spatial_check",
    ]
    assert data.pop("angle_uncertainty_deg") == pytest.approx(0.094)
    assert data == {
        "schema_version": 2,
        "angle_cw_deg": 6.21,
        "angle_ci95": [6.12, 6.30],
        "scale": 1.004,
        "translation_px": [12.5, -3.1],
        "inliers": 612,
        "matches": 905,
        "residual_rms_px": 1.8,
        "warnings": [],
        "spatial_check": {
            "angle_dev_deg": 0.004,
            "regions": [
                {
                    "name": r.name,
                    "inliers": 150,
                    "angle_cw_deg": r.angle_cw_deg,
                    "angle_ci95": list(r.angle_ci95),
                    "scale": 1.004,
                }
                for r in spatial.regions
            ],
        },
    }
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "[6.120, 6.300]" in summary
    assert "None." in summary
    assert "## Spatial consistency" in summary
    assert summary.index("## Result") < summary.index("## Spatial consistency")
    assert "| Region | Inliers | Angle (deg) | 95% interval (deg) | Scale |" in summary
    assert "Spatial deviation beyond noise (deg): 0.004" in summary
    assert "Total angle uncertainty (deg): +-0.094 (bootstrap half-width + spatial deviation)" in summary


def test_images_are_gray_at_reference_size(tmp_path: Path) -> None:
    moving, reference = _prepared(1, (100, 150)), _prepared(2, (200, 300))
    write_report(tmp_path, moving, reference, _result(), (6.0, 7.0), _spatial())
    for name in ("overlay.png", "checkerboard.png"):
        img = cv2.imread(str(tmp_path / name), cv2.IMREAD_UNCHANGED)
        assert img.shape == (200, 300)


def test_identity_overlay_and_checkerboard(tmp_path: Path) -> None:
    ref = _prepared(3)
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    write_report(tmp_path, ref, ref, _result(matrix_work=eye), (0.0, 0.0), _spatial())
    overlay = cv2.imread(str(tmp_path / "overlay.png"), cv2.IMREAD_UNCHANGED)
    board = cv2.imread(str(tmp_path / "checkerboard.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(overlay, ref.image)
    assert np.array_equal(board, ref.image)


def test_checkerboard_alternates(tmp_path: Path) -> None:
    ref = Prepared(np.full((128, 128), 200, np.uint8), None, 1.0, (128, 128))
    mov = Prepared(np.full((128, 128), 50, np.uint8), None, 1.0, (128, 128))
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    write_report(tmp_path, mov, ref, _result(matrix_work=eye), (0.0, 0.0), _spatial())
    board = cv2.imread(str(tmp_path / "checkerboard.png"), cv2.IMREAD_UNCHANGED)
    assert board[0, 0] == 200 and board[0, 64] == 50 and board[64, 0] == 50 and board[64, 64] == 200


def test_warnings_appear_when_due(tmp_path: Path) -> None:
    rigid = spatial_check(_result(), n_boot=30)
    assert rigid.angle_dev_deg is not None
    assert quality_warnings(_result(), rigid) == []
    assert len(quality_warnings(_result(inliers=29), rigid)) == 1
    assert len(quality_warnings(_result(residual_rms_px=3.5), rigid)) == 1
    assert len(quality_warnings(_result(scale=2.5), rigid)) == 1
    assert len(quality_warnings(_result(scale=0.4), rigid)) == 1
    assert len(quality_warnings(_result(inliers=5, residual_rms_px=9.0, scale=3.0), rigid)) == 3

    bad = _result(inliers=10)
    write_report(tmp_path, _prepared(1), _prepared(2), bad, (1.0, 2.0), _spatial())
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(data["warnings"]) == 1
    assert data["warnings"][0] in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_ci_not_wrapped(tmp_path: Path) -> None:
    write_report(tmp_path, _prepared(1), _prepared(2), _result(angle_cw_deg=179.5), (178.9, 181.2), _spatial())
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert data["angle_ci95"] == [178.9, 181.2]
    assert "[178.900, 181.200]" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_spatial_warnings() -> None:
    assert quality_warnings(_result(), _spatial(dev=0.06)) == [
        ("spatial inconsistency: region angles deviate up to 0.060 deg beyond their noise "
         "(> 0.05 deg); possible non-rigid distortion")
    ]
    assert quality_warnings(_result(), _spatial(dev=0.05)) == []
    assert quality_warnings(_result(), _spatial(dev=None)) == [
        ("spatial check skipped: fewer than 2 regions with >= 30 inliers; "
         "systematic error not estimated")
    ]


def test_unfitted_region_is_null(tmp_path: Path) -> None:
    spatial = SpatialCheck((Region("top_left", 5, None, None, None),), None)
    write_report(tmp_path, _prepared(1), _prepared(2), _result(), (6.0, 7.0), spatial)
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert data["spatial_check"]["angle_dev_deg"] is None
    assert data["spatial_check"]["regions"][0] == {
        "name": "top_left", "inliers": 5, "angle_cw_deg": None, "angle_ci95": None, "scale": None
    }
    assert data["angle_uncertainty_deg"] == pytest.approx(0.5)
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| top_left | 5 | n/a | n/a | n/a |" in summary
