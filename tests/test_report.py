"""Report output tests on synthetic data, writing to tmp_path (no network)."""

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from rotest.geometry import to_matrix
from rotest.io import Prepared
from rotest.register import Result
from rotest.report import quality_warnings, write_report


def _prepared(seed: int, shape: tuple[int, int] = (200, 300)) -> Prepared:
    img = np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)
    return Prepared(img, None, 1.0, (shape[1], shape[0]))


def _result(**overrides: object) -> Result:
    base = Result(
        angle_cw_deg=6.21,
        scale=1.004,
        translation_px=(12.5, -3.1),
        inliers=612,
        matches=905,
        residual_rms_px=1.8,
        matrix_work=to_matrix(6.21, 1.004, 12.5, -3.1),
        src_pts=np.zeros((612, 2)),
        dst_pts=np.zeros((612, 2)),
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_all_outputs_and_schema(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "out"
    moving, reference = _prepared(1), _prepared(2)
    warnings = write_report(out, moving, reference, _result(), (6.12, 6.30))
    assert warnings == []
    for name in ("results.json", "overlay.png", "checkerboard.png", "summary.md"):
        assert (out / name).is_file()

    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data == {
        "schema_version": 1,
        "angle_cw_deg": 6.21,
        "angle_ci95": [6.12, 6.30],
        "scale": 1.004,
        "translation_px": [12.5, -3.1],
        "inliers": 612,
        "matches": 905,
        "residual_rms_px": 1.8,
        "warnings": [],
    }
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "[6.120, 6.300]" in summary
    assert "None." in summary


def test_images_are_gray_at_reference_size(tmp_path: Path) -> None:
    moving, reference = _prepared(1, (100, 150)), _prepared(2, (200, 300))
    write_report(tmp_path, moving, reference, _result(), (6.0, 7.0))
    for name in ("overlay.png", "checkerboard.png"):
        img = cv2.imread(str(tmp_path / name), cv2.IMREAD_UNCHANGED)
        assert img.shape == (200, 300)


def test_identity_overlay_and_checkerboard(tmp_path: Path) -> None:
    ref = _prepared(3)
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    write_report(tmp_path, ref, ref, _result(matrix_work=eye), (0.0, 0.0))
    overlay = cv2.imread(str(tmp_path / "overlay.png"), cv2.IMREAD_UNCHANGED)
    board = cv2.imread(str(tmp_path / "checkerboard.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(overlay, ref.image)
    assert np.array_equal(board, ref.image)


def test_checkerboard_alternates(tmp_path: Path) -> None:
    ref = Prepared(np.full((128, 128), 200, np.uint8), None, 1.0, (128, 128))
    mov = Prepared(np.full((128, 128), 50, np.uint8), None, 1.0, (128, 128))
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    write_report(tmp_path, mov, ref, _result(matrix_work=eye), (0.0, 0.0))
    board = cv2.imread(str(tmp_path / "checkerboard.png"), cv2.IMREAD_UNCHANGED)
    assert board[0, 0] == 200 and board[0, 64] == 50 and board[64, 0] == 50 and board[64, 64] == 200


def test_warnings_appear_when_due(tmp_path: Path) -> None:
    assert quality_warnings(_result()) == []
    assert len(quality_warnings(_result(inliers=29))) == 1
    assert len(quality_warnings(_result(residual_rms_px=3.5))) == 1
    assert len(quality_warnings(_result(scale=2.5))) == 1
    assert len(quality_warnings(_result(scale=0.4))) == 1
    assert len(quality_warnings(_result(inliers=5, residual_rms_px=9.0, scale=3.0))) == 3

    bad = _result(inliers=10)
    write_report(tmp_path, _prepared(1), _prepared(2), bad, (1.0, 2.0))
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(data["warnings"]) == 1
    assert data["warnings"][0] in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_ci_not_wrapped(tmp_path: Path) -> None:
    write_report(tmp_path, _prepared(1), _prepared(2), _result(angle_cw_deg=179.5), (178.9, 181.2))
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert data["angle_ci95"] == [178.9, 181.2]
    assert "[178.900, 181.200]" in (tmp_path / "summary.md").read_text(encoding="utf-8")
