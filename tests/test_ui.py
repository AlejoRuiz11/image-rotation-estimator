# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fabian Alejandro Ruiz Osorio (AlejoRuiz11)
"""UI tests: pure helpers and the Streamlit app via AppTest (no browser, no network)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("streamlit")

from rotest.cli import main
from rotest.geometry import to_matrix
from rotest.ui import (
    MAX_DRAWN_MATCHES,
    STAGE_LABELS,
    STAGES,
    VIEW_MAX_SIDE,
    Views,
    blend,
    build_views,
    checkerboard_view,
    draw_matches,
    histogram_figure,
    overall_fraction,
    read_outputs,
    residual_figure,
    run_pipeline,
    select_matches,
    shrink_view,
    sign_test,
    split_view,
)


def _pair(tmp_path: Path, rotate_deg: float = 10.0) -> tuple[Path, Path]:
    rng = np.random.default_rng(3)
    size = 500
    ref = np.full((size, size), 128, np.uint8)
    for _ in range(300):
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.circle(ref, c, int(rng.integers(3, 15)), int(rng.integers(0, 256)), -1)
    ref = cv2.GaussianBlur(ref, (3, 3), 0)
    rot = cv2.getRotationMatrix2D((size / 2, size / 2), rotate_deg, 1.0)
    mov = cv2.warpAffine(ref, rot, (size, size), borderValue=128)
    ref_path, mov_path = tmp_path / "ref.png", tmp_path / f"mov{rotate_deg}.png"
    cv2.imwrite(str(ref_path), ref)
    cv2.imwrite(str(mov_path), mov)
    return mov_path, ref_path


def test_overall_fraction_by_hand() -> None:
    assert overall_fraction("load", 0.0) == 0.0
    assert overall_fraction("report", 1.0) == 1.0
    assert overall_fraction("match", 0.5) == pytest.approx(3.5 / 9)
    assert set(STAGES) == set(STAGE_LABELS) and len(STAGES) == 9
    assert STAGE_LABELS["spatial"] == "Checking spatial consistency"


def test_run_pipeline_matches_cli_and_orders_events(tmp_path: Path) -> None:
    mov, ref = _pair(tmp_path)
    events: list[tuple[str, float]] = []
    out_ui, out_cli = tmp_path / "ui", tmp_path / "cli"
    run = run_pipeline(mov, ref, out_ui, work_size=512, progress=lambda s, f: events.append((s, f)))
    assert main(["compare", str(mov), str(ref), "--work-size", "512", "--out", str(out_cli)]) == 0
    for name in ("results.json", "overlay.png", "checkerboard.png", "summary.md"):
        assert (out_ui / name).is_file()
    assert json.loads((out_ui / "results.json").read_text()) == json.loads(
        (out_cli / "results.json").read_text()
    )
    order = [STAGES.index(s) for s, _ in events]
    assert order == sorted(order)
    assert events[0] == ("load", 0.0) and events[-1] == ("report", 1.0)
    assert {s for s, _ in events} == set(STAGES)
    assert run.uncertainty > 0


def _rotate_cw(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate about the image center with the project convention (clockwise positive)."""
    c = (img.shape[0] - 1) / 2
    rot = to_matrix(deg, 1.0, 0.0, 0.0)
    tx = c - (rot[0, 0] * c + rot[0, 1] * c)
    ty = c - (rot[1, 0] * c + rot[1, 1] * c)
    return cv2.warpAffine(img, to_matrix(deg, 1.0, tx, ty), img.shape[::-1], borderValue=128)


def test_synthetic_sign_test_passes_and_inverted_fails(tmp_path: Path) -> None:
    mov_path, ref_path = _pair(tmp_path, 0.0)
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    # mov = ref rotated +10 (clockwise); test image = mov rotated 10 deg counter-clockwise.
    mov = _rotate_cw(ref, 10.0)
    rotated = _rotate_cw(mov, -10.0)
    mov_path, rot_path = tmp_path / "m.png", tmp_path / "r.png"
    cv2.imwrite(str(mov_path), mov)
    cv2.imwrite(str(rot_path), rotated)
    base = run_pipeline(mov_path, ref_path, work_size=512)
    rot = run_pipeline(rot_path, ref_path, work_size=512)
    assert not (tmp_path / "results.json").exists()  # out_dir=None writes nothing
    args = (base.result.angle_cw_deg, rot.result.angle_cw_deg)
    u = (base.uncertainty, rot.uncertainty)
    good = sign_test(*args, 10.0, *u)
    assert good.difference_deg == pytest.approx(10.0, abs=0.5)
    assert good.passed
    assert not sign_test(*args, -10.0, *u).passed


def test_sign_test_numbers_by_hand() -> None:
    s = sign_test(0.1766, 10.0663, 10.0, 0.0423, 0.0436)
    assert s.passed and s.tolerance_deg == 0.2 and s.tolerance_source == "floor"
    assert s.difference_deg == pytest.approx(9.8897)
    s = sign_test(0.0, 10.0, 10.0, 0.15, 0.10)
    assert s.tolerance_deg == pytest.approx(0.25) and s.tolerance_source == "U1 + U2"
    assert not sign_test(0.0, 10.5, 10.0, 0.05, 0.05).passed
    wrapped = sign_test(175.0, -175.0, 10.0, 0.0, 0.0)
    assert wrapped.difference_deg == pytest.approx(10.0) and wrapped.passed


def test_app_loads_with_uploaders_and_button() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).parents[1] / "src" / "rotest" / "ui.py"))
    at.run(timeout=30)
    assert not at.exception
    assert [b.label for b in at.button] == ["Run"]
    assert at.number_input[0].label == "Expected angle change (deg)"
    assert at.number_input[0].value == 10.0


def _views_run(tmp_path: Path, work_size: int = 512):
    mov, ref = _pair(tmp_path)
    run = run_pipeline(mov, ref, None, work_size=work_size)
    return run, build_views(run)


def test_shrink_view_limits_longest_side() -> None:
    big = np.zeros((900, 2000), np.uint8)
    small, factor = shrink_view(big, 1600)
    assert max(small.shape) == 1600 and factor == pytest.approx(0.8)
    same, one = shrink_view(np.zeros((10, 20), np.uint8))
    assert same.shape == (10, 20) and one == 1.0


def test_select_matches_caps_and_is_deterministic() -> None:
    idx = select_matches(1000)
    assert len(idx) == MAX_DRAWN_MATCHES == 300
    assert np.array_equal(idx, select_matches(1000))
    assert np.array_equal(select_matches(50), np.arange(50))


def test_draw_matches_size_within_view_limit(tmp_path: Path) -> None:
    run, views = _views_run(tmp_path)
    canvas = draw_matches(views, run.result)
    assert canvas.ndim == 3 and max(canvas.shape[:2]) <= VIEW_MAX_SIDE
    big = Views(
        np.zeros((1600, 1600), np.uint8), np.zeros((1600, 1600), np.uint8),
        np.zeros((1600, 1600), np.uint8), 1.0, 1.0,
    )  # fmt: skip
    assert max(draw_matches(big, run.result).shape[:2]) <= VIEW_MAX_SIDE


def test_blend_and_split_ends_equal_the_images() -> None:
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (40, 60), dtype=np.uint8)
    aligned = rng.integers(0, 256, (40, 60), dtype=np.uint8)
    assert np.array_equal(blend(ref, aligned, 0.0), ref)
    assert np.array_equal(blend(ref, aligned, 1.0), aligned)
    assert np.array_equal(split_view(ref, aligned, 1.0), ref)
    assert np.array_equal(split_view(ref, aligned, 0.0), aligned)
    half = split_view(ref, aligned, 0.5)
    assert np.array_equal(half[:, :30], ref[:, :30]) and np.array_equal(half[:, 30:], aligned[:, 30:])


def test_views_and_figures(tmp_path: Path) -> None:
    run, views = _views_run(tmp_path)
    assert views.aligned.shape == views.reference.shape
    assert checkerboard_view(views).shape == views.reference.shape
    assert run.boot_angles.shape == (1000,)
    assert histogram_figure(run.boot_angles, run.ci, run.result.angle_cw_deg).axes
    assert residual_figure(views, run.result).axes


def test_read_outputs_returns_the_four_files(tmp_path: Path) -> None:
    mov, ref = _pair(tmp_path)
    run_pipeline(mov, ref, tmp_path / "out", work_size=512)
    assert set(read_outputs(tmp_path / "out")) == {
        "results.json", "summary.md", "overlay.png", "checkerboard.png"
    }  # fmt: skip
