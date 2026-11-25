# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fabian Alejandro Ruiz Osorio (AlejoRuiz11)
"""Tests for scripts/diagnose_pair.py on synthetic pairs (no network, outputs in tmp dirs)."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from synthetic import make_pair, make_warped_pair

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diagnose_pair.py"
MOVING_NAME, REFERENCE_NAME = "pair_moving_xyz.png", "pair_reference_xyz.png"


def _run(moving, reference, out: Path, tmp: Path) -> tuple[subprocess.CompletedProcess, Path]:
    mov_path, ref_path = tmp / MOVING_NAME, tmp / REFERENCE_NAME
    cv2.imwrite(str(mov_path), moving.image)
    cv2.imwrite(str(ref_path), reference.image)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(mov_path), str(ref_path), "--out", str(out),
         "--threads", "2"],
        capture_output=True, text=True, timeout=600, check=False,
    )
    return proc, out


@pytest.fixture(scope="module")
def rigid(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rigid")
    mov, ref, _ = make_pair(15.0, 1.25)
    proc, out = _run(mov, ref, tmp / "out", tmp)
    return proc, out


@pytest.fixture(scope="module")
def warped(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("warped")
    mov, ref, _ = make_warped_pair(10.0, 1.0)
    proc, out = _run(mov, ref, tmp / "out", tmp)
    return proc, out


def _variants(out: Path) -> dict:
    return json.loads((out / "diagnosis.json").read_text(encoding="utf-8"))["variants"]


def test_rigid_pair_variants_agree(rigid):
    proc, out = rigid
    assert proc.returncode == 0, proc.stderr
    assert (out / "diagnosis.json").exists() and (out / "diagnosis.md").exists()
    variants = _variants(out)
    assert set(variants) == {"bf", "flann", "flann_precise"}
    angles = [v["angle_cw_deg"] for v in variants.values()]
    assert max(angles) - min(angles) <= 0.02
    for v in variants.values():
        assert v["status"] == "ok"
        assert v["angle_dev_deg"] is None or v["angle_dev_deg"] < 0.05


def test_warped_pair_flags_top_left(warped):
    proc, out = warped
    assert proc.returncode == 0, proc.stderr
    for v in _variants(out).values():
        assert v["angle_dev_deg"] > 0.1
        deltas = {k: abs(f["delta_deg"]) for k, f in v["spatial_check"].items() if f and f["delta_deg"] is not None}
        assert max(deltas, key=deltas.get) == "top_left"


def test_out_inside_repo_is_rejected(tmp_path):
    inside = REPO_ROOT / "diagnose_test_out_should_not_exist"
    mov, ref, _ = make_pair(15.0, 1.25)
    proc, _ = _run(mov, ref, inside, tmp_path)
    assert proc.returncode == 2
    assert not inside.exists()


def test_outputs_do_not_contain_input_names(rigid):
    proc, out = rigid
    for text in (proc.stdout, (out / "diagnosis.md").read_text(encoding="utf-8"),
                 (out / "diagnosis.json").read_text(encoding="utf-8")):
        assert "xyz" not in text
        assert MOVING_NAME not in text and REFERENCE_NAME not in text
