"""CLI tests with synthetic images written to tmp_path (no network)."""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from rotest.cli import main


def _textured(seed: int = 3, size: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 128, np.uint8)
    for _ in range(400):
        center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.circle(img, center, int(rng.integers(3, 15)), int(rng.integers(0, 256)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    ref = _textured()
    size = ref.shape[0]
    rot = cv2.getRotationMatrix2D((size / 2, size / 2), 10, 1.0)
    moving = cv2.warpAffine(ref, rot, (size, size), borderValue=128)
    ref_path, mov_path = tmp_path / "ref.png", tmp_path / "mov.png"
    cv2.imwrite(str(ref_path), ref)
    cv2.imwrite(str(mov_path), moving)
    return mov_path, ref_path


def test_compare_writes_all_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mov, ref = _write_pair(tmp_path)
    mask = tmp_path / "mask.png"
    cv2.imwrite(str(mask), np.full((600, 600), 255, np.uint8))
    out = tmp_path / "out"
    code = main(["compare", str(mov), str(ref), "--mask-reference", str(mask), "--out", str(out)])
    assert code == 0
    for name in ("results.json", "overlay.png", "checkerboard.png", "summary.md"):
        assert (out / name).is_file()
    assert "angle:" in capsys.readouterr().out


def test_featureless_images_exit_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    flat = tmp_path / "flat.png"
    cv2.imwrite(str(flat), np.full((200, 200), 100, np.uint8))
    code = main(["compare", str(flat), str(flat), "--out", str(tmp_path / "out")])
    assert code != 0
    assert "error:" in capsys.readouterr().err


def test_missing_file_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["compare", str(tmp_path / "nope.png"), str(tmp_path / "nope2.png")])
    assert code != 0
    assert "cannot read image" in capsys.readouterr().err


def test_help_mentions_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--mask-reference" in out and "--out" in out


def test_compare_output_format_and_json_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Safety net: stdout format and results.json keys are a contract."""
    mov, ref = _write_pair(tmp_path)
    out = tmp_path / "out"
    code = main(
        [
            "compare", str(mov), str(ref),
            "--mask-moving", str(_write_mask(tmp_path)),
            "--work-size", "512",
            "--out", str(out),
        ]
    )
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    num = r"-?\d+\.\d+"
    assert re.fullmatch(
        rf"angle: {num} deg clockwise \(95% CI {num} to {num}\), scale: {num}", lines[0]
    )
    assert re.fullmatch(
        rf"angle uncertainty: \+-{num} deg \(bootstrap {num} \+ spatial ({num}|n/a)\)", lines[1]
    )
    assert lines[-1] == f"outputs written to {out}"
    for name in ("results.json", "overlay.png", "checkerboard.png", "summary.md"):
        assert (out / name).is_file()
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert set(data) == {
        "schema_version", "angle_cw_deg", "angle_ci95", "scale", "translation_px",
        "inliers", "matches", "residual_rms_px", "warnings",
        "angle_uncertainty_deg", "spatial_check",
    }
    assert data["schema_version"] == 2


def _write_mask(tmp_path: Path) -> Path:
    path = tmp_path / "mask_mov.png"
    cv2.imwrite(str(path), np.full((600, 600), 255, np.uint8))
    return path


def test_threads_zero_exits_2(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare", "a.png", "b.png", "--threads", "0"])
    assert exc.value.code == 2
    assert "--threads" in capsys.readouterr().err


def test_set_threads_validates_and_default_is_half_of_cores() -> None:
    import os

    import cv2

    from rotest.register import default_threads, set_threads

    assert default_threads() == max(1, (os.cpu_count() or 2) // 2)
    before = cv2.getNumThreads()
    try:
        with pytest.raises(ValueError):
            set_threads(0)
        set_threads(2)
        assert cv2.getNumThreads() == 2
    finally:
        cv2.setNumThreads(before)


def test_angle_identical_across_thread_counts() -> None:
    import cv2

    from rotest.register import register, set_threads
    from synthetic import make_pair

    mov, ref, _ = make_pair(12.0, 1.0)
    before = cv2.getNumThreads()
    try:
        angles = []
        for n in (1, 2, 4):
            set_threads(n)
            angles.append(register(mov, ref).angle_cw_deg)
    finally:
        cv2.setNumThreads(before)
    assert angles[1] == pytest.approx(angles[0], abs=1e-9)
    assert angles[2] == pytest.approx(angles[0], abs=1e-9)


def test_ui_without_extra_fails_with_clear_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert main(["ui"]) == 1
    assert capsys.readouterr().err.strip() == (
        "error: the UI needs the optional 'ui' extra; install it with: uv sync --extra ui"
    )


def test_ui_launches_streamlit_with_forced_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import sys

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("subprocess.run", fake_run)
    assert main(["ui", "--port", "8600"]) == 0
    cmd = calls[0]
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert cmd[4].endswith("ui.py")
    flags = cmd[5:]
    for pair in (
        ["--server.address", "localhost"],
        ["--browser.gatherUsageStats", "false"],
        ["--server.maxUploadSize", "2048"],
        ["--server.showEmailPrompt", "false"],
        ["--server.port", "8600"],
    ):
        i = flags.index(pair[0])
        assert flags[i : i + 2] == pair
