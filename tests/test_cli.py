"""CLI tests with synthetic images written to tmp_path (no network)."""

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
