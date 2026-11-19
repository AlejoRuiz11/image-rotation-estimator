import numpy as np
import pytest

from rotest.geometry import from_matrix, normalize_angle, to_matrix


def _apply(m: np.ndarray, x: float, y: float) -> np.ndarray:
    return m @ np.array([x, y, 1.0])


def test_sign_point_above_center_goes_right_for_plus_90() -> None:
    # Center (0, 0); "above" is y < 0 because y points down.
    out = _apply(to_matrix(90.0, 1.0, 0.0, 0.0), 0.0, -1.0)
    assert out == pytest.approx([1.0, 0.0], abs=1e-9)


def test_sign_point_right_goes_down_for_plus_90() -> None:
    out = _apply(to_matrix(90.0, 1.0, 0.0, 0.0), 1.0, 0.0)
    assert out == pytest.approx([0.0, 1.0], abs=1e-9)


def test_scale_and_translation_applied() -> None:
    out = _apply(to_matrix(0.0, 2.0, 5.0, -3.0), 1.0, 1.0)
    assert out == pytest.approx([7.0, -1.0], abs=1e-9)


@pytest.mark.parametrize("angle", [-179.5, -90.0, -0.01, 0.0, 12.345, 90.0, 179.9, 180.0])
@pytest.mark.parametrize("scale", [0.25, 1.0, 3.7])
def test_round_trip(angle: float, scale: float) -> None:
    got = from_matrix(to_matrix(angle, scale, 12.5, -3.1))
    assert got == pytest.approx((angle, scale, 12.5, -3.1), abs=1e-9)


def test_from_matrix_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        from_matrix(np.eye(3))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.0, 0.0), (180.0, 180.0), (-180.0, 180.0), (190.0, -170.0), (-190.0, 170.0), (720.0, 0.0)],
)
def test_normalize_angle(raw: float, expected: float) -> None:
    assert normalize_angle(raw) == pytest.approx(expected, abs=1e-9)
