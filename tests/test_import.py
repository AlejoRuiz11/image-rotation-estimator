import rotest


def test_package_importable() -> None:
    assert rotest.__version__
