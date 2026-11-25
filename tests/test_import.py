# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Fabian Alejandro Ruiz Osorio (AlejoRuiz11)
import rotest


def test_package_importable() -> None:
    assert rotest.__version__
