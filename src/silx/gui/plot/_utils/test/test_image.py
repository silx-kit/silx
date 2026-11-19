"""Tests of silx.gui.plot._utils.image"""

import pytest

from silx.gui.plot._utils.image import imageIndexSlices


@pytest.mark.parametrize(
    "origin, scale, bounds, expected",
    [
        # Default origin and scale
        ((0, 0), (1, 1), (2, 5, 3, 7), (slice(3, 7), slice(2, 5))),
        # Partial pixels are included
        ((0, 0), (1, 1), (2.5, 4.5, 3.2, 6.8), (slice(3, 7), slice(2, 5))),
        # Clipped to the image
        ((0, 0), (1, 1), (-5, 3, 8, 50), (slice(8, 10), slice(0, 3))),
        # Origin and scale
        ((-10, 100), (0.5, 2), (-9, -8, 104, 110), (slice(2, 5), slice(2, 4))),
        # Negative scale
        ((10, 0), (-1, 1), (5, 8, 0, 10), (slice(0, 10), slice(2, 5))),
        # Outside of the image
        ((0, 0), (1, 1), (20, 30, 0, 10), None),
        ((0, 0), (1, 1), (0, 10, -20, -10), None),
    ],
)
def testImageIndexSlices(origin, scale, bounds, expected):
    assert imageIndexSlices(origin, scale, (10, 10), bounds) == expected
