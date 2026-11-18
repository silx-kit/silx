# /*##########################################################################
#
# Copyright (c) 2026 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
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
