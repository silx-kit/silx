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
"""Utilities for image items"""

from __future__ import annotations

import math


def imageIndexSlices(
    origin: tuple[float, float],
    scale: tuple[float, float],
    shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[slice, slice] | None:
    """Returns the image pixels intersecting a rectangle in data coordinates.

    It takes the image origin and scale (possibly negative) into account.

    :param origin: (ox, oy) Image origin in data coordinates
    :param scale: (sx, sy) Image pixel size in data coordinates
    :param shape: (height, width) of the image
    :param bounds: (xmin, xmax, ymin, ymax) Rectangle in data coordinates
    :returns: (row slice, column slice) or None if the intersection is empty
    """
    xmin, xmax, ymin, ymax = bounds
    result = []
    for vmin, vmax, o, s, size in (
        (ymin, ymax, origin[1], scale[1], shape[0]),
        (xmin, xmax, origin[0], scale[0], shape[1]),
    ):
        if s == 0:
            return None
        start, stop = sorted(((vmin - o) / s, (vmax - o) / s))
        start = max(0, math.floor(start))
        stop = min(size, math.ceil(stop))
        if start >= stop:
            return None
        result.append(slice(start, stop))
    return tuple(result)
