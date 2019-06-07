# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
"""Wrapper over Delaunay implementation"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/05/2019"


import logging
import sys

import numpy


_logger = logging.getLogger(__name__)


def triangulation(x, y, dtype=numpy.uint32):
    """Run Delaunay tesselation and returns triangle indices.

    :param numpy.ndarray x: X coordinates of points
    :param numpy.ndarray y: Y coordinates of points
    :param numpy.dtype dtype: Data type of output indices
    :return: Point indices for triangles as a (N, 3) array
    :rtype: Union[numpy.ndarray,None]
    """
    # Lazy loading of Delaunay
    from silx.third_party.scipy_spatial import Delaunay as _Delaunay

    coordinates = numpy.array((x, y)).T

    if len(coordinates) > 3:
        # Enough points to try a Delaunay tesselation

        try:
            tri = _Delaunay(coordinates)
        except RuntimeError:
            _logger.error("Delaunay tesselation failed: %s",
                          sys.exc_info()[1])
            return None

        triangles = tri.simplices.astype(dtype)

    else:
        # 3 or less points: Draw one triangle
        triangles = numpy.array(
            [[0, 1, 2]], dtype=dtype) % len(coordinates)

    return triangles
