# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""This module provides functions related to polygon filling.

The :class:`Polygon` class provides checking if a point is inside a polygon
and a way to generate a mask of the polygon.

The :func:`polygon_fill` function generates a mask from a set of points
defining a polygon.

The whole module uses the (row, col) (i.e., (y, x))) convention
for 2D coordinates.
"""

__authors__ = ["Jérôme Kieffer", "T. Vincent"]
__license__ = "MIT"
__date__ = "03/06/2016"


cimport cython
import numpy
from cython.parallel import prange


cdef class Polygon(object):
    """Define a polygon that provides inside check and mask generation.

    :param vertices: corners of the polygon
    :type vertices: Nx2 array of floats of (row, col)
    """

    cdef float[:,:] vertices
    cdef int nvert

    def __init__(self, vertices):
        self.vertices = numpy.ascontiguousarray(vertices, dtype=numpy.float32)
        self.nvert = self.vertices.shape[0]

    def isInside(self, row, col):
        """isInside(self, row, col)

        Check if (row, col) is inside or outside the polygon

        :param float row:
        :param float col:
        :return: True if position is inside polygon, False otherwise
        """
        return self.c_isInside(row, col)

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bint c_isInside(self, float row, float col) nogil:
        """Check if (row, col) is inside or outside the polygon

        Pure C_Cython class implementation.

        :param float row:
        :param float col:
        :return: True if position is inside polygon, False otherwise
        """
        cdef int index, is_inside
        cdef float pt1x, pt1y, pt2x, pt2y, xinters
        is_inside = 0

        pt1x = self.vertices[self.nvert-1, 1]
        pt1y = self.vertices[self.nvert-1, 0]
        for index in range(self.nvert):
            pt2x = self.vertices[index, 1]
            pt2y = self.vertices[index, 0]

            if (((pt1y <= row and row < pt2y) or
                    (pt2y <= row and row < pt1y)) and
                    # Extra (optional) condition to avoid some computation
                    (col <= pt1x or col <= pt2x)):
                xinters = (row - pt1y) * (pt2x - pt1x) / (pt2y - pt1y) + pt1x
                is_inside ^= col < xinters
            pt1x, pt1y = pt2x, pt2y
        return is_inside

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def make_mask(self, int height, int width):
        """make_mask(self, height, width)

        Create a mask array representing the filled polygon

        :param int height: Height of the mask array
        :param int width: Width of the mask array
        :return: 2D array (height, width)
        """
        # Possible optimization for fill mask:
        # treat it line by line, see http://alienryderflex.com/polygon_fill/
        cdef unsigned char[:, :] mask = numpy.zeros((height, width),
                                                    dtype=numpy.uint8)
        cdef int row, col
        cdef int row_min, row_max, col_min, col_max
        cdef float[:] rows, cols

        rows = self.vertices[:, 0]
        cols = self.vertices[:, 1]

        row_min = max(int(min(rows)), 0)
        row_max = min(int(max(rows)) + 1, height)
        col_min = max(int(min(cols)), 0)
        col_max = min(int(max(cols)) + 1, width)

        for row in prange(row_min, row_max, nogil=True):
            for col in range(col_min, col_max):
                mask[row, col] = self.c_isInside(row, col)

        # Ensures the result is exported as numpy array and not memory view.
        return numpy.asarray(mask)


def polygon_fill(vertices, shape):
    """polygon_fill(vertices, shape)

    Return a mask of boolean, True for pixels inside a polygon.

    :param vertices: Strip of segments end points (row, column) or (y, x)
    :type vertices: numpy.ndarray like container of dimension Nx2
    :param shape: size of the mask as (height, width)
    :type shape: 2-tuple of int
    :return: Mask corresponding to the polygon
    :rtype: numpy.ndarray of dimension shape
    """
    return Polygon(vertices).make_mask(shape[0], shape[1])
