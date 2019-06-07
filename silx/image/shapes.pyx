# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
"""This module provides functions making masks on an image.

- :func:`circle_fill` function generates coordinates of a circle in an image.
- :func:`draw_line` function generates coordinates of a line in an image.
- :func:`polygon_fill_mask` function generates a mask from a set of points
  defining a polygon.

The :class:`Polygon` class provides checking if a point is inside a polygon.

The whole module uses the (row, col) (i.e., (y, x))) convention
for 2D coordinates.
"""

__authors__ = ["Jérôme Kieffer", "T. Vincent"]
__license__ = "MIT"
__date__ = "15/02/2019"
__status__ = "dev"


cimport cython
import numpy
from libc.math cimport ceil, fabs


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

    def is_inside(self, row, col):
        """Check if (row, col) is inside or outside the polygon

        :param float row:
        :param float col:
        :return: True if position is inside polygon, False otherwise
        """
        return self.c_is_inside(row, col)

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bint c_is_inside(self, float row, float col) nogil:
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

    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def make_mask(self, int height, int width):
        """Create a mask array representing the filled polygon

        :param int height: Height of the mask array
        :param int width: Width of the mask array
        :return: 2D array (height, width)
        """
        cdef unsigned char[:, :] mask = numpy.zeros((height, width),
                                                    dtype=numpy.uint8)
        cdef int row_min, row_max, col_min, col_max  # mask subpart to update
        cdef int row, col, index  # Loop indixes
        cdef float pt1x, pt1y, pt2x, pt2y  # segment end points
        cdef int xinters, is_inside, current

        row_min = max(int(min(self.vertices[:, 0])), 0)
        row_max = min(int(max(self.vertices[:, 0])) + 1, height)

        # Can be replaced by prange(row_min, row_max, nogil=True)
        with nogil:
            for row in range(row_min, row_max):
                # For each line of the image, mark intersection of all segments
                # in the line and then run a xor scan to fill inner parts
                # Adapted from http://alienryderflex.com/polygon_fill/
                pt1x = self.vertices[self.nvert-1, 1]
                pt1y = self.vertices[self.nvert-1, 0]
                col_min = width - 1
                col_max = 0
                is_inside = 0  # Init with whether first col is inside or not

                for index in range(self.nvert):
                    pt2x = self.vertices[index, 1]
                    pt2y = self.vertices[index, 0]

                    if ((pt1y <= row and row < pt2y) or
                            (pt2y <= row and row < pt1y)):
                        # Intersection casted to int so that ]x, x+1] => x
                        xinters = (<int>ceil(pt1x + (row - pt1y) *
                                   (pt2x - pt1x) / (pt2y - pt1y))) - 1

                        # Update column range to patch
                        if xinters < col_min:
                            col_min = xinters
                        if xinters > col_max:
                            col_max = xinters

                        if xinters < 0:
                            # Add an intersection to init value of xor scan
                            is_inside ^= 1
                        elif xinters < width:
                            # Mark intersection in mask
                            mask[row, xinters] ^= 1
                        # else: do not consider intersection on the right

                    pt1x, pt1y = pt2x, pt2y

                if col_min < col_max:
                    # Clip column range to mask
                    if col_min < 0:
                        col_min = 0
                    if col_max > width - 1:
                        col_max = width - 1

                    # xor exclusive scan
                    for col in range(col_min, col_max + 1):
                        current = mask[row, col]
                        mask[row, col] = is_inside
                        is_inside = current ^ is_inside

        # Ensures the result is exported as numpy array and not memory view.
        return numpy.asarray(mask)


def polygon_fill_mask(vertices, shape):
    """Return a mask of boolean, True for pixels inside a polygon.

    :param vertices: Strip of segments end points (row, column) or (y, x)
    :type vertices: numpy.ndarray like container of dimension Nx2
    :param shape: size of the mask as (height, width)
    :type shape: 2-tuple of int
    :return: Mask corresponding to the polygon
    :rtype: numpy.ndarray of dimension shape
    """
    return Polygon(vertices).make_mask(shape[0], shape[1])


@cython.wraparound(False)
@cython.boundscheck(False)
def draw_line(int row0, int col0, int row1, int col1, int width=1):
    """Line includes both end points.
    Width is handled by drawing parallel lines, so junctions of lines belonging
    to different octant with width > 1 will not look nice.

    Using Bresenham line algorithm:
    Bresenham, J. E.
    Algorithm for computer control of a digital plotter.
    IBM Systems Journal. Vol 4 No 1. 1965. pp 25-30

    :param int row0: Start point row
    :param int col0: Start point col
    :param int row1: End point row
    :param int col1: End point col
    :param int width: Thickness of the line in pixels (default 1)
                      Width must be at least 1.
    :return: Array coordinates of points inside the line (might be negative)
    :rtype: 2-tuple of numpy.ndarray (rows, cols)
    """
    cdef int drow, dcol, invert_coords
    cdef int db, da, delta, b, a, step_a, step_b
    cdef int index, offset  # Loop indices

    # Store coordinates of points of the line
    cdef int[:, :] b_coords
    cdef int[:, :] a_coords

    dcol = abs(col1 - col0)
    drow = abs(row1 - row0)
    invert_coords = dcol < drow

    if dcol == 0 and drow == 0:
        return (numpy.array((row0,), dtype=numpy.int32),
                numpy.array((col0,), dtype=numpy.int32))

    if width < 1:
        width = 1

    # Set a and b according to segment octant
    if not invert_coords:
        da = dcol
        db = drow
        step_a = 1 if col1 > col0 else -1
        step_b = 1 if row1 > row0 else -1
        a = col0
        b = row0

    else:
        da = drow
        db = dcol
        step_a = 1 if row1 > row0 else -1
        step_b = 1 if col1 > col0 else -1
        a = row0
        b = col0

    b_coords = numpy.empty((da + 1, width), dtype=numpy.int32)
    a_coords = numpy.empty((da + 1, width), dtype=numpy.int32)

    with nogil:
        b -= (width - 1) // 2
        delta = 2 * db - da
        for index in range(da + 1):
            for offset in range(width):
                b_coords[index, offset] = b + offset
                a_coords[index, offset] = a

            if delta >= 0:  # M2: Move by step_a + step_b
                b += step_b
                delta -= 2 * da
            # else M1: Move by step_a

            a += step_a
            delta += 2 * db

    if not invert_coords:
        return (numpy.asarray(b_coords).reshape(-1),
                numpy.asarray(a_coords).reshape(-1))
    else:
        return (numpy.asarray(a_coords).reshape(-1),
                numpy.asarray(b_coords).reshape(-1))


def circle_fill(int crow, int ccol, float radius):
    """Generates coordinate of image points lying in a disk.

    :param int crow: Row of the center of the disk
    :param int ccol: Column of the center of the disk
    :param float radius: Radius of the disk
    :return: Array coordinates of points inside the disk (might be negative)
    :rtype: 2-tuple of numpy.ndarray (rows, cols)
    """
    cdef int i_radius, len_coords

    radius = fabs(radius)
    i_radius = <int>radius

    coords = numpy.arange(-i_radius, ceil(radius) + 1,
                          dtype=numpy.float32) ** 2
    len_coords = len(coords)
    # rows, cols = where(row**2 + col**2 < radius**2)
    rows, cols = numpy.where(coords.reshape(1, len_coords) +
                             coords.reshape(len_coords, 1) < radius ** 2)
    return rows + crow - i_radius, cols + ccol - i_radius


def ellipse_fill(int crow, int ccol, float radius_r, float radius_c):
    """Generates coordinate of image points lying in a ellipse.

    :param int crow: Row of the center of the ellipse
    :param int ccol: Column of the center of the ellipse
    :param float radius_r: Radius of the ellipse in the row
    :param float radius_c: Radius of the ellipse in the column
    :return: Array coordinates of points inside the ellipse (might be negative)
    :rtype: 2-tuple of numpy.ndarray (rows, cols)
    """
    cdef int i_radius_r
    cdef int i_radius_c

    i_radius_r = <int>fabs(radius_r)
    i_radius_c = <int>fabs(radius_c)

    x_coords = numpy.arange(-i_radius_r, ceil(radius_r) + 1, dtype=numpy.float32).reshape(-1, 1)
    y_coords = numpy.arange(-i_radius_c, ceil(radius_c) + 1, dtype=numpy.float32).reshape(1, -1)

    # rows, cols = where(x**2 + col**2 < 1)
    rows, cols = numpy.where(x_coords**2 / radius_r**2 + y_coords**2 / radius_c**2 < 1.0)
    return rows + crow - i_radius_r, cols + ccol - i_radius_c
