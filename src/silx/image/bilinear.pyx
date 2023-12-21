#cython: embedsignature=True, language_level=3
## This is for optimisation
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#
#
#    Project: silx (originally pyFAI)
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2023  European Synchrotron Radiation Facility, Grenoble, France
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

"""Bilinear interpolator, peak finder, line-profile for images"""
__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "21/12/2023"

# C-level imports
from libc.stdint cimport uint8_t
from libc.math cimport floor, ceil, sqrt, NAN, isfinite
from libc.float cimport FLT_MAX

import cython
import numpy
import logging
logger = logging.getLogger(__name__)


#Definition of some constants
# How data are stored 
ctypedef float data_t
data_d = numpy.float32

#How the mask is stored
ctypedef uint8_t mask_t
mask_d = numpy.uint8


cdef class BilinearImage:
    """Bilinear interpolator for images ... or any data on a regular grid
    """
    cdef:
        readonly data_t[:, ::1] data
        readonly mask_t[:, ::1] mask
        readonly data_t maxi, mini
        readonly Py_ssize_t width, height
        readonly bint has_mask 
    
    # C-level declarations
    cpdef Py_ssize_t coarse_local_maxi(self, Py_ssize_t)
    cdef Py_ssize_t c_local_maxi(self, Py_ssize_t) noexcept nogil
    cdef data_t c_funct(self, data_t, data_t) noexcept nogil
    cdef void _init_min_max(self) noexcept nogil
    
    def __cinit__(self, data not None, mask=None):
        """Constructor

        :param data: image as a 2D array
        """
        assert data.ndim == 2
        self.height = data.shape[0]
        self.width = data.shape[1]
        self.data = numpy.ascontiguousarray(data, dtype=data_d)
        if mask is not None:
            self.mask = numpy.ascontiguousarray(mask, dtype=mask_d)
            self.has_mask=True
        else:
            self.mask = None
            self.has_mask = False
        self._init_min_max()

    def __dealloc__(self):
        self.data = None
        self.mask = None

    def __call__(self, coord):
        """Function f((y, x)) where f is a continuous function
        made from the image and (y,x)=(row, column) is the pixel coordinates
        in natural C-order

        :param x: 2-tuple of float (row, column)
        :return: Interpolated signal from the image
        """
        return self.c_funct(coord[1], coord[0])
    
    cdef void _init_min_max(self) noexcept nogil:
        "Calculate the min & max"
        cdef:
            Py_ssize_t i, j
            data_t maxi, mini, value
        mini = FLT_MAX
        maxi = -FLT_MAX
        for i in range(self.height):
            for j in range(self.width):
                if not (self.has_mask and self.mask[i,j]):
                    value = self.data[i, j] 
                    maxi = max(value, maxi)
                    mini = min(value, mini)
        self.maxi = maxi
        self.mini = mini 

    cdef data_t c_funct(self, data_t x, data_t y) noexcept nogil:
        """Function f(x, y) where f is a continuous function
        made from the image.

        :param x (float): column coordinate
        :param y (float): row coordinate
        :return: Interpolated signal from the image (nearest for outside)

        Cython only function due to NOGIL
        """
        cdef:
            data_t d0 = min(max(y, 0.0), (self.height - 1.0))
            data_t d1 = min(max(x, 0.0), (self.width - 1.0))
            mask_t m0, m1, m2, m3
            Py_ssize_t i0, i1, j0, j1
            data_t x0, x1, y0, y1, res, scale

        x0 = floor(d0)
        x1 = ceil(d0)
        y0 = floor(d1)
        y1 = ceil(d1)
        i0 = < int > x0
        i1 = < int > x1
        j0 = < int > y0
        j1 = < int > y1
        if (i0 == i1) and (j0 == j1):
            if not (self.has_mask and self.mask[i0,j0]):
                res = self.data[i0, j0]
            else:
                res = NAN
        elif i0 == i1:
            if self.has_mask:
                m0 = self.mask[i0, j0]
                m1 = self.mask[i0, j1]
                if m0 and m1:
                    res = NAN
                elif m0:
                    res = self.data[i0, j1]
                elif m1:
                    res = self.data[i0, j0]
                else:
                    res = (self.data[i0, j0] * (y1 - d1)) + (self.data[i0, j1] * (d1 - y0))                
            else:
                res = (self.data[i0, j0] * (y1 - d1)) + (self.data[i0, j1] * (d1 - y0))
        elif j0 == j1:
            if self.has_mask:
                m0 = self.mask[i0, j0]
                m1 = self.mask[i1, j0]
                if m0 and m1:
                    res = NAN
                elif m0:
                    res = self.data[i1, j0]
                elif m1:
                    res = self.data[i0, j0]
                else:
                    res = (self.data[i0, j0] * (x1 - d0)) + (self.data[i1, j0] * (d0 - x0))                
            else:
                res = (self.data[i0, j0] * (x1 - d0)) + (self.data[i1, j0] * (d0 - x0))
        else:
            if self.has_mask:
                m0 = self.mask[i0, j0]
                m1 = self.mask[i1, j0]
                m2 = self.mask[i0, j1]
                m3 = self.mask[i1, j1]
                if m0 and m1 and m2 and m3:
                    res = NAN
                else:
                    m0 = not m0
                    m1 = not m1
                    m2 = not m2
                    m3 = not m3
                    if m0 and m1 and m2 and m3:
                        res = (self.data[i0, j0] * (x1 - d0) * (y1 - d1))  \
                            + (self.data[i1, j0] * (d0 - x0) * (y1 - d1))  \
                            + (self.data[i0, j1] * (x1 - d0) * (d1 - y0))  \
                            + (self.data[i1, j1] * (d0 - x0) * (d1 - y0))                    
                    else:
                        res = (m0 * self.data[i0, j0] * (x1 - d0) * (y1 - d1))  \
                            + (m1 * self.data[i1, j0] * (d0 - x0) * (y1 - d1))  \
                            + (m2 * self.data[i0, j1] * (x1 - d0) * (d1 - y0))  \
                            + (m3 * self.data[i1, j1] * (d0 - x0) * (d1 - y0))
                        scale = ((m0 * (x1 - d0) * (y1 - d1)) 
                               + (m1 * (d0 - x0) * (y1 - d1))
                               + (m2 * (x1 - d0) * (d1 - y0))
                               + (m3 * (d0 - x0) * (d1 - y0)))
                        res /= scale
            else:
                res = (self.data[i0, j0] * (x1 - d0) * (y1 - d1))  \
                    + (self.data[i1, j0] * (d0 - x0) * (y1 - d1))  \
                    + (self.data[i0, j1] * (x1 - d0) * (d1 - y0))  \
                    + (self.data[i1, j1] * (d0 - x0) * (d1 - y0))                    
                
        return res

    def opp_f(self, coord):
        """Function -f((y,x)) for peak finding via minimizer.

        Gives large number outside the boundaries to return into the image

        :param x: 2-tuple of float in natural C order, i.e (row, column)
        :return: Negative interpolated signal from the image
        """
        cdef:
            data_t d0, d1, res
        d0, d1 = coord
        if d0 < 0:
            res = self.mini + d0
        elif d1 < 0:
            res = self.mini + d1
        elif d0 > (self.height - 1):
            res = self.mini - d0 + self.height - 1
        elif d1 > self.width - 1:
            res = self.mini - d1 + self.width - 1
        else:
            res = self.c_funct(d1, d0)
        return - res

    def local_maxi(self, coord):
        """Return the nearest local maximum ... with sub-pixel refinement

        Nearest maximum search:
            steepest ascent

        Sub-pixel refinement:
            Second order Taylor expansion of the function;
            At the maximum, the first derivative is null
            delta = x-i = -Inverse[Hessian].gradient
            if Hessian is singular or \|delta\|>1: use a center of mass.

        :param coord: 2-tuple of scalar (row, column)
        :return: 2-tuple of float with the nearest local maximum
        """
        cdef:
            int res, current0, current1
            int i0, i1
            data_t tmp, sum0 = 0, sum1 = 0, sum = 0
            data_t a00, a01, a02, a10, a11, a12, a20, a21, a22
            data_t d00, d11, d01, denom, delta0, delta1
        res = self.c_local_maxi(round(coord[0]) * self.width + round(coord[1]))

        current0 = res // self.width
        current1 = res % self.width
        if (current0 > 0) and (current0 < self.height - 1) and (current1 > 0) and (current1 < self.width - 1):
            # Use second order polynomial Taylor expansion
            a00 = self.data[current0 - 1, current1 - 1]
            a01 = self.data[current0 - 1, current1    ]
            a02 = self.data[current0 - 1, current1 + 1]
            a10 = self.data[current0    , current1 - 1]
            a11 = self.data[current0    , current1    ]
            a12 = self.data[current0    , current1 + 1]
            a20 = self.data[current0 + 1, current1 - 1]
            a21 = self.data[current0 + 1, current1    ]
            a22 = self.data[current0 + 1, current1 - 1]
            d00 = a12 - 2.0 * a11 + a10
            d11 = a21 - 2.0 * a11 + a01
            d01 = (a00 - a02 - a20 + a22) / 4.0
            denom = 2.0 * (d00 * d11 - d01 * d01)
            if abs(denom) < 1e-10:
                logger.debug("Singular determinant, Hessian undefined")
            else:
                delta0 = ((a12 - a10) * d01 + (a01 - a21) * d11) / denom
                delta1 = ((a10 - a12) * d00 + (a21 - a01) * d01) / denom
                if abs(delta0) <= 1.0 and abs(delta1) <= 1.0:
                    # Result is OK if lower than 0.5.
                    return (delta0 + float(current0), delta1 + float(current1))
                else:
                    logger.debug("Failed to find root using second order expansion")
            # refinement of the position by a simple center of mass of the last valid region used
            for i0 in range(current0 - 1, current0 + 2):
                for i1 in range(current1 - 1, current1 + 2):
                    tmp = self.data[i0, i1]
                    sum0 += tmp * i0
                    sum1 += tmp * i1
                    sum += tmp
            if sum > 0:
                return (sum0 / sum, sum1 / sum)

        return (float(current0), float(current1))

    cpdef Py_ssize_t coarse_local_maxi(self, Py_ssize_t x):
        """Return the nearest local maximum ... without sub-pixel refinement

        :param idx: start index (=row*width+column)
        :return: local maximum index
        """
        return self.c_local_maxi(x)

    cdef Py_ssize_t c_local_maxi(self, Py_ssize_t idx) noexcept nogil:
        """Return the nearest local maximum without sub-pixel refinement

        :param idx: start index (=row*width+column)
        :return: local maximum index

        This method is Cython only due to the NOGIL
        """
        cdef:
            Py_ssize_t current0 = idx // self.width
            Py_ssize_t current1 = idx % self.width
            Py_ssize_t i0, i1, start0, stop0, start1, stop1, new0, new1, rng, cnt
            mask_t m
            data_t tmp, value, old_value

        if self.has_mask and self.mask[current0, current1]:
            #Start searching for a non masked pixel.
            rng = 0
            cnt = 0
            value = self.mini
            new0, new1 = current0, current1
            while cnt == 0:
                rng += 1
                cnt = 0
                start0 = max(0, current0 - rng)
                stop0 = min(self.height, current0 + rng + 1)
                start1 = max(0, current1 - rng)
                stop1 = min(self.width, current1 + rng + 1)
                for i0 in range(start0, stop0):
                    for i1 in range(start1, stop1):
                        m = not self.mask[i0, i1] 
                        cnt += m
                        if m:
                            tmp = self.data[i0, i1]
                            if tmp > value:
                                new0, new1 = i0, i1
                                value = tmp
            current0, current1 = new0, new1
        else:
            value = self.data[current0, current1]
        
        old_value = value -1
        new0, new1 = current0, current1

        while value > old_value:
            old_value = value
            start0 = max(0, current0 - 1)
            stop0 = min(self.height, current0 + 2)
            start1 = max(0, current1 - 1)
            stop1 = min(self.width, current1 + 2)
            for i0 in range(start0, stop0):
                for i1 in range(start1, stop1):
                    if self.has_mask and self.mask[current0, current1]:
                        continue
                    tmp = self.data[i0, i1]
                    if tmp > value:
                        new0, new1 = i0, i1
                        value = tmp
            current0, current1 = new0, new1
        return self.width * current0 + current1

    def map_coordinates(self, coordinates):
        """Map coordinates of the array on the image

        :param coordinates: 2-tuple of array of the same size (row_array, column_array)
        :return: array of values at given coordinates
        """
        cdef:
            data_t[:] d0, d1, res
            Py_ssize_t size, i
        shape = coordinates[0].shape
        size = coordinates[0].size
        d0 = numpy.ascontiguousarray(coordinates[0].ravel(), dtype=data_d)
        d1 = numpy.ascontiguousarray(coordinates[1].ravel(), dtype=data_d)
        assert size == d1.size
        res = numpy.empty(size, dtype=data_d)
        with nogil:
            for i in range(size):
                res[i] = self.c_funct(d1[i], d0[i])
        return numpy.asarray(res).reshape(shape)

    def profile_line(self, src, dst, int linewidth=1, method='mean'):
        """Return the mean or sum of intensity profile of an image measured
        along a scan line.

        :param src: The start point of the scan line.
        :type src: 2-tuple of numeric scalar
        :param dst: The end point of the scan line.
            The destination point is included in the profile,
            in contrast to standard numpy indexing.
        :type dst: 2-tuple of numeric scalar
        :param int linewidth: Width of the scanline (unit image pixel).
        :param str method: 'mean' or 'sum' depending if we want to compute the
            mean intensity along the line or the sum.
        :return: The intensity profile along the scan line.
            The length of the profile is the ceil of the computed length
            of the scan line.
        :rtype: 1d array

        Inspired from skimage
        """
        cdef:
            data_t src_row, src_col, dst_row, dst_col, d_row, d_col
            data_t length, col_width, row_width, sum, row, col, new_row, new_col, val
            Py_ssize_t lengt, i, j, cnt
            bint compute_mean
            data_t[::1] result
        src_row, src_col = src
        dst_row, dst_col = dst
        if (src_row == dst_row) and (src_col == dst_col):
            logger.warning("Source and destination points are the same")
            return numpy.array([self.c_funct(src_col, src_row)])
        d_row = dst_row - src_row
        d_col = dst_col - src_col

        # Offsets to deal with linewidth
        length = sqrt(d_row * d_row + d_col * d_col)
        row_width = d_col / length
        col_width = - d_row / length

        lengt = <int> ceil(length + 1)
        d_row /= <data_t> (lengt -1)
        d_col /= <data_t> (lengt -1)

        result = numpy.zeros(lengt, dtype=data_d)

        # Offset position to the center of the bottom pixels of the profile
        src_row -= row_width * (linewidth - 1) / 2.
        src_col -= col_width * (linewidth - 1) / 2.

        compute_mean = (method == 'mean')
        with nogil:
            for i in range(lengt):
                sum = 0
                cnt = 0

                row = src_row + i * d_row
                col = src_col + i * d_col

                for j in range(linewidth):
                    new_row = row + j * row_width
                    new_col = col + j * col_width
                    if ((new_col >= 0) and (new_col < self.width) and
                            (new_row >= 0) and (new_row < self.height)):
                        val = self.c_funct(new_col, new_row)
                        if isfinite(val):
                            cnt += 1
                            sum += val  
                if cnt:
                        if compute_mean:
                            result[i] += sum / cnt
                        else:
                            result[i] += sum
                elif compute_mean:
                    result[i] += NAN
        # Ensures the result is exported as numpy array and not memory view.
        return numpy.asarray(result)
