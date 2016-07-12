# -*- coding: utf-8 -*-
#
#    Project: silx (originally pyFAI)
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2016  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "09/05/2016"
__doc__ = "Bilinear interpolator, peak finder, line-profile for images"

import cython
from cython.view cimport array as cvarray
import numpy
from libc.math cimport floor, ceil, sin, cos, sqrt, atan2  
import logging
logger = logging.getLogger("silx.image.bilinear") 


cdef class BilinearImage:
    """Bilinear interpolator for images ... or any data on a regular grid
    """
    cdef:
        readonly float[:, ::1] data
        readonly float maxi, mini
        readonly size_t width, height

    cpdef size_t coarse_local_maxi(self, size_t)
    cdef size_t c_local_maxi(self, size_t) nogil
    cdef float c_funct(self, float, float) nogil

    def __cinit__(self, data not None):
        """ Constructor

        :param data: image as a 2D array    
        """
        assert data.ndim == 2
        self.height = data.shape[0]
        self.width = data.shape[1]
        self.maxi = data.max()
        self.mini = data.min()
        self.data = numpy.ascontiguousarray(data, dtype=numpy.float32)
    
    def __dealloc__(self):
        self.data = None

    def __call__(self, coord):
        """Function f((y, x)) where f is a continuous function 
        made from the image and (y,x)=(row, column) is the pixel coordinates 
        in natural C-order

        :param x: 2-tuple of float (row, column)
        :return: Interpolated signal from the image 
        """
        return self.c_funct(coord[1], coord[0])
   
    @cython.boundscheck(False)
    @cython.wraparound(False)            
    cdef float c_funct(self, float x, float y) nogil:
        """Function f(x, y) where f is a continuous function 
        made from the image.

        :param x (float): column coordinate
        :param y (float): row coordinate
        :return: Interpolated signal from the image (nearest for outside)

        Cython only function due to NOGIL 
        """
        cdef:
            float d0 = min(max(y, 0.0), (self.height - 1.0))
            float d1 = min(max(x, 0.0), (self.width - 1.0))
            int i0, i1, j0, j1
            float x0, x1, y0, y1, res
            
        x0 = floor(d0)
        x1 = ceil(d0)
        y0 = floor(d1)
        y1 = ceil(d1)
        i0 = < int > x0
        i1 = < int > x1
        j0 = < int > y0
        j1 = < int > y1
        if (i0 == i1) and (j0 == j1):
            res = self.data[i0, j0]
        elif i0 == i1:
            res = (self.data[i0, j0] * (y1 - d1)) + (self.data[i0, j1] * (d1 - y0))
        elif j0 == j1:
            res = (self.data[i0, j0] * (x1 - d0)) + (self.data[i1, j0] * (d0 - x0))
        else:
            res = (self.data[i0, j0] * (x1 - d0) * (y1 - d1))  \
                + (self.data[i1, j0] * (d0 - x0) * (y1 - d1))  \
                + (self.data[i0, j1] * (x1 - d0) * (d1 - y0))  \
                + (self.data[i1, j1] * (d0 - x0) * (d1 - y0))
        return res
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def opp_f(self, coord):
        """opp_f(self, coord)

        Function -f((y,x)) for peak finding via minimizer.

        Gives large number outside the boundaries to return into the image  

        :param x: 2-tuple of float in natural C order, i.e (row, column)
        :return: Negative interpolated signal from the image
        """
        cdef:
            float d0, d1, res
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
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def local_maxi(self, coord):
        """local_maxi(self, coord)

        Return the nearest local maximum ... with sub-pixel refinement

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
            float tmp, sum0 = 0, sum1 = 0, sum = 0
            float a00, a01, a02, a10, a11, a12, a20, a21, a22
            float d00, d11, d01, denom, delta0, delta1        
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

    cpdef size_t coarse_local_maxi(self, size_t x):
        """coarse_local_maxi(self, idx)

        Return the nearest local maximum ... without sub-pixel refinement

        :param idx: start index (=row*width+column)
        :return: local maximum index
        """
        return self.c_local_maxi(x)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef size_t c_local_maxi(self, size_t idx) nogil:
        """Return the nearest local maximum ... 
        ... without sub-pixel refinement

        :param idx: start index (=row*width+column)
        :return: local maximum index

        This method is Cython only due to the NOGIL
        """
        cdef:
            int current0 = idx // self.width
            int current1 = idx % self.width
            int i0, i1, start0, stop0, start1, stop1, new0, new1
            float tmp, value, old_value

        value = self.data[current0, current1]
        old_value = value - 1.0
        new0, new1 = current0, current1

        while value > old_value:
            old_value = value
            start0 = max(0, current0 - 1)
            stop0 = min(self.height, current0 + 2)
            start1 = max(0, current1 - 1)
            stop1 = min(self.width, current1 + 2)
            for i0 in range(start0, stop0):
                for i1 in range(start1, stop1):
                    tmp = self.data[i0, i1]
                    if tmp > value:
                        new0, new1 = i0, i1
                        value = tmp
            current0, current1 = new0, new1
        return self.width * current0 + current1

    @cython.boundscheck(False)
    def map_coordinates(self, coordinates):
        """map_coordinates(self, coordinates)

        Map coordinates of the array on the image

        :param coordinates: 2-tuple of array of the same size (row_array, column_array) 
        :return: array of values at given coordinates
        """
        cdef:
            float[:] d0, d1, res
            size_t size, i
        shape = coordinates[0].shape
        size = coordinates[0].size
        d0 = numpy.ascontiguousarray(coordinates[0].ravel(), dtype=numpy.float32)
        d1 = numpy.ascontiguousarray(coordinates[1].ravel(), dtype=numpy.float32)
        assert size == d1.size
        res = numpy.empty(size, dtype=numpy.float32)
        with nogil:
            for i in range(size):
                res[i] = self.c_funct(d1[i], d0[i])
        return numpy.asarray(res).reshape(shape)  
    
    @cython.boundscheck(False)
    def profile_line(self, src, dst, int linewidth=1):
        """profile_line(self, src, dst, linewidth=1)

        Return the intensity profile of an image measured along a scan line.

        :param src: The start point of the scan line.
        :type src: 2-tuple of numeric scalar
        :param dst: The end point of the scan line.
            The destination point is included in the profile,
            in contrast to standard numpy indexing.
        :type dst: 2-tuple of numeric scalar
        :param int linewidth: Width of the scanline (unit image pixel).
        :return: The intensity profile along the scan line.
            The length of the profile is the ceil of the computed length
            of the scan line.
        :rtype: 1d array

        Inspired from skimage
        """
        cdef:
            float src_row, src_col, dst_row, dst_col, d_row, d_col
            float length, col_width, row_width, sum, row, col, new_row, new_col
            int lengt, i, j, cnt
            float[::1] result
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
        d_row /= <float> (lengt -1)
        d_col /= <float> (lengt -1)

        result = numpy.zeros(lengt, dtype=numpy.float32)

        # Offset position to the center of the bottom pixels of the profile
        src_row -= row_width * (linewidth - 1) / 2.
        src_col -= col_width * (linewidth - 1) / 2.

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
                        cnt = cnt + 1
                        sum = sum + self.c_funct(new_col, new_row)
                if cnt:
                    result[i] += sum / cnt

        # Ensures the result is exported as numpy array and not memory view.
        return numpy.asarray(result)
