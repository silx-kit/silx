# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This module provides a function to apply a color LUT (colormap) to a dataset.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "16/05/2018"


cimport cython
from cython.parallel import prange
cimport numpy as cnumpy
from libc.math cimport lrint, HUGE_VAL, isfinite, isnan, frexp, NAN
from libc.math cimport asinh, sqrt

import logging
import numpy


_logger = logging.getLogger(__name__)


# Supported data types
ctypedef fused data_types:
    cnumpy.uint8_t
    cnumpy.int8_t
    cnumpy.uint16_t
    cnumpy.int16_t
    cnumpy.uint32_t
    cnumpy.int32_t
    cnumpy.uint64_t
    cnumpy.int64_t
    float
    double
    long double


# Data types using a LUT to apply the colormap
ctypedef fused lut_types:
    cnumpy.uint8_t
    cnumpy.int8_t
    cnumpy.uint16_t
    cnumpy.int16_t


# Data types using default colormap implementation
ctypedef fused default_types:
    cnumpy.uint32_t
    cnumpy.int32_t
    cnumpy.uint64_t
    cnumpy.int64_t
    float
    double
    long double


# Supported colors/output types
ctypedef fused image_types:
    cnumpy.uint8_t
    float


# Colormap

cdef class Colormap:
    """Class for processing of linear normalized colormap."""

    def __cinit__(self):
        pass

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double normalize(self, double value) nogil:
        """For linear colormap, this is a No-Op.

        Override in subclass to perform some normalization.
        This MUST be a monotonic function.

        :param value: Value to normalize
        :return: Normalized value
        """
        return value

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    def apply(self,
              image_types[:, ::1] output,
              data_types[:] data,
              image_types[:, ::1] colors,
              double vmin,
              double vmax,
              image_types[::1] nan_color):
        """Apply colormap to data.

        :param output: Memory view where to store the result
        :param data: Input data
        :param colors: Colors look-up-table
        :param vmin: Lower bound of the colormap range
        :param vmax: Upper bound of the colormap range
        :param nan_color: Color to use for NaN value.
        """
        cdef double normalized_vmin, normalized_vmax

        normalized_vmin = self.normalize(vmin)
        normalized_vmax = self.normalize(vmax)

        if not isfinite(normalized_vmin) or not isfinite(normalized_vmax):
            raise ValueError('Colormap range is not valid')

        # Proxy for calling the right implementation depending on data type
        if data_types in lut_types:  # Use LUT implementation
            self._cmap_lut(output, data, colors,
                           normalized_vmin, normalized_vmax, nan_color)

        elif data_types in default_types:  # Use default implementation
            self._cmap(output, data, colors,
                       normalized_vmin, normalized_vmax, nan_color)

        else:
            raise ValueError('Unsupported data type')

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef _cmap(self,
               image_types[:, ::1] output,
               default_types[:] data,
               image_types[:, ::1] colors,
               double normalized_vmin,
               double normalized_vmax,
               image_types[::1] nan_color):
        """Apply colormap to data.

        :param output: Memory view where to store the result
        :param data: Input data
        :param colors: Colors look-up-table
        :param normalized_vmin: Normalized lower bound of the colormap range
        :param normalized_vmax: Normalized upper bound of the colormap range
        :param nan_color: Color to use for NaN value
        """
        cdef double scale, value
        cdef unsigned int length, channel, nb_channels, nb_colors
        cdef int index, lut_index

        nb_colors = colors.shape[0]
        nb_channels = colors.shape[1]
        length = data.size

        if normalized_vmin == normalized_vmax:
            scale = 0.
        else:
            # TODO check this
            #scale = (nb_colors - 1) / (vmax - vmin)
            scale = nb_colors / (normalized_vmax - normalized_vmin)

        with nogil:
            for index in prange(length):
                value = self.normalize(data[index])

                # Handle NaN
                if isnan(value):
                    for channel in range(nb_channels):
                        output[index, channel] = nan_color[channel]
                    continue

                if value <= normalized_vmin:
                    lut_index = 0
                elif value >= normalized_vmax:
                    lut_index = nb_colors - 1
                else:
                    lut_index = <int>((value - normalized_vmin) * scale)
                    # Safety net, duplicate previous checks
                    # TODO needed?
                    if lut_index < 0:
                        lut_index = 0
                    elif lut_index >= nb_colors:
                        lut_index = nb_colors - 1

                for channel in range(nb_channels):
                    output[index, channel] = colors[lut_index, channel]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef _cmap_lut(self,
                   image_types[:, ::1] output,
                   lut_types[:] data,
                   image_types[:, ::1] colors,
                   double normalized_vmin,
                   double normalized_vmax,
                   image_types[::1] nan_color):
        """Convert data to colors using look-up table to speed the process.

        Only supports data of types: uint8, uint16, int8, int16.

        :param output: Memory view where to store the result
        :param data: Input data
        :param colors: Colors look-up-table
        :param normalized_vmin: Normalized lower bound of the colormap range
        :param normalized_vmax: Normalized upper bound of the colormap range
        :param nan_color: Color to use for NaN values
        """
        cdef double[:] values
        cdef image_types[:, ::1] lut
        cdef int type_min, type_max
        cdef unsigned int nb_channels, length, channel
        cdef int index, lut_index

        length = data.size
        nb_channels = colors.shape[1]

        if lut_types is cnumpy.int8_t:
            type_min = -128
            type_max = 127
        elif lut_types is cnumpy.uint8_t:
            type_min = 0
            type_max = 255
        elif lut_types is cnumpy.int16_t:
            type_min = -32768
            type_max = 32767
        else:  # uint16_t
            type_min = 0
            type_max = 65535

        values = numpy.arange(type_min, type_max + 1, dtype=numpy.float64)
        lut = numpy.empty((length, nb_channels),
                          dtype=numpy.array(colors, copy=False).dtype)
        self._cmap(lut, values, colors,
                   normalized_vmin, normalized_vmax, nan_color)

        with nogil:
            # Apply LUT
            for index in prange(length):
                lut_index = data[index] - type_min
                for channel in range(nb_channels):
                    output[index, channel] = lut[lut_index, channel]


DEF LOG_LUT_SIZE = 4096

cdef class ColormapLog(Colormap):
    """Class for processing of log normalized colormap."""

    # Size +1 as index_lut can overflow of 1
    cdef readonly double _log_lut[LOG_LUT_SIZE + 1]
    """LUT used for fast log approximation"""

    def __cinit__(self):
        # Initialize log approximation LUT
        self._log_lut = numpy.log2(
            numpy.linspace(0.5, 1., LOG_LUT_SIZE + 1,
                           endpoint=True, dtype=numpy.float64))
        # Handle  indexLUT == 1 overflow
        self._log_lut[LOG_LUT_SIZE] = self._log_lut[LOG_LUT_SIZE - 1]

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double normalize(self, double value) nogil:
        """Return log10(value) fast approximation based on LUT"""
        cdef double result = NAN  # if value < 0.0 or value == NAN
        cdef int exponent, index_lut
        cdef double mantissa  # in [0.5, 1) unless value == 0 NaN or +/-inf

        if value <= 0.0 or not isfinite(value):
            if value == 0.0:
                result = - HUGE_VAL
            elif value > 0.0:  # i.e., value = +INFINITY
                result = value  # i.e. +INFINITY
        else:
            mantissa = frexp(value, &exponent)
            index_lut = lrint(LOG_LUT_SIZE * 2 * (mantissa - 0.5))
            # 1/log2(10) = 0.30102999566398114
            result = 0.30102999566398114 * (<double> exponent +
                                            self._log_lut[index_lut])
        return result


cdef class ColormapArcsinh:
    """Class for processing of arcsinh normalized colormap."""

    def __cinit__(self):
        pass

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double normalize(self, double value) nogil:
        """Returns arcsinh(value)"""
        return asinh(value)


cdef class ColormapSqrt(Colormap):
    """Class for processing of sqrt normalized colormap."""

    def __cinit__(self):
        pass

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double normalize(self, double value) nogil:
        """Returns sqrt(value)"""
        return sqrt(value)


# Colormap objects to use for conversion
_colormaps = {
    'linear': Colormap(),
    'log': ColormapLog(),
    'arcsinh': ColormapArcsinh(),
    'sqrt': ColormapSqrt(),
}


def cmap(data,
         colors,
         double vmin,
         double vmax,
         str normalization='linear',
         nan_color=None):
    """Convert data to colors.

    :param numpy.ndarray data: The data to convert to colors
    :param numpy.ndarray colors: Color look-up table as a 2D array.
    :param vmin: Data value to map to the beginning of colormap.
    :param vmax: Data value to map to the end of the colormap.
    :param str normalization: The normalization to apply:
                              'linear' (default) or 'log'
    :param nan_color: Color to use for NaN value.
        Default: A color with all channels set to 0
    :return: The colors corresponding to data. The shape of the
        returned array is that of data array + the 2nd dimension of colors.
        The dtype of the returned array is that of the colors array.
    :rtype: numpy.ndarray
    """
    cdef int nb_channels

    assert normalization in _colormaps.keys()

    # Make data a numpy array of native endian type (no need for contiguity)
    data = numpy.array(data, copy=False)
    data = numpy.array(data, copy=False, dtype=data.dtype.newbyteorder('N'))

    # Make colors a contiguous array of native endian type
    colors = numpy.array(colors, copy=False)
    nb_channels = colors.shape[colors.ndim - 1]
    colors = numpy.ascontiguousarray(colors,
                                     dtype=colors.dtype.newbyteorder('N'))

    # Check nan_color
    if nan_color is None:
        nan_color = numpy.zeros((nb_channels,), dtype=colors.dtype)
    else:
        nan_color = numpy.ascontiguousarray(
            nan_color, dtype=colors.dtype).reshape(-1)
    assert nan_color.shape == (nb_channels,)

    # Allocate output image array
    image = numpy.empty(data.shape + (nb_channels,), dtype=colors.dtype)

    _colormaps[normalization].apply(
        image.reshape(-1, nb_channels),
        data.reshape(-1),
        colors.reshape(-1, nb_channels),
        vmin, vmax, nan_color)

    return image
