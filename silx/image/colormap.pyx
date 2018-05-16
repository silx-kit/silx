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
__date__ = "02/03/2018"


# TODO make _cmap work with integers
# TODO nanColor with if type in cython.floating: handle nan
# TODO test
# TODO compare result to mpl
# TODO if only RGBA8888 is supported, copy color as a int32 instead of 4 char, probably faster

cimport cython
from cython.parallel import prange
cimport numpy as cnumpy
from libc.math cimport lrint, HUGE_VAL, isfinite, frexp, NAN, asinh

from silx.math.combo import min_max

import logging
import numpy


_logger = logging.getLogger(__name__)


# Types using a LUT to apply the colormap
ctypedef fused _lut_types:
    cnumpy.uint8_t
    cnumpy.int8_t
    cnumpy.uint16_t
    cnumpy.int16_t


# Supported data types
ctypedef fused _data_types:
    cnumpy.uint8_t
    cnumpy.int8_t
    cnumpy.uint16_t
    cnumpy.int16_t
    #cnumpy.uint32_t
    #cnumpy.int32_t
    #cnumpy.uint64_t
    #cnumpy.int64_t
    float
    double
    long double


# Supported colors/output types
ctypedef fused _image_types:
    cnumpy.uint8_t
    # cnumpy.int8_t
    # cnumpy.uint16_t
    # cnumpy.int16_t
    # cnumpy.uint32_t
    # cnumpy.int32_t
    # cnumpy.uint64_t
    # cnumpy.int64_t
    float
    # double
    # long double


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

        :param value: Value to normalize
        :return: Normalized value
        """
        return value

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    def apply(self,
              _image_types[:, ::1] output,
              cython.floating[:] data,
              _image_types[:, ::1] colors,
              double vmin,
              double vmax):
        """Apply colormap to data.

        :param output: Memory view where to store the result
        :param data: Input data
        :param colors: Colors look-up-table
        :param vmin: Lower bound of the colormap range
        :param vmax: Upper bound of the colormap range
        """
        cdef double scale, normed_vmin
        cdef unsigned int length, channel, nb_channels, nb_colors
        cdef int index, lut_index

        nb_colors = colors.shape[0]
        nb_channels = colors.shape[1]
        length = data.size

        normed_vmin = self.normalize(vmin)

        if vmin == vmax:
            scale = 0.
        else:
            # TODO check this
            #scale = (nb_colors - 1) / (vmax - vmin)
            scale = nb_colors / (self.normalize(vmax) - normed_vmin)

        with nogil:
            for index in prange(length):
                if data[index] <= vmin:
                    lut_index = 0
                elif data[index] >= vmax:
                    lut_index = nb_colors - 1
                else:
                    lut_index = <int>((self.normalize(data[index]) - normed_vmin) * scale)
                    # Safety net, duplicate previous checks
                    # TODO needed?
                    if lut_index < 0:
                        lut_index = 0
                    elif lut_index >= nb_colors:
                        lut_index = nb_colors - 1

                for channel in range(nb_channels):
                    output[index, channel] = colors[lut_index, channel]


DEF LOG_LUT_SIZE = 4096

cdef class ColormapLog(Colormap):
    """Class for processing of log normalized colormap."""

    cdef readonly double _log_lut[LOG_LUT_SIZE + 1]  # index_lut can overflow of 1 !
    """LUT used for fast log approximation"""

    def __cinit__(self):
        # Initialize log approximation LUT
        self._log_lut = numpy.log2(
            numpy.linspace(0.5, 1., LOG_LUT_SIZE + 1, endpoint=True, dtype=numpy.float64))
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
            result = 0.30102999566398114 * (<double> exponent + self._log_lut[index_lut])
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


_colormaps = {
    'linear': Colormap(),
    'log': ColormapLog(),
    'arcsinh': ColormapArcsinh()
}


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _cmap_lut(
    _image_types[:, ::1] output,
    _lut_types[:] data,
    _image_types[:, ::1] colors,
    double vmin,
    double vmax,
    str normalization):
    """Convert data to colors using look-up table to speed the process.

    Only supports data of type: uint8, uint16, int8, int16.
    """
    cdef float[:] values
    cdef _image_types[:, ::1] lut
    cdef int type_min, type_max
    cdef unsigned int nb_channels, length, channel
    cdef int index
    cdef _lut_types lut_index

    length = data.size
    nb_channels = colors.shape[1]

    if _lut_types is cnumpy.int8_t:
        type_min = -128
        type_max = 127
    elif _lut_types is cnumpy.uint8_t:
        type_min = 0
        type_max = 255
    elif _lut_types is cnumpy.int16_t:
        type_min = -32768
        type_max = 32767
    else:  # uint16_t
        type_min = 0
        type_max = 65535

    values = numpy.arange(type_min, type_max + 1, dtype=numpy.float32)
    lut = numpy.empty((length, nb_channels), dtype=numpy.array(colors, copy=False).dtype)
    _colormaps[normalization].apply(lut, values, colors, vmin, vmax)

    with nogil:
        # Apply LUT
        for index in prange(length):
            lut_index = data[index] - type_min
            for channel in range(nb_channels):
                output[index, channel] = lut[lut_index, channel]


def _cmap_proxy(
    _image_types[:, ::1] output,
    _data_types[:] data,
    _image_types[:, ::1] colors,
    double vmin,
    double vmax,
    str normalization):
    """Proxy for calling the right implementation depending on data type"""
    if _data_types in _lut_types:  # Use LUT implementation
        _cmap_lut(output, data, colors, vmin, vmax, normalization)

    elif _data_types in cython.floating:  # Use float implementation
        _colormaps[normalization].apply(output, data, colors, vmin, vmax)

    else:
        raise NotImplementedError() #TODO (u)int32|64


def cmap(data,
         colors,
         vmin=None,
         vmax=None,
         str normalization='linear'):
    """Convert data to colors.

    :param numpy.ndarray data: The data to convert to colors
    :param numpy.ndarray colors: Color look-up table as a 2D array.
    :param vmin:
        Data value to map to the beginning of colormap.
        Default: Min of the dataset.
    :param vmax:
        Data value to map to the end of the colormap.
        Default: Max of the dataset.
    :param str normalization: The normalization to apply:
                              'linear' (default) or 'log'
    :return: The colors corresponding to data. The shape of the
        returned array is that of data array + the 2nd dimension of colors.
        The dtype of the returned array is that of the colors array.
    :rtype: numpy.ndarray
    """
    cdef int nb_channels

    assert normalization in _colormaps.keys()

    # Make sure data is a numpy array (no need for a contiguous array)
    # TODO check if endianness is an issue
    data = numpy.array(data, copy=False)

    # Make colors a contiguous array (and take care of endianness)
    colors = numpy.array(colors, copy=False)
    nb_channels = colors.shape[colors.ndim - 1]
    colors = numpy.ascontiguousarray(colors,
                                     dtype=colors.dtype.newbyteorder('N'))

    # Allocate output image array
    image = numpy.empty(data.shape + (nb_channels,), dtype=colors.dtype)

    # Init vmin, vmax if not set
    # TODO improve + check fallback with log with only nan, only inf, etc...
    if vmin is None or vmax is None:
        if vmin is None and vmax is None:
            if normalization == 'log':
                result = min_max(data, min_positive=True)
                vmin = result.min_positive
                vmax = result.maximum
                if vmin is None:  # Only negative data
                    _logger.warning(
                        'Only negative data, auto min/max error for log scale')
                    vmin, vmax = 1., 1.
            else:
                result = min_max(data, min_positive=False)
                vmin = result.minimum
                vmax = result.maximum

        elif vmin is None:
            if normalization == 'log':
                positive_data = data[data > 0]
                if positive_data.size > 0:
                    vmin = numpy.nanmin(positive_data)
                else:
                    _logger.warning(
                        'Only negative data, auto min error for log scale')
                    vmin = vmax
            else:
                vmin = numpy.nanmin(data)

        elif vmax is None:
            vmax = numpy.nanmax(data)

    _cmap_proxy(image.reshape(-1, nb_channels),
                data.reshape(-1),
                colors.reshape(-1, nb_channels),
                vmin, vmax, normalization)

    return image
