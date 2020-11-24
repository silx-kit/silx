# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
"""This module provides :func:`cmap` which applies a colormap to a dataset.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "16/05/2018"


import os
cimport cython
from cython.parallel import prange
cimport numpy as cnumpy
from libc.math cimport frexp, sinh, sqrt
from .math_compatibility cimport asinh, isnan, isfinite, lrint, INFINITY, NAN

import logging
import numbers

import numpy

__all__ = ['cmap']

_logger = logging.getLogger(__name__)


cdef int DEFAULT_NUM_THREADS
if hasattr(os, 'sched_getaffinity'):
    DEFAULT_NUM_THREADS = min(4, len(os.sched_getaffinity(0)))
elif os.cpu_count() is not None:
    DEFAULT_NUM_THREADS = min(4, os.cpu_count())
else:  # Fallback
    DEFAULT_NUM_THREADS = 1
# Number of threads to use for the computation (initialized to up to 4)

cdef int USE_OPENMP_THRESHOLD = 1000
"""OpenMP is not used for arrays with less elements than this threshold"""

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


# Normalization

ctypedef double (*NormalizationFunction)(double) nogil


cdef class Normalization:
    """Base class for colormap normalization"""

    def apply(self, data, double vmin, double vmax):
        """Apply normalization.

        :param Union[float,numpy.ndarray] data:
        :param float vmin: Lower bound of the range
        :param float vmax: Upper bound of the range
        :rtype: Union[float,numpy.ndarray]
        """
        cdef int length
        cdef double[:] result

        if isinstance(data, numbers.Real):
            return self.apply_double(<double> data, vmin, vmax)
        else:
            data = numpy.array(data, copy=False)
            length = <int> data.size
            result = numpy.empty(length, dtype=numpy.float64)
            data1d = numpy.ravel(data)
            for index in range(length):
                result[index] = self.apply_double(
                    <double> data1d[index], vmin, vmax)
            return numpy.array(result).reshape(data.shape)

    def revert(self, data, double vmin, double vmax):
        """Revert normalization.

        :param Union[float,numpy.ndarray] data:
        :param float vmin: Lower bound of the range
        :param float vmax: Upper bound of the range
        :rtype: Union[float,numpy.ndarray]
        """
        cdef int length
        cdef double[:] result

        if isinstance(data, numbers.Real):
            return self.revert_double(<double> data, vmin, vmax)
        else:
            data = numpy.array(data, copy=False)
            length = <int> data.size
            result = numpy.empty(length, dtype=numpy.float64)
            data1d = numpy.ravel(data)
            for index in range(length):
                result[index] = self.revert_double(
                    <double> data1d[index], vmin, vmax)
            return numpy.array(result).reshape(data.shape)

    cdef double apply_double(self, double value, double vmin, double vmax) nogil:
        """Apply normalization to a floating point value

        Override in subclass

        :param float value:
        :param float vmin: Lower bound of the range
        :param float vmax: Upper bound of the range
        """
        return value

    cdef double revert_double(self, double value, double vmin, double vmax) nogil:
        """Apply inverse of normalization to a floating point value

        Override in subclass

        :param float value:
        :param float vmin: Lower bound of the range
        :param float vmax: Upper bound of the range
        """
        return value


cdef class LinearNormalization(Normalization):
    """Linear normalization"""

    cdef double apply_double(self, double value, double vmin, double vmax) nogil:
        return value

    cdef double revert_double(self, double value, double vmin, double vmax) nogil:
        return value


cdef class LogarithmicNormalization(Normalization):
    """Logarithmic normalization using a fast log approximation"""
    cdef:
        readonly int lutsize
        readonly double[::1] lut # LUT used for fast log approximation

    def __cinit__(self, int lutsize=4096):
        # Initialize log approximation LUT
        self.lutsize = lutsize
        self.lut = numpy.log2(
            numpy.linspace(0.5, 1., lutsize + 1,
                           endpoint=True).astype(numpy.float64))
        # index_lut can overflow of 1
        self.lut[lutsize] = self.lut[lutsize - 1]

    def __dealloc__(self):
        self.lut = None

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef double apply_double(self, double value, double vmin, double vmax) nogil:
        """Return log10(value) fast approximation based on LUT"""
        cdef double result = NAN  # if value < 0.0 or value == NAN
        cdef int exponent, index_lut
        cdef double mantissa  # in [0.5, 1) unless value == 0 NaN or +/-inf

        if value <= 0.0 or not isfinite(value):
            if value == 0.0:
                result = - INFINITY
            elif value > 0.0:  # i.e., value = +INFINITY
                result = value  # i.e. +INFINITY
        else:
            mantissa = frexp(value, &exponent)
            index_lut = lrint(self.lutsize * 2 * (mantissa - 0.5))
            # 1/log2(10) = 0.30102999566398114
            result = 0.30102999566398114 * (<double> exponent +
                                            self.lut[index_lut])
        return result

    cdef double revert_double(self, double value, double vmin, double vmax) nogil:
        return 10**value


cdef class ArcsinhNormalization(Normalization):
    """Inverse hyperbolic sine normalization"""

    cdef double apply_double(self, double value, double vmin, double vmax) nogil:
        return asinh(value)

    cdef double revert_double(self, double value, double vmin, double vmax) nogil:
        return sinh(value)


cdef class SqrtNormalization(Normalization):
    """Square root normalization"""

    cdef double apply_double(self, double value, double vmin, double vmax) nogil:
        return sqrt(value)

    cdef double revert_double(self, double value, double vmin, double vmax) nogil:
        return value**2


cdef class PowerNormalization(Normalization):
    """Gamma correction:

    Linear normalization to [0, 1] followed by power normalization.

    :param gamma: Gamma correction factor
    """

    cdef:
        readonly double gamma

    def __cinit__(self, double gamma):
        self.gamma = gamma

    def __init__(self, gamma):
        # Needed for multiple inheritance to work
        pass

    cdef double apply_double(self, double value, double vmin, double vmax) nogil:
        if vmin == vmax:
            return 0.
        elif value <= vmin:
            return 0.
        elif value >= vmax:
            return 1.
        else:
            return ((value - vmin) / (vmax - vmin))**self.gamma

    cdef double revert_double(self, double value, double vmin, double vmax) nogil:
        if value <= 0.:
            return vmin
        elif value >= 1.:
            return vmax
        else:
            return vmin + (vmax - vmin) * value**(1.0/self.gamma)


# Colormap

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef image_types[:, ::1] compute_cmap(
           default_types[:] data,
           image_types[:, ::1] colors,
           Normalization normalization,
           double vmin,
           double vmax,
           image_types[::1] nan_color):
    """Apply colormap to data.

    :param data: Input data
    :param colors: Colors look-up-table
    :param vmin: Lower bound of the colormap range
    :param vmax: Upper bound of the colormap range
    :param nan_color: Color to use for NaN value
    :param normalization: Normalization to apply
    :return: Data converted to colors
    """
    cdef image_types[:, ::1] output
    cdef double scale, value, normalized_vmin, normalized_vmax
    cdef int length, nb_channels, nb_colors
    cdef int channel, index, lut_index, num_threads

    nb_colors = <int> colors.shape[0]
    nb_channels = <int> colors.shape[1]
    length = <int> data.size

    output = numpy.empty((length, nb_channels),
                         dtype=numpy.array(colors, copy=False).dtype)

    normalized_vmin = normalization.apply_double(vmin, vmin, vmax)
    normalized_vmax = normalization.apply_double(vmax, vmin, vmax)

    if not isfinite(normalized_vmin) or not isfinite(normalized_vmax):
        raise ValueError('Colormap range is not valid')

    if normalized_vmin == normalized_vmax:
        scale = 0.
    else:
        scale = nb_colors / (normalized_vmax - normalized_vmin)

    if length < USE_OPENMP_THRESHOLD:
        num_threads = 1
    else:
        num_threads = min(
            DEFAULT_NUM_THREADS,
            int(os.environ.get("OMP_NUM_THREADS", DEFAULT_NUM_THREADS)))

    with nogil:
        for index in prange(length, num_threads=num_threads):
            value = normalization.apply_double(
                <double> data[index], vmin, vmax)

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
                # Index can overflow of 1
                if lut_index >= nb_colors:
                    lut_index = nb_colors - 1

            for channel in range(nb_channels):
                output[index, channel] = colors[lut_index, channel]

    return output

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef image_types[:, ::1] compute_cmap_with_lut(
               lut_types[:] data,
               image_types[:, ::1] colors,
               Normalization normalization,
               double vmin,
               double vmax,
               image_types[::1] nan_color):
    """Convert data to colors using look-up table to speed the process.

    Only supports data of types: uint8, uint16, int8, int16.

    :param data: Input data
    :param colors: Colors look-up-table
    :param vmin: Lower bound of the colormap range
    :param vmax: Upper bound of the colormap range
    :param nan_color: Color to use for NaN values
    :param normalization: Normalization to apply
    :return: The generated image
    """
    cdef image_types[:, ::1] output
    cdef double[:] values
    cdef image_types[:, ::1] lut
    cdef int type_min, type_max
    cdef int nb_channels, length
    cdef int channel, index, lut_index, num_threads

    length = <int> data.size
    nb_channels = <int> colors.shape[1]

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

    colors_dtype = numpy.array(colors).dtype

    values = numpy.arange(type_min, type_max + 1, dtype=numpy.float64)
    lut = compute_cmap(
        values, colors, normalization, vmin, vmax, nan_color)

    output = numpy.empty((length, nb_channels), dtype=colors_dtype)

    if length < USE_OPENMP_THRESHOLD:
        num_threads = 1
    else:
        num_threads = min(
            DEFAULT_NUM_THREADS,
            int(os.environ.get("OMP_NUM_THREADS", DEFAULT_NUM_THREADS)))

    with nogil:
        # Apply LUT
        for index in prange(length, num_threads=num_threads):
            lut_index = data[index] - type_min
            for channel in range(nb_channels):
                output[index, channel] = lut[lut_index, channel]

    return output


# Normalizations without parameters
_BASIC_NORMALIZATIONS = {
    'linear': LinearNormalization(),
    'log': LogarithmicNormalization(),
    'arcsinh': ArcsinhNormalization(),
    'sqrt': SqrtNormalization(),
    }


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _cmap(data_types[:] data,
          image_types[:, ::1] colors,
          Normalization normalization,
          double vmin,
          double vmax,
          image_types[::1] nan_color):
    """Implementation of colormap.

    Use :func:`cmap`.

    :param data: Input data
    :param colors: Colors look-up-table
    :param normalization: Normalization object to apply
    :param vmin: Lower bound of the colormap range
    :param vmax: Upper bound of the colormap range
    :param nan_color: Color to use for NaN value.
    :return: The generated image
    """
    cdef image_types[:, ::1] output

    # Proxy for calling the right implementation depending on data type
    if data_types in lut_types:  # Use LUT implementation
        output = compute_cmap_with_lut(
            data, colors, normalization, vmin, vmax, nan_color)

    elif data_types in default_types:  # Use default implementation
        output = compute_cmap(
            data, colors, normalization, vmin, vmax, nan_color)

    else:
        raise ValueError('Unsupported data type')

    return numpy.array(output, copy=False)


def cmap(data,
         colors,
         double vmin,
         double vmax,
         normalization='linear',
         nan_color=None):
    """Convert data to colors with provided colors look-up table.

    :param numpy.ndarray data: The input data
    :param numpy.ndarray colors: Color look-up table as a 2D array.
       It MUST be of type uint8 or float32
    :param vmin: Data value to map to the beginning of colormap.
    :param vmax: Data value to map to the end of the colormap.
    :param Union[str,Normalization] normalization:
        Either a :class:`Normalization` instance or a str in:

        - 'linear' (default)
        - 'log'
        - 'arcsinh'
        - 'sqrt'
        - 'gamma'

    :param nan_color: Color to use for NaN value.
        Default: A color with all channels set to 0
    :return: Array of colors. The shape of the
        returned array is that of data array + the last dimension of colors.
        The dtype of the returned array is that of the colors array.
    :rtype: numpy.ndarray
    """
    cdef int nb_channels
    cdef Normalization norm

    # Make data a numpy array of native endian type (no need for contiguity)
    data = numpy.array(data, copy=False)
    native_endian_dtype = data.dtype.newbyteorder('N')
    if native_endian_dtype.kind == 'f' and native_endian_dtype.itemsize == 2:
        native_endian_dtype = "=f4"  # Use native float32 instead of float16
    data = numpy.array(data, copy=False, dtype=native_endian_dtype)

    # Make colors a contiguous array of native endian type
    colors = numpy.array(colors, copy=False)
    nb_channels = colors.shape[colors.ndim - 1]
    colors = numpy.ascontiguousarray(colors,
                                     dtype=colors.dtype.newbyteorder('N'))

    # Make normalization a Normalization object
    if isinstance(normalization, str):
        norm = _BASIC_NORMALIZATIONS.get(normalization, None)
        if norm is None:
            raise ValueError('Unsupported normalization %s' % normalization)
    else:
        norm = normalization

    # Check nan_color
    if nan_color is None:
        nan_color = numpy.zeros((nb_channels,), dtype=colors.dtype)
    else:
        nan_color = numpy.ascontiguousarray(
            nan_color, dtype=colors.dtype).reshape(-1)
    assert nan_color.shape == (nb_channels,)

    image = _cmap(
        data.reshape(-1),
        colors.reshape(-1, nb_channels),
        norm,
        vmin,
        vmax,
        nan_color)
    image.shape = data.shape + (nb_channels,)

    return image
