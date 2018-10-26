# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""This module provides median filter function for 1D and 2D arrays.
"""

__authors__ = ["H. Payno", "J. Kieffer"]
__license__ = "MIT"
__date__ = "02/05/2017"


from cython.parallel import prange
cimport cython
cimport silx.math.medianfilter.median_filter as median_filter
import numpy
cimport numpy as cnumpy
from libcpp cimport bool

import numbers

ctypedef unsigned long uint64
ctypedef unsigned int uint32
ctypedef unsigned short uint16


MODES = {'nearest': 0, 'reflect': 1, 'mirror': 2, 'shrink': 3, 'constant': 4}


def medfilt1d(data,
              kernel_size=3,
              bool conditional=False,
              mode='nearest',
              cval=0):
    """Function computing the median filter of the given input.

    Behavior at boundaries: the algorithm is reducing the size of the
    window/kernel for pixels at boundaries (there is no mirroring).

    Not-a-Number (NaN) float values are ignored.
    If the window only contains NaNs, it evaluates to NaN.

    In event of an even number of valid values in the window (either
    because of NaN values or on image border in shrink mode),
    the highest of the 2 central sorted values is taken.

    :param numpy.ndarray data: the array for which we want to apply
        the median filter. Should be 1d.
    :param kernel_size: the dimension of the kernel.
    :type kernel_size: int
    :param bool conditional: True if we want to apply a conditional median
        filtering.
    :param str mode: the algorithm used to determine how values at borders
        are determined: 'nearest', 'reflect', 'mirror', 'shrink', 'constant'
    :param cval: Value used outside borders in 'constant' mode

    :returns: the array with the median value for each pixel.
    """
    return medfilt(data, kernel_size, conditional, mode, cval)


def medfilt2d(image,
              kernel_size=3,
              bool conditional=False,
              mode='nearest',
              cval=0):
    """Function computing the median filter of the given input.
    Behavior at boundaries: the algorithm is reducing the size of the
    window/kernel for pixels at boundaries (there is no mirroring).

    Not-a-Number (NaN) float values are ignored.
    If the window only contains NaNs, it evaluates to NaN.

    In event of an even number of valid values in the window (either
    because of NaN values or on image border in shrink mode),
    the highest of the 2 central sorted values is taken.

    :param numpy.ndarray data: the array for which we want to apply
        the median filter. Should be 2d.
    :param kernel_size: the dimension of the kernel.
    :type kernel_size: For 1D should be an int for 2D should be a tuple or
        a list of (kernel_height, kernel_width)
    :param bool conditional: True if we want to apply a conditional median
        filtering.
    :param str mode: the algorithm used to determine how values at borders
        are determined: 'nearest', 'reflect', 'mirror', 'shrink', 'constant'
    :param cval: Value used outside borders in 'constant' mode

    :returns: the array with the median value for each pixel.
    """
    return medfilt(image, kernel_size, conditional, mode, cval)


def medfilt(data,
            kernel_size=3,
            bool conditional=False,
            mode='nearest',
            cval=0):
    """Function computing the median filter of the given input.
    Behavior at boundaries: the algorithm is reducing the size of the
    window/kernel for pixels at boundaries (there is no mirroring).

    Not-a-Number (NaN) float values are ignored.
    If the window only contains NaNs, it evaluates to NaN.

    In event of an even number of valid values in the window (either
    because of NaN values or on image border in shrink mode),
    the highest of the 2 central sorted values is taken.

    :param numpy.ndarray data: the array for which we want to apply
        the median filter. Should be 1d or 2d.
    :param kernel_size: the dimension of the kernel.
    :type kernel_size: For 1D should be an int for 2D should be a tuple or
        a list of (kernel_height, kernel_width)
    :param bool conditional: True if we want to apply a conditional median
        filtering.
    :param str mode: the algorithm used to determine how values at borders
        are determined: 'nearest', 'reflect', 'mirror', 'shrink', 'constant'
    :param cval: Value used outside borders in 'constant' mode

    :returns: the array with the median value for each pixel.
    """
    if mode not in MODES:
        err = 'Requested mode %s is unknown.' % mode
        raise ValueError(err)

    if data.ndim > 2:
        raise ValueError(
            "Invalid data shape. Dimension of the array should be 1 or 2")

    # Handle case of scalar kernel size
    if isinstance(kernel_size, numbers.Integral):
        kernel_size = [kernel_size] * data.ndim

    assert len(kernel_size) == data.ndim

    # Convert 1D arrays to 2D
    reshaped = False
    if len(data.shape) == 1:
        data = data.reshape(1, data.shape[0])
        kernel_size = [1, kernel_size[0]]
        reshaped = True

    # simple median filter apply into a 2D buffer
    output_buffer = numpy.zeros_like(data)
    check(data, output_buffer)

    ker_dim = numpy.array(kernel_size, dtype=numpy.int32)

    if data.dtype == numpy.float64:
        medfilterfc = _median_filter_float64
    elif data.dtype == numpy.float32:
        medfilterfc = _median_filter_float32
    elif data.dtype == numpy.int64:
        medfilterfc = _median_filter_int64
    elif data.dtype == numpy.uint64:
        medfilterfc = _median_filter_uint64
    elif data.dtype == numpy.int32:
        medfilterfc = _median_filter_int32
    elif data.dtype == numpy.uint32:
        medfilterfc = _median_filter_uint32
    elif data.dtype == numpy.int16:
        medfilterfc = _median_filter_int16
    elif data.dtype == numpy.uint16:
        medfilterfc = _median_filter_uint16
    else:
        raise ValueError("%s type is not managed by the median filter" % data.dtype)

    medfilterfc(input_buffer=data,
                output_buffer=output_buffer,
                kernel_size=ker_dim,
                conditional=conditional,
                mode=MODES[mode],
                cval=cval)

    if reshaped:
        output_buffer.shape = -1  # Convert to 1D array

    return output_buffer


def check(input_buffer, output_buffer):
    """Simple check on the two buffers to make sure we can apply the median filter
    """
    if (input_buffer.flags['C_CONTIGUOUS'] is False):
        raise ValueError('<input_buffer> must be a C_CONTIGUOUS numpy array.')

    if (output_buffer.flags['C_CONTIGUOUS'] is False):
        raise ValueError('<output_buffer> must be a C_CONTIGUOUS numpy array.')

    if not (len(input_buffer.shape) <= 2):
        raise ValueError('<input_buffer> dimension must mo higher than 2.')

    if not (len(output_buffer.shape) <= 2):
        raise ValueError('<output_buffer> dimension must mo higher than 2.')

    if not(input_buffer.dtype == output_buffer.dtype):
        raise ValueError('input buffer and output_buffer must be of the same type')

    if not (input_buffer.shape == output_buffer.shape):
        raise ValueError('input buffer and output_buffer must be of the same dimension and same dimension')


######### implementations of the include/median_filter.hpp function ############
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def reflect(int index, int length_max):
    """find the correct index into [0, length_max-1] for index in reflect mode

    :param int index: the index to move into [0, length_max-1] in reflect mode
    :param int length_max: the higher bound limit
    """
    return median_filter.reflect(index, length_max)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def mirror(int index, int length_max):
    """find the correct index into [0, length_max-1] for index in mirror mode

    :param int index: the index to move into [0, length_max-1] in mirror mode
    :param int length_max: the higher bound limit
    """
    return median_filter.mirror(index, length_max)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _median_filter_float32(float[:, ::1] input_buffer not None,
                           float[:, ::1] output_buffer not None,
                           cnumpy.int32_t[::1] kernel_size not None,
                           bool conditional,
                           int mode,
                           float cval):

    cdef:
        int y = 0
        int image_dim = input_buffer.shape[1] - 1
        int[2] buffer_shape
    buffer_shape[0] = input_buffer.shape[0]
    buffer_shape[1] = input_buffer.shape[1]

    for y in prange(input_buffer.shape[0], nogil=True):
            median_filter.median_filter[float](<float*> & input_buffer[0,0], 
                                               <float*> & output_buffer[0,0], 
                                               <int*>& kernel_size[0],
                                               <int*>buffer_shape,
                                               y,
                                               0,
                                               image_dim,
                                               conditional,
                                               mode,
                                               cval)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _median_filter_float64(double[:, ::1] input_buffer not None,
                           double[:, ::1] output_buffer not None,
                           cnumpy.int32_t[::1] kernel_size not None,
                           bool conditional,
                           int mode,
                           double cval):

    cdef:
        int y = 0
        int image_dim = input_buffer.shape[1] - 1
        int[2] buffer_shape
    buffer_shape[0] = input_buffer.shape[0]
    buffer_shape[1] = input_buffer.shape[1]

    for y in prange(input_buffer.shape[0], nogil=True):
            median_filter.median_filter[double](<double*> & input_buffer[0, 0], 
                                                <double*> & output_buffer[0, 0], 
                                                <int*>&kernel_size[0],
                                                <int*>buffer_shape,
                                                y,
                                                0,
                                                image_dim,
                                                conditional,
                                                mode,
                                                cval)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _median_filter_int64(cnumpy.int64_t[:, ::1] input_buffer not None,
                         cnumpy.int64_t[:, ::1] output_buffer not None,
                         cnumpy.int32_t[::1] kernel_size not None,
                         bool conditional,
                         int mode,
                         cnumpy.int64_t cval):

    cdef:
        int y = 0
        int image_dim = input_buffer.shape[1] - 1
        int[2] buffer_shape
    buffer_shape[0] = input_buffer.shape[0]
    buffer_shape[1] = input_buffer.shape[1]

    for y in prange(input_buffer.shape[0], nogil=True):
            median_filter.median_filter[long](<long*> & input_buffer[0,0], 
                                              <long*>  & output_buffer[0, 0], 
                                              <int*>&kernel_size[0],
                                              <int*>buffer_shape,
                                              y,
                                              0,
                                              image_dim,
                                              conditional,
                                              mode,
                                              cval)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _median_filter_uint64(cnumpy.uint64_t[:, ::1] input_buffer not None,
                          cnumpy.uint64_t[:, ::1] output_buffer not None,
                          cnumpy.int32_t[::1] kernel_size not None,
                          bool conditional,
                          int mode,
                          cnumpy.uint64_t cval):

    cdef: 
        int y = 0
        int image_dim = input_buffer.shape[1] - 1
        int[2] buffer_shape
    buffer_shape[0] = input_buffer.shape[0]
    buffer_shape[1] = input_buffer.shape[1]

    for y in prange(input_buffer.shape[0], nogil=True):
            median_filter.median_filter[uint64](<uint64*> & input_buffer[0,0], 
                                                <uint64*> & output_buffer[0, 0],
                                                <int*>&kernel_size[0],
                                                <int*>buffer_shape,
                                                y,
                                                0,
                                                image_dim,
                                                conditional,
                                                mode,
                                                cval)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _median_filter_int32(cnumpy.int32_t[:, ::1] input_buffer not None,
                         cnumpy.int32_t[:, ::1] output_buffer not None,
                         cnumpy.int32_t[::1] kernel_size not None,
                         bool conditional,
                         int mode,
                         cnumpy.int32_t cval):

    cdef:
        int y = 0
        int image_dim = input_buffer.shape[1] - 1
        int[2] buffer_shape
    buffer_shape[0] = input_buffer.shape[0]
    buffer_shape[1] = input_buffer.shape[1]

    for y in prange(input_buffer.shape[0], nogil=True):
            median_filter.median_filter[int](<int*> & input_buffer[0,0], 
                                             <int*>  & output_buffer[0, 0],
                                             <int*>&kernel_size[0],
                                             <int*>buffer_shape,
                                             y,
                                             0,
                                             image_dim,
                                             conditional,
                                             mode,
                                             cval)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _median_filter_uint32(cnumpy.uint32_t[:, ::1] input_buffer not None,
                          cnumpy.uint32_t[:, ::1] output_buffer not None,
                          cnumpy.int32_t[::1] kernel_size not None,
                          bool conditional,
                          int mode,
                          cnumpy.uint32_t cval):

    cdef:
        int y = 0
        int image_dim = input_buffer.shape[1] - 1
        int[2] buffer_shape
    buffer_shape[0] = input_buffer.shape[0]
    buffer_shape[1] = input_buffer.shape[1]

    for y in prange(input_buffer.shape[0], nogil=True):
            median_filter.median_filter[uint32](<uint32*> & input_buffer[0,0], 
                                                <uint32*>  & output_buffer[0, 0],
                                                <int*>&kernel_size[0],
                                                <int*>buffer_shape,
                                                y,
                                                0,
                                                image_dim,
                                                conditional,
                                                mode,
                                                cval)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _median_filter_int16(cnumpy.int16_t[:, ::1] input_buffer not None,
                         cnumpy.int16_t[:, ::1] output_buffer not None,
                         cnumpy.int32_t[::1] kernel_size not None,
                         bool conditional,
                         int mode,
                         cnumpy.int16_t cval):

    cdef:
        int y = 0
        int image_dim = input_buffer.shape[1] - 1
        int[2] buffer_shape
    buffer_shape[0] = input_buffer.shape[0]
    buffer_shape[1] = input_buffer.shape[1]

    for y in prange(input_buffer.shape[0], nogil=True):
            median_filter.median_filter[short](<short*> & input_buffer[0,0], 
                                               <short*>  & output_buffer[0, 0],
                                               <int*>&kernel_size[0],
                                               <int*>buffer_shape,
                                               y,
                                               0,
                                               image_dim,
                                               conditional,
                                               mode,
                                               cval)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _median_filter_uint16(
      cnumpy.uint16_t[:, ::1] input_buffer not None,
      cnumpy.uint16_t[:, ::1] output_buffer not None,
      cnumpy.int32_t[::1] kernel_size not None,
      bool conditional,
      int mode,
      cnumpy.uint16_t cval):

    cdef:
        int y = 0
        int image_dim = input_buffer.shape[1] - 1
        int[2] buffer_shape, 
    buffer_shape[0] = input_buffer.shape[0]
    buffer_shape[1] = input_buffer.shape[1]

    for y in prange(input_buffer.shape[0], nogil=True):
            median_filter.median_filter[uint16](<uint16*> & input_buffer[0, 0],
                                                <uint16*> & output_buffer[0, 0],
                                                <int*>&kernel_size[0],
                                                <int*>buffer_shape,
                                                y,
                                                0,
                                                image_dim,
                                                conditional,
                                                mode,
                                                cval)
