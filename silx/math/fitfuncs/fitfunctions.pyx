# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
""" """

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "14/06/2016"

import logging
import numpy

logging.basicConfig()
_logger = logging.getLogger(__name__)

cimport cython

# fitfunctions.pxd
from fitfunctions cimport snip1d as _snip1d
from fitfunctions cimport snip2d as _snip2d
from fitfunctions cimport snip3d as _snip3d
from fitfunctions cimport sum_gauss as _sum_gauss


def snip1d(data, int width):
    """snip1d(data, width) -> numpy.ndarray
    Estimate the baseline (background) of a 1D data vector by clipping peaks.

    The implementation of the algorithm SNIP in 1D is described in *Miroslav
    Morhac et al. Nucl. Instruments and Methods in Physics Research A401
    (1997) 113-132*.

    The original idea for 1D and the low-statistics-digital-filter (lsdf) come
    from *C.G. Ryan et al. Nucl. Instruments and Methods in Physics Research
    B34 (1988) 396-402*.

    :param data: Data array, preferably 1D and of type *numpy.float64*.
        Else, the data array will be flattened and converted to
        *dtype=numpy.float64* prior to applying the snip filter.
    :type data: numpy.ndarray
    :param width: Width of the snip operator, in number of samples. A wider
        snip operator will result in a smoother result (lower frequency peaks
        will be clipped), and a longer computation time.
    :type width: int
    :return: Baseline of the input array, as an array of the same shape.
    :rtype: numpy.ndarray
    """
    cdef double[::1] data_c

    # Ensure we are dealing with a 1D array in contiguous memory
    data_c = numpy.array(data, copy=False, dtype=numpy.float64, order='C').reshape(-1)

    _snip1d(&data_c[0], data.size, width)

    return numpy.asarray(data_c).reshape(data.shape)


def snip2d(data, int width, nrows=None, ncolumns=None):
    """snip2d(data, width, nrows=None, ncolumns=None) -> numpy.ndarray
    Estimate the baseline (background) of a 2D data signal by clipping peaks.

    Implementation of the algorithm SNIP in 2D described in
    *Miroslav Morhac et al. Nucl. Instruments and Methods in Physics Research
    A401 (1997) 113-132.*

    :param data: Data array, preferably 1D and of type *numpy.float64*.
        Else, the data array will be flattened and converted to
        *dtype=numpy.float64* prior to applying the snip filter.
        If the data is a 2D array, ``nrows`` and ``ncolumns`` don't
        need to be specified.
    :type data: numpy.ndarray
    :param width: Width of the snip operator, in number of samples. A wider
        snip operator will result in a smoother result (lower frequency peaks
        will be clipped), and a longer computation time.
    :type width: int
    :param nrows: Number of rows (second dimension) in array.
        If ``None``, it will be inferred from the shape of the data if it
        is a 2D array.
    :type nrows: int or None
    :param ncolumns: Number of columns (first dimension) in array
        If ``None``, it will be inferred from the shape of the data if it
        is a 2D array.
    :type ncolumns: int or None
    :return: Baseline of the input array, as an array of the same shape.
    :rtype: numpy.ndarray
    """
    cdef double[::1] data_c


    if nrows is None or ncolumns is None:
        if len(data.shape) == 2:
            nrows, ncolumns = data.shape
        else:
            raise TypeError("nrows and ncolumns must both be specified " +
                            "if the data array is not 2D.")

    # Convert data to a 1D array in contiguous memory
    data_c = numpy.array(data, copy=False, dtype=numpy.float64, order='C').reshape(-1)

    _snip2d(&data_c[0], nrows, ncolumns, width)

    return numpy.asarray(data_c).reshape(data.shape)


def snip3d(data, int width, nx=None, ny=None, nz=None):
    """snip3d(data, width, nx=None, ny=None, nz=None) -> numpy.ndarray
    Estimate the baseline (background) of a 3D data signal by clipping peaks.

    Implementation of the algorithm SNIP in 2D described in
    *Miroslav Morhac et al. Nucl. Instruments and Methods in Physics Research
    A401 (1997) 113-132.*

    :param data: Data array, preferably 1D and of type *numpy.float64*.
        Else, the data array will be flattened and converted to
        *dtype=numpy.float64* prior to applying the snip filter.
        If the data is a 3D array, arguments ``nx``, ``ny`` and ``nz`` can
        be omitted.
    :type data: numpy.ndarray
    :param width: Width of the snip operator, in number of samples. A wider
        snip operator will result in a smoother result (lower frequency peaks
        will be clipped), and a longer computation time.
    :type width: int
    :param nx: Size of first dimension in array.
        If ``None``, it can be inferred from the shape of the data if it
        is a 3D array.
    :type nx: int or None
    :param ny: Size of second dimension in array.
        If ``None``, it can be inferred from the shape of the data if it
        is a 3D array.
    :type ny: int or None
    :param nz: Size of third dimension in array.
        If ``None``, it can be inferred from the shape of the data if it
        is a 3D array.
    :type ny: int or None
    :return: Baseline of the input array, as an array of the same shape.
    :rtype: numpy.ndarray
    """
    cdef double[::1] data_c


    if nx is None or ny is None or nz is None:
        if len(data.shape) == 3:
            nx, ny, nz = data.shape
        else:
            raise TypeError("nx, ny and nz must all be specified " +
                            "if the data array is not 3D.")

    # Convert data to a 1D array in contiguous memory
    data_c = numpy.array(data, copy=False, dtype=numpy.float64, order='C').reshape(-1)

    _snip3d(&data_c[0], nx, ny, nz, width)

    return numpy.asarray(data_c).reshape(data.shape)


def sum_gauss(x, *params):
    """gauss(x, *params) -> numpy.ndarray

    Return a sum of gaussian functions.

    :param x: Independant variable where the gaussians are calculated
    :type x: 1D numpy.ndarray
    :param params: Array of gaussian parameters (length must be a multiple
        of 3):
        *(height1, centroid1, fwhm1, height2, centroid2, fwhm2,...)*
    :return: Array of sum of gaussian functions at each ``x`` coordinates.
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)


    _sum_gauss(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    # cdef numpy.ndarray ret_array = numpy.empty((x.size,),
    #                                            dtype=numpy.float64)
    # for i in range(x.size):
    #     ret_array[i] = y_c[i]

    # free(y_c)
    return numpy.asarray(y_c).reshape(x.shape)

