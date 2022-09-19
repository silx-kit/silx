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
# ############################################################################*/
"""This module provides :func:`interp3d` to perform trilinear interpolation.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "11/07/2019"


import cython
from cython.parallel import prange
import numpy

cimport cython
from libc.math cimport floor
cimport numpy as cnumpy


ctypedef fused _floating:
    float
    double

ctypedef fused _floating_pts:
    float
    double


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double trilinear_interpolation(
    _floating[:, :, :] values,
    _floating_pts pos0,
    _floating_pts pos1,
    _floating_pts pos2,
    double fill_value) nogil:
    """Evaluate the trilinear interpolation at a given position

    :param values: 3D dataset from which to do the interpolation
    :param pos0: Dimension 0 coordinate at which to evaluate the interpolation
    :param pos1: Dimension 1 coordinate at which to evaluate the interpolation
    :param pos2: Dimension 2 coordinate at which to evaluate the interpolation
    :param fill_value: Value to return for points outside data
    """
    cdef:
        int i0, i1, i2  # Indices
        int i0_plus1, i1_plus1, i2_plus1  # Indices+1
        double delta
        double c00, c01, c10, c11, c0, c1
        double c

    if (pos0 < 0. or pos0 > (values.shape[0] -1) or
            pos1 < 0. or pos1 > (values.shape[1] -1) or
            pos2 < 0. or pos2 > (values.shape[2] -1)):
        return fill_value

    i0 = < int > floor(pos0)
    i1 = < int > floor(pos1)
    i2 = < int > floor(pos2)

    # Clip i+1 indices to data volume
    # In this case, corresponding dX is 0.
    i0_plus1 = min(i0 + 1, values.shape[0] - 1)
    i1_plus1 = min(i1 + 1, values.shape[1] - 1)
    i2_plus1 = min(i2 + 1, values.shape[2] - 1)

    if pos2 == i2:  # Avoids multiplication by 0 (which yields to NaN with inf)
        c00 = <double> values[i0, i1, i2]
        c10 = <double> values[i0, i1_plus1, i2]
        c01 = <double> values[i0_plus1, i1, i2]
        c11 = <double> values[i0_plus1, i1_plus1, i2]
    else:
        delta = pos2 - i2
        c00 = (<double> values[i0, i1, i2]) * (1. - delta) + (<double> values[i0, i1, i2_plus1]) * delta
        c10 = (<double> values[i0, i1_plus1, i2]) * (1. - delta) + (<double> values[i0, i1_plus1, i2_plus1]) * delta
        c01 = (<double> values[i0_plus1, i1, i2]) * (1. - delta) + (<double> values[i0_plus1, i1, i2_plus1]) * delta
        c11 = (<double> values[i0_plus1, i1_plus1, i2]) * (1. - delta) + (<double> values[i0_plus1, i1_plus1, i2_plus1]) * delta

    if pos1 == i1:  # Avoids multiplication by 0 (which yields to NaN with inf)
        c0 = c00
        c1 = c01
    else:
        delta = pos1 - i1
        c0 = c00 * (1. - delta) + c10 * delta
        c1 = c01 * (1. - delta) + c11 * delta

    if pos0 == i0:  # Avoids multiplication by 0 (which yields to NaN with inf)
        c = c0
    else:
        delta = pos0 - i0
        c = c0 * (1 - delta) + c1 * delta

    return c


@cython.boundscheck(False)
@cython.wraparound(False)
def interp3d(_floating[:, :, :] values not None,
             _floating_pts[:, :] xi not None,
             str method='linear',
             double fill_value=numpy.nan):
    """Trilinear interpolation in a regular grid.

    Perform trilinear interpolation of the 3D dataset at given points

    :param numpy.ndarray values: 3D dataset of floating point values
    :param numpy.ndarray xi: (N, 3) sampling points
    :param str method: Interpolation method to use in:
        - 'linear': Trilinear interpolation
        - 'linear_omp': Trilinear interpolation with OpenMP parallelism
    :param float fill_value:
        Value to use for points outside the volume (default: nan)
    :return: Values evaluated at given input points.
    :rtype: numpy.ndarray
    """
    if _floating is cnumpy.float32_t:
        dtype = numpy.float32
    elif _floating is cnumpy.float64_t:
        dtype = numpy.float64
    else:  # This should not happen
        raise ValueError("Unsupported input dtype")

    cdef:
        int npoints = xi.shape[0]
        _floating[:] result = numpy.empty((npoints,), dtype=dtype)
        int index
        double c_fill_value = fill_value

    if method == 'linear':
        with nogil:
            for index in range(npoints):
                result[index] = < _floating > trilinear_interpolation(
                    values, xi[index, 0], xi[index, 1], xi[index, 2], c_fill_value)

    elif method == 'linear_omp':
        for index in prange(npoints, nogil=True):
            result[index] = < _floating > trilinear_interpolation(
                values, xi[index, 0], xi[index, 1], xi[index, 2], c_fill_value)
    else:
        raise ValueError("Unsupported method: %s" % method)

    return numpy.array(result, copy=False)