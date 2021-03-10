# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
"""This module provides background extraction functions and smoothing
functions. These functions are extracted from PyMca module SpecFitFuns.

Index of background extraction functions:
------------------------------------------

    - :func:`strip`
    - :func:`snip1d`
    - :func:`snip2d`
    - :func:`snip3d`

Smoothing functions:
--------------------

    - :func:`savitsky_golay`
    - :func:`smooth1d`
    - :func:`smooth2d`
    - :func:`smooth3d`

References:
-----------

.. [Morhac97] Miroslav Morháč et al.
   Background elimination methods for multidimensional coincidence γ-ray spectra.
   Nucl. Instruments and Methods in Physics Research A401 (1997) 113-132.
   https://doi.org/10.1016/S0168-9002(97)01023-1

.. [Ryan88] C.G. Ryan et al.
   SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications.
   Nucl. Instruments and Methods in Physics Research B34 (1988) 396-402*.
   https://doi.org/10.1016/0168-583X(88)90063-8

API documentation:
-------------------

"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "15/05/2017"

import logging
import numpy

_logger = logging.getLogger(__name__)

cimport cython
cimport silx.math.fit.filters_wrapper as filters_wrapper


def strip(data, w=1, niterations=1000, factor=1.0, anchors=None):
    """Extract background from data using the strip algorithm, as explained at
    http://pymca.sourceforge.net/stripbackground.html.

    In its simplest implementation it is just as an iterative procedure
    depending on two parameters. These parameters are the strip background
    width ``w``, and the number of iterations. At each iteration, if the
    contents of channel ``i``, ``y(i)``, is above the average of the contents
    of the channels at ``w`` channels of distance, ``y(i-w)`` and
    ``y(i+w)``, ``y(i)`` is replaced by the average.
    At the end of the process we are left with something that resembles a spectrum
    in which the peaks have been stripped.

    :param data: Data array
    :type data: numpy.ndarray
    :param w: Strip width
    :param niterations: number of iterations
    :param factor: scaling factor applied to the average of ``y(i-w)`` and
        ``y(i+w)`` before comparing to ``y(i)``
    :param anchors: Array of anchors, indices of points that will not be
          modified during the stripping procedure.
    :return: Data with peaks stripped away
    """
    cdef:
        double[::1] input_c
        double[::1] output
        long[::1] anchors_c

    if not isinstance(data, numpy.ndarray):
        if not hasattr(data, "__len__"):
            raise TypeError("data must be a sequence (list, tuple) " +
                            "or a numpy array")
        data_shape = (len(data), )
    else:
        data_shape = data.shape

    input_c = numpy.array(data,
                          copy=True,
                          dtype=numpy.float64,
                          order='C').reshape(-1)

    output = numpy.empty(shape=(input_c.size,),
                         dtype=numpy.float64)

    if anchors is not None and len(anchors):
        # numpy.int_ is the same as C long (http://docs.scipy.org/doc/numpy/user/basics.types.html)
        anchors_c = numpy.array(anchors,
                                copy=False,
                                dtype=numpy.int_,
                                order='C')
        len_anchors = anchors_c.size
    else:
        # Make a dummy length-1 array, because if I use shape=(0,) I get the error
        # IndexError: Out of bounds on buffer access (axis 0)
        anchors_c = numpy.empty(shape=(1,),
                                dtype=numpy.int_)
        len_anchors = 0


    status = filters_wrapper.strip(&input_c[0], input_c.size,
                                    factor, niterations, w,
                                    &anchors_c[0], len_anchors, &output[0])

    return numpy.asarray(output).reshape(data_shape)


def snip1d(data, snip_width):
    """Estimate the baseline (background) of a 1D data vector by clipping peaks.

    Implementation of the algorithm SNIP in 1D is described in [Morhac97]_.
    The original idea for 1D and the low-statistics-digital-filter (lsdf) comes
    from [Ryan88]_.

    :param data: Data array, preferably 1D and of type *numpy.float64*.
        Else, the data array will be flattened and converted to
        *dtype=numpy.float64* prior to applying the snip filter.
    :type data: numpy.ndarray
    :param snip_width: Width of the snip operator, in number of samples.
        A sample will be iteratively compared to it's neighbors up to a
        distance of ``snip_width`` samples. This parameters has a direct
        influence on the speed of the algorithm.
    :type width: int
    :return: Baseline of the input array, as an array of the same shape.
    :rtype: numpy.ndarray
    """
    cdef:
        double[::1] data_c

    if not isinstance(data, numpy.ndarray):
        if not hasattr(data, "__len__"):
            raise TypeError("data must be a sequence (list, tuple) " +
                            "or a numpy array")
        data_shape = (len(data), )
    else:
        data_shape = data.shape

    data_c =  numpy.array(data,
                          copy=True,
                          dtype=numpy.float64,
                          order='C').reshape(-1)

    filters_wrapper.snip1d(&data_c[0], data_c.size, snip_width)

    return numpy.asarray(data_c).reshape(data_shape)


def snip2d(data, snip_width):
    """Estimate the baseline (background) of a 2D data signal by clipping peaks.

    Implementation of the algorithm SNIP in 2D described in [Morhac97]_.

    :param data: 2D array
    :type data: numpy.ndarray
    :param width: Width of the snip operator, in number of samples. A wider
        snip operator will result in a smoother result (lower frequency peaks
        will be clipped), and a longer computation time.
    :type width: int
    :return: Baseline of the input array, as an array of the same shape.
    :rtype: numpy.ndarray
    """
    cdef:
        double[::1] data_c

    if not isinstance(data, numpy.ndarray):
        if not hasattr(data, "__len__") or not hasattr(data[0], "__len__"):
            raise TypeError("data must be a 2D sequence (list, tuple) " +
                            "or a 2D numpy array")
        nrows = len(data)
        ncolumns = len(data[0])
        data_shape = (len(data), len(data[0]))

    else:
        data_shape = data.shape
        nrows =  data_shape[0]
        if len(data_shape) == 2:
            ncolumns = data_shape[1]
        else:
            raise TypeError("data array must be 2-dimensional")

    data_c =  numpy.array(data,
                          copy=True,
                          dtype=numpy.float64,
                          order='C').reshape(-1)

    filters_wrapper.snip2d(&data_c[0], nrows, ncolumns, snip_width)

    return numpy.asarray(data_c).reshape(data_shape)


def snip3d(data, snip_width):
    """Estimate the baseline (background) of a 3D data signal by clipping peaks.

    Implementation of the algorithm SNIP in 3D described in [Morhac97]_.

    :param data: 3D array
    :type data: numpy.ndarray
    :param width: Width of the snip operator, in number of samples. A wider
        snip operator will result in a smoother result (lower frequency peaks
        will be clipped), and a longer computation time.
    :type width: int

    :return: Baseline of the input array, as an array of the same shape.
    :rtype: numpy.ndarray
    """
    cdef:
        double[::1] data_c

    if not isinstance(data, numpy.ndarray):
        if not hasattr(data, "__len__") or not hasattr(data[0], "__len__") or\
                not hasattr(data[0][0], "__len__"):
            raise TypeError("data must be a 3D sequence (list, tuple) " +
                            "or a 3D numpy array")
        nx = len(data)
        ny = len(data[0])
        nz = len(data[0][0])
        data_shape = (len(data), len(data[0]),  len(data[0][0]))
    else:
        data_shape = data.shape
        nrows =  data_shape[0]
        if len(data_shape) == 3:
            nx =  data_shape[0]
            ny = data_shape[1]
            nz = data_shape[2]
        else:
            raise TypeError("data array must be 3-dimensional")

    data_c =  numpy.array(data,
                          copy=True,
                          dtype=numpy.float64,
                          order='C').reshape(-1)

    filters_wrapper.snip3d(&data_c[0], nx, ny, nz, snip_width)

    return numpy.asarray(data_c).reshape(data_shape)


def savitsky_golay(data, npoints=5):
    """Smooth a curve using a Savitsky-Golay filter.

    :param data: Input data
    :type data: 1D numpy array
    :param npoints: Size of the smoothing operator in number of samples
        Must be between 3 and 100.
    :return: Smoothed data
    """
    cdef:
        double[::1] data_c
        double[::1] output

    data_c =  numpy.array(data,
                          dtype=numpy.float64,
                          order='C').reshape(-1)

    output = numpy.empty(shape=(data_c.size,),
                         dtype=numpy.float64)

    status = filters_wrapper.SavitskyGolay(&data_c[0], data_c.size,
                                           npoints, &output[0])

    if status:
        _logger.error("Smoothing failed. Check that npoints is greater " +
                      "than 3 and smaller than 100.")

    return numpy.asarray(output).reshape(data.shape)


def smooth1d(data):
    """Simple smoothing for 1D data.

    For a data array :math:`y` of length :math:`n`, the smoothed array
    :math:`ys` is calculated as a weighted average of neighboring samples:

    :math:`ys_0 = 0.75 y_0 + 0.25 y_1`

    :math:`ys_i = 0.25 (y_{i-1} + 2 y_i + y_{i+1})` for :math:`0 < i < n-1`

    :math:`ys_{n-1} = 0.25 y_{n-2} + 0.75 y_{n-1}`


    :param data: 1D data array
    :type data: numpy.ndarray
    :return: Smoothed data
    :rtype: numpy.ndarray(dtype=numpy.float64)
    """
    cdef:
        double[::1] data_c

    if not isinstance(data, numpy.ndarray):
        if not hasattr(data, "__len__"):
            raise TypeError("data must be a sequence (list, tuple) " +
                            "or a numpy array")
        data_shape = (len(data), )
    else:
        data_shape = data.shape

    data_c =  numpy.array(data,
                          copy=True,
                          dtype=numpy.float64,
                          order='C').reshape(-1)

    filters_wrapper.smooth1d(&data_c[0], data_c.size)

    return numpy.asarray(data_c).reshape(data_shape)


def smooth2d(data):
    """Simple smoothing for 2D data:
    :func:`smooth1d` is applied succesively along both axis

    :param data: 2D data array
    :type data: numpy.ndarray
    :return: Smoothed data
    :rtype: numpy.ndarray(dtype=numpy.float64)
    """
    cdef:
        double[::1] data_c

    if not isinstance(data, numpy.ndarray):
        if not hasattr(data, "__len__") or not hasattr(data[0], "__len__"):
            raise TypeError("data must be a 2D sequence (list, tuple) " +
                            "or a 2D numpy array")
        nrows = len(data)
        ncolumns = len(data[0])
        data_shape = (len(data), len(data[0]))

    else:
        data_shape = data.shape
        nrows =  data_shape[0]
        if len(data_shape) == 2:
            ncolumns = data_shape[1]
        else:
            raise TypeError("data array must be 2-dimensional")

    data_c =  numpy.array(data,
                          copy=True,
                          dtype=numpy.float64,
                          order='C').reshape(-1)

    filters_wrapper.smooth2d(&data_c[0], nrows, ncolumns)

    return numpy.asarray(data_c).reshape(data_shape)


def smooth3d(data):
    """Simple smoothing for 3D data:
    :func:`smooth2d` is applied on each 2D slice of the data volume along all
    3 axis

    :param data: 2D data array
    :type data: numpy.ndarray
    :return: Smoothed data
    :rtype: numpy.ndarray(dtype=numpy.float64)
    """
    cdef:
        double[::1] data_c

    if not isinstance(data, numpy.ndarray):
        if not hasattr(data, "__len__") or not hasattr(data[0], "__len__") or\
                not hasattr(data[0][0], "__len__"):
            raise TypeError("data must be a 3D sequence (list, tuple) " +
                            "or a 3D numpy array")
        nx = len(data)
        ny = len(data[0])
        nz = len(data[0][0])
        data_shape = (len(data), len(data[0]),  len(data[0][0]))
    else:
        data_shape = data.shape
        nrows =  data_shape[0]
        if len(data_shape) == 3:
            nx =  data_shape[0]
            ny = data_shape[1]
            nz = data_shape[2]
        else:
            raise TypeError("data array must be 3-dimensional")

    data_c =  numpy.array(data,
                          copy=True,
                          dtype=numpy.float64,
                          order='C').reshape(-1)

    filters_wrapper.smooth3d(&data_c[0], nx, ny, nz)

    return numpy.asarray(data_c).reshape(data_shape)
