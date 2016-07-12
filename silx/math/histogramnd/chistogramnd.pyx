# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "01/02/2016"

cimport numpy  # noqa
cimport cython
import numpy as np

cimport histogramnd_c


def chistogramnd(sample,
                 histo_range,
                 n_bins,
                 weights=None,
                 weight_min=None,
                 weight_max=None,
                 last_bin_closed=False,
                 histo=None,
                 weighted_histo=None,
                 wh_dtype=None):
    """
    histogramnd(sample, histo_range, n_bins, weights=None, weight_min=None, weight_max=None, last_bin_closed=False, histo=None, weighted_histo=None)

    Computes the multidimensional histogram of some data.

    :param sample:
        The data to be histogrammed.
        Its shape must be either
        (N,) if it contains one dimensional coordinates,
        or an (N,D) array where the rows are the
        coordinates of points in a D dimensional space.
        The following dtypes are supported : :class:`numpy.float64`,
        :class:`numpy.float32`, :class:`numpy.int32`.

        .. warning:: if sample is not a C_CONTIGUOUS ndarray (e.g : a non
            contiguous slice) then histogramnd will have to do make an internal
            copy.
    :type sample: :class:`numpy.array`

    :param histo_range:
        A (N, 2) array containing the histogram range along each dimension,
        where N is the sample's number of dimensions.
    :type histo_range: array_like

    :param n_bins:
        The number of bins :
            * a scalar (same number of bins for all dimensions)
            * a D elements array (number of bins for each dimensions)
    :type n_bins: scalar or array_like

    :param weights:
        A N elements numpy array of values associated with
        each sample.
        The values of the *weighted_histo* array
        returned by the function are equal to the sum of
        the weights associated with the samples falling
        into each bin.
        The following dtypes are supported : :class:`numpy.float64`,
        :class:`numpy.float32`, :class:`numpy.int32`.

        .. note:: If None, the weighted histogram returned will be None.
    :type weights: *optional*, :class:`numpy.array`

    :param weight_min:
        Use this parameter to filter out all samples whose
        weights are lower than this value.

        .. note:: This value will be cast to the same type
            as *weights*.
    :type weight_min: *optional*, scalar

    :param weight_max:
        Use this parameter to filter out all samples whose
        weights are higher than this value.

        .. note:: This value will be cast to the same type
            as *weights*.

    :type weight_max: *optional*, scalar

    :param last_bin_closed:
        By default the last bin is half
        open (i.e.: [x,y) ; x included, y
        excluded), like all the other bins.
        Set this parameter to true if you want
        the LAST bin to be closed.
    :type last_bin_closed: *optional*, :class:`python.boolean`

    :param histo:
        Use this parameter if you want to pass your
        own histogram array instead of the one
        created by this function. New values
        will be added to this array. The returned array
        will then be this one (same reference).

        .. warning:: If the histo array was created by a previous
            call to histogramnd then the user is
            responsible for providing the same parameters
            (*n_bins*, *histo_range*, ...).
    :type histo: *optional*, :class:`numpy.array`

    :param weighted_histo:
        Use this parameter if you want to pass your
        own weighted histogram array instead of
        the created by this function. New
        values will be added to this array. The returned array
        will then be this one (same reference).

        .. warning:: If the weighted_histo array was created by a previous
            call to histogramnd then the user is
            responsible for providing the same parameters
            (*n_bins*, *histo_range*, ...).

        .. warning:: if weighted_histo is not a C_CONTIGUOUS ndarray (e.g : a
            non contiguous slice) then histogramnd will have to do make an
            internal copy.
    :type weighted_histo: *optional*, :class:`numpy.array`
    
    :param wh_dtype: type of the weighted histogram array. This parameter is
        ignored if *weighted_histo* is provided. If not provided, the
        weighted histogram array will contain values of the same type as
        *weights*. Allowed values are : `numpu.double` and `numpy.float32`.
    :type wh_dtype: *optional*, numpy data type

    :return: Histogram (bin counts, always returned), weighted histogram of
        the sample (or *None* if weights is *None*) and bin edges for each
        dimension.
    :rtype: *tuple* (:class:`numpy.array`, :class:`numpy.array`, `tuple`) or
        (:class:`numpy.array`, None, `tuple`)
    """  # noqa
    
    if wh_dtype is None:
        wh_dtype = np.double
    elif wh_dtype not in (np.double, np.float32):
        raise ValueError('<wh_dtype> type not supported : {0}.'.format(wh_dtype))

    if (weighted_histo is not None and
        weighted_histo.flags['C_CONTIGUOUS'] is False):
        raise ValueError('<weighted_histo> must be a C_CONTIGUOUS numpy array.')

    if histo is not None and histo.flags['C_CONTIGUOUS'] is False:
        raise ValueError('<histo> must be a C_CONTIGUOUS numpy array.')

    s_shape = sample.shape

    n_dims = 1 if len(s_shape) == 1 else s_shape[1]

    if weights is not None:
        w_shape = weights.shape

        # making sure the sample and weights sizes are coherent
        # 2 different cases : 2D sample (N,M) and 1D (N)
        if len(w_shape) != 1 or w_shape[0] != s_shape[0]:
                raise ValueError('<weights> must be an array whose length '
                                 'is equal to the number of samples.')

        weights_type = weights.dtype
    else:
        weights_type = None

    # just in case those arent numpy arrays
    # (this allows the user to provide native python lists,
    #   => easier for testing)
    i_histo_range = histo_range
    histo_range = np.array(histo_range)
    err_histo_range = False

    if n_dims == 1:
        if histo_range.shape == (2,):
            pass
        elif histo_range.shape == (1, 2):
            histo_range.shape = -1
        else:
            err_histo_range = True
    elif n_dims != 1 and histo_range.shape != (n_dims, 2):
        err_histo_range = True

    if err_histo_range:
        raise ValueError('<histo_range> error : expected {n_dims} sets of '
                         'lower and upper bin edges, '
                         'got the following instead : {histo_range}. '
                         '(provided <sample> contains '
                         '{n_dims}D values)'
                         ''.format(histo_range=i_histo_range,
                                   n_dims=n_dims))

    # checking n_bins size
    n_bins = np.array(n_bins, ndmin=1)
    if len(n_bins) == 1:
        n_bins = np.tile(n_bins, n_dims)
    elif n_bins.shape != (n_dims,):
        raise ValueError('n_bins must be either a scalar (same number '
                         'of bins for all dimensions) or '
                         'an array (number of bins for each '
                         'dimension).')

    # checking if None is in n_bins, otherwise a rather cryptic
    #   exception is thrown when calling np.zeros
    # also testing for negative/null values
    if np.any(np.equal(n_bins, None)) or np.any(n_bins <= 0):
        raise ValueError('<n_bins> : only positive values allowed.')

    output_shape = tuple(n_bins)

    # checking the histo array, if provided
    if histo is None:
        histo = np.zeros(output_shape, dtype=np.uint32)
    else:
        if histo.shape != output_shape:
            raise ValueError('Provided <histo> array doesn\'t have '
                             'a shape compatible with <n_bins> '
                             ': should be {0} instead of {1}.'
                             ''.format(output_shape, histo.shape))
        if histo.dtype != np.uint32:
            raise ValueError('Provided <histo> array doesn\'t have '
                             'the expected type '
                             ': should be {0} instead of {1}.'
                             ''.format(np.uint32, histo.dtype))

    # checking the weighted_histo array, if provided
    if weights_type is None:
        weighted_histo = None
    elif weighted_histo is None:
        if wh_dtype is None:
            wh_dtype = weights_type
        weighted_histo = np.zeros(output_shape, dtype=wh_dtype)
    else:
        if weighted_histo.shape != output_shape:
            raise ValueError('Provided <weighted_histo> array doesn\'t have '
                             'a shape compatible with <n_bins> '
                             ': should be {0} instead of {1}.'
                             ''.format(output_shape, weighted_histo.shape))
        if (weighted_histo.dtype != np.float64 and
            weighted_histo.dtype != np.float32):
            raise ValueError('Provided <weighted_histo> array doesn\'t have '
                             'the expected type '
                             ': should be {0} or {1} instead of {2}.'
                             ''.format(np.double,
                                       np.float32,
                                       weighted_histo.dtype))

    option_flags = 0

    if weight_min is not None:
        option_flags |= histogramnd_c.HISTO_WEIGHT_MIN
    else:
        weight_min = 0

    if weight_max is not None:
        option_flags |= histogramnd_c.HISTO_WEIGHT_MAX
    else:
        weight_max = 0

    if last_bin_closed is not None and last_bin_closed:
        option_flags |= histogramnd_c.HISTO_LAST_BIN_CLOSED

    sample_type = sample.dtype

    n_elem = sample.size // n_dims

    bin_edges = np.zeros(n_bins.sum() + n_bins.size, dtype=np.double)

    # wanted to store the functions in a dict (with the supported types
    # as keys, but I couldn't find a way to make it work with cdef
    # functions. so I have to explicitly list them all...

    def raise_unsupported_type():
        raise TypeError('Case not supported - sample:{0} '
                        'and weights:{1}.'
                        ''.format(sample_type, weights_type))

    sample_c = np.ascontiguousarray(sample.reshape((sample.size,)))

    weights_c = (np.ascontiguousarray(weights.reshape((weights.size,)))
                 if weights is not None else None)

    histo_range_c = np.ascontiguousarray(histo_range.reshape((histo_range.size,)),
                                      dtype=np.double)

    n_bins_c = np.ascontiguousarray(n_bins.reshape((n_bins.size,)),
                                    dtype=np.int32)

    histo_c = histo.reshape((histo.size,))

    if weighted_histo is not None:
        cumul_c = weighted_histo.reshape((weighted_histo.size,))
    else:
        cumul_c = None

    bin_edges_c = np.ascontiguousarray(bin_edges.reshape((bin_edges.size,)))

    rc = 0

    if weighted_histo is None or weighted_histo.dtype == np.double:

        if sample_type == np.float64:

            if weights_type == np.float64 or weights_type is None:

                rc = _histogramnd_double_double_double(sample_c,
                                                       weights_c,
                                                       n_dims,
                                                       n_elem,
                                                       histo_range_c,
                                                       n_bins_c,
                                                       histo_c,
                                                       cumul_c,
                                                       bin_edges_c,
                                                       option_flags,
                                                       weight_min=weight_min,
                                                       weight_max=weight_max)

            elif weights_type == np.float32:

                rc = _histogramnd_double_float_double(sample_c,
                                                      weights_c,
                                                      n_dims,
                                                      n_elem,
                                                      histo_range_c,
                                                      n_bins_c,
                                                      histo_c,
                                                      cumul_c,
                                                      bin_edges_c,
                                                      option_flags,
                                                      weight_min=weight_min,
                                                      weight_max=weight_max)

            elif weights_type == np.int32:

                rc = _histogramnd_double_int32_t_double(sample_c,
                                                        weights_c,
                                                        n_dims,
                                                        n_elem,
                                                        histo_range_c,
                                                        n_bins_c,
                                                        histo_c,
                                                        cumul_c,
                                                        bin_edges_c,
                                                        option_flags,
                                                        weight_min=weight_min,
                                                        weight_max=weight_max)

            else:
                raise_unsupported_type()

        # endif sample_type == np.float64
        elif sample_type == np.float32:

            if weights_type == np.float64 or weights_type is None:

                rc = _histogramnd_float_double_double(sample_c,
                                                      weights_c,
                                                      n_dims,
                                                      n_elem,
                                                      histo_range_c,
                                                      n_bins_c,
                                                      histo_c,
                                                      cumul_c,
                                                      bin_edges_c,
                                                      option_flags,
                                                      weight_min=weight_min,
                                                      weight_max=weight_max)

            elif weights_type == np.float32:

                rc = _histogramnd_float_float_double(sample_c,
                                                     weights_c,
                                                     n_dims,
                                                     n_elem,
                                                     histo_range_c,
                                                     n_bins_c,
                                                     histo_c,
                                                     cumul_c,
                                                     bin_edges_c,
                                                     option_flags,
                                                     weight_min=weight_min,
                                                     weight_max=weight_max)

            elif weights_type == np.int32:

                rc = _histogramnd_float_int32_t_double(sample_c,
                                                       weights_c,
                                                       n_dims,
                                                       n_elem,
                                                       histo_range_c,
                                                       n_bins_c,
                                                       histo_c,
                                                       cumul_c,
                                                       bin_edges_c,
                                                       option_flags,
                                                       weight_min=weight_min,
                                                       weight_max=weight_max)

            else:
                raise_unsupported_type()

        # endif sample_type == np.float32
        elif sample_type == np.int32:

            if weights_type == np.float64 or weights_type is None:

                rc = _histogramnd_int32_t_double_double(sample_c,
                                                        weights_c,
                                                        n_dims,
                                                        n_elem,
                                                        histo_range_c,
                                                        n_bins_c,
                                                        histo_c,
                                                        cumul_c,
                                                        bin_edges_c,
                                                        option_flags,
                                                        weight_min=weight_min,
                                                        weight_max=weight_max)

            elif weights_type == np.float32:

                rc = _histogramnd_int32_t_float_double(sample_c,
                                                       weights_c,
                                                       n_dims,
                                                       n_elem,
                                                       histo_range_c,
                                                       n_bins_c,
                                                       histo_c,
                                                       cumul_c,
                                                       bin_edges_c,
                                                       option_flags,
                                                       weight_min=weight_min,
                                                       weight_max=weight_max)

            elif weights_type == np.int32:

                rc = _histogramnd_int32_t_int32_t_double(sample_c,
                                                         weights_c,
                                                         n_dims,
                                                         n_elem,
                                                         histo_range_c,
                                                         n_bins_c,
                                                         histo_c,
                                                         cumul_c,
                                                         bin_edges_c,
                                                         option_flags,
                                                         weight_min=weight_min,
                                                         weight_max=weight_max)

            else:
                raise_unsupported_type()

        # endif sample_type == np.int32:
        else:
            raise_unsupported_type()

    # endif weighted_histo is None or weighted_histo.dtype == np.double:
    elif weighted_histo.dtype == np.float32:

        if sample_type == np.float64:

            if weights_type == np.float64 or weights_type is None:

                rc = _histogramnd_double_double_float(sample_c,
                                                      weights_c,
                                                      n_dims,
                                                      n_elem,
                                                      histo_range_c,
                                                      n_bins_c,
                                                      histo_c,
                                                      cumul_c,
                                                      bin_edges_c,
                                                      option_flags,
                                                      weight_min=weight_min,
                                                      weight_max=weight_max)

            elif weights_type == np.float32:

                rc = _histogramnd_double_float_float(sample_c,
                                                     weights_c,
                                                     n_dims,
                                                     n_elem,
                                                     histo_range_c,
                                                     n_bins_c,
                                                     histo_c,
                                                     cumul_c,
                                                     bin_edges_c,
                                                     option_flags,
                                                     weight_min=weight_min,
                                                     weight_max=weight_max)

            elif weights_type == np.int32:

                rc = _histogramnd_double_int32_t_float(sample_c,
                                                       weights_c,
                                                       n_dims,
                                                       n_elem,
                                                       histo_range_c,
                                                       n_bins_c,
                                                       histo_c,
                                                       cumul_c,
                                                       bin_edges_c,
                                                       option_flags,
                                                       weight_min=weight_min,
                                                       weight_max=weight_max)

            else:
                raise_unsupported_type()

        # endif sample_type == np.float64
        elif sample_type == np.float32:

            if weights_type == np.float64 or weights_type is None:

                rc = _histogramnd_float_double_float(sample_c,
                                                     weights_c,
                                                     n_dims,
                                                     n_elem,
                                                     histo_range_c,
                                                     n_bins_c,
                                                     histo_c,
                                                     cumul_c,
                                                     bin_edges_c,
                                                     option_flags,
                                                     weight_min=weight_min,
                                                     weight_max=weight_max)

            elif weights_type == np.float32:

                rc = _histogramnd_float_float_float(sample_c,
                                                    weights_c,
                                                    n_dims,
                                                    n_elem,
                                                    histo_range_c,
                                                    n_bins_c,
                                                    histo_c,
                                                    cumul_c,
                                                    bin_edges_c,
                                                    option_flags,
                                                    weight_min=weight_min,
                                                    weight_max=weight_max)

            elif weights_type == np.int32:

                rc = _histogramnd_float_int32_t_float(sample_c,
                                                      weights_c,
                                                      n_dims,
                                                      n_elem,
                                                      histo_range_c,
                                                      n_bins_c,
                                                      histo_c,
                                                      cumul_c,
                                                      bin_edges_c,
                                                      option_flags,
                                                      weight_min=weight_min,
                                                      weight_max=weight_max)

            else:
                raise_unsupported_type()

        # endif sample_type == np.float32
        elif sample_type == np.int32:

            if weights_type == np.float64 or weights_type is None:

                rc = _histogramnd_int32_t_double_float(sample_c,
                                                       weights_c,
                                                       n_dims,
                                                       n_elem,
                                                       histo_range_c,
                                                       n_bins_c,
                                                       histo_c,
                                                       cumul_c,
                                                       bin_edges_c,
                                                       option_flags,
                                                       weight_min=weight_min,
                                                       weight_max=weight_max)

            elif weights_type == np.float32:

                rc = _histogramnd_int32_t_float_float(sample_c,
                                                      weights_c,
                                                      n_dims,
                                                      n_elem,
                                                      histo_range_c,
                                                      n_bins_c,
                                                      histo_c,
                                                      cumul_c,
                                                      bin_edges_c,
                                                      option_flags,
                                                      weight_min=weight_min,
                                                      weight_max=weight_max)

            elif weights_type == np.int32:

                rc = _histogramnd_int32_t_int32_t_float(sample_c,
                                                        weights_c,
                                                        n_dims,
                                                        n_elem,
                                                        histo_range_c,
                                                        n_bins_c,
                                                        histo_c,
                                                        cumul_c,
                                                        bin_edges_c,
                                                        option_flags,
                                                        weight_min=weight_min,
                                                        weight_max=weight_max)

            else:
                raise_unsupported_type()

        # endif sample_type == np.int32:
        else:
            raise_unsupported_type()

    # end elseif weighted_histo.dtype == np.float32:
    else:
        # this isnt supposed to happen since weighted_histo type was checked earlier
        raise_unsupported_type()

    if rc != histogramnd_c.HISTO_OK:
        if rc == histogramnd_c.HISTO_ERR_ALLOC:
            raise MemoryError('histogramnd failed to allocate memory.')
        else:
            raise Exception('histogramnd returned an error : {0}'
                            ''.format(rc))

    edges = []
    offset = 0
    for i_dim in range(n_dims):
        edges.append(bin_edges[offset:offset + n_bins[i_dim] + 1])
        offset += n_bins[i_dim] + 1

    return histo, weighted_histo, tuple(edges)

# =====================
#  double sample, double cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_double_double(double[:] sample,
                                           double[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           double[:] histo_range,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           double[:] cumul,
                                           double[:] bin_edges,
                                           int option_flags,
                                           double weight_min,
                                           double weight_max) nogil:

    return histogramnd_c.histogramnd_double_double_double(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &histo_range[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          &bin_edges[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_float_double(double[:] sample,
                                          float[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          double[:] histo_range,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          double[:] cumul,
                                          double[:] bin_edges,
                                          int option_flags,
                                          float weight_min,
                                          float weight_max) nogil:

    return histogramnd_c.histogramnd_double_float_double(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &histo_range[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         &bin_edges[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_int32_t_double(double[:] sample,
                                            numpy.int32_t[:] weights,
                                            int n_dims,
                                            int n_elem,
                                            double[:] histo_range,
                                            int[:] n_bins,
                                            numpy.uint32_t[:] histo,
                                            double[:] cumul,
                                            double[:] bin_edges,
                                            int option_flags,
                                            numpy.int32_t weight_min,
                                            numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_double_int32_t_double(&sample[0],
                                                           &weights[0],
                                                           n_dims,
                                                           n_elem,
                                                           &histo_range[0],
                                                           &n_bins[0],
                                                           &histo[0],
                                                           &cumul[0],
                                                           &bin_edges[0],
                                                           option_flags,
                                                           weight_min,
                                                           weight_max)


# =====================
#  float sample, double cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_double_double(float[:] sample,
                                          double[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          double[:] histo_range,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          double[:] cumul,
                                          double[:] bin_edges,
                                          int option_flags,
                                          double weight_min,
                                          double weight_max) nogil:

    return histogramnd_c.histogramnd_float_double_double(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &histo_range[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         &bin_edges[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_float_double(float[:] sample,
                                         float[:] weights,
                                         int n_dims,
                                         int n_elem,
                                         double[:] histo_range,
                                         int[:] n_bins,
                                         numpy.uint32_t[:] histo,
                                         double[:] cumul,
                                         double[:] bin_edges,
                                         int option_flags,
                                         float weight_min,
                                         float weight_max) nogil:

    return histogramnd_c.histogramnd_float_float_double(&sample[0],
                                                        &weights[0],
                                                        n_dims,
                                                        n_elem,
                                                        &histo_range[0],
                                                        &n_bins[0],
                                                        &histo[0],
                                                        &cumul[0],
                                                        &bin_edges[0],
                                                        option_flags,
                                                        weight_min,
                                                        weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_int32_t_double(float[:] sample,
                                           numpy.int32_t[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           double[:] histo_range,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           double[:] cumul,
                                           double[:] bin_edges,
                                           int option_flags,
                                           numpy.int32_t weight_min,
                                           numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_float_int32_t_double(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &histo_range[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          &bin_edges[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


# =====================
#  numpy.int32_t sample, double cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_double_double(numpy.int32_t[:] sample,
                                            double[:] weights,
                                            int n_dims,
                                            int n_elem,
                                            double[:] histo_range,
                                            int[:] n_bins,
                                            numpy.uint32_t[:] histo,
                                            double[:] cumul,
                                            double[:] bin_edges,
                                            int option_flags,
                                            double weight_min,
                                            double weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_double_double(&sample[0],
                                                           &weights[0],
                                                           n_dims,
                                                           n_elem,
                                                           &histo_range[0],
                                                           &n_bins[0],
                                                           &histo[0],
                                                           &cumul[0],
                                                           &bin_edges[0],
                                                           option_flags,
                                                           weight_min,
                                                           weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_float_double(numpy.int32_t[:] sample,
                                           float[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           double[:] histo_range,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           double[:] cumul,
                                           double[:] bin_edges,
                                           int option_flags,
                                           float weight_min,
                                           float weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_float_double(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &histo_range[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          &bin_edges[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_int32_t_double(numpy.int32_t[:] sample,
                                             numpy.int32_t[:] weights,
                                             int n_dims,
                                             int n_elem,
                                             double[:] histo_range,
                                             int[:] n_bins,
                                             numpy.uint32_t[:] histo,
                                             double[:] cumul,
                                             double[:] bin_edges,
                                             int option_flags,
                                             numpy.int32_t weight_min,
                                             numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_int32_t_double(&sample[0],
                                                            &weights[0],
                                                            n_dims,
                                                            n_elem,
                                                            &histo_range[0],
                                                            &n_bins[0],
                                                            &histo[0],
                                                            &cumul[0],
                                                            &bin_edges[0],
                                                            option_flags,
                                                            weight_min,
                                                            weight_max)


# =====================
#  double sample, float cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_double_float(double[:] sample,
                                          double[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          double[:] histo_range,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          float[:] cumul,
                                          double[:] bin_edges,
                                          int option_flags,
                                          double weight_min,
                                          double weight_max) nogil:

    return histogramnd_c.histogramnd_double_double_float(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &histo_range[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         &bin_edges[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_float_float(double[:] sample,
                                         float[:] weights,
                                         int n_dims,
                                         int n_elem,
                                         double[:] histo_range,
                                         int[:] n_bins,
                                         numpy.uint32_t[:] histo,
                                         float[:] cumul,
                                         double[:] bin_edges,
                                         int option_flags,
                                         float weight_min,
                                         float weight_max) nogil:

    return histogramnd_c.histogramnd_double_float_float(&sample[0],
                                                        &weights[0],
                                                        n_dims,
                                                        n_elem,
                                                        &histo_range[0],
                                                        &n_bins[0],
                                                        &histo[0],
                                                        &cumul[0],
                                                        &bin_edges[0],
                                                        option_flags,
                                                        weight_min,
                                                        weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_double_int32_t_float(double[:] sample,
                                           numpy.int32_t[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           double[:] histo_range,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           float[:] cumul,
                                           double[:] bin_edges,
                                           int option_flags,
                                           numpy.int32_t weight_min,
                                           numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_double_int32_t_float(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &histo_range[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          &bin_edges[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


# =====================
#  float sample, float cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_double_float(float[:] sample,
                                         double[:] weights,
                                         int n_dims,
                                         int n_elem,
                                         double[:] histo_range,
                                         int[:] n_bins,
                                         numpy.uint32_t[:] histo,
                                         float[:] cumul,
                                         double[:] bin_edges,
                                         int option_flags,
                                         double weight_min,
                                         double weight_max) nogil:

    return histogramnd_c.histogramnd_float_double_float(&sample[0],
                                                        &weights[0],
                                                        n_dims,
                                                        n_elem,
                                                        &histo_range[0],
                                                        &n_bins[0],
                                                        &histo[0],
                                                        &cumul[0],
                                                        &bin_edges[0],
                                                        option_flags,
                                                        weight_min,
                                                        weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_float_float(float[:] sample,
                                        float[:] weights,
                                        int n_dims,
                                        int n_elem,
                                        double[:] histo_range,
                                        int[:] n_bins,
                                        numpy.uint32_t[:] histo,
                                        float[:] cumul,
                                        double[:] bin_edges,
                                        int option_flags,
                                        float weight_min,
                                        float weight_max) nogil:

    return histogramnd_c.histogramnd_float_float_float(&sample[0],
                                                       &weights[0],
                                                       n_dims,
                                                       n_elem,
                                                       &histo_range[0],
                                                       &n_bins[0],
                                                       &histo[0],
                                                       &cumul[0],
                                                       &bin_edges[0],
                                                       option_flags,
                                                       weight_min,
                                                       weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_float_int32_t_float(float[:] sample,
                                          numpy.int32_t[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          double[:] histo_range,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          float[:] cumul,
                                          double[:] bin_edges,
                                          int option_flags,
                                          numpy.int32_t weight_min,
                                          numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_float_int32_t_float(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &histo_range[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         &bin_edges[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


# =====================
#  numpy.int32_t sample, float cumul
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_double_float(numpy.int32_t[:] sample,
                                           double[:] weights,
                                           int n_dims,
                                           int n_elem,
                                           double[:] histo_range,
                                           int[:] n_bins,
                                           numpy.uint32_t[:] histo,
                                           float[:] cumul,
                                           double[:] bin_edges,
                                           int option_flags,
                                           double weight_min,
                                           double weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_double_float(&sample[0],
                                                          &weights[0],
                                                          n_dims,
                                                          n_elem,
                                                          &histo_range[0],
                                                          &n_bins[0],
                                                          &histo[0],
                                                          &cumul[0],
                                                          &bin_edges[0],
                                                          option_flags,
                                                          weight_min,
                                                          weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_float_float(numpy.int32_t[:] sample,
                                          float[:] weights,
                                          int n_dims,
                                          int n_elem,
                                          double[:] histo_range,
                                          int[:] n_bins,
                                          numpy.uint32_t[:] histo,
                                          float[:] cumul,
                                          double[:] bin_edges,
                                          int option_flags,
                                          float weight_min,
                                          float weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_float_float(&sample[0],
                                                         &weights[0],
                                                         n_dims,
                                                         n_elem,
                                                         &histo_range[0],
                                                         &n_bins[0],
                                                         &histo[0],
                                                         &cumul[0],
                                                         &bin_edges[0],
                                                         option_flags,
                                                         weight_min,
                                                         weight_max)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cdef int _histogramnd_int32_t_int32_t_float(numpy.int32_t[:] sample,
                                            numpy.int32_t[:] weights,
                                            int n_dims,
                                            int n_elem,
                                            double[:] histo_range,
                                            int[:] n_bins,
                                            numpy.uint32_t[:] histo,
                                            float[:] cumul,
                                            double[:] bin_edges,
                                            int option_flags,
                                            numpy.int32_t weight_min,
                                            numpy.int32_t weight_max) nogil:

    return histogramnd_c.histogramnd_int32_t_int32_t_float(&sample[0],
                                                           &weights[0],
                                                           n_dims,
                                                           n_elem,
                                                           &histo_range[0],
                                                           &n_bins[0],
                                                           &histo[0],
                                                           &cumul[0],
                                                           &bin_edges[0],
                                                           option_flags,
                                                           weight_min,
                                                           weight_max)


if __name__=='__main__':
    pass
