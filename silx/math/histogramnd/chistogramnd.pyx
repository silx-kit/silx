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
                 bins_rng,
                 n_bins,
                 weights=None,
                 weight_min=None,
                 weight_max=None,
                 last_bin_closed=False,
                 histo=None,
                 cumul=None):

    # see silx.math.histogramnd for the doc

    if cumul is not None and cumul.flags['C_CONTIGUOUS'] is False:
        raise ValueError('<cumul> must be a C_CONTIGUOUS numpy array.')

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
    i_bins_rng = bins_rng
    bins_rng = np.array(bins_rng)
    err_bins_rng = False

    if n_dims == 1:
        if bins_rng.shape == (2,):
            pass
        elif bins_rng.shape == (1, 2):
            bins_rng.shape = -1
        else:
            err_bins_rng = True
    elif n_dims != 1 and bins_rng.shape != (n_dims, 2):
        err_bins_rng = True

    if err_bins_rng:
        raise ValueError('<bins_rng> error : expected {n_dims} sets of '
                         'lower and upper bin edges, '
                         'got the following instead : {bins_rng}. '
                         '(provided <sample> contains '
                         '{n_dims}D values)'
                         ''.format(bins_rng=i_bins_rng,
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

    # checking the cumul array, if provided
    if weights_type is None:
        cumul = None
    elif cumul is None:
        cumul = np.zeros(output_shape, dtype=np.double)
    else:
        if cumul.shape != output_shape:
            raise ValueError('Provided <cumul> array doesn\'t have '
                             'a shape compatible with <n_bins> '
                             ': should be {0} instead of {1}.'
                             ''.format(output_shape, cumul.shape))
        if cumul.dtype != np.float64 and cumul.dtype != np.float32:
            raise ValueError('Provided <cumul> array doesn\'t have '
                             'the expected type '
                             ': should be {0} or {1} instead of {2}.'
                             ''.format(np.double, np.float32, cumul.dtype))

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

    bins_rng_c = np.ascontiguousarray(bins_rng.reshape((bins_rng.size,)),
                                      dtype=np.double)

    n_bins_c = np.ascontiguousarray(n_bins.reshape((n_bins.size,)),
                                    dtype=np.int32)

    histo_c = np.ascontiguousarray(histo.reshape((histo.size,)))

    if cumul is not None:
        cumul_c = np.ascontiguousarray(cumul.reshape((cumul.size,)))
    else:
        cumul_c = None

    bin_edges_c = np.ascontiguousarray(bin_edges.reshape((bin_edges.size,)))

    rc = 0

    if cumul is None or cumul.dtype == np.double:

        if sample_type == np.float64:

            if weights_type == np.float64 or weights_type is None:

                rc = _histogramnd_double_double_double(sample_c,
                                                       weights_c,
                                                       n_dims,
                                                       n_elem,
                                                       bins_rng_c,
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
                                                      bins_rng_c,
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
                                                        bins_rng_c,
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
                                                      bins_rng_c,
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
                                                     bins_rng_c,
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
                                                       bins_rng_c,
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
                                                        bins_rng_c,
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
                                                       bins_rng_c,
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
                                                         bins_rng_c,
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

    # endif cumul is None or cumul.dtype == np.double:
    elif cumul.dtype == np.float32:

        if sample_type == np.float64:

            if weights_type == np.float64 or weights_type is None:

                rc = _histogramnd_double_double_float(sample_c,
                                                      weights_c,
                                                      n_dims,
                                                      n_elem,
                                                      bins_rng_c,
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
                                                     bins_rng_c,
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
                                                       bins_rng_c,
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
                                                     bins_rng_c,
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
                                                    bins_rng_c,
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
                                                      bins_rng_c,
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
                                                       bins_rng_c,
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
                                                      bins_rng_c,
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
                                                        bins_rng_c,
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

    # end elseif cumul.dtype == np.float32:
    else:
        # this isnt supposed to happen since cumul type was checked earlier
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

    return histo, cumul, tuple(edges)

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
                                           double[:] bins_rng,
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
                                                          &bins_rng[0],
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
                                          double[:] bins_rng,
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
                                                         &bins_rng[0],
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
                                            double[:] bins_rng,
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
                                                           &bins_rng[0],
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
                                          double[:] bins_rng,
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
                                                         &bins_rng[0],
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
                                         double[:] bins_rng,
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
                                                        &bins_rng[0],
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
                                           double[:] bins_rng,
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
                                                          &bins_rng[0],
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
                                            double[:] bins_rng,
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
                                                           &bins_rng[0],
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
                                           double[:] bins_rng,
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
                                                          &bins_rng[0],
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
                                             double[:] bins_rng,
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
                                                            &bins_rng[0],
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
                                          double[:] bins_rng,
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
                                                         &bins_rng[0],
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
                                         double[:] bins_rng,
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
                                                        &bins_rng[0],
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
                                           double[:] bins_rng,
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
                                                          &bins_rng[0],
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
                                         double[:] bins_rng,
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
                                                        &bins_rng[0],
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
                                        double[:] bins_rng,
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
                                                       &bins_rng[0],
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
                                          double[:] bins_rng,
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
                                                         &bins_rng[0],
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
                                           double[:] bins_rng,
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
                                                          &bins_rng[0],
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
                                          double[:] bins_rng,
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
                                                         &bins_rng[0],
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
                                            double[:] bins_rng,
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
                                                           &bins_rng[0],
                                                           &n_bins[0],
                                                           &histo[0],
                                                           &cumul[0],
                                                           &bin_edges[0],
                                                           option_flags,
                                                           weight_min,
                                                           weight_max)
