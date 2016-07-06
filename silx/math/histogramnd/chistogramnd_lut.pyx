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
__date__ = "15/05/2016"


cimport numpy as np  # noqa
cimport cython
import numpy as np

ctypedef fused sample_t:
    np.float64_t
    np.float32_t
    np.int32_t
    np.int64_t

ctypedef fused cumul_t:
    np.float64_t
    np.float32_t
    np.int32_t
    np.int64_t

ctypedef fused weights_t:
    np.float64_t
    np.float32_t
    np.int32_t
    np.int64_t

ctypedef fused lut_t:
    np.int64_t
    np.int32_t
    np.int16_t


def histogramnd_get_lut(sample,
                        histo_range,
                        n_bins,
                        last_bin_closed=False):
    """
    histogramnd_get_lut(sample, histo_range, n_bins, last_bin_closed=False)

    TBD

    :param sample:
        The data to be histogrammed.
        Its shape must be either (N,) if it contains one dimensional
        coordinates, or an (N, D) array where the rows are the
        coordinates of points in a D dimensional space.
        The following dtypes are supported : :class:`numpy.float64`,
        :class:`numpy.float32`, :class:`numpy.int32`.
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

    :param last_bin_closed:
        By default the last bin is half
        open (i.e.: [x,y) ; x included, y
        excluded), like all the other bins.
        Set this parameter to true if you want
        the LAST bin to be closed.
    :type last_bin_closed: *optional*, :class:`python.boolean`

    :return: The indices for each sample and the histogram (bin counts).
    :rtype: tuple : (:class:`numpy.array`, :class:`numpy.array`)
    """

    s_shape = sample.shape

    n_dims = 1 if len(s_shape) == 1 else s_shape[1]

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
            histo_range.reshape(-1)
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

    histo_range = np.double(histo_range)

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

    sample_type = sample.dtype

    n_elem = sample.size // n_dims

    if n_bins.prod(dtype=np.uint64) < 2**15:
        lut_dtype = np.int16
    elif n_bins.prod(dtype=np.uint64) < 2**31:
        lut_dtype = np.int32
    else:
        lut_dtype = np.int64

    # allocating the output arrays
    lut = np.zeros(n_elem, dtype=lut_dtype)
    histo = np.zeros(n_bins, dtype=np.uint32)

    sample_c = np.ascontiguousarray(sample.reshape((sample.size,)))

    histo_range_c = np.ascontiguousarray(histo_range.reshape((histo_range.size,)))

    n_bins_c = np.ascontiguousarray(n_bins.reshape((n_bins.size,)),
                                    dtype=np.int32)

    lut_c = np.ascontiguousarray(lut.reshape((lut.size,)))
    histo_c = np.ascontiguousarray(histo.reshape((histo.size,)))

    rc = 0

    try:
        rc = _histogramnd_get_lut_fused(sample_c,
                                        n_dims,
                                        n_elem,
                                        histo_range_c,
                                        n_bins_c,
                                        lut_c,
                                        histo_c,
                                        last_bin_closed)
    except TypeError as ex:
        raise TypeError('Type not supported - sample : {0}'
                        ''.format(sample_type))

    if rc != 0:
        raise Exception('histogramnd returned an error : {0}'
                        ''.format(rc))

    edges = []
    histo_range = histo_range.reshape(-1)
    for i_dim in range(n_dims):
        dim_edges = np.zeros(n_bins[i_dim] + 1)
        rng_min = histo_range[2 * i_dim]
        rng_max = histo_range[2 * i_dim + 1]
        dim_edges[:-1] = (rng_min + np.arange(n_bins[i_dim]) *
                          ((rng_max - rng_min) / n_bins[i_dim]))
        dim_edges[-1] = rng_max
        edges.append(dim_edges)

    return lut, histo, tuple(edges)


# =====================
# =====================


def histogramnd_from_lut(weights,
                         histo_lut,
                         histo=None,
                         weighted_histo=None,
                         shape=None,
                         dtype=None,
                         weight_min=None,
                         weight_max=None):
    """
    dtype ignored if weighted_histo provided
    """

    if histo is None and weighted_histo is None:
        if shape is None:
            raise ValueError('At least one of the following parameters has to '
                             'be provided : <shape> or <histo> or '
                             '<weighted_histo>')

    if shape is not None:
        if histo is not None and list(histo.shape) != list(shape):
            raise ValueError('The <shape> value does not match'
                             'the <histo> shape.')

        if(weighted_histo is not None and
           list(weighted_histo.shape) != list(shape)):
            raise ValueError('The <shape> value does not match'
                             'the <weighted_histo> shape.')
    else:
        if histo is not None:
            shape = histo.shape
        else:
            shape = weighted_histo.shape

    if histo is not None:
        if histo.dtype != np.uint32:
            raise ValueError('Provided <histo> array doesn\'t have '
                             'the expected type '
                             ': should be {0} instead of {1}.'
                             ''.format(np.uint32, histo.dtype))

        if weighted_histo is not None:
            if histo.shape != weighted_histo.shape:
                raise ValueError('The <histo> shape does not match'
                                 'the <weighted_histo> shape.')
    else:
        histo = np.zeros(shape, dtype=np.uint32)

    w_dtype = weights.dtype

    if dtype is None:
        if weighted_histo is None:
            dtype = w_dtype
        else:
            dtype = weighted_histo.dtype
    elif weighted_histo is not None:
        if weighted_histo.dtype != dtype:
            raise ValueError('Provided <dtype> and <weighted_histo>\'s dtype'
                             ' do not match.')
        dtype = weighted_histo.dtype

    if weighted_histo is None:
        weighted_histo = np.zeros(shape, dtype=dtype)

    if histo_lut.size != weights.size:
        raise ValueError('The LUT and weights arrays must have the same '
                         'number of elements.')

    w_c = np.ascontiguousarray(weights.reshape((weights.size,)))

    h_c = np.ascontiguousarray(histo.reshape((histo.size,)))

    w_h_c = np.ascontiguousarray(weighted_histo.reshape((weighted_histo.size,)))  # noqa

    h_lut_c = np.ascontiguousarray(histo_lut.reshape((histo_lut.size,)))

    rc = 0

    if weight_min is None:
        weight_min = 0
        filt_min_weights = False
    else:
        filt_min_weights = True

    if weight_max is None:
        weight_max = 0
        filt_max_weights = False
    else:
        filt_max_weights = True

    try:
        _histogramnd_from_lut_fused(w_c,
                                    h_lut_c,
                                    h_c,
                                    w_h_c,
                                    weights.size,
                                    filt_min_weights,
                                    w_dtype.type(weight_min),
                                    filt_max_weights,
                                    w_dtype.type(weight_max))
    except TypeError as ex:
        print(ex)
        raise TypeError('Case not supported - weights:{0} '
                        'and histo:{1}.'
                        ''.format(weights.dtype, histo.dtype))

    return histo, weighted_histo


# =====================
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _histogramnd_from_lut_fused(weights_t[:] i_weights,
                                lut_t[:] i_lut,
                                np.uint32_t[:] o_histo,
                                cumul_t[:] o_weighted_histo,
                                int i_n_elems,
                                bint i_filt_min_weights,
                                weights_t i_weight_min,
                                bint i_filt_max_weights,
                                weights_t i_weight_max):
    with nogil:
        for i in range(i_n_elems):
            if (i_lut[i] >= 0):
                if i_filt_min_weights and i_weights[i] < i_weight_min:
                    continue
                if i_filt_max_weights and i_weights[i] > i_weight_max:
                    continue
                o_histo[i_lut[i]] += 1
                o_weighted_histo[i_lut[i]] += <cumul_t>i_weights[i]  # noqa


# =====================
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _histogramnd_get_lut_fused(sample_t[:] i_sample,
                               int i_n_dims,
                               int i_n_elems,
                               double[:] i_histo_range,
                               int[:] i_n_bins,
                               lut_t[:] o_lut,
                               np.uint32_t[:] o_histo,
                               bint last_bin_closed):

    cdef:
        int i = 0
        long elem_idx = 0
        long max_idx = 0
        long lut_idx = -1

        # computed bin index (i_sample -> grid)
        long bin_idx = 0

        sample_t elem_coord = 0

        double[50] g_min
        double[50] g_max
        double[50] bins_range

    for i in range(i_n_dims):
        g_min[i] = i_histo_range[2*i]
        g_max[i] = i_histo_range[2*i+1]
        bins_range[i] = g_max[i] - g_min[i]

    elem_idx = 0 - i_n_dims
    max_idx = i_n_elems * i_n_dims - i_n_dims

    with nogil:
        while elem_idx < max_idx:
            elem_idx += i_n_dims
            lut_idx += 1

            bin_idx = 0

            for i in range(i_n_dims):
                elem_coord = i_sample[elem_idx+i]
                # =====================
                # Element is rejected if any of the following is NOT true :
                # 1. coordinate is >= than the minimum value
                # 2. coordinate is <= than the maximum value
                # 3. coordinate==maximum value and last_bin_closed is True
                # =====================
                if elem_coord < g_min[i]:
                    bin_idx = -1
                    break

                # Here we make the assumption that most of the time
                # there will be more coordinates inside the grid interval
                #  (one test)
                #  than coordinates higher or equal to the max
                #  (two tests)
                if elem_coord < g_max[i]:
                    bin_idx = <long>(bin_idx * i_n_bins[i] +  # noqa
                                     (((elem_coord - g_min[i]) * i_n_bins[i]) /
                                      bins_range[i]))
                else:
                    # if equal and the last bin is closed :
                    #  put it in the last bin
                    # else : discard
                    if last_bin_closed and elem_coord == g_max[i]:
                        bin_idx = (bin_idx + 1) * i_n_bins[i] - 1
                    else:
                        bin_idx = -1
                        break

            o_lut[lut_idx] = bin_idx
            if bin_idx >= 0:
                o_histo[bin_idx] += 1

    return 0
