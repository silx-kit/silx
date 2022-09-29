# /*##########################################################################
# Copyright (C) 2016-2021 European Synchrotron Radiation Facility
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
"""
Tests for the histogramnd function.
Results are compared to numpy's histogramdd.
"""

import os
import unittest
import operator
import pytest

import numpy as np

from silx.math.chistogramnd import chistogramnd as histogramnd

# ==============================================================
# ==============================================================
# ==============================================================

_RTOL_DICT = {np.float64: 10**-13,
              np.float32: 10**-5}

# ==============================================================
# ==============================================================
# ==============================================================


def _add_values_to_array_if_missing(array, values, n_values):
    max_in_col = np.any(array[:, ...] == values, axis=0)

    if len(array.shape) == 1:
        if not max_in_col:
            rnd_idx = np.random.randint(0,
                                                high=len(array)-1,
                                                size=(n_values,))
            array[rnd_idx] = values
    else:
        for i in range(len(max_in_col)):
            if not max_in_col[i]:
                rnd_idx = np.random.randint(0,
                                                    high=len(array)-1,
                                                    size=(n_values,))
                array[rnd_idx, i] = values[i]


def _get_values_index(array, values, op=operator.lt):
    idx = op(array[:, ...], values)
    if array.ndim > 1:
        idx = np.all(idx, axis=1)
    return np.where(idx)[0]


def _get_in_range_indices(array,
                          minvalues,
                          maxvalues,
                          minop=operator.ge,
                          maxop=operator.lt):
    idx = np.logical_and(minop(array, minvalues),
                         maxop(array, maxvalues))
    if array.ndim > 1:
        idx = np.all(idx, axis=1)
    return np.where(idx)[0]


class _TestHistogramnd(unittest.TestCase):
    """
    Unit tests of the histogramnd function.
    """
    __test__ = False  # ignore abstract class

    sample_rng = None
    weights_rng = None
    n_dims = None

    filter_min = None
    filter_max = None

    histo_range = None
    n_bins = None

    dtype_sample = None
    dtype_weights = None

    def generate_data(self):

        self.longMessage = True

        int_min = 0
        int_max = 100000
        n_elements = 10**5

        if self.n_dims == 1:
            shape = (n_elements,)
        else:
            shape = (n_elements, self.n_dims,)

        self.rng_state = np.random.get_state()

        self.state_msg = ('Current RNG state :\n'
                          '{0}'.format(self.rng_state))

        sample = np.random.randint(int_min,
                                           high=int_max,
                                           size=shape)

        sample = sample.astype(self.dtype_sample)
        sample = (self.sample_rng[0] +
                  (sample-int_min) *
                  (self.sample_rng[1]-self.sample_rng[0]) /
                  (int_max-int_min)).astype(self.dtype_sample)

        weights = np.random.randint(int_min,
                                            high=int_max,
                                            size=(n_elements,))
        weights = weights.astype(self.dtype_weights)
        weights = (self.weights_rng[0] +
                   (weights-int_min) *
                   (self.weights_rng[1]-self.weights_rng[0]) /
                   (int_max-int_min)).astype(self.dtype_weights)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # the bins range are cast to the same type as the sample
        # in order to get the same results as numpy
        # (which doesnt cast the range)
        self.histo_range = np.array(self.histo_range).astype(self.dtype_sample)

        # adding some values that are equal to the max
        #   in order to test the opened/closed last bin
        bins_max = [b[1] for b in self.histo_range]
        _add_values_to_array_if_missing(sample,
                                        bins_max,
                                        100)

        # adding some values that are equal to the min weight value
        #   in order to test the filters
        _add_values_to_array_if_missing(weights,
                                        self.weights_rng[0],
                                        100)

        # adding some values that are equal to the max weight value
        #   in order to test the filters
        _add_values_to_array_if_missing(weights,
                                        self.weights_rng[1],
                                        100)

        return sample, weights

    def setUp(self):
        if type(self).__name__.startswith("_"):
            self.skipTest("Abstract class")
        self.sample, self.weights = self.generate_data()
        self.rtol = _RTOL_DICT.get(self.dtype_weights, None)

    def array_compare(self, ar_a, ar_b):
        if self.rtol is None:
            return np.array_equal(ar_a, ar_b)
        return np.allclose(ar_a, ar_b, self.rtol)

    def test_bin_ranges(self):
        """

        """
        result_c = histogramnd(self.sample,
                               self.histo_range,
                               self.n_bins,
                               weights=self.weights,
                               last_bin_closed=True)

        result_np = np.histogramdd(self.sample,
                                   bins=self.n_bins,
                                   range=self.histo_range)

        for i_edges, edges in enumerate(result_c[2]):
            # allclose for now until I can try with the latest version (TBD)
            # of numpy
            self.assertTrue(np.allclose(edges,
                                        result_np[1][i_edges]),
                            msg='{0}. Testing bin_edges for dim {1}.'
                                ''.format(self.state_msg, i_edges+1))

    def test_last_bin_closed(self):
        """

        """
        result_c = histogramnd(self.sample,
                               self.histo_range,
                               self.n_bins,
                               weights=self.weights,
                               last_bin_closed=True)

        result_np = np.histogramdd(self.sample,
                                   bins=self.n_bins,
                                   range=self.histo_range)

        result_np_w = np.histogramdd(self.sample,
                                     bins=self.n_bins,
                                     range=self.histo_range,
                                     weights=self.weights)

        # comparing "hits"
        hits_cmp = np.array_equal(result_c[0],
                                  result_np[0])
        # comparing weights
        weights_cmp = np.array_equal(result_c[1],
                                     result_np_w[0])

        self.assertTrue(hits_cmp, msg=self.state_msg)
        self.assertTrue(weights_cmp, msg=self.state_msg)

        bins_min = [rng[0] for rng in self.histo_range]
        bins_max = [rng[1] for rng in self.histo_range]
        inrange_idx = _get_in_range_indices(self.sample,
                                            bins_min,
                                            bins_max,
                                            minop=operator.ge,
                                            maxop=operator.le)

        self.assertEqual(result_c[0].sum(), inrange_idx.shape[0],
                         msg=self.state_msg)

        # we have to sum the weights using the same precision as the
        # histogramnd function
        weights_sum = self.weights[inrange_idx].astype(result_c[1].dtype).sum()
        self.assertTrue(self.array_compare(result_c[1].sum(), weights_sum),
                        msg=self.state_msg)

    def test_last_bin_open(self):
        """

        """
        result_c = histogramnd(self.sample,
                               self.histo_range,
                               self.n_bins,
                               weights=self.weights,
                               last_bin_closed=False)

        bins_max = [rng[1] for rng in self.histo_range]
        filtered_idx = _get_values_index(self.sample, bins_max)

        result_np = np.histogramdd(self.sample[filtered_idx],
                                   bins=self.n_bins,
                                   range=self.histo_range)

        result_np_w = np.histogramdd(self.sample[filtered_idx],
                                     bins=self.n_bins,
                                     range=self.histo_range,
                                     weights=self.weights[filtered_idx])

        # comparing "hits"
        hits_cmp = np.array_equal(result_c[0], result_np[0])
        # comparing weights
        weights_cmp = np.array_equal(result_c[1],
                                     result_np_w[0])

        self.assertTrue(hits_cmp, msg=self.state_msg)
        self.assertTrue(weights_cmp, msg=self.state_msg)

        bins_min = [rng[0] for rng in self.histo_range]
        bins_max = [rng[1] for rng in self.histo_range]
        inrange_idx = _get_in_range_indices(self.sample,
                                            bins_min,
                                            bins_max,
                                            minop=operator.ge,
                                            maxop=operator.lt)

        self.assertEqual(result_c[0].sum(), len(inrange_idx),
                         msg=self.state_msg)
        # we have to sum the weights using the same precision as the
        # histogramnd function
        weights_sum = self.weights[inrange_idx].astype(result_c[1].dtype).sum()
        self.assertTrue(self.array_compare(result_c[1].sum(), weights_sum),
                        msg=self.state_msg)

    def test_filter_min(self):
        """

        """
        result_c = histogramnd(self.sample,
                               self.histo_range,
                               self.n_bins,
                               weights=self.weights,
                               last_bin_closed=True,
                               weight_min=self.filter_min)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        filter_min = self.dtype_weights(self.filter_min)

        weight_idx = _get_values_index(self.weights,
                                       filter_min,  # <------ !!!
                                       operator.ge)

        result_np = np.histogramdd(self.sample[weight_idx],
                                   bins=self.n_bins,
                                   range=self.histo_range)

        result_np_w = np.histogramdd(self.sample[weight_idx],
                                     bins=self.n_bins,
                                     range=self.histo_range,
                                     weights=self.weights[weight_idx])

        # comparing "hits"
        hits_cmp = np.array_equal(result_c[0],
                                  result_np[0])
        # comparing weights
        weights_cmp = np.array_equal(result_c[1], result_np_w[0])

        self.assertTrue(hits_cmp, msg=self.state_msg)
        self.assertTrue(weights_cmp, msg=self.state_msg)

        bins_min = [rng[0] for rng in self.histo_range]
        bins_max = [rng[1] for rng in self.histo_range]
        inrange_idx = _get_in_range_indices(self.sample[weight_idx],
                                            bins_min,
                                            bins_max,
                                            minop=operator.ge,
                                            maxop=operator.le)

        inrange_idx = weight_idx[inrange_idx]

        self.assertEqual(result_c[0].sum(), len(inrange_idx),
                         msg=self.state_msg)

        # we have to sum the weights using the same precision as the
        # histogramnd function
        weights_sum = self.weights[inrange_idx].astype(result_c[1].dtype).sum()
        self.assertTrue(self.array_compare(result_c[1].sum(), weights_sum),
                        msg=self.state_msg)

    def test_filter_max(self):
        """

        """
        result_c = histogramnd(self.sample,
                               self.histo_range,
                               self.n_bins,
                               weights=self.weights,
                               last_bin_closed=True,
                               weight_max=self.filter_max)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        filter_max = self.dtype_weights(self.filter_max)

        weight_idx = _get_values_index(self.weights,
                                       filter_max,  # <------ !!!
                                       operator.le)

        result_np = np.histogramdd(self.sample[weight_idx],
                                   bins=self.n_bins,
                                   range=self.histo_range)

        result_np_w = np.histogramdd(self.sample[weight_idx],
                                     bins=self.n_bins,
                                     range=self.histo_range,
                                     weights=self.weights[weight_idx])

        # comparing "hits"
        hits_cmp = np.array_equal(result_c[0],
                                  result_np[0])
        # comparing weights
        weights_cmp = np.array_equal(result_c[1], result_np_w[0])

        self.assertTrue(hits_cmp, msg=self.state_msg)
        self.assertTrue(weights_cmp, msg=self.state_msg)

        bins_min = [rng[0] for rng in self.histo_range]
        bins_max = [rng[1] for rng in self.histo_range]
        inrange_idx = _get_in_range_indices(self.sample[weight_idx],
                                            bins_min,
                                            bins_max,
                                            minop=operator.ge,
                                            maxop=operator.le)

        inrange_idx = weight_idx[inrange_idx]

        self.assertEqual(result_c[0].sum(), len(inrange_idx),
                         msg=self.state_msg)

        # we have to sum the weights using the same precision as the
        # histogramnd function
        weights_sum = self.weights[inrange_idx].astype(result_c[1].dtype).sum()
        self.assertTrue(self.array_compare(result_c[1].sum(), weights_sum),
                        msg=self.state_msg)

    def test_filter_minmax(self):
        """

        """
        result_c = histogramnd(self.sample,
                               self.histo_range,
                               self.n_bins,
                               weights=self.weights,
                               last_bin_closed=True,
                               weight_min=self.filter_min,
                               weight_max=self.filter_max)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        filter_min = self.dtype_weights(self.filter_min)
        filter_max = self.dtype_weights(self.filter_max)

        weight_idx = _get_in_range_indices(self.weights,
                                           filter_min,  # <------ !!!
                                           filter_max,  # <------ !!!
                                           minop=operator.ge,
                                           maxop=operator.le)

        result_np = np.histogramdd(self.sample[weight_idx],
                                   bins=self.n_bins,
                                   range=self.histo_range)

        result_np_w = np.histogramdd(self.sample[weight_idx],
                                     bins=self.n_bins,
                                     range=self.histo_range,
                                     weights=self.weights[weight_idx])

        # comparing "hits"
        hits_cmp = np.array_equal(result_c[0],
                                  result_np[0])
        # comparing weights
        weights_cmp = np.array_equal(result_c[1], result_np_w[0])

        self.assertTrue(hits_cmp)
        self.assertTrue(weights_cmp)

        bins_min = [rng[0] for rng in self.histo_range]
        bins_max = [rng[1] for rng in self.histo_range]
        inrange_idx = _get_in_range_indices(self.sample[weight_idx],
                                            bins_min,
                                            bins_max,
                                            minop=operator.ge,
                                            maxop=operator.le)

        inrange_idx = weight_idx[inrange_idx]

        self.assertEqual(result_c[0].sum(), len(inrange_idx),
                         msg=self.state_msg)

        # we have to sum the weights using the same precision as the
        # histogramnd function
        weights_sum = self.weights[inrange_idx].astype(result_c[1].dtype).sum()
        self.assertTrue(self.array_compare(result_c[1].sum(), weights_sum),
                        msg=self.state_msg)

    def test_reuse_histo(self):
        """

        """
        result_c_1 = histogramnd(self.sample,
                                 self.histo_range,
                                 self.n_bins,
                                 weights=self.weights,
                                 last_bin_closed=True)

        result_np_1 = np.histogramdd(self.sample,
                                     bins=self.n_bins,
                                     range=self.histo_range)

        np.histogramdd(self.sample,
                       bins=self.n_bins,
                       range=self.histo_range,
                       weights=self.weights)

        sample_2, weights_2 = self.generate_data()

        result_c_2 = histogramnd(sample_2,
                                 self.histo_range,
                                 self.n_bins,
                                 weights=weights_2,
                                 last_bin_closed=True,
                                 histo=result_c_1[0])

        result_np_2 = np.histogramdd(sample_2,
                                     bins=self.n_bins,
                                     range=self.histo_range)

        result_np_w_2 = np.histogramdd(sample_2,
                                       bins=self.n_bins,
                                       range=self.histo_range,
                                       weights=weights_2)

        # comparing "hits"
        hits_cmp = np.array_equal(result_c_2[0],
                                  result_np_1[0] +
                                  result_np_2[0])
        # comparing weights
        weights_cmp = np.array_equal(result_c_2[1],
                                     result_np_w_2[0])

        self.assertTrue(hits_cmp, msg=self.state_msg)
        self.assertTrue(weights_cmp, msg=self.state_msg)

    def test_reuse_cumul(self):
        """

        """
        result_c = histogramnd(self.sample,
                               self.histo_range,
                               self.n_bins,
                               weights=self.weights,
                               last_bin_closed=True)

        np.histogramdd(self.sample,
                       bins=self.n_bins,
                       range=self.histo_range)

        result_np_w = np.histogramdd(self.sample,
                                     bins=self.n_bins,
                                     range=self.histo_range,
                                     weights=self.weights)

        sample_2, weights_2 = self.generate_data()

        result_c_2 = histogramnd(sample_2,
                                 self.histo_range,
                                 self.n_bins,
                                 weights=weights_2,
                                 last_bin_closed=True,
                                 weighted_histo=result_c[1])

        result_np_2 = np.histogramdd(sample_2,
                                     bins=self.n_bins,
                                     range=self.histo_range)

        result_np_w_2 = np.histogramdd(sample_2,
                                       bins=self.n_bins,
                                       range=self.histo_range,
                                       weights=weights_2)

        # comparing "hits"
        hits_cmp = np.array_equal(result_c_2[0],
                                  result_np_2[0])
        # comparing weights

        self.assertTrue(hits_cmp, msg=self.state_msg)
        self.assertTrue(self.array_compare(result_c_2[1],
                                           result_np_w[0] + result_np_w_2[0]),
                        msg=self.state_msg)

    def test_reuse_cumul_float(self):
        """

        """
        n_bins = np.array(self.n_bins, ndmin=1)
        if len(self.sample.shape) == 2:
            if len(n_bins) == self.sample.shape[1]:
                shp = tuple([x for x in n_bins])
            else:
                shp = (self.n_bins,) * self.sample.shape[1]
            cumul = np.zeros(shp, dtype=np.float32)
        else:
            shp = (self.n_bins,)
            cumul = np.zeros(shp, dtype=np.float32)

        result_c_1 = histogramnd(self.sample,
                                 self.histo_range,
                                 self.n_bins,
                                 weights=self.weights,
                                 last_bin_closed=True,
                                 weighted_histo=cumul)

        result_np_1 = np.histogramdd(self.sample,
                                     bins=self.n_bins,
                                     range=self.histo_range)

        result_np_w_1 = np.histogramdd(self.sample,
                                       bins=self.n_bins,
                                       range=self.histo_range,
                                       weights=self.weights)

        # comparing "hits"
        hits_cmp = np.array_equal(result_c_1[0],
                                  result_np_1[0])

        self.assertTrue(hits_cmp, msg=self.state_msg)
        self.assertEqual(result_c_1[1].dtype, np.float32, msg=self.state_msg)

        bins_min = [rng[0] for rng in self.histo_range]
        bins_max = [rng[1] for rng in self.histo_range]
        inrange_idx = _get_in_range_indices(self.sample,
                                            bins_min,
                                            bins_max,
                                            minop=operator.ge,
                                            maxop=operator.le)
        weights_sum = \
            self.weights[inrange_idx].astype(np.float32).sum(dtype=np.float64)
        self.assertTrue(np.allclose(result_c_1[1].sum(dtype=np.float64),
                        weights_sum), msg=self.state_msg)
        self.assertTrue(np.allclose(result_c_1[1].sum(dtype=np.float64),
                                    result_np_w_1[0].sum(dtype=np.float64)),
                        msg=self.state_msg)

    @pytest.mark.usefixtures("use_large_memory")
    def test_histo_big_array(self):
        """
        Test histogram on arrays with more than 2**31-1 samples.
        """
        if self.sample.ndim > 1:
            self.skipTest("Test only many samples along one dimension")
        if self.sample.dtype.itemsize > 4:
            self.skipTest("Test only many samples for itemsize < 4")
        n_repeat = (2**31 + 10) // self.sample.size
        sample = np.repeat(self.sample, n_repeat)
        n_bins = int(1e6)
        result_c = histogramnd(
            sample,
            self.histo_range,
            n_bins,
            last_bin_closed=True
        )
        result_np = np.histogramdd(
            sample,
            n_bins,
            range=self.histo_range
        )
        for i_edges, edges in enumerate(result_c[2]):
            self.assertTrue(
                np.allclose(
                    edges,
                    result_np[1][i_edges]
                ),
                msg='{0}. Testing bin_edges for dim {1}.'''.format(self.state_msg, i_edges+1)
            )




class _TestHistogramnd_1d(_TestHistogramnd):
    """
    Unit tests of the 1D histogramnd function.
    """
    sample_rng = [-55., 100.]
    weights_rng = [-70., 150.]
    n_dims = 1
    filter_min = -15.6
    filter_max = 85.7

    histo_range = [[-30.2, 90.3]]
    n_bins = 30

    dtype = None


class _TestHistogramnd_2d(_TestHistogramnd):
    """
    Unit tests of the 1D histogramnd function.
    """
    sample_rng = [-50.2, 100.99]
    weights_rng = [70., 150.]
    n_dims = 2
    filter_min = 81.7
    filter_max = 135.3

    histo_range = [[10., 90.], [20., 70.]]
    n_bins = 30

    dtype = None


class _TestHistogramnd_3d(_TestHistogramnd):
    """
    Unit tests of the 1D histogramnd function.
    """
    sample_rng = [10.2, 200.9]
    weights_rng = [0., 100.]
    n_dims = 3
    filter_min = 31.5
    filter_max = 83.7

    histo_range = [[30.8, 150.2], [20.1, 90.9], [10.1, 195.]]
    n_bins = 30

    dtype = None


# ################################################################
# ################################################################
# ################################################################
# ################################################################


class TestHistogramnd_1d_double_double(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.double


class TestHistogramnd_1d_double_float(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.float32


class TestHistogramnd_1d_double_int32(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.int32


class TestHistogramnd_1d_float_double(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.double


class TestHistogramnd_1d_float_float(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.float32


class TestHistogramnd_1d_float_int32(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.int32


class TestHistogramnd_1d_int32_double(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.double


class TestHistogramnd_1d_int32_float(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.float32


class TestHistogramnd_1d_int32_int32(_TestHistogramnd_1d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.int32


class TestHistogramnd_2d_double_double(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.double


class TestHistogramnd_2d_double_float(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.float32


class TestHistogramnd_2d_double_int32(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.int32


class TestHistogramnd_2d_float_double(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.double


class TestHistogramnd_2d_float_float(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.float32


class TestHistogramnd_2d_float_int32(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.int32


class TestHistogramnd_2d_int32_double(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.double


class TestHistogramnd_2d_int32_float(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.float32


class TestHistogramnd_2d_int32_int32(_TestHistogramnd_2d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.int32


class TestHistogramnd_3d_double_double(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.double


class TestHistogramnd_3d_double_float(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.float32


class TestHistogramnd_3d_double_int32(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.double
    dtype_weights = np.int32


class TestHistogramnd_3d_float_double(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.double


class TestHistogramnd_3d_float_float(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.float32


class TestHistogramnd_3d_float_int32(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.float32
    dtype_weights = np.int32


class TestHistogramnd_3d_int32_double(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.double


class TestHistogramnd_3d_int32_float(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.float32


class TestHistogramnd_3d_int32_int32(_TestHistogramnd_3d):
    __test__ = True  # because _TestHistogramnd is ignored
    dtype_sample = np.int32
    dtype_weights = np.int32
