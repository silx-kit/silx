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
"""
Nominal tests of the HistogramndLut function.
"""

import unittest

import numpy as np

from silx.math import HistogramndLut


def _get_bin_edges(histo_range, n_bins, n_dims):
    edges = []
    for i_dim in range(n_dims):
        edges.append(histo_range[i_dim, 0] +
                     np.arange(n_bins[i_dim] + 1) *
                     (histo_range[i_dim, 1] - histo_range[i_dim, 0]) /
                     n_bins[i_dim])
    return tuple(edges)


# ==============================================================
# ==============================================================
# ==============================================================


class _TestHistogramndLut_nominal(unittest.TestCase):
    """
    Unit tests of the HistogramndLut class.
    """

    ndims = None

    def setUp(self):
        ndims = self.ndims
        self.tested_dim = ndims-1

        if ndims is None:
            raise ValueError('ndims class member not set.')

        sample = np.array([5.5,        -3.3,
                           0.,         -0.5,
                           3.3,        8.8,
                           -7.7,       6.0,
                           -4.0])

        weights = np.array([500.5,    -300.3,
                            0.01,      -0.5,
                            300.3,     800.8,
                            -700.7,    600.6,
                            -400.4])

        n_elems = len(sample)

        if ndims == 1:
            shape = (n_elems,)
        else:
            shape = (n_elems, ndims)

        self.sample = np.zeros(shape=shape, dtype=sample.dtype)
        if ndims == 1:
            self.sample = sample
        else:
            self.sample[..., ndims-1] = sample

        self.weights = weights

        # the tests are performed along one dimension,
        #   all the other bins indices along the other dimensions
        #   are expected to be 2
        # (e.g : when testing a 2D sample : [0, x] will go into
        # bin [2, y] because of the bin ranges [-2, 2] and n_bins = 4
        # for the first dimension)
        self.other_axes_index = 2
        self.histo_range = np.repeat([[-2., 2.]], ndims, axis=0)
        self.histo_range[ndims-1] = [-4., 6.]

        self.n_bins = np.array([4]*ndims)
        self.n_bins[ndims-1] = 5

        if ndims == 1:
            def fill_histo(h, v, dim, op=None):
                if op:
                    h[:] = op(h[:], v)
                else:
                    h[:] = v
            self.fill_histo = fill_histo
        else:
            def fill_histo(h, v, dim, op=None):
                idx = [self.other_axes_index]*len(h.shape)
                idx[dim] = slice(0, None)
                if op:
                    h[idx] = op(h[idx], v)
                else:
                    h[idx] = v
            self.fill_histo = fill_histo

    def test_nominal_bin_edges(self):

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        bin_edges = instance.bins_edges

        expected_edges = _get_bin_edges(self.histo_range,
                                        self.n_bins,
                                        self.ndims)

        for i_edges, edges in enumerate(expected_edges):
            self.assertTrue(np.array_equal(bin_edges[i_edges],
                                           expected_edges[i_edges]),
                            msg='Testing bin_edges for dim {0}'
                                ''.format(i_edges+1))

    def test_nominal_histo_range(self):

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        histo_range = instance.histo_range

        self.assertTrue(np.array_equal(histo_range, self.histo_range))

    def test_nominal_last_bin_closed(self):

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        last_bin_closed = instance.last_bin_closed

        self.assertEqual(last_bin_closed, False)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins,
                                  last_bin_closed=True)

        last_bin_closed = instance.last_bin_closed

        self.assertEqual(last_bin_closed, True)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins,
                                  last_bin_closed=False)

        last_bin_closed = instance.last_bin_closed

        self.assertEqual(last_bin_closed, False)

    def test_nominal_n_bins_array(self):

        test_n_bins = np.arange(self.ndims) + 10
        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  test_n_bins)

        n_bins = instance.n_bins

        self.assertTrue(np.array_equal(test_n_bins, n_bins))

    def test_nominal_n_bins_scalar(self):

        test_n_bins = 10
        expected_n_bins = np.array([test_n_bins] * self.ndims)
        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  test_n_bins)

        n_bins = instance.n_bins

        self.assertTrue(np.array_equal(expected_n_bins, n_bins))

    def test_nominal_histo_ref(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700.7, -0.5, 0.01, 300.3, 500.5])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        instance.accumulate(self.weights)

        histo = instance.histo()
        w_histo = instance.weighted_histo()
        histo_ref = instance.histo(copy=False)
        w_histo_ref = instance.weighted_histo(copy=False)

        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))
        self.assertTrue(np.array_equal(histo_ref, expected_h))
        self.assertTrue(np.array_equal(w_histo_ref, expected_c))

        histo_ref[0, ...] = histo_ref[0, ...] + 10
        w_histo_ref[0, ...] = w_histo_ref[0, ...] + 20

        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))
        self.assertFalse(np.array_equal(histo_ref, expected_h))
        self.assertFalse(np.array_equal(w_histo_ref, expected_c))

        histo_2 = instance.histo()
        w_histo_2 = instance.weighted_histo()

        self.assertFalse(np.array_equal(histo_2, expected_h))
        self.assertFalse(np.array_equal(w_histo_2, expected_c))
        self.assertTrue(np.array_equal(histo_2, histo_ref))
        self.assertTrue(np.array_equal(w_histo_2, w_histo_ref))

    def test_nominal_accumulate_once(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700.7, -0.5, 0.01, 300.3, 500.5])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        instance.accumulate(self.weights)

        histo = instance.histo()
        w_histo = instance.weighted_histo()

        self.assertEqual(w_histo.dtype, np.float64)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))
        self.assertTrue(np.array_equal(instance.histo(), expected_h))
        self.assertTrue(np.array_equal(instance.weighted_histo(),
                                       expected_c))

    def test_nominal_accumulate_twice(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700.7, -0.5, 0.01, 300.3, 500.5])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        # calling accumulate twice
        expected_h *= 2
        expected_c *= 2

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        instance.accumulate(self.weights)

        instance.accumulate(self.weights)

        histo = instance.histo()
        w_histo = instance.weighted_histo()

        self.assertEqual(w_histo.dtype, np.float64)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))
        self.assertTrue(np.array_equal(instance.histo(), expected_h))
        self.assertTrue(np.array_equal(instance.weighted_histo(),
                                       expected_c))

    def test_nominal_apply_lut_once(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700.7, -0.5, 0.01, 300.3, 500.5])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        histo, w_histo = instance.apply_lut(self.weights)

        self.assertEqual(w_histo.dtype, np.float64)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))
        self.assertEqual(instance.histo(), None)
        self.assertEqual(instance.weighted_histo(), None)

    def test_nominal_apply_lut_twice(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700.7, -0.5, 0.01, 300.3, 500.5])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        # calling apply_lut twice
        expected_h *= 2
        expected_c *= 2

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        histo, w_histo = instance.apply_lut(self.weights)
        histo_2, w_histo_2 = instance.apply_lut(self.weights,
                                                histo=histo,
                                                weighted_histo=w_histo)

        self.assertEqual(id(histo), id(histo_2))
        self.assertEqual(id(w_histo), id(w_histo_2))
        self.assertEqual(w_histo.dtype, np.float64)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))
        self.assertEqual(instance.histo(), None)
        self.assertEqual(instance.weighted_histo(), None)

    def test_nominal_accumulate_last_bin_closed(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 2])
        expected_c_tpl = np.array([-700.7, -0.5, 0.01, 300.3, 1101.1])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins,
                                  last_bin_closed=True)

        instance.accumulate(self.weights)

        histo = instance.histo()
        w_histo = instance.weighted_histo()

        self.assertEqual(w_histo.dtype, np.float64)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))

    def test_nominal_accumulate_weight_min_max(self):
        """
        """
        weight_min = -299.9
        weight_max = 499.9

        expected_h_tpl = np.array([0, 1, 1, 1, 0])
        expected_c_tpl = np.array([0., -0.5, 0.01, 300.3, 0.])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        instance.accumulate(self.weights,
                            weight_min=weight_min,
                            weight_max=weight_max)

        histo = instance.histo()
        w_histo = instance.weighted_histo()

        self.assertEqual(w_histo.dtype, np.float64)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))

    def test_nominal_accumulate_forced_int32(self):
        """
        double weights, int32 weighted_histogram
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700, 0, 0, 300, 500])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins,
                                  dtype=np.int32)

        instance.accumulate(self.weights)

        histo = instance.histo()
        w_histo = instance.weighted_histo()

        self.assertEqual(w_histo.dtype, np.int32)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))

    def test_nominal_accumulate_forced_float32(self):
        """
        int32 weights, float32 weighted_histogram
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700., 0., 0., 300., 500.])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.float32)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins,
                                  dtype=np.float32)

        instance.accumulate(self.weights.astype(np.int32))

        histo = instance.histo()
        w_histo = instance.weighted_histo()

        self.assertEqual(w_histo.dtype, np.float32)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))

    def test_nominal_accumulate_int32(self):
        """
        int32 weights
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700, 0, 0, 300, 500])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.int32)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        instance.accumulate(self.weights.astype(np.int32))

        histo = instance.histo()
        w_histo = instance.weighted_histo()

        self.assertEqual(w_histo.dtype, np.int32)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))

    def test_nominal_accumulate_int32_double(self):
        """
        int32 weights
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700, 0, 0, 300, 500])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.int32)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        instance = HistogramndLut(self.sample,
                                  self.histo_range,
                                  self.n_bins)

        instance.accumulate(self.weights.astype(np.int32))
        instance.accumulate(self.weights)

        histo = instance.histo()
        w_histo = instance.weighted_histo()

        expected_h *= 2
        expected_c *= 2

        self.assertEqual(w_histo.dtype, np.int32)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(w_histo, expected_c))

    def testNoneNativeTypes(self):
        type = self.sample.dtype.newbyteorder("B")
        sampleB = self.sample.astype(type)

        type = self.sample.dtype.newbyteorder("L")
        sampleL = self.sample.astype(type)

        histo_inst = HistogramndLut(sampleB,
                                 self.histo_range,
                                 self.n_bins)

        histo_inst = HistogramndLut(sampleL,
                                 self.histo_range,
                                 self.n_bins)


class TestHistogramndLut_nominal_1d(_TestHistogramndLut_nominal):
    ndims = 1


class TestHistogramndLut_nominal_2d(_TestHistogramndLut_nominal):
    ndims = 2


class TestHistogramndLut_nominal_3d(_TestHistogramndLut_nominal):
    ndims = 3


# ==============================================================
# ==============================================================
# ==============================================================


test_cases = (TestHistogramndLut_nominal_1d,
              TestHistogramndLut_nominal_2d,
              TestHistogramndLut_nominal_3d,)


def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
