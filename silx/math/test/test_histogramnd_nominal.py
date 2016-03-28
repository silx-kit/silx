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
Nominal tests of the histogramnd function.
"""

import unittest

import numpy as np

from silx.math import histogramnd

# ==============================================================
# ==============================================================
# ==============================================================


class _TestHistogramnd_nominal(unittest.TestCase):
    """
    Unit tests of the histogramnd function.
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
        self.other_axes_index = 2
        self.bins_rng = np.repeat([[-2., 2.]], ndims, axis=0)
        self.bins_rng[ndims-1] = [-4., 6.]

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

    def test_nominal(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])
        expected_c_tpl = np.array([-700.7, -0.5, 0.01, 300.3, 500.5])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=self.weights)

        self.assertEqual(cumul.dtype, np.float64)
        self.assertEqual(histo.dtype, np.uint32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(cumul, expected_c))

    def test_nominal_wo_weights(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=None)

        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(cumul is None)

    def test_nominal_wo_weights_w_cumul(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)

        cumul_in = np.zeros(self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=None,
                                   cumul=cumul_in)

        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(cumul is None)
        self.assertTrue(np.array_equal(cumul_in,
                                       np.zeros(shape=self.n_bins,
                                                dtype=np.double)))

    def test_nominal_wo_weights_w_histo(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 1])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)

        # creating an array of ones just to make sure that
        # it is not cleared by histogramnd
        histo_in = np.ones(self.n_bins, dtype=np.uint32)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=None,
                                   histo=histo_in)

        self.assertTrue(np.array_equal(histo, expected_h + 1))
        self.assertTrue(cumul is None)
        self.assertEqual(id(histo), id(histo_in))

    def test_nominal_last_bin_closed(self):
        """
        """
        expected_h_tpl = np.array([2, 1, 1, 1, 2])
        expected_c_tpl = np.array([-700.7, -0.5, 0.01, 300.3, 1101.1])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=self.weights,
                                   last_bin_closed=True)

        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(cumul, expected_c))

    def test_int32_weights_double_weights_range(self):
        """
        """
        weight_min = -299.9  # ===> will be cast to -299
        weight_max = 499.9  # ===> will be cast to 499

        expected_h_tpl = np.array([0, 1, 1, 1, 0])
        expected_c_tpl = np.array([0., 0., 0., 300., 0.])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=self.weights.astype(np.int32),
                                   weight_min=weight_min,
                                   weight_max=weight_max)

        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(cumul, expected_c))

    def test_reuse_histo(self):
        """
        """

        expected_h_tpl = np.array([2, 3, 2, 2, 2])
        expected_c_tpl = np.array([0.0, -7007, -5.0, 0.1, 3003.0])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=self.weights)

        sample_2 = self.sample[:]
        if len(sample_2.shape) == 1:
            idx = [slice(0, None)]
        else:
            idx = [slice(0, None), self.tested_dim]

        sample_2[idx] += 2

        histo_2, cumul = histogramnd(sample_2,          # <==== !!
                                     self.bins_rng,
                                     self.n_bins,
                                     weights=10 * self.weights,  # <==== !!
                                     histo=histo)

        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.array_equal(cumul, expected_c))
        self.assertEqual(id(histo), id(histo_2))

    def test_reuse_cumul(self):
        """
        """

        expected_h_tpl = np.array([0, 2, 1, 1, 1])
        expected_c_tpl = np.array([-700.7, -7007.5, -4.99, 300.4, 3503.5])

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=self.weights)

        sample_2 = self.sample[:]
        if len(sample_2.shape) == 1:
            idx = [slice(0, None)]
        else:
            idx = [slice(0, None), self.tested_dim]

        sample_2[idx] += 2

        histo, cumul_2 = histogramnd(sample_2,           # <==== !!
                                     self.bins_rng,
                                     self.n_bins,
                                     weights=10 * self.weights,  # <==== !!
                                     cumul=cumul)

        self.assertEqual(cumul.dtype, np.float64)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertTrue(np.allclose(cumul, expected_c, rtol=10e-15))
        self.assertEqual(id(cumul), id(cumul_2))

    def test_reuse_cumul_float(self):
        """
        """

        expected_h_tpl = np.array([0, 2, 1, 1, 1])
        expected_c_tpl = np.array([-700.7, -7007.5, -4.99, 300.4, 3503.5],
                                  dtype=np.float32)

        expected_h = np.zeros(shape=self.n_bins, dtype=np.double)
        expected_c = np.zeros(shape=self.n_bins, dtype=np.double)

        self.fill_histo(expected_h, expected_h_tpl, self.ndims-1)
        self.fill_histo(expected_c, expected_c_tpl, self.ndims-1)

        histo, cumul = histogramnd(self.sample,
                                   self.bins_rng,
                                   self.n_bins,
                                   weights=self.weights)

        # converting the cumul array to float
        cumul = cumul.astype(np.float32)

        sample_2 = self.sample[:]
        if len(sample_2.shape) == 1:
            idx = [slice(0, None)]
        else:
            idx = [slice(0, None), self.tested_dim]

        sample_2[idx] += 2

        histo, cumul_2 = histogramnd(sample_2,           # <==== !!
                                     self.bins_rng,
                                     self.n_bins,
                                     weights=10 * self.weights,  # <==== !!
                                     cumul=cumul)

        self.assertEqual(cumul.dtype, np.float32)
        self.assertTrue(np.array_equal(histo, expected_h))
        self.assertEqual(id(cumul), id(cumul_2))
        self.assertTrue(np.allclose(cumul, expected_c, rtol=10e-15))


class TestHistogram_nominal_1d(_TestHistogramnd_nominal):
    ndims = 1


class TestHistogram_nominal_2d(_TestHistogramnd_nominal):
    ndims = 2


class TestHistogram_nominal_3d(_TestHistogramnd_nominal):
    ndims = 3


# ==============================================================
# ==============================================================
# ==============================================================


test_cases = (TestHistogram_nominal_1d,
              TestHistogram_nominal_2d,
              TestHistogram_nominal_3d,)


def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
