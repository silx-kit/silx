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

"""
Tests of the histogramnd function, error cases.
"""
import sys
import platform
import unittest

import numpy as np

from silx.math.chistogramnd import chistogramnd as histogramnd
from silx.math import Histogramnd


# ==============================================================
# ==============================================================
# ==============================================================


class _Test_chistogramnd_errors(unittest.TestCase):
    """
    Unit tests of the chistogramnd error cases.
    """
    __test__ = False  # ignore abstract class

    def setUp(self):
        self.skipTest("Abstract class")

    def test_weights_shape(self):
        """
        """

        for err_w_shape in self.err_weights_shapes:
            test_msg = ('Testing invalid weights shape : {0}'
                        ''.format(err_w_shape))

            err_weights = np.random.randint(0,
                                                    high=10,
                                                    size=err_w_shape)
            err_weights = err_weights.astype(np.double)

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           self.histo_range,
                                           self.n_bins,
                                           weights=err_weights)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str,
                             '<weights> must be an array whose length '
                             'is equal to the number of samples.')

    def test_histo_range_shape(self):
        """
        """
        n_dims = 1 if len(self.s_shape) == 1 else self.s_shape[1]
        expected_txt_tpl = ('<histo_range> error : expected {n_dims} sets '
                            'of lower and upper bin edges, '
                            'got the following instead : {histo_range}. '
                            '(provided <sample> contains '
                            '{n_dims}D values)')

        for err_histo_range in self.err_histo_range_shapes:
            test_msg = ('Testing invalid histo_range shape : {0}'
                        ''.format(err_histo_range))

            expected_txt = expected_txt_tpl.format(histo_range=err_histo_range,
                                                   n_dims=n_dims)

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           err_histo_range,
                                           self.n_bins,
                                           weights=self.weights)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)

    def test_nbins_shape(self):
        """
        """

        expected_txt = ('n_bins must be either a scalar (same number '
                        'of bins for all dimensions) or '
                        'an array (number of bins for each '
                        'dimension).')

        for err_n_bins in self.err_n_bins_shapes:
            test_msg = ('Testing invalid n_bins shape : {0}'
                        ''.format(err_n_bins))

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           self.histo_range,
                                           err_n_bins,
                                           weights=self.weights)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)

    def test_nbins_values(self):
        """
        """
        expected_txt = ('<n_bins> : only positive values allowed.')

        for err_n_bins in self.err_n_bins_values:
            test_msg = ('Testing invalid n_bins value : {0}'
                        ''.format(err_n_bins))

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           self.histo_range,
                                           err_n_bins,
                                           weights=self.weights)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)

    def test_histo_shape(self):
        """
        """
        for err_h_shape in self.err_histo_shapes:

            # windows & python 2.7 : numpy shapes are long values
            if platform.system() == 'Windows':
                version = (sys.version_info.major, sys.version_info.minor)
                if version <= (2, 7):
                    err_h_shape = tuple([long(val) for val in err_h_shape])

            test_msg = ('Testing invalid histo shape : {0}'
                        ''.format(err_h_shape))

            expected_txt = ('Provided <histo> array doesn\'t have '
                            'a shape compatible with <n_bins> '
                            ': should be {0} instead of {1}.'
                            ''.format(self.h_shape, err_h_shape))

            histo = np.zeros(shape=err_h_shape, dtype=np.uint32)

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           self.histo_range,
                                           self.n_bins,
                                           weights=self.weights,
                                           histo=histo)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)

    def test_histo_dtype(self):
        """
        """
        for err_h_dtype in self.err_histo_dtypes:
            test_msg = ('Testing invalid histo dtype : {0}'
                        ''.format(err_h_dtype))

            histo = np.zeros(shape=self.h_shape, dtype=err_h_dtype)

            expected_txt = ('Provided <histo> array doesn\'t have '
                            'the expected type '
                            ': should be {0} instead of {1}.'
                            ''.format(np.uint32, histo.dtype))

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           self.histo_range,
                                           self.n_bins,
                                           weights=self.weights,
                                           histo=histo)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)

    def test_weighted_histo_shape(self):
        """
        """
        # using the same values as histo
        for err_h_shape in self.err_histo_shapes:

            # windows & python 2.7 : numpy shapes are long values
            if platform.system() == 'Windows':
                version = (sys.version_info.major, sys.version_info.minor)
                if version <= (2, 7):
                    err_h_shape = tuple([long(val) for val in err_h_shape])

            test_msg = ('Testing invalid weighted_histo shape : {0}'
                        ''.format(err_h_shape))

            expected_txt = ('Provided <weighted_histo> array doesn\'t have '
                            'a shape compatible with <n_bins> '
                            ': should be {0} instead of {1}.'
                            ''.format(self.h_shape, err_h_shape))

            cumul = np.zeros(shape=err_h_shape, dtype=np.double)

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           self.histo_range,
                                           self.n_bins,
                                           weights=self.weights,
                                           weighted_histo=cumul)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)

    def test_cumul_dtype(self):
        """
        """
        # using the same values as histo
        for err_h_dtype in self.err_histo_dtypes:
            test_msg = ('Testing invalid weighted_histo dtype : {0}'
                        ''.format(err_h_dtype))

            cumul = np.zeros(shape=self.h_shape, dtype=err_h_dtype)

            expected_txt = ('Provided <weighted_histo> array doesn\'t have '
                            'the expected type '
                            ': should be {0} or {1} instead of {2}.'
                            ''.format(np.float64, np.float32, cumul.dtype))

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           self.histo_range,
                                           self.n_bins,
                                           weights=self.weights,
                                           weighted_histo=cumul)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)
            
    def test_wh_histo_dtype(self):
        """
        """
        # using the same values as histo
        for err_h_dtype in self.err_histo_dtypes:
            test_msg = ('Testing invalid wh_dtype dtype : {0}'
                        ''.format(err_h_dtype))

            expected_txt = ('<wh_dtype> type not supported : {0}.'
                            ''.format(err_h_dtype))

            ex_str = None
            try:
                histo, cumul = histogramnd(self.sample,
                                           self.histo_range,
                                           self.n_bins,
                                           weights=self.weights,
                                           wh_dtype=err_h_dtype)[0:2]
            except ValueError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)

    def test_unmanaged_dtypes(self):
        """
        """
        for err_unmanaged_dtype in self.err_unmanaged_dtypes:
            test_msg = ('Testing unmanaged dtypes : {0}'
                        ''.format(err_unmanaged_dtype))

            sample = self.sample.astype(err_unmanaged_dtype[0])
            weights = self.weights.astype(err_unmanaged_dtype[1])

            expected_txt = ('Case not supported - sample:{0} '
                            'and weights:{1}.'
                            ''.format(sample.dtype,
                                      weights.dtype))

            ex_str = None
            try:
                histogramnd(sample,
                            self.histo_range,
                            self.n_bins,
                            weights=weights)
            except TypeError as ex:
                ex_str = str(ex)

            self.assertIsNotNone(ex_str, msg=test_msg)
            self.assertEqual(ex_str, expected_txt, msg=test_msg)

    def test_uncontiguous_histo(self):
        """
        """
        # non contiguous array
        shape = np.array(self.n_bins, ndmin=1)
        shape[0] *= 2
        histo_tmp = np.zeros(shape)
        histo = histo_tmp[::2, ...]

        expected_txt = ('<histo> must be a C_CONTIGUOUS numpy array.')

        ex_str = None
        try:
            histogramnd(self.sample,
                        self.histo_range,
                        self.n_bins,
                        weights=self.weights,
                        histo=histo)
        except ValueError as ex:
            ex_str = str(ex)

        self.assertIsNotNone(ex_str)
        self.assertEqual(ex_str, expected_txt)

    def test_uncontiguous_weighted_histo(self):
        """
        """
        # non contiguous array
        shape = np.array(self.n_bins, ndmin=1)
        shape[0] *= 2
        cumul_tmp = np.zeros(shape)
        cumul = cumul_tmp[::2, ...]

        expected_txt = ('<weighted_histo> must be a C_CONTIGUOUS numpy array.')

        ex_str = None
        try:
            histogramnd(self.sample,
                        self.histo_range,
                        self.n_bins,
                        weights=self.weights,
                        weighted_histo=cumul)
        except ValueError as ex:
            ex_str = str(ex)

        self.assertIsNotNone(ex_str)
        self.assertEqual(ex_str, expected_txt)


class Test_chistogramnd_1D_errors(_Test_chistogramnd_errors):
    """
    Unit tests of the 1D histogramnd error cases.
    """
    __test__ = True  # because _Test_chistogramnd_errors is ignored

    def setUp(self):
        # nominal values
        self.n_elements = 1000
        self.s_shape = (self.n_elements,)
        self.w_shape = (self.n_elements,)

        self.histo_range = [0., 100.]
        self.n_bins = 10

        self.h_shape = (self.n_bins,)

        self.sample = np.random.randint(0,
                                                high=10,
                                                size=self.s_shape)
        self.sample = self.sample.astype(np.double)

        self.weights = np.random.randint(0,
                                                 high=10,
                                                 size=self.w_shape)
        self.weights = self.weights.astype(np.double)

        self.err_weights_shapes = ((self.n_elements+1,),
                                   (self.n_elements-1,),
                                   (self.n_elements-1, 3))
        self.err_histo_range_shapes = ([0.],
                                    [0., 1., 2.],
                                    [[0.], [1.]])
        self.err_n_bins_shapes = ([10, 2],
                                  [[10], [2]])
        self.err_n_bins_values = (0,
                                  [-10],
                                  None)
        self.err_histo_shapes = ((self.n_bins+1,),
                                 (self.n_bins-1,),
                                 (self.n_bins, self.n_bins))
        # these are used for testing the histo parameter as well
        #   as the weighted_histo parameter.
        self.err_histo_dtypes = (np.uint16,
                                 np.float16)

        self.err_unmanaged_dtypes = ((np.double, np.uint16),
                                     (np.uint16, np.double),
                                     (np.uint16, np.uint16))

class Test_chistogramnd_ND_range(unittest.TestCase):
    """

    """

    def test_invalid_histo_range(self):
        data = np.random.random((60, 60))
        nbins = 10

        with self.assertRaises(ValueError):
            histo_range = data.min(), np.inf

            Histogramnd(sample=data.ravel(),
                        histo_range=histo_range,
                        n_bins=nbins)

            histo_range = data.min(), np.nan

            Histogramnd(sample=data.ravel(),
                        histo_range=histo_range,
                        n_bins=nbins)


class Test_chistogramnd_ND_errors(_Test_chistogramnd_errors):
    """
    Unit tests of the 3D histogramnd error cases.
    """
    __test__ = True  # because _Test_chistogramnd_errors is ignored

    def setUp(self):
        # nominal values
        self.n_elements = 1000
        self.s_shape = (self.n_elements, 3)
        self.w_shape = (self.n_elements,)

        self.histo_range = [[0., 100.], [0., 100.], [0., 100.]]
        self.n_bins = (10, 20, 30)

        self.h_shape = self.n_bins

        self.sample = np.random.randint(0,
                                                high=10,
                                                size=self.s_shape)
        self.sample = self.sample.astype(np.double)

        self.weights = np.random.randint(0,
                                                 high=10,
                                                 size=self.w_shape)
        self.weights = self.weights.astype(np.double)

        self.err_weights_shapes = ((self.n_elements+1,),
                                   (self.n_elements-1,),
                                   (self.n_elements-1, 3))
        self.err_histo_range_shapes = ([0.],
                                    [0., 1.],
                                    [[0., 10.], [0., 10.]],
                                    [0., 10., 0, 10., 0, 10.])
        self.err_n_bins_shapes = ([10, 2],
                                  [[10], [20], [30]])
        self.err_n_bins_values = (0,
                                  [-10],
                                  [10, 20, -4],
                                  None,
                                  [10, None, 30])
        self.err_histo_shapes = ((self.n_bins[0]+1,
                                  self.n_bins[1],
                                  self.n_bins[2]),
                                 (self.n_bins[0],
                                  self.n_bins[1],
                                  self.n_bins[2]-1),
                                 (self.n_bins[0],
                                  self.n_bins[1]),
                                 (self.n_bins[1],
                                  self.n_bins[0],
                                  self.n_bins[2]),
                                 (self.n_bins[0],
                                  self.n_bins[1],
                                  self.n_bins[2],
                                  10)
                                 )
        # these are used for testing the histo parameter as well
        #   as the weighted_histo parameter.
        self.err_histo_dtypes = (np.uint16,
                                 np.float16)

        self.err_unmanaged_dtypes = ((np.double, np.uint16),
                                     (np.uint16, np.double),
                                     (np.uint16, np.uint16))
