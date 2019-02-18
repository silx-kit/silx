#!/usr/bin/env python
# coding: utf-8
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
# ###########################################################################*/

"""
Test of the Convolution class.
"""

from __future__ import division, print_function

__authors__ = ["Pierre Paleo"]
__contact__ = "pierre.paleo@esrf.fr"
__license__ = "MIT"
__copyright__ = "2019 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/02/2019"

import logging
import numpy as np
from math import ceil
try:
    from scipy.ndimage import convolve, convolve1d
    from scipy.misc import ascent
    scipy_convolve = convolve
    scipy_convolve1d = convolve1d
except ImportError:
    scipy_convolve = None
import unittest
from ..common import ocl
if ocl:
    import pyopencl
    import pyopencl.array
    from ..convolution import Convolution
logger = logging.getLogger(__name__)


# TODO move elsewhere
def gaussian_kernel(sigma, cutoff=4, force_odd_size=False):
    size = int(ceil(2 * cutoff * sigma + 1))
    if force_odd_size and size % 2 == 0:
        size += 1
    x = np.arange(size) - (size - 1.0) / 2.0
    g = np.exp(-(x / sigma) ** 2 / 2.0)
    g /= g.sum()
    return g


@unittest.skipUnless(ocl and scipy_convolve, "PyOpenCl/scipy is missing")
class TestConvolution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestConvolution, cls).setUpClass()
        cls.image = np.ascontiguousarray(ascent()[:, :511], dtype="f")
        cls.kernel = gaussian_kernel(1.)
        cls.kernel2d = np.outer(cls.kernel, cls.kernel)
        cls.kernel3d = np.multiply.outer(cls.kernel2d, cls.kernel)
        cls.ctx = ocl.create_context()
        cls.tol = {
            "1D": 1e-4,
            "2D": 1e-3,
            "3D": 1e-3,
        }


    @classmethod
    def tearDownClass(cls):
        cls.image = None


    @staticmethod
    def compare(arr1, arr2):
        return np.max(np.abs(arr1 - arr2))


    def test_1D(self):
        data = self.image[0]
        conv = Convolution(data.shape, self.kernel, ctx=self.ctx)
        res = conv(data)
        ref = scipy_convolve1d(data, self.kernel, mode="wrap")
        metric = self.compare(res, ref)
        logger.info("test_1D: max error = %.2e" % metric)
        self.assertLess(metric, self.tol["1D"], "Something wrong with 1D convolution")


    def test_separable_2D(self):
        data = self.image
        conv = Convolution(data.shape, self.kernel, ctx=self.ctx)
        res = conv(data)
        ref1 = scipy_convolve1d(data, self.kernel, mode="wrap", axis=1)
        ref = scipy_convolve1d(ref1, self.kernel, mode="wrap", axis=0)
        metric = self.compare(res, ref)
        logger.info("test_separable_2D: max error = %.2e" % metric)
        self.assertLess(
            metric,
            self.tol["1D"],
            "Something wrong with separable 2D convolution"
        )


    def test_separable_3D(self):
        data = np.tile(self.image, (64, 1, 1))
        conv = Convolution(data.shape, self.kernel, ctx=self.ctx)
        res = conv(data)
        ref1 = scipy_convolve1d(data, self.kernel, mode="wrap", axis=2)
        ref2 = scipy_convolve1d(ref1, self.kernel, mode="wrap", axis=1)
        ref = scipy_convolve1d(ref2, self.kernel, mode="wrap", axis=0)
        metric = self.compare(res, ref)
        logger.info("test_separable_3D: max error = %.2e" % metric)
        self.assertLess(
            metric,
            self.tol["1D"],
            "Something wrong with separable 3D convolution"
        )


    def test_nonseparable_2D(self):
        data = self.image
        kernel = np.outer(self.kernel, self.kernel) # "non-separable" kernel
        conv = Convolution(data.shape, kernel, ctx=self.ctx)
        res = conv(data)
        ref = scipy_convolve(data, kernel, mode="wrap")
        metric = self.compare(res, ref)
        logger.info("test_nonseparable_2D: max error = %.2e" % metric)
        self.assertLess(
            metric,
            self.tol["2D"],
            "Something wrong with nonseparable 2D convolution"
        )


    def test_nonseparable_3D(self):
        data = np.tile(self.image[224:-224, 224:-224], (62, 1, 1))
        kernel = self.kernel3d
        conv = Convolution(data.shape, kernel, ctx=self.ctx)
        res = conv(data)
        ref = scipy_convolve(data, kernel, mode="wrap")
        metric = self.compare(res, ref)
        logger.info("test_nonseparable_3D: max error = %.2e" % metric)
        self.assertLess(
            metric,
            self.tol["3D"],
            "Something wrong with nonseparable 3D convolution"
        )



def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestConvolution("test_1D"))
    testSuite.addTest(TestConvolution("test_separable_2D"))
    testSuite.addTest(TestConvolution("test_separable_3D"))
    testSuite.addTest(TestConvolution("test_nonseparable_2D"))
    testSuite.addTest(TestConvolution("test_nonseparable_3D"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
