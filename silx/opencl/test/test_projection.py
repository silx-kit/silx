#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Test of the forward projection module"""

from __future__ import division, print_function

__authors__ = ["Pierre paleo"]
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/01/2018"


import logging
import numpy as np
import unittest
from time import time
try:
    import mako
except ImportError:
    mako = None
from ..common import ocl
if ocl:
    from .. import projection
    import pyopencl.array as parray
from silx.test.utils import utilstest

logger = logging.getLogger(__name__)


@unittest.skipUnless(ocl and mako, "PyOpenCl is missing")
class TestProj(unittest.TestCase):

    def setUp(self):
        if ocl is None:
            return
        self.getfiles()
        n_angles = self.sino_ref.shape[0]
        self.proj = projection.Projection(self.phantom.shape, n_angles)
        logger.debug("Using device %s" % self.proj.device)
        if self.proj.compiletime_workgroup_size < 16 * 16:
            self.skipTest("Current implementation of OpenCL projection is not supported on this platform yet")
        self.rtol = 1e-3

    def tearDown(self):
        self.phantom = None
        self.sino_ref = None
        self.proj = None

    def getfiles(self):
        # load 512x512 MRI phantom
        self.phantom = np.load(utilstest.getfile("Brain512.npz"))["data"]
        # load sinogram computed with PyHST
        self.sino_ref = np.load(utilstest.getfile("sino500_pyhst.npz"))["data"]


    @staticmethod
    def compare(arr1, arr2):
        return np.max(np.abs(arr1 - arr2)/arr1.max())

    def check_result(self, res):
        errmax = self.compare(res, self.sino_ref)
        msg = str("Max error = %e" % errmax)
        logger.info(msg)
        self.assertTrue(errmax < self.rtol, "Max error is too high")

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_proj(self):
        """
        tests Projection
        """
        P = self.proj
        # Test single reconstruction
        # --------------------------
        t0 = time()
        res = P.projection(self.phantom)
        el_ms = (time() - t0)*1e3
        logger.info("test_proj: time = %.3f ms" % el_ms)
        self.check_result(res)

        # Test multiple projections
        # -----------------------------
        res0 = np.copy(res)
        for i in range(10):
            res = P.projection(self.phantom)
            errmax = np.max(np.abs(res - res0))
            self.assertTrue(errmax < 1.e-6, "Max error is too high")

        # Test projection with input and/or output on device
        # --------------------------------------------------
        d_input = parray.to_device(P.queue, self.phantom)
        d_output = parray.zeros(P.queue, P.sino_shape, "f")
        res = P.projection(d_input)
        self.check_result(res)
        P.projection(self.phantom, output=d_output)
        self.check_result(d_output.get())
        P.projection(d_input, output=d_output)
        self.check_result(d_output.get())



def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestProj("test_proj"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
