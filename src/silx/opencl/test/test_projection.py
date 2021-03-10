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


import time
import logging
import numpy as np
import unittest
try:
    import mako
except ImportError:
    mako = None
from ..common import ocl
if ocl:
    from .. import projection
from silx.test.utils import utilstest

logger = logging.getLogger(__name__)


@unittest.skipUnless(ocl and mako, "PyOpenCl is missing")
class TestProj(unittest.TestCase):

    def setUp(self):
        if ocl is None:
            return
        # ~ if sys.platform.startswith('darwin'):
            # ~ self.skipTest("Projection is not implemented on CPU for OS X yet")
        self.getfiles()
        n_angles = self.sino.shape[0]
        self.proj = projection.Projection(self.phantom.shape, n_angles)
        if self.proj.compiletime_workgroup_size < 16 * 16:
            self.skipTest("Current implementation of OpenCL projection is not supported on this platform yet")

    def tearDown(self):
        self.phantom = None
        self.sino = None
        self.proj = None

    def getfiles(self):
        # load 512x512 MRI phantom
        self.phantom = np.load(utilstest.getfile("Brain512.npz"))["data"]
        # load sinogram computed with PyHST
        self.sino = np.load(utilstest.getfile("sino500_pyhst.npz"))["data"]

    def measure(self):
        "Common measurement of timings"
        t1 = time.time()
        try:
            result = self.proj.projection(self.phantom)
        except RuntimeError as msg:
            logger.error(msg)
            return
        t2 = time.time()
        return t2 - t1, result

    def compare(self, res):
        """
        Compare a result with the reference reconstruction.
        Only the valid reconstruction zone (inscribed circle) is taken into account
        """
        # Compare with the original phantom.
        # TODO: compare a standard projection
        ref = self.sino
        return np.max(np.abs(res - ref))

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_proj(self):
        """
        tests Projection
        """
        # Test single reconstruction
        # --------------------------
        t, res = self.measure()
        if t is None:
            logger.info("test_proj: skipped")
        else:
            logger.info("test_proj: time = %.3fs" % t)
            err = self.compare(res)
            msg = str("Max error = %e" % err)
            logger.info(msg)
            # Interpolation differs at some lines, giving relative error of 10/50000
            self.assertTrue(err < 20., "Max error is too high")
        # Test multiple reconstructions
        # -----------------------------
        res0 = np.copy(res)
        for i in range(10):
            res = self.proj.projection(self.phantom)
            errmax = np.max(np.abs(res - res0))
            self.assertTrue(errmax < 1.e-6, "Max error is too high")


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestProj("test_proj"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
