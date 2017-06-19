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
"""Test of the filtered backprojection module"""

from __future__ import division, print_function

__authors__ = ["Pierre paleo"]
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "14/06/2017"


import sys
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
    import pyopencl
    import pyopencl.array
    from .. import backprojection

logger = logging.getLogger(__name__)




@unittest.skipUnless(ocl and mako, "PyOpenCl is missing")
class TestFBP(unittest.TestCase):

    def setUp(self):
        if ocl is None:
            return
        # Create a dummy sinogram
        self.sino = np.random.randn(500, 512)
        self.fbp = backprojection.Backprojection(self.sino.shape)
        #~ if self.fbp.device.type == "CPU":
            #~ self.skipTest("Backprojection is not implemented on CPU yet")
        if sys.platform.startswith('darwin'):
            self.skipTest("Backprojection is not implemented on CPU for OS X yet")

    def tearDown(self):
        self.sino = None
        self.fbp = None

    def measure(self):
        "Common measurement of timings"
        t1 = time.time()
        try:
            result = self.fbp.filtered_backprojection(self.sino)
        except RuntimeError as msg:
            logger.error(msg)
            return
        t2 = time.time()
        return t2 - t1

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_fbp(self):
        """
        tests FBP
        """
        r = self.measure()
        if r is None:
            logger.info("test_fp: skipped")
        else:
            logger.info("test_medfilt: time = %.3fs" % r)



def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestFBP("test_fbp"))
    return testSuite



if __name__ == '__main__':
    unittest.main(defaultTest="suite")
