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

__authors__ = ["Jerome Kieffer"]
__license__ = "MIT"
__copyright__ = "2013-2018 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/10/2018"

import os
import scipy.misc
import logging
import numpy
import unittest
from .. import ocl
if ocl:
    from .. import fft
    from .. import pyopencl

logger = logging.getLogger(__name__)


@unittest.skipUnless(fft.gpyfft, "gpyfft is missing")
class TestFFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFFT, cls).setUpClass()
        if ocl:
            cls.ctx = ocl.create_context()
            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
                cls.queue = pyopencl.CommandQueue(
                                cls.ctx,
                                properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                cls.PROFILE = False
                cls.queue = pyopencl.CommandQueue(cls.ctx)
            cls.max_valid_wg = 0

    @classmethod
    def tearDownClass(cls):
        super(TestFFT, cls).tearDownClass()
        print("Maximum valid workgroup size %s on device %s" % (cls.max_valid_wg, cls.ctx.devices[0]))
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        if ocl is None:
            return
        self.data = scipy.misc.ascent().astype("float32")
        self.shape = self.data.shape

    def tearDown(self):
        self.img = self.shape = None
        self.d_array_img = self.d_array_5 = self.program = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_fft(self):
        """
        tests the fft  kernel
        """
        f = fft.FFT(self.shape, profile=True)
        res_ocl = f.fft(self.data)
        res_np = numpy.fft.rfft2(self.data)
        err = abs(res_ocl - res_np).max()
        self.assertLess(err, 1e-3 * self.data.max(), "Results are roughly the same")
        logger.debug(os.linesep.join(f.log_profile(verbose=False)))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestFFT("test_fft"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
