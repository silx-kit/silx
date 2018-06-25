#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2017  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Test suite for alignment module
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/06/2018"

import unittest
import logging
import numpy
try:
    import scipy.misc
    import scipy.ndimage
except ImportError:
    scipy = None

from ...common import ocl
if ocl:
    import pyopencl

from ..alignment import LinearAlign
logger = logging.getLogger(__name__)
PRINT_KEYPOINTS = False


class TestLinalign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestLinalign, cls).setUpClass()
        if ocl:
            cls.ctx = ocl.create_context()
            print(cls.ctx, logger.getEffectiveLevel() <= logging.INFO)

            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
                cls.queue = pyopencl.CommandQueue(cls.ctx,
                                                  properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                cls.PROFILE = False
                cls.queue = pyopencl.CommandQueue(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        super(TestLinalign, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        if scipy and ocl is None:
            return

        if hasattr(scipy.misc, "ascent"):
            self.lena = scipy.misc.ascent().astype(numpy.float32)
        else:
            self.lena = scipy.misc.lena().astype(numpy.float32)

        self.shape = self.lena.shape
        self.extra = (10, 11)
        self.img = scipy.ndimage.affine_transform(self.lena, [[1.1, -0.1], [0.05, 0.9]], [7, 5])
        self.align = LinearAlign(self.lena, ctx=self.ctx)

    def tearDown(self):
        self.img = self.lena = None

    @unittest.skipUnless(scipy and ocl, "scipy or pyopencl are missing")
    def test_align(self):
        """
        tests the combine (linear combination) kernel
        """
        out = self.align.align(self.img, 0, 1)
        self.align.log_profile()
        out = out["result"]

        if self.PROFILE and (out is not None):
            delta = (out - self.lena)[100:400, 100:400]
            logger.info({"min": delta.min(), "max:": delta.max(), "mean": delta.mean(), "std:": delta.std()})


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestLinalign("test_align"))
    return testSuite
