#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: image manipulation in  OpenCL
#             https://github.com/silx-kit/silx
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
Simple test of image manipulation
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/10/2017"

import logging
import numpy

import unittest
from ..common import ocl, _measure_workgroup_size
if ocl:
    import pyopencl
    import pyopencl.array
from ...test.utils import utilstest
from ..image import ImageProcessing
logger = logging.getLogger(__name__)
try:
    from PIL import Image
except ImportWarning:
    Image = None


@unittest.skipUnless(ocl and Image, "PyOpenCl/Image is missing")
class TestImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestImage, cls).setUpClass()
        if ocl:
            cls.ctx = ocl.create_context()
            cls.lena = utilstest.getfile("lena.png")
            cls.data = numpy.asarray(Image.open(cls.lena))
            cls.ip = ImageProcessing(ctx=cls.ctx, template=cls.data, profile=True)

    @classmethod
    def tearDownClass(cls):
        super(TestImage, cls).tearDownClass()
        cls.ctx = None
        cls.lena = None
        cls.data = None
        if logger.level <= logging.INFO:
            logger.warning("\n".join(cls.ip.log_profile()))
        cls.ip = None

    def setUp(self):
        if ocl is None:
            return

    def tearDown(self):
        self.img = self.data = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_cast(self):
        """
        tests the cast kernel
        """

        res = self.ip.to_float(self.data)
        self.assertEqual(res.shape, self.data.shape, "shape")
        self.assertEqual(res.dtype, numpy.float32, "dtype")
        self.assertEqual(abs(res - self.data).max(), 0, "content")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_normalize(self):
        """
        tests that all devices are working properly ...
        """
        res = self.ip.to_float(self.data)
        res2 = self.ip.normalize(res, -100, 100)
        norm = (self.data.astype(numpy.float32) - self.data.min()) / (self.data.max() - self.data.min())
        ref2 = 200 * norm - 100
        self.assertLess(abs(res2 - ref2).max(), 3e-5, "content")


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestImage("test_cast"))
    testSuite.addTest(TestImage("test_normalize"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
