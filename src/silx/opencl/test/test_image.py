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
__date__ = "13/02/2018"

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
except ImportError:
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
        self.data = numpy.asarray(Image.open(self.lena))

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
        tmp = pyopencl.array.empty(self.ip.ctx, self.data.shape, "float32")
        res = self.ip.to_float(self.data, out=tmp)
        res2 = self.ip.normalize(tmp, -100, 100, copy=False)
        norm = (self.data.astype(numpy.float32) - self.data.min()) / (self.data.max() - self.data.min())
        ref2 = 200 * norm - 100
        self.assertLess(abs(res2 - ref2).max(), 3e-5, "content")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_histogram(self):
        """
        Test on a greyscaled image ... of Lena :)
        """
        lena_bw = (0.2126 * self.data[:, :, 0] +
                   0.7152 * self.data[:, :, 1] +
                   0.0722 * self.data[:, :, 2]).astype("int32")
        ref = numpy.histogram(lena_bw, 255)
        ip = ImageProcessing(ctx=self.ctx, template=lena_bw, profile=True)
        res = ip.histogram(lena_bw, 255)
        ip.log_profile()
        delta = (ref[0] - res[0])
        deltap = (ref[1] - res[1])
        self.assertEqual(delta.sum(), 0, "errors are self-compensated")
        self.assertLessEqual(abs(delta).max(), 1, "errors are small")
        self.assertLessEqual(abs(deltap).max(), 3e-5, "errors on position are small: %s" % (abs(deltap).max()))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestImage("test_cast"))
    testSuite.addTest(TestImage("test_normalize"))
    testSuite.addTest(TestImage("test_histogram"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
