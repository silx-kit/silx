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
__date__ = "11/10/2017"

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
from PIL import Image


@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestImage, cls).setUpClass()
        if ocl:
            cls.ctx = ocl.create_context()
            cls.lena = utilstest.getfile("lena.png")

    @classmethod
    def tearDownClass(cls):
        super(TestImage, cls).tearDownClass()
        cls.ctx = None
        cls.lena = None

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
        ip = ImageProcessing(ctx=self.ctx, template=self.data, profile=True)
        res = ip.to_float(self.data)
        self.assertEqual(res.shape, self.data.shape, "shape")
        self.assertEqual(res.dtype, numpy.float32, "dtype")
        self.assertEqual(abs(res - self.data).max(), 0, "content")

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_measurement(self):
        """
        tests that all devices are working properly ...
        """
        for platform in ocl.platforms:
            for did, device in enumerate(platform.devices):
                meas = _measure_workgroup_size((platform.id, device.id))
                self.assertEqual(meas, device.max_work_group_size,
                                 "Workgroup size for %s/%s: %s == %s" % (platform, device, meas, device.max_work_group_size))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestImage("test_cast"))
    # testSuite.addTest(TestAddition("test_measurement"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
