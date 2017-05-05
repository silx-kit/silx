#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
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
Simple test of an addition
"""

from __future__ import division, print_function

__authors__ = ["Henri Payno, Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/03/2017"

import logging
import numpy

import unittest
from ..common import ocl, _measure_workgroup_size
if ocl:
    import pyopencl
    import pyopencl.array
from ..utils import get_opencl_code
logger = logging.getLogger(__name__)


@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestAddition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestAddition, cls).setUpClass()
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
        super(TestAddition, cls).tearDownClass()
        print("Maximum valid workgroup size %s on device %s" % (cls.max_valid_wg, cls.ctx.devices[0]))
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        if ocl is None:
            return
        self.shape = 4096
        self.data = numpy.random.random(self.shape).astype(numpy.float32)
        self.d_array_img = pyopencl.array.to_device(self.queue, self.data)
        self.d_array_5 = pyopencl.array.zeros_like(self.d_array_img) - 5
        self.program = pyopencl.Program(self.ctx, get_opencl_code("addition")).build()

    def tearDown(self):
        self.img = self.data = None
        self.d_array_img = self.d_array_5 = self.program = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_add(self):
        """
        tests the addition  kernel
        """
        maxi = int(round(numpy.log2(self.shape)))
        for i in range(maxi):
            d_array_result = pyopencl.array.empty_like(self.d_array_img)
            wg = 1 << i
            try:
                evt = self.program.addition(self.queue, (self.shape,), (wg,),
                       self.d_array_img.data, self.d_array_5.data, d_array_result.data, numpy.int32(self.shape))
                evt.wait()
            except Exception as error:
                max_valid_wg = self.program.addition.get_work_group_info(pyopencl.kernel_work_group_info.WORK_GROUP_SIZE, self.ctx.devices[0])
                msg = "Error %s on WG=%s: %s" % (error, wg, max_valid_wg)
                self.assertLess(max_valid_wg, wg, msg)
                break
            else:
                res = d_array_result.get()
                good = numpy.allclose(res, self.data - 5)
                if good and wg>self.max_valid_wg:
                    self.__class__.max_valid_wg = wg
                self.assert_(good, "calculation is correct for WG=%s" % wg)

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
    testSuite.addTest(TestAddition("test_add"))
    testSuite.addTest(TestAddition("test_measurement"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
