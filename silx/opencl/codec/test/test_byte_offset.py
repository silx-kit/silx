#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Byte-offset decompression in OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2017  European Synchrotron Radiation Facility, Grenoble, France
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
Test suite for byte-offset decompression 
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/10/2017"

import os
import time
import logging
import numpy
from silx.opencl import ocl
import unittest
logger = logging.getLogger(__name__)


@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestAlgebra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestAlgebra, cls).setUpClass()
        if ocl:
            cls.ctx = ocl.create_context()
            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                cls.PROFILE = False
                cls.queue = pyopencl.CommandQueue(cls.ctx)
            device = cls.ctx.devices[0]
            device_id = device.platform.get_devices().index(device)
            platform_id = pyopencl.get_platforms().index(device.platform)
            cls.maxwg = ocl.platforms[platform_id].devices[device_id].max_work_group_size
#             logger.warning("max_work_group_size: %s on (%s, %s)", cls.maxwg, platform_id, device_id)

    @classmethod
    def tearDownClass(cls):
        super(TestAlgebra, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        kernel_src = get_opencl_code(os.path.join("sift", "algebra.cl"))
        self.program = pyopencl.Program(self.ctx, kernel_src).build()
        self.wg = (32, 4)
        if self.maxwg < self.wg[0] * self.wg[1]:
            self.wg = (self.maxwg, 1)

    def tearDown(self):
        self.mat1 = None
        self.mat2 = None
        self.program = None

    def test_combine(self):
        """
        tests the combine (linear combination) kernel
        """
        width = numpy.int32(157)
        height = numpy.int32(147)
        coeff1 = numpy.random.rand(1)[0].astype(numpy.float32)
        coeff2 = numpy.random.rand(1)[0].astype(numpy.float32)
        mat1 = numpy.random.rand(height, width).astype(numpy.float32)
        mat2 = numpy.random.rand(height, width).astype(numpy.float32)

        gpu_mat1 = pyopencl.array.to_device(self.queue, mat1)
        gpu_mat2 = pyopencl.array.to_device(self.queue, mat2)
        gpu_out = pyopencl.array.empty(self.queue, mat1.shape, dtype=numpy.float32, order="C")
        shape = calc_size((width, height), self.wg)

        t0 = time.time()
        try:
            k1 = self.program.combine(self.queue, shape, None,
                                      gpu_mat1.data, coeff1, gpu_mat2.data, coeff2,
                                      gpu_out.data, numpy.int32(0),
                                      width, height)
        except pyopencl.LogicError as error:
            logger.warning("%s in test_combine", error)
        res = gpu_out.get()
        t1 = time.time()
        ref = my_combine(mat1, coeff1, mat2, coeff2)
        t2 = time.time()
        delta = abs(ref - res).max()
        logger.debug("delta=%s" % delta)
        self.assert_(delta < 1e-4, "delta=%s" % (delta))
        if self.PROFILE:
            logger.debug("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.debug("Linear combination took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))


def suite():
    testSuite = unittest.TestSuite()
     testSuite.addTest(TestByteOffset("test_byte_offset"))
#     testSuite.addTest(TestAlgebra("test_compact"))
    return testSuite
