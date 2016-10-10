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
Test suite for all reductionsessing kernels.
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/10/2016"

import os
import unittest
import time
import logging
import math
import numpy
try:
    import scipy
except ImportError:
    scipy = None
else:
    import scipy.misc
from silx.opencl import ocl
if ocl:
    import pyopencl, pyopencl.array

from ..utils import get_opencl_code

logger = logging.getLogger(__name__)


@unittest.skipUnless(scipy and ocl, "scipy or ocl missing")
class TestReduction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestReduction, cls).setUpClass()
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
        super(TestReduction, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        kernel_src = get_opencl_code("reductions")
        self.program = pyopencl.Program(self.ctx, kernel_src).build(options="")

    def tearDown(self):
        self.program = None

    def test_max_min_rnd(self):
        self.test_max_min(numpy.random.randint(1000), -numpy.random.randint(1000))

    @unittest.skipIf(os.environ.get("SILX_TEST_LOW_MEM") == "True", "low mem")
    def test_max_min_rnd_big(self):
        self.test_max_min(512, 0, (1980, 2560))

    def test_max_min(self, val_max=1.0, val_min=0.0, shape=((512, 512)), data=None):
        """
        Test global_max_min kernel
        """

        if data is None:
            logger.info("values: %s -> %s" % (val_min, val_max))
            data = ((val_max - val_min) * numpy.random.random(shape) + val_min).astype(numpy.float32)
#            data = numpy.arange(shape[0] * shape[1], dtype="float32").reshape(shape)
            #        data = numpy.zeros(shape, dtype=numpy.float32)
        inp_gpu = pyopencl.array.to_device(self.queue, data)
        wg_float = min(512.0, numpy.sqrt(data.size))
        wg = 2 ** (int(math.ceil(math.log(wg_float, 2))))
        if self.maxwg < wg:
            logger.info("Skip test_max_min as wg=% < red_size=%s", self.maxwg, wg)
            return

        size = wg * wg
        max_min_gpu = pyopencl.array.zeros(self.queue, (wg, 2), dtype=numpy.float32, order="C")
#        max_min_gpu = pyopencl.array.empty(self.queue, (wg, 2), dtype=numpy.float32, order="C")
        max_gpu = pyopencl.array.empty(self.queue, (1,), dtype=numpy.float32, order="C")
        min_gpu = pyopencl.array.empty(self.queue, (1,), dtype=numpy.float32, order="C")
        logger.info("workgroup: %s, size: %s" % (wg, size))
        t = time.time()
        nmin = data.min()
        nmax = data.max()
        t0 = time.time()
        k1 = self.program.max_min_global_stage1(self.queue, (size,), (wg,), inp_gpu.data, max_min_gpu.data, numpy.uint32(data.size))
        k2 = self.program.max_min_global_stage2(self.queue, (wg,), (wg,), max_min_gpu.data, max_gpu.data, min_gpu.data)
        k2.wait()
        t1 = time.time()
        min_res = min_gpu.get()
        max_res = max_gpu.get()

        logger.info("temp res: max %s min %s", max_min_gpu.get().max(), max_min_gpu.get().min())
#        logger.info( max_min_gpu.get()

        logger.info("Fina res: max %s min %s", min_res, max_res)
        t1 = time.time()
        min_pyocl = pyopencl.array.min(inp_gpu, self.queue).get()
        max_pyocl = pyopencl.array.max(inp_gpu, self.queue).get()
        t2 = time.time()
        max_res = max_res.max()
        min_res = min_res.min()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms, pyopencl: %.3fms." % (1000.0 * (t0 - t), 1000.0 * (t1 - t0), 1000.0 * (t2 - t1)))
            logger.info("reduction took %.3fms + %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start), 1e-6 * (k2.profile.end - k2.profile.start)))
        logger.info("Minimum: ref %s obt %s other %s", nmin, min_res, min_pyocl)
        logger.info("Maximum: ref %s obt %s other %s", nmax, max_res, max_pyocl)
        logger.info("where min %s %s ", numpy.where(data == nmin), numpy.where(data.ravel() == nmin))
        logger.info("where max %s %s ", numpy.where(data == nmax), numpy.where(data.ravel() == nmax))
        self.assertEqual(nmin, min_res, "min: numpy vs ours")
        self.assertEqual(nmax, max_res, "max: numpy vs ours")
        self.assertEqual(nmin, min_pyocl, "min: numpy vs pyopencl")
        self.assertEqual(nmax, max_pyocl, "max: numpy vs pyopencl")
        self.assertEqual(min_pyocl, min_res, "min: ours vs pyopencl")
        self.assertEqual(max_pyocl, max_res, "max: ours vs pyopencl")


def suite():
    testSuite = unittest.TestSuite()
    if ocl:
        testSuite.addTest(TestReduction("test_max_min_rnd"))
        testSuite.addTest(TestReduction("test_max_min"))
        testSuite.addTest(TestReduction("test_max_min_rnd_big"))


    return testSuite
