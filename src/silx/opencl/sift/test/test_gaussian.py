#!/usr/bin/env python
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2018  European Synchrotron Radiation Facility, Grenoble, France
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
Test suite for all preprocessing kernels.
"""

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/07/2018"

import os
import time
import numpy
import unittest
import logging
from silx.opencl import ocl, kernel_workgroup_size
try:
    import scipy
except ImportError:
    scipy = None

try:
    import mako
except ImportError:
    mako = None


from ..utils import get_opencl_code
logger = logging.getLogger(__name__)

if ocl:
    import pyopencl.array


def gaussian_cpu(sigma, size=None, PROFILE=False):
    """
    Calculate a 1D gaussian using numpy.
    This is the same as scipy.signal.gaussian

    :param sigma: width of the gaussian
    :param size: can be calculated as 1 + 2 * 4sigma
    """
    t0 = time.time()
    if not size:
        size = int(1 + 8 * sigma)
    x = numpy.arange(size) - (size - 1.0) / 2.0
    g = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
    g /= g.sum(dtype=numpy.float32)

    if PROFILE:
        logger.info("execution time: %.3fms on CPU" % (1e3 * (time.time() - t0)))
    return g


@unittest.skipUnless(mako and ocl and scipy, "ocl or scipy is missing")
class TestGaussian(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestGaussian, cls).setUpClass()
        cls.ctx = ocl.create_context()

        if logger.getEffectiveLevel() <= logging.INFO:
            cls.PROFILE = True
            cls.queue = pyopencl.CommandQueue(cls.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            cls.PROFILE = False
            cls.queue = pyopencl.CommandQueue(cls.ctx)

        cls.kernels = {"preprocess": 8,
                       "gaussian": 512}

        device = cls.ctx.devices[0]
        device_id = device.platform.get_devices().index(device)
        platform_id = pyopencl.get_platforms().index(device.platform)
        maxwg = ocl.platforms[platform_id].devices[device_id].max_work_group_size
#         logger.warning("max_work_group_size: %s on (%s, %s)", maxwg, platform_id, device_id)
        for kernel in list(cls.kernels.keys()):
            if cls.kernels[kernel] < maxwg:
                logger.warning("%s Limiting workgroup size to %s", kernel, maxwg)
                cls.kernels[kernel] = maxwg
        cls.max_wg = maxwg

        for kernel in list(cls.kernels.keys()):
            kernel_src = get_opencl_code(os.path.join("sift", kernel))
            program = pyopencl.Program(cls.ctx, kernel_src).build("-D WORKGROUP=%s" % cls.kernels[kernel])
            cls.kernels[kernel] = program

    @classmethod
    def tearDownClass(cls):
        super(TestGaussian, cls).tearDownClass()
        cls.ctx = cls.kernels = cls.queue = None

    @classmethod
    def gaussian_gpu_v1(cls, sigma, size=None):
        """
        Calculate a 1D gaussian using pyopencl.
        This is the same as scipy.signal.gaussian

        :param sigma: width of the gaussian
        :param size: can be calculated as 1 + 2 * 4sigma
        """
        if not size:
            size = int(1 + 8 * sigma)
        g_gpu = pyopencl.array.empty(cls.queue, size, dtype=numpy.float32, order="C")
        t0 = time.time()
        evt1 = cls.kernels["gaussian"].gaussian_nosync(cls.queue, (size,), (1,),
                                                       g_gpu.data,  # __global     float     *data,
                                                       numpy.float32(sigma),  # const        float     sigma,
                                                       numpy.int32(size))  # const        int     SIZE
        sum_data = pyopencl.array.sum(g_gpu, dtype=numpy.dtype(numpy.float32), queue=cls.queue)
        evt2 = cls.kernels["preprocess"].divide_cst(cls.queue, (size,), (1,),
                                                    g_gpu.data,  # __global     float     *data,
                                                    sum_data.data,  # const        float     sigma,
                                                    numpy.int32(size))  # const        int     SIZE
        g = g_gpu.get()
        if cls.PROFILE:
            logger.info("execution time: %.3fms; Kernel took %.3fms and %.3fms", 1e3 * (time.time() - t0), 1e-6 * (evt1.profile.end - evt1.profile.start), 1e-6 * (evt2.profile.end - evt2.profile.start))

        return g

    @classmethod
    def gaussian_gpu_v2(cls, sigma, size=None):
        """
        Calculate a 1D gaussian using pyopencl.
        This is the same as scipy.signal.gaussian.
        Only one kernel to

        :param sigma: width of the gaussian
        :param size: can be calculated as 1 + 2 * 4sigma
        """
        if not size:
            size = int(1 + 8 * sigma)
        g_gpu = pyopencl.array.empty(cls.queue, size, dtype=numpy.float32, order="C")
        t0 = time.time()
        evt = cls.kernels["gaussian"].gaussian(cls.queue, (64,), (64,),
                                               g_gpu.data,  # __global     float     *data,
                                               numpy.float32(sigma),  # const        float     sigma,
                                               numpy.int32(size),  # const        int     SIZE
                                               pyopencl.LocalMemory(64 * 4),
                                               pyopencl.LocalMemory(64 * 4),)
        g = g_gpu.get()
        if cls.PROFILE:
            logger.info("execution time: %.3fms; Kernel took %.3fms", 1e3 * (time.time() - t0), 1e-6 * (evt.profile.end - evt.profile.start))
        return g

    def test_v1_odd(self):
        """
        test odd kernel size
        """
        sigma = 3.0
        size = 27
        ref = gaussian_cpu(sigma, size)
        res = self.gaussian_gpu_v1(sigma, size)
        delta = ref - res
        self.assertLess(abs(delta).max(), 1e-6, "gaussian are the same ")

    def test_v1_even(self):
        """
        test odd kernel size
        """
        sigma = 3.0
        size = 28
        ref = gaussian_cpu(sigma, size)
        res = self.gaussian_gpu_v1(sigma, size)
        delta = ref - res
        self.assertLess(abs(delta).max(), 1e-6, "gaussian are the same ")

    def test_v2_odd(self):
        """
        test odd kernel size
        """
        sigma = 3.0
        size = 27
        ref = gaussian_cpu(sigma, size)
        max_wg = kernel_workgroup_size(self.kernels["gaussian"], "gaussian")
        if max_wg < size:
            logger.warning("Skipping test of WG=%s when maximum is %s", size, max_wg)
            return
        res = self.gaussian_gpu_v2(sigma, size)
        delta = ref - res
        self.assertLess(abs(delta).max(), 1e-6, "gaussian are the same ")

    def test_v2_even(self):
        """
        test odd kernel size
        """
        sigma = 3.0
        size = 28
        ref = gaussian_cpu(sigma, size)
        max_wg = kernel_workgroup_size(self.kernels["gaussian"], "gaussian")
        if max_wg < size:
            logger.warning("Skipping test of WG=%s when maximum is %s", size, max_wg)
            return
        res = self.gaussian_gpu_v2(sigma, size)
        delta = ref - res
        self.assertLess(abs(delta).max(), 1e-6, "gaussian are the same ")
