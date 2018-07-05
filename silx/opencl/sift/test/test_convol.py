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
Test suite for all preprocessing kernels.
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/07/2018"

import os
import time
import logging
import numpy

try:
    import scipy.misc
    import scipy.ndimage
except ImportError:
    scipy = None

import unittest
from silx.opencl import ocl
if ocl:
    import pyopencl.array
from ..utils import calc_size, get_opencl_code
logger = logging.getLogger(__name__)


def my_blur(img, kernel):
    """
    hand made implementation of gaussian blur with OUR kernel
    which differs from Scipy's if ksize is even
    """
    tmp1 = scipy.ndimage.filters.convolve1d(img, kernel, axis=-1, mode="reflect")
    return scipy.ndimage.filters.convolve1d(tmp1, kernel, axis=0, mode="reflect")


@unittest.skipUnless(scipy and ocl, "scipy or opencl not available")
class TestConvol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestConvol, cls).setUpClass()
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
            cls.max_wg = ocl.platforms[platform_id].devices[device_id].max_work_group_size
#             logger.warning("max_work_group_size: %s on (%s, %s)", cls.max_wg, platform_id, device_id)

    @classmethod
    def tearDownClass(cls):
        super(TestConvol, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        if scipy and ocl is None:
            return

        if hasattr(scipy.misc, "ascent"):
            self.input = scipy.misc.ascent().astype(numpy.float32)
        else:
            self.input = scipy.misc.lena().astype(numpy.float32)

        self.input = numpy.ascontiguousarray(self.input[0:507, 0:209])

        self.gpu_in = pyopencl.array.to_device(self.queue, self.input)
        self.gpu_tmp = pyopencl.array.empty(self.queue, self.input.shape, dtype=numpy.float32, order="C")
        self.gpu_out = pyopencl.array.empty(self.queue, self.input.shape, dtype=numpy.float32, order="C")
        kernel_src = get_opencl_code(os.path.join("sift", "convolution.cl"))
        self.program = pyopencl.Program(self.ctx, kernel_src).build()
        self.IMAGE_W = numpy.int32(self.input.shape[-1])
        self.IMAGE_H = numpy.int32(self.input.shape[0])
        if self.max_wg < 512:
            if self.max_wg > 1:
                self.wg = (self.max_wg, 1)
            else:
                self.wg = (1, 1)
        else:
            self.wg = (256, 2)
        self.shape = calc_size((self.input.shape[1], self.input.shape[0]), self.wg)

    def tearDown(self):
        self.input = None
        # self.gpudata.release()
        self.program = None
        self.gpu_in = self.gpu_tmp = self.gpu_out = None

    def test_convol_hor(self):
        """
        tests the convolution kernel
        """
        for sigma in [2, 15 / 8.]:
            ksize = int(8 * sigma + 1)
            x = numpy.arange(ksize) - (ksize - 1.0) / 2.0
            gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
            gaussian /= gaussian.sum(dtype=numpy.float32)
            gpu_filter = pyopencl.array.to_device(self.queue, gaussian)
            t0 = time.time()
            k1 = self.program.horizontal_convolution(self.queue, self.shape, self.wg,
                                self.gpu_in.data, self.gpu_out.data, gpu_filter.data, numpy.int32(ksize), self.IMAGE_W, self.IMAGE_H)
            res = self.gpu_out.get()
            t1 = time.time()
            ref = scipy.ndimage.filters.convolve1d(self.input, gaussian, axis=-1, mode="reflect")
            t2 = time.time()
            delta = abs(ref - res).max()
            if ksize % 2 == 0:  # we have a problem with even kernels !!!
                self.assertLess(delta, 50, "sigma= %s delta=%s" % (sigma, delta))
            else:
                self.assertLess(delta, 1e-4, "sigma= %s delta=%s" % (sigma, delta))
            logger.info("sigma= %s delta=%s" % (sigma, delta))
            if self.PROFILE:
                logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
                logger.info("Horizontal convolution took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))

    @unittest.skipIf(scipy and ocl is None, "scipy or opencl not available")
    def test_convol_vert(self):
        """
        tests the convolution kernel
        """
        for sigma in [2, 15 / 8.]:
            ksize = int(8 * sigma + 1)
            x = numpy.arange(ksize) - (ksize - 1.0) / 2.0
            gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
            gaussian /= gaussian.sum(dtype=numpy.float32)
            gpu_filter = pyopencl.array.to_device(self.queue, gaussian)
            t0 = time.time()
            k1 = self.program.vertical_convolution(self.queue, self.shape, self.wg,
                                                   self.gpu_in.data,
                                                   self.gpu_out.data,
                                                   gpu_filter.data,
                                                   numpy.int32(ksize),
                                                   self.IMAGE_W, self.IMAGE_H)
            res = self.gpu_out.get()
            t1 = time.time()
            ref = scipy.ndimage.filters.convolve1d(self.input, gaussian, axis=0, mode="reflect")
            t2 = time.time()
            delta = abs(ref - res).max()
            if ksize % 2 == 0:  # we have a problem with even kernels !!!
                self.assertLess(delta, 50, "sigma= %s delta=%s" % (sigma, delta))
            else:
                self.assertLess(delta, 1e-4, "sigma= %s delta=%s" % (sigma, delta))
            logger.info("sigma= %s delta=%s" % (sigma, delta))
            if self.PROFILE:
                logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
                logger.info("Vertical convolution took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))

    def test_convol(self):
        """
        tests the convolution kernel
        """
        for sigma in [2, 15 / 8.]:
            ksize = int(8 * sigma + 1)
            x = numpy.arange(ksize) - (ksize - 1.0) / 2.0
            gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
            gaussian /= gaussian.sum(dtype=numpy.float32)
            gpu_filter = pyopencl.array.to_device(self.queue, gaussian)
            t0 = time.time()
            k1 = self.program.horizontal_convolution(self.queue, self.shape, self.wg,
                                self.gpu_in.data, self.gpu_tmp.data, gpu_filter.data, numpy.int32(ksize), self.IMAGE_W, self.IMAGE_H)
            k2 = self.program.vertical_convolution(self.queue, self.shape, self.wg,
                                self.gpu_tmp.data, self.gpu_out.data, gpu_filter.data, numpy.int32(ksize), self.IMAGE_W, self.IMAGE_H)
            res = self.gpu_out.get()
            k2.wait()
            t1 = time.time()
            ref = my_blur(self.input, gaussian)
            # ref = scipy.ndimage.filters.gaussian_filter(self.input, sigma, mode="reflect")
            t2 = time.time()
            delta = abs(ref - res).max()
            if ksize % 2 == 0:  # we have a problem with even kernels !!!
                self.assertLess(delta, 50, "sigma= %s delta=%s" % (sigma, delta))
            else:
                self.assertLess(delta, 1e-4, "sigma= %s delta=%s" % (sigma, delta))
            logger.info("sigma= %s delta=%s" % (sigma, delta))
            if self.PROFILE:
                logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
                logger.info("Horizontal convolution took %.3fms and vertical convolution took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start),
                                                                                          1e-6 * (k2.profile.end - k2.profile.start)))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestConvol("test_convol"))
    testSuite.addTest(TestConvol("test_convol_hor"))
    testSuite.addTest(TestConvol("test_convol_vert"))
    return testSuite
