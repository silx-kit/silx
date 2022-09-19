#!/usr/bin/env python
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
except ImportError:
    scipy = None

import math
from silx.opencl import ocl, kernel_workgroup_size
if ocl:
    import pyopencl.array
import unittest
from ..utils import calc_size, get_opencl_code

logger = logging.getLogger(__name__)


def normalize(img, max_out=255):
    """
    Numpy implementation of the normalization
    """
    fimg = img.astype("float32")
    img_max = fimg.max()
    img_min = fimg.min()
    return max_out * (fimg - img_min) / (img_max - img_min)


def shrink(img, xs, ys):
    return img[0::ys, 0::xs]


def shrink_cython(img, xs, ys):
    try:
        import feature
    except ImportError:
        return img[0::ys, 0::xs]
    else:
        return feature.shrink(img, xs)


def binning(input_img, binsize):
    """
    :param input_img: input ndarray
    :param binsize: int or 2-tuple representing the size of the binning
    :return: binned input ndarray

    TODO: Not used here
    """
    inputSize = input_img.shape
    assert(len(inputSize) == 2)
    if isinstance(binsize, int):
        binsize = (binsize, binsize)
    outputSize = [int(math.ceil(float(i) / j)) for i, j in zip(inputSize, binsize)]
    bigSize = [i * j for i, j in zip(outputSize, binsize)]
    delta = [j - i for i, j in zip(inputSize, bigSize)]
    big_array = numpy.empty(bigSize, input_img.dtype)
    big_array[:inputSize[0], :inputSize[1]] = input_img
    # corner
    big_array[inputSize[0]:, inputSize[1]:] = input_img[-1:-delta[0] - 1:-1, -1:-delta[1] - 1:-1]
    # 2 sides
    big_array[inputSize[0]:, :inputSize[1]] = input_img[-1:-delta[0] - 1:-1, :]
    big_array[:inputSize[0], inputSize[1]:] = input_img[:, -1:-delta[1] - 1:-1]

    if numpy.array(binsize).prod() < 50:
        out = numpy.zeros(tuple(outputSize))
#        print input_img.shape, out.shape, big_array.shape, binsize
        for i in range(binsize[0]):
            for j in range(binsize[1]):
                out += big_array[i::binsize[0], j::binsize[1]]
    else:
        temp = big_array.copy()
        temp.shape = (outputSize[0], binsize[0], outputSize[1], binsize[1])
        out = temp.sum(axis=3).sum(axis=1)
    return out


@unittest.skipUnless(scipy and ocl, "no scipy or ocl")
class TestPreproc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestPreproc, cls).setUpClass()
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
        super(TestPreproc, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        if not (ocl and scipy):
            return
        try:
            testdata = scipy.misc.ascent()
        except:
            # for very old version of scipy
            testdata = scipy.misc.lena()
        self.input = numpy.ascontiguousarray(testdata[:510, :511])
        self.gpudata = pyopencl.array.empty(self.queue, self.input.shape, dtype=numpy.float32, order="C")
        kernel_src = get_opencl_code(os.path.join("sift", "preprocess"))
        reduct_src = get_opencl_code(os.path.join("sift", "reductions"))
        self.program = pyopencl.Program(self.ctx, kernel_src).build()
        self.reduction = pyopencl.Program(self.ctx, reduct_src).build()
        self.IMAGE_W = numpy.int32(self.input.shape[-1])
        self.IMAGE_H = numpy.int32(self.input.shape[0])
        if self.max_wg < 32 * 16:
            self.wg = self.max_wg, 1
        else:
            self.wg = (32, 16)  # (256, 2) #(32, 16) # (2, 256)

        self.shape = calc_size((self.IMAGE_W, self.IMAGE_H), self.wg)
#        print self.shape
        self.binning = (4, 2)  # Nota if wg < ouptup size weired results are expected !
#        self.binning = (2, 2)
        self.red_size = 128  # reduction size
        self.twofivefive = pyopencl.array.to_device(self.queue, numpy.array([255], numpy.float32))
        self.buffers_max_min = pyopencl.array.empty(self.queue, (self.red_size, 2), dtype=numpy.float32)  # temporary buffer for max/min reduction
        self.buffers_min = pyopencl.array.empty(self.queue, (1), dtype=numpy.float32)
        self.buffers_max = pyopencl.array.empty(self.queue, (1), dtype=numpy.float32)

    def tearDown(self):
        self.input = None
        self.program = None
        self.twofivefive = None
        self.buffers_max_min = None
        self.buffers_max = None
        self.buffers_min = None

    def test_uint8(self):
        """
        tests the uint8 kernel
        """
        max_wg = kernel_workgroup_size(self.reduction, "max_min_global_stage1")
        if max_wg < self.red_size:
            logger.warning("test_uint8: Skipping test of WG=%s when maximum is %s (%s)", self.red_size, max_wg, self.max_wg)
            return

        lint = self.input.astype(numpy.uint8)
        t0 = time.time()
        au8 = pyopencl.array.to_device(self.queue, lint)
        k1 = self.program.u8_to_float(self.queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
#        print abs(au8.get() - self.gpudata.get()).max()
        k2 = self.reduction.max_min_global_stage1(self.queue, (self.red_size * self.red_size,), (self.red_size,),
                                                  self.gpudata.data,
                                                  self.buffers_max_min.data,
                                                  (self.IMAGE_W * self.IMAGE_H),
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k3 = self.reduction.max_min_global_stage2(self.queue, (self.red_size,), (self.red_size,),
                                                  self.buffers_max_min.data,
                                                  self.buffers_max.data,
                                                  self.buffers_min.data,
                                                  pyopencl.LocalMemory(8 * self.red_size))
#        print self.buffers_max.get(), self.buffers_min.get(), self.input.min(), self.input.max()
        k4 = self.program.normalizes(self.queue, self.shape, self.wg,
                                     self.gpudata.data,
                                     self.buffers_min.data,
                                     self.buffers_max.data,
                                     self.twofivefive.data,
                                     self.IMAGE_W, self.IMAGE_H)
        k4.wait()
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Conversion uint8->float took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            logger.info("Reduction stage1 took        %.3fms" % (1e-6 * (k2.profile.end - k2.profile.start)))
            logger.info("Reduction stage2 took        %.3fms" % (1e-6 * (k3.profile.end - k3.profile.start)))
            logger.info("Normalization                %.3fms" % (1e-6 * (k4.profile.end - k4.profile.start)))
            logger.info("--------------------------------------")

        self.assertLess(delta, 1e-4, "delta=%s" % delta)

    def test_uint16(self):
        """
        tests the uint16 kernel
        """
        max_wg = kernel_workgroup_size(self.reduction, "max_min_global_stage1")
        if max_wg < self.red_size:
            logger.warning("test_uint16: Skipping test of WG=%s when maximum is %s (%s)", self.red_size, max_wg, self.max_wg)
            return

        lint = self.input.astype(numpy.uint16)
        t0 = time.time()
        au8 = pyopencl.array.to_device(self.queue, lint)
        k1 = self.program.u16_to_float(self.queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        k2 = self.reduction.max_min_global_stage1(self.queue, (self.red_size * self.red_size,), (self.red_size,),
                                                  self.gpudata.data,
                                                  self.buffers_max_min.data,
                                                  (self.IMAGE_W * self.IMAGE_H),
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k3 = self.reduction.max_min_global_stage2(self.queue, (self.red_size,), (self.red_size,),
                                                  self.buffers_max_min.data,
                                                  self.buffers_max.data,
                                                  self.buffers_min.data,
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k4 = self.program.normalizes(self.queue, self.shape, self.wg,
                                     self.gpudata.data,
                                     self.buffers_min.data,
                                     self.buffers_max.data,
                                     self.twofivefive.data,
                                     self.IMAGE_W, self.IMAGE_H)
        k4.wait()
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Conversion uint16->float took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            logger.info("Reduction stage1 took         %.3fms" % (1e-6 * (k2.profile.end - k2.profile.start)))
            logger.info("Reduction stage2 took         %.3fms" % (1e-6 * (k3.profile.end - k3.profile.start)))
            logger.info("Normalization                 %.3fms" % (1e-6 * (k4.profile.end - k4.profile.start)))
            logger.info("--------------------------------------")
        self.assertLess(delta, 1e-4, "delta=%s" % delta)

    def test_uint32(self):
        """
        tests the uint32 kernel
        """
        max_wg = kernel_workgroup_size(self.reduction, "max_min_global_stage1")
        if max_wg < self.red_size:
            logger.warning("test_uint32: Skipping test of WG=%s when maximum is %s (%s)", self.red_size, max_wg, self.max_wg)
            return

        lint = self.input.astype(numpy.uint32)
        t0 = time.time()
        au8 = pyopencl.array.to_device(self.queue, lint)
        k1 = self.program.u32_to_float(self.queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        k2 = self.reduction.max_min_global_stage1(self.queue, (self.red_size * self.red_size,), (self.red_size,),
                                                  self.gpudata.data,
                                                  self.buffers_max_min.data,
                                                  (self.IMAGE_W * self.IMAGE_H),
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k3 = self.reduction.max_min_global_stage2(self.queue, (self.red_size,), (self.red_size,),
                                                  self.buffers_max_min.data,
                                                  self.buffers_max.data,
                                                  self.buffers_min.data,
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k4 = self.program.normalizes(self.queue, self.shape, self.wg,
                                     self.gpudata.data,
                                     self.buffers_min.data,
                                     self.buffers_max.data,
                                     self.twofivefive.data,
                                     self.IMAGE_W, self.IMAGE_H)
        k4.wait()
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Conversion uint32->float took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            logger.info("Reduction stage1 took         %.3fms" % (1e-6 * (k2.profile.end - k2.profile.start)))
            logger.info("Reduction stage2 took         %.3fms" % (1e-6 * (k3.profile.end - k3.profile.start)))
            logger.info("Normalization                 %.3fms" % (1e-6 * (k4.profile.end - k4.profile.start)))
            logger.info("--------------------------------------")
        self.assertLess(delta, 1e-4, "delta=%s" % delta)

    def test_uint64(self):
        """
        tests the uint64 kernel
        """
        max_wg = kernel_workgroup_size(self.reduction, "max_min_global_stage1")
        if max_wg < self.red_size:
            logger.warning("test_uint64: Skipping test of WG=%s when maximum is %s (%s)", self.red_size, max_wg, self.max_wg)
            return

        lint = self.input.astype(numpy.uint64)
        t0 = time.time()
        au8 = pyopencl.array.to_device(self.queue, lint)
        k1 = self.program.u64_to_float(self.queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        k2 = self.reduction.max_min_global_stage1(self.queue, (self.red_size * self.red_size,), (self.red_size,),
                                                  self.gpudata.data,
                                                  self.buffers_max_min.data,
                                                  (self.IMAGE_W * self.IMAGE_H),
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k3 = self.reduction.max_min_global_stage2(self.queue, (self.red_size,), (self.red_size,),
                                                  self.buffers_max_min.data,
                                                  self.buffers_max.data,
                                                  self.buffers_min.data,
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k4 = self.program.normalizes(self.queue, self.shape, self.wg,
                                     self.gpudata.data,
                                     self.buffers_min.data,
                                     self.buffers_max.data,
                                     self.twofivefive.data,
                                     self.IMAGE_W, self.IMAGE_H)
        k4.wait()
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Conversion uint64->float took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            logger.info("Reduction stage1 took         %.3fms" % (1e-6 * (k2.profile.end - k2.profile.start)))
            logger.info("Reduction stage2 took         %.3fms" % (1e-6 * (k3.profile.end - k3.profile.start)))
            logger.info("Normalization                 %.3fms" % (1e-6 * (k4.profile.end - k4.profile.start)))
            logger.info("--------------------------------------")
        self.assertLess(delta, 1e-4, "delta=%s" % delta)

    def test_int32(self):
        """
        tests the int32 kernel
        """
        max_wg = kernel_workgroup_size(self.reduction, "max_min_global_stage1")
        if max_wg < self.red_size:
            logger.warning("test_int32: Skipping test of WG=%s when maximum is %s (%s)", self.red_size, max_wg, self.max_wg)
            return

        lint = self.input.astype(numpy.int32)
        t0 = time.time()
        au8 = pyopencl.array.to_device(self.queue, lint)
        k1 = self.program.s32_to_float(self.queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        k2 = self.reduction.max_min_global_stage1(self.queue, (self.red_size * self.red_size,), (self.red_size,),
                                                  self.gpudata.data,
                                                  self.buffers_max_min.data,
                                                  (self.IMAGE_W * self.IMAGE_H),
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k3 = self.reduction.max_min_global_stage2(self.queue, (self.red_size,), (self.red_size,),
                                                  self.buffers_max_min.data,
                                                  self.buffers_max.data,
                                                  self.buffers_min.data,
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k4 = self.program.normalizes(self.queue, self.shape, self.wg,
                                     self.gpudata.data,
                                     self.buffers_min.data,
                                     self.buffers_max.data,
                                     self.twofivefive.data,
                                     self.IMAGE_W, self.IMAGE_H)
        k4.wait()
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Conversion int32->float took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            logger.info("Reduction stage1 took        %.3fms" % (1e-6 * (k2.profile.end - k2.profile.start)))
            logger.info("Reduction stage2 took        %.3fms" % (1e-6 * (k3.profile.end - k3.profile.start)))
            logger.info("Normalization                %.3fms" % (1e-6 * (k4.profile.end - k4.profile.start)))
            logger.info("--------------------------------------")
        self.assertLess(delta, 1e-4, "delta=%s" % delta)

    def test_int64(self):
        """
        tests the int64 kernel
        """
        max_wg = kernel_workgroup_size(self.reduction, "max_min_global_stage1")
        if max_wg < self.red_size:
            logger.warning("test_int64: Skipping test of WG=%s when maximum is %s (%s)", self.red_size, max_wg, self.max_wg)
            return

        lint = self.input.astype(numpy.int64)
        t0 = time.time()
        au8 = pyopencl.array.to_device(self.queue, lint)
        k1 = self.program.s64_to_float(self.queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        k2 = self.reduction.max_min_global_stage1(self.queue, (self.red_size * self.red_size,), (self.red_size,),
                                                  self.gpudata.data,
                                                  self.buffers_max_min.data,
                                                  (self.IMAGE_W * self.IMAGE_H),
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k3 = self.reduction.max_min_global_stage2(self.queue, (self.red_size,), (self.red_size,),
                                                  self.buffers_max_min.data,
                                                  self.buffers_max.data,
                                                  self.buffers_min.data,
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k4 = self.program.normalizes(self.queue, self.shape, self.wg,
                                     self.gpudata.data,
                                     self.buffers_min.data,
                                     self.buffers_max.data,
                                     self.twofivefive.data,
                                     self.IMAGE_W, self.IMAGE_H)
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint)
        t2 = time.time()
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Conversion int64->float took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            logger.info("Reduction stage1 took        %.3fms" % (1e-6 * (k2.profile.end - k2.profile.start)))
            logger.info("Reduction stage2 took        %.3fms" % (1e-6 * (k3.profile.end - k3.profile.start)))
            logger.info("Normalization                %.3fms" % (1e-6 * (k4.profile.end - k4.profile.start)))
            logger.info("--------------------------------------")
        self.assertLess(delta, 1e-4, "delta=%s" % delta)

    def test_rgb(self):
        """
        tests the int64 kernel
        """
        max_wg = kernel_workgroup_size(self.reduction, "max_min_global_stage1")
        if max_wg < self.red_size:
            logger.warning("test_uint16: Skipping test of WG=%s when maximum is %s (%s)", self.red_size, max_wg, self.max_wg)
            return

        lint = numpy.empty((self.input.shape[0], self.input.shape[1], 3), dtype=numpy.uint8)
        lint[:, :, 0] = self.input.astype(numpy.uint8)
        lint[:, :, 1] = self.input.astype(numpy.uint8)
        lint[:, :, 2] = self.input.astype(numpy.uint8)
        t0 = time.time()
        au8 = pyopencl.array.to_device(self.queue, lint)
        k1 = self.program.rgb_to_float(self.queue, self.shape, self.wg, au8.data, self.gpudata.data, self.IMAGE_W, self.IMAGE_H)
        k2 = self.reduction.max_min_global_stage1(self.queue, (self.red_size * self.red_size,), (self.red_size,),
                                                  self.gpudata.data,
                                                  self.buffers_max_min.data,
                                                  (self.IMAGE_W * self.IMAGE_H),
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k3 = self.reduction.max_min_global_stage2(self.queue, (self.red_size,), (self.red_size,),
                                                  self.buffers_max_min.data,
                                                  self.buffers_max.data,
                                                  self.buffers_min.data,
                                                  pyopencl.LocalMemory(8 * self.red_size))
        k4 = self.program.normalizes(self.queue, self.shape, self.wg,
                                     self.gpudata.data,
                                     self.buffers_min.data,
                                     self.buffers_max.data,
                                     self.twofivefive.data,
                                     self.IMAGE_W, self.IMAGE_H)
        res = self.gpudata.get()
        t1 = time.time()
        ref = normalize(lint.max(axis=-1))
        t2 = time.time()
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Conversion  RGB ->float took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
            logger.info("Reduction stage1 took        %.3fms" % (1e-6 * (k2.profile.end - k2.profile.start)))
            logger.info("Reduction stage2 took        %.3fms" % (1e-6 * (k3.profile.end - k3.profile.start)))
            logger.info("Normalization                %.3fms" % (1e-6 * (k4.profile.end - k4.profile.start)))
            logger.info("--------------------------------------")
        self.assertLess(delta, 1e-4, "delta=%s" % delta)

    def test_shrink(self):
        """
        Test shrinking kernel
        """
        lint = self.input.astype(numpy.float32)
        out_shape = tuple(int(math.ceil(float(i) / j)) for i, j in zip((self.IMAGE_H, self.IMAGE_W), self.binning))
        t0 = time.time()
        inp_gpu = pyopencl.array.to_device(self.queue, lint)
        out_gpu = pyopencl.array.empty(self.queue, out_shape, dtype=numpy.float32, order="C")
        k1 = self.program.shrink(self.queue, calc_size((out_shape[1], out_shape[0]), self.wg), self.wg,
                                 inp_gpu.data, out_gpu.data,
                                 numpy.int32(self.binning[1]), numpy.int32(self.binning[0]),
                                 self.IMAGE_W, self.IMAGE_H,
                                 numpy.int32(out_shape[1]), numpy.int32(out_shape[0]))
        res = out_gpu.get()
        t1 = time.time()
        ref = shrink(lint, xs=self.binning[1], ys=self.binning[0])
        t2 = time.time()
#        print ref.shape, res.shape
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Shrinking  took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
        self.assertLess(delta, 1e-6, "delta=%s" % delta)

    def test_bin(self):
        """
        Test binning kernel
        """
        lint = numpy.ascontiguousarray(self.input, numpy.float32)

        out_shape = tuple(int(math.ceil((float(i) / j))) for i, j in zip(self.input.shape, self.binning))
        t0 = time.time()
        inp_gpu = pyopencl.array.to_device(self.queue, lint)
        out_gpu = pyopencl.array.empty(self.queue, out_shape, dtype=numpy.float32, order="C")
        k1 = self.program.bin(self.queue, calc_size((out_shape[1], out_shape[0]), self.wg), self.wg, inp_gpu.data, out_gpu.data,
                              numpy.int32(self.binning[1]), numpy.int32(self.binning[0]),
                              numpy.int32(lint.shape[1]), numpy.int32(lint.shape[0]),
                              numpy.int32(out_shape[1]), numpy.int32(out_shape[0]))
        res = out_gpu.get()
        t1 = time.time()
        ref = binning(lint, self.binning) / self.binning[0] / self.binning[1]
        t2 = time.time()
#        print ref.shape, res.shape
        delta = abs(ref - res).max()
        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Binning took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
        self.assertLess(delta, 1e-6, "delta=%s" % delta)
