#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
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
Test suite for algebra kernels
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/07/2018"

import os
import time
import logging
import numpy

import unittest
from silx.opencl import ocl, kernel_workgroup_size
if ocl:
    import pyopencl.array
from ..utils import calc_size, get_opencl_code
logger = logging.getLogger(__name__)


def my_combine(mat1, a1, mat2, a2):
    """
    reference linear combination
    """
    return a1 * mat1 + a2 * mat2


def my_compact(keypoints, nbkeypoints):
    '''
    Reference compacting
    '''
    output = -numpy.ones_like(keypoints)
    idx = numpy.where(keypoints[:, 1] != -1)[0]
    length = idx.size
    output[:length, 0] = keypoints[idx, 0]
    output[:length, 1] = keypoints[idx, 1]
    output[:length, 2] = keypoints[idx, 2]
    output[:length, 3] = keypoints[idx, 3]
    return output, length


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
        kernel_src = os.linesep.join(get_opencl_code(os.path.join("sift", i)) for i in ("sift.cl", "algebra.cl"))
        self.program = pyopencl.Program(self.ctx, kernel_src).build()
        self.wg_compact = kernel_workgroup_size(self.program, "compact")

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
        t0 = time.time()
        try:
            k1 = self.program.combine(self.queue, (int(width), int(height)), None,
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
        self.assertLess(delta, 1e-4, "delta=%s" % (delta))
        if self.PROFILE:
            logger.debug("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.debug("Linear combination took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))

    def test_compact(self):
        """
        tests the "compact" kernel
        """

        nbkeypoints = 10000  # constant value
        keypoints = numpy.random.rand(nbkeypoints, 4).astype(numpy.float32)
        nb_ones = 0
        for i in range(nbkeypoints):
            if ((numpy.random.rand(1))[0] < 0.25):  # discard about 1 out of 4
                keypoints[i] = (-1, -1, i, -1)
                nb_ones += 1
            else:
                keypoints[i, 2] = i

        gpu_keypoints = pyopencl.array.to_device(self.queue, keypoints)
        output = pyopencl.array.empty(self.queue, (nbkeypoints, 4), dtype=numpy.float32, order="C")
        output.fill(-1.0, self.queue)
        counter = pyopencl.array.zeros(self.queue, (1,), dtype=numpy.int32, order="C")
        wg = self.wg_compact,
        shape = calc_size((keypoints.shape[0],), wg)
        nbkeypoints = numpy.int32(nbkeypoints)
        startkeypoints = numpy.int32(0)
        t0 = time.time()
        try:
            k1 = self.program.compact(self.queue, shape, wg,
                                      gpu_keypoints.data, output.data, counter.data, startkeypoints, nbkeypoints)
        except pyopencl.LogicError as error:
            logger.warning("%s in test_compact", error)
        res = output.get()
        count = counter.get()[0]
        t1 = time.time()
        ref, count_ref = my_compact(keypoints, nbkeypoints)
        t2 = time.time()

        logger.debug("Kernel counter : %s / Python counter : %s / True value : %s",
                     count, count_ref, nbkeypoints - nb_ones)

        res_sort_arg = res[:, 2].argsort(axis=0)
        res_sort = res[res_sort_arg]
        ref_sort_arg = ref[:, 2].argsort(axis=0)
        ref_sort = ref[ref_sort_arg]
        delta = abs((res_sort - ref_sort)).max()
        self.assertLess(delta, 1e-5, "delta=%s" % (delta))
        self.assertEqual(count, count_ref, "counters are the same")
        logger.debug("delta=%s", delta)
        if self.PROFILE:
            logger.debug("Global execution time: CPU %.3fms, GPU: %.3fms.", 1000.0 * (t2 - t1), 1000.0 * (t1 - t0))
            logger.debug("Compact operation took %.3fms", 1e-6 * (k1.profile.end - k1.profile.start))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestAlgebra("test_combine"))
    testSuite.addTest(TestAlgebra("test_compact"))
    return testSuite
