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
Test suite for image kernels

For Python implementation of tested functions, see "test_image_functions.py"

"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/07/2018"

import os
import unittest
import time
import logging
import numpy
try:
    import scipy.misc
except ImportError:
    scipy = None

from silx.utils.testutils import parameterize
# for Python implementation of tested functions
from .test_image_functions import my_compact, my_orientation, keypoints_compare, my_descriptor, descriptors_compare
from .test_image_setup import orientation_setup, descriptor_setup
from silx.opencl import ocl, pyopencl, kernel_workgroup_size
from ..utils import get_opencl_code
logger = logging.getLogger(__name__)


@unittest.skipUnless(ocl and scipy, "opencl or scipy missing")
class TestKeypoints(unittest.TestCase):

    def __init__(self, methodName='runTest',
                 orientation_script=None, orientation_param=None,
                 keypoint_script=None, keypoint_param=None):
        super(TestKeypoints, self).__init__(methodName)
        self.orientation_script = orientation_script
        self.orientation_param = orientation_param
        self.keypoint_script = keypoint_script
        self.keypoint_param = keypoint_param

    @classmethod
    def setUpClass(cls):
        super(TestKeypoints, cls).setUpClass()
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
            # logger.warning("max_work_group_size: %s on (%s, %s)", cls.maxwg, platform_id, device_id)

    @classmethod
    def tearDownClass(cls):
        super(TestKeypoints, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        self.abort = False
        if scipy and ocl is None:
            return
        try:
            self.testdata = scipy.misc.ascent()
        except Exception:
            # for very old versions of scipy
            self.testdata = scipy.misc.lena()
        kernel_base = get_opencl_code(os.path.join("sift", "sift"))
        if self.orientation_script is None:
            self.skipTest("Uninitialized parametric class")

        if "cpu" in self.orientation_script:
            self.USE_CPU = True
        else:
            self.USE_CPU = False

        if self.USE_CPU:
            assert("cpu" in self.keypoint_script)
        else:
            assert("gpu" in self.keypoint_script)

        self.wg_orient = self.orientation_param
        prod_wg = 1
        for i in self.wg_orient:
            prod_wg *= i

        kernel_src = kernel_base + get_opencl_code(os.path.join("sift", self.orientation_script))
        try:
            self.program_orient = pyopencl.Program(self.ctx, kernel_src).build()
        except Exception:
            logger.warning("Failed to compile kernel '%s': aborting" % self.orientation_script)
            self.abort = True
            return

        if prod_wg > self.maxwg:
            logger.warning("max_work_group_size: %s %s needed", self.maxwg, prod_wg)
            self.abort = True

        self.wg_keypoint = self.keypoint_param
        kernel_src = kernel_base + get_opencl_code(os.path.join("sift", self.keypoint_script))
        prod_wg = 1
        for i in self.wg_keypoint:
            prod_wg *= i

        try:
            self.program_keypoint = pyopencl.Program(self.ctx, kernel_src).build()
        except Exception:
            logger.warning("Failed to compile kernel '%s': aborting" % self.keypoint_script)
            self.abort = True
            return

        if prod_wg > self.maxwg:
            logger.warning("max_work_group_size: %s %s needed", self.maxwg, prod_wg)
            self.abort = True

    def tearDown(self):
        self.mat = None
        self.program = None
        self.testdata = None

    def test_orientation(self):
        '''
        #tests keypoints orientation assignment kernel
        '''
        if self.abort:
            return
        # orientation_setup :
        keypoints, nb_keypoints, updated_nb_keypoints, grad, ori, octsize = orientation_setup()
        keypoints, compact_cnt = my_compact(numpy.copy(keypoints), nb_keypoints)
        updated_nb_keypoints = compact_cnt
        logger.info("Number of keypoints before orientation assignment : %s", updated_nb_keypoints)

        # Prepare kernel call
        wg = self.wg_orient
        kernel = self.program_orient.all_kernels()[0]
        max_wg = kernel_workgroup_size(self.program_orient, kernel)
        if max_wg < wg[0]:
            logger.warning("test_orientation: Skipping test of WG=%s when maximum for this kernel is %s ", wg, max_wg)
            return
        shape = keypoints.shape[0] * wg[0],  # shape = calc_size(keypoints.shape, self.wg)
        gpu_keypoints = pyopencl.array.to_device(self.queue, keypoints)
        actual_nb_keypoints = numpy.int32(updated_nb_keypoints)
        gpu_grad = pyopencl.array.to_device(self.queue, grad)
        gpu_ori = pyopencl.array.to_device(self.queue, ori)
        orisigma = numpy.float32(1.5)  # SIFT
        grad_height, grad_width = numpy.int32(grad.shape)
        keypoints_start = numpy.int32(0)
        keypoints_end = numpy.int32(actual_nb_keypoints)
        counter = pyopencl.array.to_device(self.queue, keypoints_end)  # actual_nb_keypoints)
        kargs = [
            gpu_keypoints.data,
            gpu_grad.data,
            gpu_ori.data,
            counter.data,
            octsize,
            orisigma,
            nb_keypoints,
            keypoints_start,
            keypoints_end,
            grad_width,
            grad_height
        ]
        if not self.USE_CPU:
            kargs += [pyopencl.LocalMemory(36 * 4),
                      pyopencl.LocalMemory(128 * 4),
                      pyopencl.LocalMemory(128 * 4)]

        # Call the kernel
        t0 = time.time()
        k1 = kernel(self.queue, shape, wg, *kargs)
        res = gpu_keypoints.get()
        cnt = counter.get()
        t1 = time.time()

        # Reference Python implemenattion
        ref, updated_nb_keypoints = my_orientation(keypoints,
                                                   nb_keypoints,
                                                   keypoints_start,
                                                   keypoints_end, grad, ori,
                                                   octsize, orisigma)
        t2 = time.time()

        # sort to compare added keypoints
        upbound = min(cnt, updated_nb_keypoints)
        d1, d2, d3, d4 = keypoints_compare(ref[0:upbound], res[0:upbound])
        self.assertLess(d1, 1e-4, "delta_cols=%s" % (d1))
        self.assertLess(d2, 1e-4, "delta_rows=%s" % (d2))
        self.assertLess(d3, 1e-4, "delta_sigma=%s" % (d3))
        self.assertLess(d4, 1e-1, "delta_angle=%s" % (d4))  # orientation has a poor precision
        logger.info("delta_cols=%s" % d1)
        logger.info("delta_rows=%s" % d2)
        logger.info("delta_sigma=%s" % d3)
        logger.info("delta_angle=%s" % d4)

        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Orientation assignment took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))

    def test_descriptor(self):
        '''
        #tests keypoints descriptors creation kernel
        '''
        if self.abort:
            return

        # Descriptor_setup
        keypoints_o, nb_keypoints, actual_nb_keypoints, grad, ori, octsize = descriptor_setup()
        # keypoints should be a compacted vector of keypoints
        keypoints_o, compact_cnt = my_compact(numpy.copy(keypoints_o), nb_keypoints)
        actual_nb_keypoints = compact_cnt
        keypoints_start, keypoints_end = 0, actual_nb_keypoints
        keypoints = keypoints_o[keypoints_start:keypoints_end + 2]  # to check if we actually stop at keypoints_end
        logger.info("Working on keypoints : [%s,%s] (octave = %s)" % (keypoints_start, keypoints_end - 1, int(numpy.log2(octsize) + 1)))

        # Prepare kernel call
        wg = self.wg_keypoint
        if len(wg) == 1:
            shape = keypoints.shape[0] * wg[0],
        else:
            shape = keypoints.shape[0] * wg[0], wg[1], wg[2]
        kernel = self.program_keypoint.all_kernels()[0]
        # kernel_name = kernel.name
        max_wg = kernel_workgroup_size(self.program_keypoint, kernel)
        if max_wg < wg[0]:
            logger.warning("test_descriptor: Skipping test of WG=%s when maximum for this kernel is %s ", wg, max_wg)
            return
        gpu_keypoints = pyopencl.array.to_device(self.queue, keypoints)
        gpu_descriptors = pyopencl.array.zeros(self.queue, (keypoints_end - keypoints_start, 128), dtype=numpy.uint8, order="C")
        gpu_grad = pyopencl.array.to_device(self.queue, grad)
        gpu_ori = pyopencl.array.to_device(self.queue, ori)
        keypoints_start, keypoints_end = numpy.int32(keypoints_start), numpy.int32(keypoints_end)
        grad_height, grad_width = numpy.int32(grad.shape)
        counter = pyopencl.array.to_device(self.queue, keypoints_end)
        kargs = [
            gpu_keypoints.data,
            gpu_descriptors.data,
            gpu_grad.data,
            gpu_ori.data,
            numpy.int32(octsize),
            keypoints_start,
            counter.data,
            grad_width,
            grad_height
        ]

        # Call the kernel
        t0 = time.time()
        k1 = kernel(self.queue, shape, wg, *kargs)
        try:
            res = gpu_descriptors.get()
        except (pyopencl.LogicError, RuntimeError) as error:
            logger.warning("Segmentation fault like error (%s) on Descriptor for %s" % (error, self.param))
            return
        t1 = time.time()

        # Reference Python implementation
        ref = my_descriptor(keypoints_o,
                            grad,
                            ori,
                            octsize,
                            keypoints_start,
                            keypoints_end)
        ref_sort = ref[numpy.argsort(keypoints[keypoints_start: keypoints_end, 1])]
        t2 = time.time()

        res_sort = res[numpy.argsort(keypoints[keypoints_start:keypoints_end, 1])]
        logger.info(res_sort[5:10])
        logger.info(ref_sort[5:10])
        logger.info("Comparing descriptors (OpenCL and cpp) :")
        match, nulldesc = descriptors_compare(ref[keypoints_start:keypoints_end], res)
        logger.info(("%s/%s match found", match, (keypoints_end - keypoints_start) - nulldesc))

        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms.", 1000.0 * (t2 - t1), 1000.0 * (t1 - t0))
            logger.info("Descriptors computation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))


@unittest.skipUnless(ocl, "opencl missing")
class TestFeature(unittest.TestCase):
    """Test a simple image with all possible path"""

    def test_keypoints(self):
        shape = 32, 32
        img = numpy.zeros(shape, dtype=numpy.uint8)
        img[:16, :15] = 255
        # results calculated by sift from feature
        xkp = 11.82945061
        ykp = 12.82944393
        sigma = 2.12008238,
        orientations = numpy.array([-2.81852078, -1.78396046])
        keypoints = numpy.array(
             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 0, 0, 0, 0, 0,
               0, 48, 67, 0, 0, 0, 0, 0, 0, 55, 79, 0, 0, 0, 0, 0,
               12, 7, 0, 0, 0, 0, 0, 0, 100, 140, 65, 0, 0, 0, 0, 8,
               21, 140, 140, 0, 0, 0, 0, 1, 0, 122, 140, 0, 0, 0, 0, 0,
               19, 0, 0, 0, 0, 0, 0, 6, 140, 43, 4, 0, 0, 0, 0, 131,
               140, 51, 28, 0, 0, 0, 0, 74, 1, 7, 10, 0, 0, 0, 0, 0,
               2, 0, 0, 0, 0, 0, 0, 1, 140, 0, 0, 0, 0, 0, 0, 107,
               140, 0, 0, 0, 0, 0, 0, 132, 5, 0, 0, 0, 0, 0, 0, 3],
              [6, 2, 0, 0, 0, 0, 0, 0, 152, 83, 0, 0, 0, 0, 0, 0,
               152, 63, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
               24, 5, 0, 0, 0, 0, 0, 0, 152, 94, 0, 0, 0, 0, 4, 38,
               152, 42, 0, 0, 0, 0, 38, 43, 0, 0, 0, 0, 0, 0, 21, 8,
               14, 0, 0, 0, 0, 0, 0, 6, 119, 6, 0, 0, 0, 0, 77, 152,
               23, 1, 0, 0, 0, 0, 152, 152, 0, 0, 0, 0, 0, 0, 152, 100,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 14,
               0, 0, 0, 0, 0, 0, 74, 29, 0, 0, 0, 0, 0, 0, 94, 35]
              ])
        epsilon = 5e-5

        def check_kp(kp, config=None):
            self.assertEqual(len(kp), 2, "two keypoints in this image")
            self.assertEqual(len(kp), 2, "two keypoints in this image")
            for i, k in enumerate(kp):
                self.assertLessEqual(abs(k[0] - xkp), epsilon, "x_coordinate matches for kp %i" % i)
                self.assertLessEqual(abs(k[1] - ykp), epsilon, "y_coordinate matches for kp %i" % i)
                self.assertLessEqual(abs(k[2] - sigma), epsilon, "sigma matches for kp %i" % i)
                self.assertLessEqual((abs(k[3] - orientations).min()), 0.03, "one orientation matches for %i" % i)
                idx = (abs(k[3] - orientations).argmin())
                l1 = abs(k[4] - keypoints[idx]).sum()
                if l1 > 100:
                    logger.warning("error is not neglectable %s\n%s\n%s", l1, k[4], keypoints[idx])
                self.assertLessEqual(l1, 500, "keypoint matches %i matches:\n%s\ngot:\n%s\nexpected:\n%s" % (i, l1, k[4], keypoints[idx]))

        from silx.opencl.sift import SiftPlan
        sp = SiftPlan(template=img, profile=True)
        kp = sp(img)
        sp.log_profile()
        check_kp(kp)


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(parameterize(TestKeypoints, "orientation_gpu", (128,), "descriptor_gpu2", (8, 8, 8)))
    testSuite.addTest(parameterize(TestKeypoints, "orientation_cpu", (1,), "descriptor_cpu", (1,)))
    testSuite.addTest(parameterize(TestKeypoints, "orientation_gpu", (128,), "descriptor_gpu1", (8, 4, 4)))
    testSuite.addTest(TestFeature("test_keypoints"))
    return testSuite
