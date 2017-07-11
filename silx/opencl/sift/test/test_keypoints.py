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
__date__ = "15/03/2017"

import os
import unittest
import time
import logging
import numpy
try:
    import scipy
except ImportError:
    scipy = None
else:
    import scipy.misc
    import scipy.ndimage

# for Python implementation of tested functions
from .test_image_functions import my_compact, my_orientation, keypoints_compare, my_descriptor, descriptors_compare
from .test_image_setup import orientation_setup, descriptor_setup
from silx.opencl import ocl, kernel_workgroup_size
if ocl:
    import pyopencl, pyopencl.array
from ..utils import calc_size, get_opencl_code
logger = logging.getLogger(__name__)


class ParameterisedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parameterised should
        inherit from this class.
        From Eli Bendersky's website
        http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
    """
    @classmethod
    def setUpClass(cls):
        super(ParameterisedTestCase, cls).setUpClass()
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
        super(ParameterisedTestCase, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def __init__(self, methodName='runTest', param=None):
        super(ParameterisedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parameterise(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite


@unittest.skipUnless(ocl and scipy, "opencl or scipy missing")
class test_keypoints(ParameterisedTestCase):
    def setUp(self):
        self.abort = False
        if scipy and ocl is None:
            return
        try:
            self.testdata = scipy.misc.ascent()
        except:
            # for very old versions of scipy
            self.testdata = scipy.misc.lena()

        for kernel_file in self.param:
            if "cpu" in kernel_file:
                self.USE_CPU = True
            else:
                self.USE_CPU = False
            if kernel_file.startswith("orient"):
                self.wg_orient = self.param[kernel_file]
                prod_wg = 1
                for i in self.wg_orient:
                    prod_wg *= i

                kernel_src = get_opencl_code(os.path.join("sift", kernel_file))
                try:
                    self.program_orient = pyopencl.Program(self.ctx, kernel_src).build()
                except:
                    logger.warning("Failed to compile kernel '%s': aborting" % kernel_file)
                    self.abort = True
                    return
            elif kernel_file.startswith("keypoint"):
                self.wg_keypoint = self.param[kernel_file]
                kernel_src = get_opencl_code(os.path.join("sift", kernel_file))
                prod_wg = 1
                for i in self.wg_keypoint:
                    prod_wg *= i

                try:
                    self.program_keypoint = pyopencl.Program(self.ctx, kernel_src).build()
                except:
                    logger.warning("Failed to compile kernel '%s': aborting" % kernel_file)
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
        max_wg = kernel_workgroup_size(self.program_orient,"orientation_assignment")
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

        # Call the kernel
        t0 = time.time()
        k1 = self.program_orient.orientation_assignment(self.queue, shape, wg, *kargs)
        res = gpu_keypoints.get()
        cnt = counter.get()
        t1 = time.time()

        # Reference Python implemenattion
        ref, updated_nb_keypoints = my_orientation(keypoints, nb_keypoints, keypoints_start, keypoints_end, grad, ori, octsize, orisigma)
        t2 = time.time()

        # sort to compare added keypoints
        upbound = min(cnt, updated_nb_keypoints)
        d1, d2, d3, d4 = keypoints_compare(ref[0:upbound], res[0:upbound])
        self.assert_(d1 < 1e-4, "delta_cols=%s" % (d1))
        self.assert_(d2 < 1e-4, "delta_rows=%s" % (d2))
        self.assert_(d3 < 1e-4, "delta_sigma=%s" % (d3))
        self.assert_(d4 < 1e-1, "delta_angle=%s" % (d4))  # orientation has a poor precision
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
        max_wg = kernel_workgroup_size(self.program_keypoint,"descriptor")
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
        k1 = self.program_keypoint.descriptor(self.queue, shape, wg, *kargs)
        try:
            res = gpu_descriptors.get()
        except (pyopencl.LogicError, RuntimeError) as error:
            logger.warning("Segmentation fault like error (%s) on Descriptor for %s" % (error, self.param))
            return
        t1 = time.time()

        # Reference Python implementation
        ref = my_descriptor(keypoints_o, grad, ori, octsize, keypoints_start, keypoints_end)
        ref_sort = ref[numpy.argsort(keypoints[keypoints_start: keypoints_end, 1])]
        t2 = time.time()

        res_sort = res[numpy.argsort(keypoints[keypoints_start:keypoints_end, 1])]
        logger.info(res_sort[5:10])
        logger.info(ref_sort[5:10])
        logger.info("Comparing descriptors (OpenCL and cpp) :")
        match, nulldesc = descriptors_compare(ref[keypoints_start:keypoints_end], res)
        logger.info(("%s/%s match found" , match, (keypoints_end - keypoints_start) - nulldesc))

        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." , 1000.0 * (t2 - t1), 1000.0 * (t1 - t0))
            logger.info("Descriptors computation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))


def suite():
    testSuite = unittest.TestSuite()
    TESTCASES = [{"orientation_gpu": (128,), "keypoints_gpu2": (8, 8, 8)},
                 {"orientation_cpu": (1,), "keypoints_cpu": (1,)},
                 {"orientation_gpu": (128,), "keypoints_gpu1": (4, 4, 8)},
                 ]
    for param in TESTCASES:
        testSuite.addTest(ParameterisedTestCase.parameterise(
                test_keypoints, param))

    return testSuite
