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
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/08/2019"

import time
import logging
import numpy
import os


from silx.opencl import ocl, kernel_workgroup_size
if ocl:
    import pyopencl, pyopencl.array

import unittest
from ..utils import calc_size, get_opencl_code
from .test_image_setup import my_gradient, my_local_maxmin, my_interp_keypoint, \
    interpolation_setup, local_maxmin_setup, scipy
from .test_image_functions import norm_L1
logger = logging.getLogger(__name__)

PRINT_KEYPOINTS = False


@unittest.skipUnless(ocl, "OpenCL missing")
class TestImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestImage, cls).setUpClass()
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

    @classmethod
    def tearDownClass(cls):
        super(TestImage, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        kernel_src = os.linesep.join((get_opencl_code(os.path.join("sift", i)) for i in ("sift", "image")))
        self.program = pyopencl.Program(self.ctx, kernel_src).build()
        self.wg = (8, 1)

    def tearDown(self):
        self.mat = None
        self.program = None

    @unittest.skipIf(scipy is None, "scipy missing")
    def test_gradient(self):
        """
        tests the gradient kernel (norm and orientation)
        """
        border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, scale, nb_keypoints, width, height, DOGS, g = local_maxmin_setup()
        self.mat = numpy.ascontiguousarray(g[1])
        self.height, self.width = numpy.int32(self.mat.shape)
        self.gpu_mat = pyopencl.array.to_device(self.queue, self.mat)
        self.gpu_grad = pyopencl.array.empty(self.queue, self.mat.shape, dtype=numpy.float32, order="C")
        self.gpu_ori = pyopencl.array.empty(self.queue, self.mat.shape, dtype=numpy.float32, order="C")
        self.shape = calc_size((self.width, self.height), self.wg)

        t0 = time.time()
        k1 = self.program.compute_gradient_orientation(self.queue, self.shape, self.wg, self.gpu_mat.data, self.gpu_grad.data, self.gpu_ori.data, self.width, self.height)
        res_norm = self.gpu_grad.get()
        res_ori = self.gpu_ori.get()
        t1 = time.time()
        ref_norm, ref_ori = my_gradient(self.mat)
        t2 = time.time()
        delta_norm = abs(ref_norm - res_norm).max()
        delta_ori = abs(ref_ori - res_ori).max()
        if (PRINT_KEYPOINTS):
            rmin, cmin = 0, 0
            rmax, cmax = rmin + 6, cmin + 6

            logger.info(res_norm[-rmax, cmin:cmax])
            logger.info("")
            logger.info(ref_norm[-rmax, cmin:cmax])

        logger.info("delta_norm=%s" % delta_norm)
        logger.info("delta_ori=%s" % delta_ori)
        self.assertLess(delta_norm, 1e-4, "delta_norm=%s" % (delta_norm))
        self.assertLess(delta_ori, 1e-4, "delta_ori=%s" % (delta_ori))

        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Gradient computation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))

    @unittest.skipIf(scipy is None, "scipy missing")
    def test_local_maxmin(self):
        """
        tests the local maximum/minimum detection kernel
        """
        # local_maxmin_setup :
        border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, s, nb_keypoints, width, height, DOGS, g = local_maxmin_setup()
        self.s = numpy.int32(s)  # 1, 2, 3 ... not 4 nor 0.
        self.gpu_dogs = pyopencl.array.to_device(self.queue, DOGS)
        self.output = pyopencl.array.empty(self.queue, (nb_keypoints, 4), dtype=numpy.float32, order="C")
        self.output.fill(-1.0)  # memset for invalid keypoints
        self.counter = pyopencl.array.empty(self.queue, (1,), dtype=numpy.int32)
        self.counter.fill(0)
        nb_keypoints = numpy.int32(nb_keypoints)
        self.shape = calc_size((DOGS.shape[1], DOGS.shape[0] * DOGS.shape[2]), self.wg)  # it's a 3D vector !!

        t0 = time.time()
        k1 = self.program.local_maxmin(self.queue, self.shape, self.wg,
                                       self.gpu_dogs.data, self.output.data,
                                       border_dist, peakthresh, octsize, EdgeThresh0, EdgeThresh,
                                       self.counter.data, nb_keypoints, self.s, width, height)

        res = self.output.get()
        self.keypoints1 = self.output  # for further use
        self.actual_nb_keypoints = self.counter.get()[0]  # for further use

        t1 = time.time()
        ref, actual_nb_keypoints2 = my_local_maxmin(DOGS, peakthresh, border_dist, octsize,
                                                    EdgeThresh0, EdgeThresh, nb_keypoints, self.s, width, height)
        t2 = time.time()

        # we have to sort the arrays, for peaks orders is unknown for GPU
        res_peaks = res[(res[:, 0].argsort(axis=0)), 0]
        ref_peaks = ref[(ref[:, 0].argsort(axis=0)), 0]
        res_r = res[(res[:, 1].argsort(axis=0)), 1]
        ref_r = ref[(ref[:, 1].argsort(axis=0)), 1]
        res_c = res[(res[:, 2].argsort(axis=0)), 2]
        ref_c = ref[(ref[:, 2].argsort(axis=0)), 2]
        # res_s = res[(res[:,3].argsort(axis=0)),3]
        # ref_s = ref[(ref[:,3].argsort(axis=0)),3]
        delta_peaks = abs(ref_peaks - res_peaks).max()
        delta_r = abs(ref_r - res_r).max()
        delta_c = abs(ref_c - res_c).max()

        if (PRINT_KEYPOINTS):
            logger.info("keypoints after 2 steps of refinement: (s= %s, octsize=%s) %s", self.s, octsize, self.actual_nb_keypoints)
            # logger.info("For ref: %s" %(ref_peaks[ref_peaks!=-1].shape))
            logger.info(res[0:self.actual_nb_keypoints])  # [0:74]
            # logger.info(ref[0:32]

        self.assertLess(delta_peaks, 1e-4, "delta_peaks=%s" % (delta_peaks))
        self.assertLess(delta_r, 1e-4, "delta_r=%s" % (delta_r))
        self.assertLess(delta_c, 1e-4, "delta_c=%s" % (delta_c))
        logger.info("delta_peaks=%s" % delta_peaks)
        logger.info("delta_r=%s" % delta_r)
        logger.info("delta_c=%s" % delta_c)

        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Local extrema search took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))

    @unittest.skipIf(scipy is None, "scipy missing")
    def test_interpolation(self):
        """
        tests the keypoints interpolation kernel
        Requires the following: "self.keypoints1", "self.actual_nb_keypoints",     "self.gpu_dog_prev", self.gpu_dog",             "self.gpu_dog_next", "self.s", "self.width", "self.height", "self.peakthresh"
        """

        # interpolation_setup :
        border_dist, peakthresh, EdgeThresh, EdgeThresh0, octsize, nb_keypoints, actual_nb_keypoints, width, height, DOGS, s, keypoints_prev, blur = interpolation_setup()

        # actual_nb_keypoints is the number of keypoints returned by "local_maxmin".
        # After the interpolation, it will be reduced, but we can still use it as a boundary.
        maxwg = kernel_workgroup_size(self.program, "interp_keypoint")
        shape = calc_size((keypoints_prev.shape[0],), maxwg)
        gpu_dogs = pyopencl.array.to_device(self.queue, DOGS)
        gpu_keypoints1 = pyopencl.array.to_device(self.queue, keypoints_prev)
        # actual_nb_keypoints = numpy.int32(len((keypoints_prev[:,0])[keypoints_prev[:,1] != -1]))
        start_keypoints = numpy.int32(0)
        actual_nb_keypoints = numpy.int32(actual_nb_keypoints)
        InitSigma = numpy.float32(1.6)  #   warning: it must be the same in my_keypoints_interpolation
        t0 = time.time()
        k1 = self.program.interp_keypoint(self.queue, shape, (maxwg,),
                                          gpu_dogs.data, gpu_keypoints1.data, start_keypoints, actual_nb_keypoints,
                                          peakthresh, InitSigma, width, height)
        res = gpu_keypoints1.get()

        t1 = time.time()
        ref = numpy.copy(keypoints_prev)  # important here
        for i, k in enumerate(ref[:nb_keypoints, :]):
            ref[i] = my_interp_keypoint(DOGS, s, k[1], k[2], 5, peakthresh, width, height)

        t2 = time.time()

        # we have to compare keypoints different from (-1,-1,-1,-1)
        res2 = res[res[:, 1] != -1]
        ref2 = ref[ref[:, 1] != -1]

        if (PRINT_KEYPOINTS):
            logger.info("[s=%s]Keypoints before interpolation: %s", s, actual_nb_keypoints)
            # logger.info(keypoints_prev[0:10,:]
            logger.info("[s=%s]Keypoints after interpolation : %s", s, res2.shape[0])
            logger.info(res[0:actual_nb_keypoints])  # [0:10,:]
            # logger.info("Ref:")
            # logger.info(ref[0:32,:]


#         print(maxwg, self.maxwg, self.wg[0], self.wg[1])
        if self.maxwg < self.wg[0] * self.wg[1]:
            logger.info("Not testing result as the WG is too little %s", self.maxwg)
            return
        self.assertLess(abs(len(ref2) - len(res2)) / (len(ref2) + len(res2)), 0.33, "the number of keypoint is almost the same")
#         print(ref2)
#         print(res2)

        delta = norm_L1(ref2, res2)
        self.assertLess(delta, 0.43, "delta=%s" % (delta))
        logger.info("delta=%s" % delta)

        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Keypoints interpolation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestImage("test_gradient"))
    testSuite.addTest(TestImage("test_local_maxmin"))
    testSuite.addTest(TestImage("test_interpolation"))
    return testSuite
