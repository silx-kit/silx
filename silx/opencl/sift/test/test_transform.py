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
Test suite for transformation kernel
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/06/2018"


import os
import unittest
import time
import logging
import numpy
try:
    import scipy.misc
    import scipy.ndimage
except ImportError:
    scipy = None

from silx.opencl import ocl, kernel_workgroup_size
if ocl:
    import pyopencl.array
# for Python implementation of tested functions
# from .test_image_functions import
# from .test_image_setup import
from ..utils import calc_size, get_opencl_code, matching_correction
from ..plan import SiftPlan
from ..match import MatchPlan
from silx.test.utils import test_options
logger = logging.getLogger(__name__)


@unittest.skipUnless(scipy and ocl, "scipy or ocl missing")
class TestTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestTransform, cls).setUpClass()
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
        super(TestTransform, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        kernel_src = get_opencl_code(os.path.join("sift", "transform"))
        self.program = pyopencl.Program(self.ctx, kernel_src).build()  # .build('-D WORKGROUP_SIZE=%s' % wg_size)
        self.wg = (1, 128)
        if hasattr(scipy.misc, "ascent"):
            self.image = scipy.misc.ascent().astype(numpy.float32)
        else:
            self.image = scipy.misc.lena().astype(numpy.float32)

    def tearDown(self):
        self.program = None

    def image_reshape(self, img, output_height, output_width, image_height, image_width):
        '''
        Reshape the image to get a bigger image with the input image in the center

        '''
        image3 = numpy.zeros((output_height, output_width), dtype=numpy.float32)
        d1 = (output_width - image_width) // 2
        d0 = (output_height - image_height) // 2
        image3[d0:-d0, d1:-d1] = numpy.copy(img)
        image = image3
        image_height, image_width = output_height, output_width
        return image, image_height, image_width

    @unittest.skipIf(test_options.TEST_LOW_MEM, "low mem")
    def test_transform(self):
        '''
        tests transform kernel
        '''

        # Transformation
        # ---------------
        matrix = numpy.array([[1.0, -0.75], [0.7, 0.5]], dtype=numpy.float32)
        offset_value = numpy.array([250.0, -150.0], dtype=numpy.float32)
        transformation = lambda img: scipy.ndimage.interpolation.affine_transform(img, matrix, offset=offset_value, order=1, mode="constant")
        image_transformed = transformation(self.image)

        fill_value = numpy.float32(0.0)
        mode = numpy.int32(1)

        # computing keypoints matching with SIFT
        sift_plan = SiftPlan(template=self.image, block_size=self.maxwg)
        kp1 = sift_plan.keypoints(self.image)
        kp2 = sift_plan.keypoints(image_transformed)  # image2 and image must have the same size
        match_plan = MatchPlan()  # cls.ctx
        matching = match_plan.match(kp2, kp1)

        # Retrieve the linear transformation from the matching pairs
        sol = matching_correction(matching)
        logger.info(sol)

        # Compute the correction matrix (inverse of transformation)
        correction_matrix = numpy.zeros((2, 2), dtype=numpy.float32)
        correction_matrix[0] = sol[0:2, 0]
        correction_matrix[1] = sol[3:5, 0]
        matrix_for_gpu = correction_matrix.reshape(4, 1)  # for float4 struct
        offset_value[0] = sol[2, 0]
        offset_value[1] = sol[5, 0]

        # Prepare the arguments for the "transform" kernel call
        maxwg = kernel_workgroup_size(self.program, "transform")
        wg = maxwg, 1
        shape = calc_size(self.image.shape[::-1], wg)
        gpu_image = pyopencl.array.to_device(self.queue, image_transformed)
        gpu_output = pyopencl.array.empty(self.queue, self.image.shape, dtype=numpy.float32, order="C")
        gpu_matrix = pyopencl.array.to_device(self.queue, matrix_for_gpu)
        gpu_offset = pyopencl.array.to_device(self.queue, offset_value)
        image_height, image_width = numpy.int32(self.image.shape)
        output_height, output_width = numpy.int32(gpu_output.shape)
        kargs = [
            gpu_image.data,
            gpu_output.data,
            gpu_matrix.data,
            gpu_offset.data,
            image_width,
            image_height,
            output_width,
            output_height,
            fill_value, mode
        ]

        # Call the kernel
        t0 = time.time()
        k1 = self.program.transform(self.queue, shape, wg, *kargs)
        res = gpu_output.get()

        # Reference result
        t1 = time.time()
        ref = scipy.ndimage.interpolation.affine_transform(image_transformed, correction_matrix,
                                                           offset=offset_value,
                                                           output_shape=(output_height, output_width),
                                                           order=1,
                                                           mode="constant",
                                                           cval=fill_value)
        t2 = time.time()

        # Compare the implementations
        delta = numpy.abs(res - ref)
        delta_arg = delta.argmax()
        delta_max = delta.max()
        at_0, at_1 = delta_arg / output_width, delta_arg % output_width
        logger.info("Max difference wrt scipy : %f at (%d, %d)", delta_max, at_0, at_1)

        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms.", 1000.0 * (t2 - t1), 1000.0 * (t1 - t0))
            logger.info("Transformation took %.3fms", 1e-6 * (k1.profile.end - k1.profile.start))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestTransform("test_transform"))
    return testSuite
