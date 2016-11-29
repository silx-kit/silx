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
Test suite for transformation kernel
"""

from __future__ import division, print_function

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/10/2016"


import os
import unittest
import time
import logging
import numpy
try:
    from silx.third_party import six
except ImportError:
    import six
try:
    import scipy
except ImportError:
    scipy = None
else:
    import scipy.misc
    import scipy.ndimage
from silx.opencl import ocl, kernel_workgroup_size
if ocl:
    import pyopencl, pyopencl.array
# for Python implementation of tested functions
# from .test_image_functions import
# from .test_image_setup import
from ..utils import calc_size, get_opencl_code
from ..plan import SiftPlan
from ..match import MatchPlan
logger = logging.getLogger(__name__)

SHOW_FIGURES = False
IMAGE_RESHAPE = True
USE_LENA = True
DEVICETYPE = "ALL"

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

#             logger.warning("max_work_group_size: %s on (%s, %s)", cls.maxwg, platform_id, device_id)

    @classmethod
    def tearDownClass(cls):
        super(TestTransform, cls).tearDownClass()
        cls.ctx = None
        cls.queue = None

    def setUp(self):
        kernel_src = get_opencl_code("transform")
        self.program = pyopencl.Program(self.ctx, kernel_src).build()  # .build('-D WORKGROUP_SIZE=%s' % wg_size)
        self.wg = (1, 128)

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

    def matching_correction(self, image, image2):
        '''
        Computes keypoints for two images and try to align image2 on image1
        '''
        # computing keypoints matching
        s = SiftPlan(template=image, devicetype=DEVICETYPE, max_workgroup_size=self.maxwg)
        kp1 = s.keypoints(image)
        kp2 = s.keypoints(image2)  # image2 and image must have the same size
        m = MatchPlan(devicetype=DEVICETYPE)
        matching = m.match(kp2, kp1)
#         print(numpy.isnan(matching))
        N = matching.shape[0]
        # solving normals equations for least square fit
        X = numpy.zeros((2 * N, 6))
        X[::2, 2:] = 1, 0, 0, 0
        X[::2, 0] = matching[:, 0].x
        X[::2, 1] = matching[:, 0].y
        X[1::2, 0:3] = 0, 0, 0
        X[1::2, 3] = matching[:, 0].x
        X[1::2, 4] = matching[:, 0].y
        X[1::2, 5] = 1
        y = numpy.zeros((2 * N, 1))
        y[::2, 0] = matching.x[:, 1]
        y[1::2, 0] = matching.y[:, 1]
        # A = numpy.dot(X.transpose(),X)
        # sol = numpy.dot(numpy.linalg.inv(A),numpy.dot(X.transpose(),y))
#         print(X.shape, y.shape)
#         print(X)
#         print(y)
        sol = numpy.dot(numpy.linalg.pinv(X), y)
        MSE = numpy.linalg.norm(y - numpy.dot(X, sol)) ** 2 / N  # value of the sum of residuals at "sol"
        return sol, MSE

    @unittest.skipIf(os.environ.get("SILX_TEST_LOW_MEM") == "True", "low mem")
    def test_transform(self):
        '''
        tests transform kernel
        '''

        if (USE_LENA):
            # original image
            if hasattr(scipy.misc, "ascent"):
                image = scipy.misc.ascent().astype(numpy.float32)
            else:
                image = scipy.misc.lena().astype(numpy.float32)

            image = numpy.ascontiguousarray(image[0:512, 0:512])


            # transformation
            angle = 1.9  # numpy.pi/5.0
    #        matrix = numpy.array([[numpy.cos(angle),-numpy.sin(angle)],[numpy.sin(angle),numpy.cos(angle)]],dtype=numpy.float32)
    #        offset_value = numpy.array([1000.0, 100.0],dtype=numpy.float32)
    #        matrix = numpy.array([[0.9,0.2],[-0.4,0.9]],dtype=numpy.float32)
    #        offset_value = numpy.array([-20.0,256.0],dtype=numpy.float32)
            matrix = numpy.array([[1.0, -0.75], [0.7, 0.5]], dtype=numpy.float32)

            offset_value = numpy.array([250.0, -150.0], dtype=numpy.float32)

            image2 = scipy.ndimage.interpolation.affine_transform(image, matrix, offset=offset_value, order=1, mode="constant")

        else:  # use images of a stack
            image = scipy.misc.imread("/home/paleo/Titanium/test/frame0.png")
            image2 = scipy.misc.imread("/home/paleo/Titanium/test/frame1.png")
            offset_value = numpy.array([0.0, 0.0], dtype=numpy.float32)
        image_height, image_width = image.shape
        image2_height, image2_width = image2.shape

        fill_value = numpy.float32(0.0)
        mode = numpy.int32(1)

        if IMAGE_RESHAPE:  # turns out that image should always be reshaped
            output_height, output_width = int(3000), int(3000)
            image, image_height, image_width = self.image_reshape(image, output_height, output_width, image_height, image_width)
            image2, image2_height, image2_width = self.image_reshape(image2, output_height, output_width, image2_height, image2_width)
        else:
            output_height, output_width = int(image_height * numpy.sqrt(2)), int(image_width * numpy.sqrt(2))
        logger.info("Image : (%s, %s) -- Output: (%s, %s)", image_height, image_width, output_height, output_width)

        # perform correction by least square
        sol, MSE = self.matching_correction(image, image2)
        logger.info(sol)

        correction_matrix = numpy.zeros((2, 2), dtype=numpy.float32)
        correction_matrix[0] = sol[0:2, 0]
        correction_matrix[1] = sol[3:5, 0]
        matrix_for_gpu = correction_matrix.reshape(4, 1)  # for float4 struct
        offset_value[0] = sol[2, 0]
        offset_value[1] = sol[5, 0]

        maxwg = kernel_workgroup_size(self.program,"transform")
        wg = maxwg, 1
        shape = calc_size((output_width, output_height), wg)
        gpu_image = pyopencl.array.to_device(self.queue, image2)
        gpu_output = pyopencl.array.empty(self.queue, (output_height, output_width), dtype=numpy.float32, order="C")
        gpu_matrix = pyopencl.array.to_device(self.queue, matrix_for_gpu)
        gpu_offset = pyopencl.array.to_device(self.queue, offset_value)
        image_height, image_width = numpy.int32((image_height, image_width))
        output_height, output_width = numpy.int32((output_height, output_width))

        t0 = time.time()
        k1 = self.program.transform(self.queue, shape, wg,
                                    gpu_image.data, gpu_output.data, gpu_matrix.data, gpu_offset.data,
                                    image_width, image_height, output_width, output_height, fill_value, mode)
        res = gpu_output.get()
        t1 = time.time()
#        logger.info(res[0,0]

        ref = scipy.ndimage.interpolation.affine_transform(image2, correction_matrix,
                                                           offset=offset_value,
                                                           output_shape=(output_height, output_width),
                                                           order=1,
                                                           mode="constant",
                                                           cval=fill_value)
        t2 = time.time()

        delta = abs(res - image)
        delta_arg = delta.argmax()
        delta_max = delta.max()
#        delta_mse_res = ((res-image)**2).sum()/image.size
#        delta_mse_ref = ((ref-image)**2).sum()/image.size
        at_0, at_1 = delta_arg / output_width, delta_arg % output_width
        logger.info("Max error: %f at (%d, %d)", delta_max, at_0, at_1)
#        print("Mean Squared Error Res/Original : %f" %(delta_mse_res))
#        print("Mean Squared Error Ref/Original: %f" %(delta_mse_ref))
        logger.info("minimal MSE according to least squares : %f", MSE)
#        logger.info(res[at_0,at_1]
#        logger.info(ref[at_0,at_1]

        if self.PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms.", 1000.0 * (t2 - t1), 1000.0 * (t1 - t0))
            logger.info("Transformation took %.3fms", 1e-6 * (k1.profile.end - k1.profile.start))


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestTransform("test_transform"))
    return testSuite

