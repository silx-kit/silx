#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for all preprocessing kernels.
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-05-28"
__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

import time, os, logging
import numpy
import pyopencl, pyopencl.array
import scipy, scipy.misc, scipy.ndimage
import sys
import unittest
from utilstest import UtilsTest, getLogger, ctx
import sift_pyocl as sift
from sift_pyocl.utils import calc_size
logger = getLogger(__file__)
if logger.getEffectiveLevel() <= logging.INFO:
    PROFILE = True
    queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
    import pylab
else:
    PROFILE = False
    queue = pyopencl.CommandQueue(ctx)

print "working on %s" % ctx.devices[0].name

def my_blur(img, kernel):
    """
    hand made implementation of gaussian blur with OUR kernel
    which differs from Scipy's if ksize is even
    """
    tmp1 = scipy.ndimage.filters.convolve1d(img, kernel, axis= -1, mode="reflect")
    return scipy.ndimage.filters.convolve1d(tmp1, kernel, axis=0, mode="reflect")


class test_convol(unittest.TestCase):
    def setUp(self):
        self.input = scipy.misc.lena().astype(numpy.float32)
        self.input = numpy.ascontiguousarray(self.input[0:507,0:209])
        
        self.gpu_in = pyopencl.array.to_device(queue, self.input)
        self.gpu_tmp = pyopencl.array.empty(queue, self.input.shape, dtype=numpy.float32, order="C")
        self.gpu_out = pyopencl.array.empty(queue, self.input.shape, dtype=numpy.float32, order="C")
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "convolution.cl")
        kernel_src = open(kernel_path).read()
#        compile_options = "-D NIMAGE=%i" % self.input.size
#        logger.info("Compiling file %s with options %s" % (kernel_path, compile_options))
#        self.program = pyopencl.Program(ctx, kernel_src).build(options=compile_options)
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.IMAGE_W = numpy.int32(self.input.shape[-1])
        self.IMAGE_H = numpy.int32(self.input.shape[0])
        self.wg = (256, 2)
        self.shape = calc_size((self.input.shape[1], self.input.shape[0]), self.wg)

    def tearDown(self):
        self.input = None
#        self.gpudata.release()
        self.program = None

    def test_convol_hor(self):
        """
        tests the convolution kernel
        """
        for sigma in [2, 15 / 8.]:
            ksize = int(8 * sigma + 1)
            x = numpy.arange(ksize) - (ksize - 1.0) / 2.0
            gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
            gaussian /= gaussian.sum(dtype=numpy.float32)
            gpu_filter = pyopencl.array.to_device(queue, gaussian)
            t0 = time.time()
            k1 = self.program.horizontal_convolution(queue, self.shape, self.wg,
                                self.gpu_in.data, self.gpu_out.data, gpu_filter.data, numpy.int32(ksize), self.IMAGE_W, self.IMAGE_H)
            res = self.gpu_out.get()
            t1 = time.time()
            ref = scipy.ndimage.filters.convolve1d(self.input, gaussian, axis= -1, mode="reflect")
            t2 = time.time()
            delta = abs(ref - res).max()
            if ksize % 2 == 0:  #we have a problem with even kernels !!!
                self.assert_(delta < 50, "sigma= %s delta=%s" % (sigma, delta))
            else:
                self.assert_(delta < 1e-4, "sigma= %s delta=%s" % (sigma, delta))
            logger.info("sigma= %s delta=%s" % (sigma, delta))
            if PROFILE:
                logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
                logger.info("Horizontal convolution took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
                fig = pylab.figure()
                fig.suptitle('convolution horizontal sigma=%s delta=%s' % (sigma, delta))
                sp1 = fig.add_subplot(221)
                sp1.imshow(self.input, interpolation="nearest")
                sp2 = fig.add_subplot(222)
                sp2.imshow(ref, interpolation="nearest")
                sp3 = fig.add_subplot(223)
                sp3.imshow(ref - res, interpolation="nearest")
                sp4 = fig.add_subplot(224)
                sp4.imshow(res, interpolation="nearest")
                fig.show()
                raw_input("enter")
    def test_convol_vert(self):
        """
        tests the convolution kernel
        """
        for sigma in [2, 15 / 8.]:
            ksize = int(8 * sigma + 1)
            x = numpy.arange(ksize) - (ksize - 1.0) / 2.0
            gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
            gaussian /= gaussian.sum(dtype=numpy.float32)
            gpu_filter = pyopencl.array.to_device(queue, gaussian)
            t0 = time.time()
            k1 = self.program.vertical_convolution(queue, self.shape, self.wg,
                                self.gpu_in.data, self.gpu_out.data, gpu_filter.data, numpy.int32(ksize), self.IMAGE_W, self.IMAGE_H)
            res = self.gpu_out.get()
            t1 = time.time()
            ref = scipy.ndimage.filters.convolve1d(self.input, gaussian, axis=0, mode="reflect")
            t2 = time.time()
            delta = abs(ref - res).max()
            if ksize % 2 == 0:  #we have a problem with even kernels !!!
                self.assert_(delta < 50, "sigma= %s delta=%s" % (sigma, delta))
            else:
                self.assert_(delta < 1e-4, "sigma= %s delta=%s" % (sigma, delta))
            logger.info("sigma= %s delta=%s" % (sigma, delta))
            if PROFILE:
                logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
                logger.info("Vertical convolution took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
                fig = pylab.figure()
                fig.suptitle('convolution horizontal sigma=%s delta=%s' % (sigma, delta))
                sp1 = fig.add_subplot(221)
                sp1.imshow(self.input, interpolation="nearest")
                sp2 = fig.add_subplot(222)
                sp2.imshow(ref, interpolation="nearest")
                sp3 = fig.add_subplot(223)
                sp3.imshow(ref - res, interpolation="nearest")
                sp4 = fig.add_subplot(224)
                sp4.imshow(res, interpolation="nearest")
                fig.show()
                raw_input("enter")


    def test_convol(self):
        """
        tests the convolution kernel
        """
        for sigma in [2, 15 / 8.]:
            ksize = int(8 * sigma + 1)
            x = numpy.arange(ksize) - (ksize - 1.0) / 2.0
            gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
            gaussian /= gaussian.sum(dtype=numpy.float32)
            gpu_filter = pyopencl.array.to_device(queue, gaussian)
            t0 = time.time()
            k1 = self.program.horizontal_convolution(queue, self.shape, self.wg,
                                self.gpu_in.data, self.gpu_tmp.data, gpu_filter.data, numpy.int32(ksize), self.IMAGE_W, self.IMAGE_H)
            k2 = self.program.vertical_convolution(queue, self.shape, self.wg,
                                self.gpu_tmp.data, self.gpu_out.data, gpu_filter.data, numpy.int32(ksize), self.IMAGE_W, self.IMAGE_H)
            res = self.gpu_out.get()
            k2.wait()
            t1 = time.time()
            ref = my_blur(self.input, gaussian)
#            ref = scipy.ndimage.filters.gaussian_filter(self.input, sigma, mode="reflect")
            t2 = time.time()
            delta = abs(ref - res).max()
            if ksize % 2 == 0:  #we have a problem with even kernels !!!
                self.assert_(delta < 50, "sigma= %s delta=%s" % (sigma, delta))
            else:
                self.assert_(delta < 1e-4, "sigma= %s delta=%s" % (sigma, delta))
            logger.info("sigma= %s delta=%s" % (sigma, delta))
            if PROFILE:
                logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
                logger.info("Horizontal convolution took %.3fms and vertical convolution took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start),
                                                                                          1e-6 * (k2.profile.end - k2.profile.start)))
                fig = pylab.figure()
                fig.suptitle('sigma=%s' % sigma)
                sp1 = fig.add_subplot(221)
                sp1.imshow(self.input, interpolation="nearest")
                sp2 = fig.add_subplot(222)
                sp2.imshow(ref, interpolation="nearest")
                sp3 = fig.add_subplot(223)
                sp3.imshow(ref - res, interpolation="nearest")
                sp4 = fig.add_subplot(224)
                sp4.imshow(res, interpolation="nearest")
                fig.show()
                raw_input("enter")

def test_suite_convol():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_convol("test_convol"))
    testSuite.addTest(test_convol("test_convol_hor"))
    testSuite.addTest(test_convol("test_convol_vert"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_convol()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

