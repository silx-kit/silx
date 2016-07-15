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
__date__ = "2013-06-20"
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
import scipy, scipy.misc
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

kernels = {"preprocess":8 , "gaussian":512}
for kernel in kernels:
    kernel_file = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), kernel + ".cl")
    kernel_src = open(kernel_file).read()
    program = pyopencl.Program(ctx, kernel_src).build("-D WORKGROUP=%s" % kernels[kernel])
    kernels[kernel] = program

def gaussian_cpu(sigma, size=None):
    """
    Calculate a 1D gaussian using numpy.
    This is the same as scipy.signal.gaussian

    @param sigma: width of the gaussian
    @param size: can be calculated as 1 + 2 * 4sigma
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

def gaussian_gpu_v1(sigma, size=None):
    """
    Calculate a 1D gaussian using pyopencl.
    This is the same as scipy.signal.gaussian

    @param sigma: width of the gaussian
    @param size: can be calculated as 1 + 2 * 4sigma
    """
    if not size:
        size = int(1 + 8 * sigma)
    g_gpu = pyopencl.array.empty(queue, size, dtype=numpy.float32, order="C")
    t0 = time.time()
    evt1 = kernels["preprocess"].gaussian(queue, (size,), (1,),
                                        g_gpu.data,  # __global     float     *data,
                                        numpy.float32(sigma),  # const        float     sigma,
                                        numpy.int32(size))  # const        int     SIZE
    sum_data = pyopencl.array.sum(g_gpu, dtype=numpy.float32, queue=queue)
    evt2 = kernels["preprocess"].divide_cst(queue, (size,), (1,),
                                          g_gpu.data,  # __global     float     *data,
                                          sum_data.data,  # const        float     sigma,
                                          numpy.int32(size))  # const        int     SIZE
    g = g_gpu.get()
    if PROFILE:
        logger.info("execution time: %.3fms; Kernel took %.3fms and %.3fms" % (1e3 * (time.time() - t0), 1e-6 * (evt1.profile.end - evt1.profile.start)
                , 1e-6 * (evt2.profile.end - evt2.profile.start)))

    return g

def gaussian_gpu_v2(sigma, size=None):
    """
    Calculate a 1D gaussian using pyopencl.
    This is the same as scipy.signal.gaussian.
    Only one kernel to

    @param sigma: width of the gaussian
    @param size: can be calculated as 1 + 2 * 4sigma
    """
    if not size:
        size = int(1 + 8 * sigma)
    g_gpu = pyopencl.array.empty(queue, size, dtype=numpy.float32, order="C")
    t0 = time.time()
    evt = kernels["gaussian"].gaussian(queue, (64,), (64,),
                                        g_gpu.data,  # __global     float     *data,
                                        numpy.float32(sigma),  # const        float     sigma,
                                        numpy.int32(size))  # const        int     SIZE
    g = g_gpu.get()
    if PROFILE:
        logger.info("execution time: %.3fms; Kernel took %.3fms" % (1e3 * (time.time() - t0), 1e-6 * (evt.profile.end - evt.profile.start)))
    return g


def show(ref, res, delta):
    pylab.ion()
    pylab.plot(ref, label="ref")
    pylab.plot(res, label="res")
    pylab.plot(delta, label="delta")
    pylab.legend()
    pylab.show()
    raw_input("enter")

class test_gaussian_v1(unittest.TestCase):

    def test_odd(self):
        """
        test odd kernel size
        """
        sigma = 3.0
        size = 27
        ref = gaussian_cpu(sigma, size)
        res = gaussian_gpu_v1(sigma, size)
        delta = ref - res
        if PROFILE:
            show (ref, res, delta)
        self.assert_(abs(ref - res).max() < 1e-6, "gaussian are the same ")

    def test_even(self):
        """
        test odd kernel size
        """
        sigma = 3.0
        size = 28
        ref = gaussian_cpu(sigma, size)
        res = gaussian_gpu_v1(sigma, size)
        delta = ref - res
        if PROFILE:
            show (ref, res, delta)
        self.assert_(abs(ref - res).max() < 1e-6, "gaussian are the same ")

class test_gaussian_v2(unittest.TestCase):

    def test_odd(self):
        """
        test odd kernel size
        """
        sigma = 3.0
        size = 27
        ref = gaussian_cpu(sigma, size)
        res = gaussian_gpu_v2(sigma, size)
        delta = ref - res
        if PROFILE:
            show (ref, res, delta)
        self.assert_(abs(ref - res).max() < 1e-6, "gaussian are the same ")

    def test_even(self):
        """
        test odd kernel size
        """
        sigma = 3.0
        size = 28
        ref = gaussian_cpu(sigma, size)
        res = gaussian_gpu_v2(sigma, size)
        delta = ref - res
        if PROFILE:
            show (ref, res, delta)
        self.assert_(abs(ref - res).max() < 1e-6, "gaussian are the same ")


def test_suite_gaussian():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_gaussian_v1("test_odd"))
    testSuite.addTest(test_gaussian_v1("test_even"))
    testSuite.addTest(test_gaussian_v2("test_odd"))
    testSuite.addTest(test_gaussian_v2("test_even"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_gaussian()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)



