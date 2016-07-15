#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for algebra kernels
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
PRINT_KEYPOINTS = False

print "working on %s" % ctx.devices[0].name




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




class test_algebra(unittest.TestCase):
    def setUp(self):

        kernel_path = os.path.join(os.path.dirname(os.path.abspath(sift.__file__)), "algebra.cl")
        kernel_src = open(kernel_path).read()
        self.program = pyopencl.Program(ctx, kernel_src).build()
        self.wg = (32, 4)


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

        gpu_mat1 = pyopencl.array.to_device(queue, mat1)
        gpu_mat2 = pyopencl.array.to_device(queue, mat2)
        gpu_out = pyopencl.array.empty(queue, mat1.shape, dtype=numpy.float32, order="C")
        shape = calc_size((width, height), self.wg)

        t0 = time.time()
        k1 = self.program.combine(queue, shape, self.wg,
                                  gpu_mat1.data, coeff1, gpu_mat2.data, coeff2,
                                  gpu_out.data, numpy.int32(0),
                                  width, height)
        res = gpu_out.get()
        t1 = time.time()
        ref = my_combine(mat1, coeff1, mat2, coeff2)
        t2 = time.time()
        delta = abs(ref - res).max()
        logger.info("delta=%s" % delta)
        self.assert_(delta < 1e-4, "delta=%s" % (delta))
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Linear combination took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))



    def test_compact(self):
        """
        tests the "compact" kernel
        """

        nbkeypoints = 10000 #constant value
        keypoints = numpy.random.rand(nbkeypoints, 4).astype(numpy.float32)
        nb_ones = 0
        for i in range(0, nbkeypoints):
            if ((numpy.random.rand(1))[0] < 0.25):
                keypoints[i] = (-1, -1, i, -1)
                nb_ones += 1
            else: keypoints[i,2] = i

        gpu_keypoints = pyopencl.array.to_device(queue, keypoints)
        output = pyopencl.array.empty(queue, (nbkeypoints, 4), dtype=numpy.float32, order="C")
        output.fill(-1.0, queue)
        counter = pyopencl.array.zeros(queue, (1,), dtype=numpy.int32, order="C")
        wg = max(self.wg),
        shape = calc_size((keypoints.shape[0],), wg)
        nbkeypoints = numpy.int32(nbkeypoints)
        startkeypoints = numpy.int32(0)
        t0 = time.time()
        k1 = self.program.compact(queue, shape, wg,
            gpu_keypoints.data, output.data, counter.data, startkeypoints, nbkeypoints)
        res = output.get()
        if (PRINT_KEYPOINTS):
            print res
        count = counter.get()[0]
        t1 = time.time()
        ref, count_ref = my_compact(keypoints, nbkeypoints)
        t2 = time.time()

        print("Kernel counter : %s / Python counter : %s / True value : %s" % (count, count_ref, nbkeypoints - nb_ones))

        res_sort_arg = res[:, 2].argsort(axis=0)
        res_sort = res[res_sort_arg]
        ref_sort_arg = ref[:, 2].argsort(axis=0)
        ref_sort = ref[ref_sort_arg]
        if (PRINT_KEYPOINTS):
            print "Delta matrix :"
            print (abs(res_sort - ref_sort) > 1e-5).sum()
        delta = abs((res_sort - ref_sort)).max()
        self.assert_(delta < 1e-5, "delta=%s" % (delta))
        self.assertEqual(count, count_ref, "counters are the same")
        logger.info("delta=%s" % delta)
        if PROFILE:
            logger.info("Global execution time: CPU %.3fms, GPU: %.3fms." % (1000.0 * (t2 - t1), 1000.0 * (t1 - t0)))
            logger.info("Compact operation took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))

















def test_suite_algebra():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_algebra("test_combine"))
    testSuite.addTest(test_algebra("test_compact"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_algebra()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

