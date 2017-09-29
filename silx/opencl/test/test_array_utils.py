#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""Test of the OpenCL array_utils"""

from __future__ import division, print_function

__authors__ = ["Pierre paleo"]
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "14/06/2017"


import time
import logging
import numpy as np
import unittest
try:
    import mako
except ImportError:
    mako = None
from ..common import ocl
if ocl:
    import pyopencl as cl
    import pyopencl.array as parray
    from .. import linalg
from ..utils import get_opencl_code
from silx.test.utils import utilstest

logger = logging.getLogger(__name__)
try:
    from scipy.ndimage.filters import laplace
    _has_scipy = True
except ImportError:
    _has_scipy = False



@unittest.skipUnless(ocl and mako, "PyOpenCl is missing")
class TestCpy2d(unittest.TestCase):

    def setUp(self):
        if ocl is None:
            return
        self.ctx = ocl.create_context()
        if logger.getEffectiveLevel() <= logging.INFO:
            self.PROFILE = True
            self.queue = cl.CommandQueue(
                            self.ctx,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.PROFILE = False
            self.queue = cl.CommandQueue(self.ctx)
        self.allocate_arrays()
        self.program = cl.Program(self.ctx, get_opencl_code("array_utils")).build()

    def allocate_arrays(self):
        """
        Allocate various types of arrays for the tests
        """
        self.prng_state = np.random.get_state()
        # Generate arrays of random shape
        self.shape1 = np.random.randint(20, high=512, size=(2,))
        self.shape2 = np.random.randint(20, high=512, size=(2,))
        self.array1 = np.random.rand(*self.shape1).astype(np.float32)
        self.array2 = np.random.rand(*self.shape2).astype(np.float32)
        self.d_array1 = parray.to_device(self.queue, self.array1)
        self.d_array2 = parray.to_device(self.queue, self.array2)
        # Generate random offsets
        offset1_y = np.random.randint(2, high=min(self.shape1[0], self.shape2[0]) - 10)
        offset1_x = np.random.randint(2, high=min(self.shape1[1], self.shape2[1]) - 10)
        offset2_y = np.random.randint(2, high=min(self.shape1[0], self.shape2[0]) - 10)
        offset2_x = np.random.randint(2, high=min(self.shape1[1], self.shape2[1]) - 10)
        self.offset1 = (offset1_y, offset1_x)
        self.offset2 = (offset2_y, offset2_x)
        # Compute the size of the rectangle to transfer
        size_y = np.random.randint(2, high=min(self.shape1[0], self.shape2[0]) - max(offset1_y, offset2_y) + 1)
        size_x = np.random.randint(2, high=min(self.shape1[1], self.shape2[1]) - max(offset1_x, offset2_x) + 1)
        self.transfer_shape = (size_y, size_x)

    def tearDown(self):
        self.array1 = None
        self.array2 = None
        self.d_array1.data.release()
        self.d_array2.data.release()
        self.d_array1 = None
        self.d_array2 = None
        self.ctx = None
        self.queue = None

    def compare(self, result, reference):
        errmax = np.max(np.abs(result - reference))
        logger.info("Max error = %e" % (errmax))
        self.assertTrue(errmax == 0, str("Max error is too high"))#. PRNG state was %s" % str(self.prng_state)))

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_cpy2d(self):
        """
        Test rectangular transfer of self.d_array1 to self.d_array2
        """
        # Reference
        o1 = self.offset1
        o2 = self.offset2
        T = self.transfer_shape
        logger.info("""Testing D->D rectangular copy with (N1_y, N1_x) = %s,
                    (N2_y, N2_x) = %s:
                    array2[%d:%d, %d:%d] = array1[%d:%d, %d:%d]""" %
                        (
                            str(self.shape1), str(self.shape2),
                            o2[0], o2[0] + T[0],
                            o2[1], o2[1] + T[1],
                            o1[0], o1[0] + T[0],
                            o1[1], o1[1] + T[1]
                        )
                    )
        self.array2[o2[0]:o2[0] + T[0], o2[1]:o2[1] + T[1]] = self.array1[o1[0]:o1[0] + T[0], o1[1]:o1[1] + T[1]]
        kernel_args = (
            self.d_array2.data,
            self.d_array1.data,
            np.int32(self.shape2[1]),
            np.int32(self.shape1[1]),
            np.int32(self.offset2[::-1]),
            np.int32(self.offset1[::-1]),
            np.int32(self.transfer_shape[::-1])
        )
        wg = None
        ndrange = self.transfer_shape[::-1]
        self.program.cpy2d(self.queue, ndrange, wg, *kernel_args)
        res = self.d_array2.get()
        self.compare(res, self.array2)


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestCpy2d("test_cpy2d"))
    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
