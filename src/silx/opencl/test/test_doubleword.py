#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2015-2019 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"test suite for OpenCL code"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/05/2021"


import unittest
import numpy
import logging
import platform

logger = logging.getLogger(__name__)
try:
    import pyopencl
except ImportError as error:
    logger.warning("OpenCL module (pyopencl) is not present, skip tests. %s.", error)
    pyopencl = None

from .. import ocl
if ocl is not None:
    from ..utils import read_cl_file
    from .. import pyopencl
    import pyopencl.array
    from pyopencl.elementwise import ElementwiseKernel
from ...test.utils import test_options


class TestDoubleWord(unittest.TestCase):
    """
    Test the kernels for compensated math in OpenCL
    """

    @classmethod
    def setUpClass(cls):
        if not test_options.WITH_OPENCL_TEST:
            raise unittest.SkipTest("User request to skip OpenCL tests")
        if pyopencl is None or ocl is None:
            raise unittest.SkipTest("OpenCL module (pyopencl) is not present or no device available")

        cls.ctx = ocl.create_context(devicetype="GPU")
        cls.queue = pyopencl.CommandQueue(cls.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)

        # this is running 32 bits OpenCL woth POCL
        if (platform.machine() in ("i386", "i686", "x86_64") and (tuple.__itemsize__ == 4) and
                cls.ctx.devices[0].platform.name == 'Portable Computing Language'):
            cls.args = "-DX87_VOLATILE=volatile"
        else:
            cls.args = ""
        size = 1024
        cls.a = numpy.random.random(size)
        cls.b = numpy.random.random(size)
        cls.ah = numpy.round(cls.a, 7)
        cls.bh = numpy.round(cls.b, 7)
        cls.al = cls.a - cls.ah
        cls.bl = cls.b - cls.bh 

    @classmethod
    def tearDownClass(cls):
        cls.queue = None
        cls.ctx = None
        cls.a = cls.al = cls.ah = None  
        cls.b = cls.bl = cls.bh = None


    # def test_fast_sum2(self):
    #     test_fast_sum2 = ElementwiseKernel(self.ctx,
    #                                  "float *a, float *b, float *res_h, float *res_l",
    #                                  "float2 tmp = fast_sum2(a[i], b[i]); res_h[i] = tmp.s0; res_l[i] = tmp.s1",
    #                                  "test_fast_sum2",
    #                                  preamble=read_cl_file("doubleword.cl"))
    #     a_g = pyopencl.array.to_device(self.queue, self.ah)
    #     b_g = pyopencl.array.to_device(self.queue, self.bh)
    #     res_l = pyopencl.array.empty_like(a_g)
    #     res_h = pyopencl.array.empty_like(a_g)
    #     test_fast_sum2(a_g, b_g, res_h, res_l)
    #     self.assertEqual(abs(self.ah + self.bh - res_h.get()).max(), 0, "Major matches")
        
    def test_sum2(self):
        test_sum2 = ElementwiseKernel(self.ctx,
                                     "float *a, float *b, float *res_h, float *res_l",
                                     "res_l[i] = ((float2)(a[i]+b[i], a[i]-b[i])).s0",
                                     "test_sum2",)
                                     #preamble=read_cl_file("doubleword.cl"))
        a_g = pyopencl.array.to_device(self.queue, self.ah)
        b_g = pyopencl.array.to_device(self.queue, self.bh)
        res_l = pyopencl.array.empty_like(a_g)
        res_h = pyopencl.array.empty_like(a_g)
        test_sum2(a_g, b_g, res_h, res_l)
        print(self.ah + self.bh)
        print(res_h.get())
        print(res_l.get())
        self.assertEqual(abs(self.ah + self.bh - res_h.get()).max(), 0, "Major matches")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestDoubleWord))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
