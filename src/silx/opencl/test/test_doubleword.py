#!/usr/bin/env python
#
#    Project: The silx project
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "31/05/2021"

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

EPS32 = numpy.finfo("float32").eps
EPS64 = numpy.finfo("float64").eps


@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestDoubleWord(unittest.TestCase):
    """
    Test the kernels for compensated math in OpenCL
    """

    @classmethod
    def setUpClass(cls):
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
        cls.a = 1.0 + numpy.random.random(size)
        cls.b = 1.0 + numpy.random.random(size)
        cls.ah = cls.a.astype(numpy.float32)
        cls.bh = cls.b.astype(numpy.float32)
        cls.al = (cls.a - cls.ah).astype(numpy.float32)
        cls.bl = (cls.b - cls.bh).astype(numpy.float32)
        cls.doubleword = read_cl_file("doubleword.cl")

    @classmethod
    def tearDownClass(cls):
        cls.queue = None
        cls.ctx = None
        cls.a = cls.al = cls.ah = None
        cls.b = cls.bl = cls.bh = None
        cls.doubleword = None

    def test_fast_sum2(self):
        test_kernel = ElementwiseKernel(self.ctx,
                      "float *a, float *b, float *res_h, float *res_l",
                      "float2 tmp = fast_fp_plus_fp(a[i], b[i]); res_h[i] = tmp.s0; res_l[i] = tmp.s1",
                      preamble=self.doubleword)
        a_g = pyopencl.array.to_device(self.queue, self.ah)
        b_g = pyopencl.array.to_device(self.queue, self.bl)
        res_l = pyopencl.array.empty_like(a_g)
        res_h = pyopencl.array.empty_like(a_g)
        test_kernel(a_g, b_g, res_h, res_l)
        self.assertEqual(abs(self.ah + self.bl - res_h.get()).max(), 0, "Major matches")
        self.assertGreater(abs(self.ah.astype(numpy.float64) + self.bl - res_h.get()).max(), 0, "Exact mismatches")
        self.assertEqual(abs(self.ah.astype(numpy.float64) + self.bl - (res_h.get().astype(numpy.float64) + res_l.get())).max(), 0, "Exact matches")

    def test_sum2(self):
        test_kernel = ElementwiseKernel(self.ctx,
                    "float *a, float *b, float *res_h, float *res_l",
                    "float2 tmp = fp_plus_fp(a[i],b[i]); res_h[i]=tmp.s0; res_l[i]=tmp.s1;",
                    preamble=self.doubleword)
        a_g = pyopencl.array.to_device(self.queue, self.ah)
        b_g = pyopencl.array.to_device(self.queue, self.bh)
        res_l = pyopencl.array.empty_like(a_g)
        res_h = pyopencl.array.empty_like(a_g)
        test_kernel(a_g, b_g, res_h, res_l)
        self.assertEqual(abs(self.ah + self.bh - res_h.get()).max(), 0, "Major matches")
        self.assertGreater(abs(self.ah.astype(numpy.float64) + self.bh - res_h.get()).max(), 0, "Exact mismatches")
        self.assertEqual(abs(self.ah.astype(numpy.float64) + self.bh - (res_h.get().astype(numpy.float64) + res_l.get())).max(), 0, "Exact matches")

    def test_prod2(self):
        test_kernel = ElementwiseKernel(self.ctx,
                    "float *a, float *b, float *res_h, float *res_l",
                    "float2 tmp = fp_times_fp(a[i],b[i]); res_h[i]=tmp.s0; res_l[i]=tmp.s1;",
                    preamble=self.doubleword)
        a_g = pyopencl.array.to_device(self.queue, self.ah)
        b_g = pyopencl.array.to_device(self.queue, self.bh)
        res_l = pyopencl.array.empty_like(a_g)
        res_h = pyopencl.array.empty_like(a_g)
        test_kernel(a_g, b_g, res_h, res_l)
        res_m = res_h.get()
        res = res_h.get().astype(numpy.float64) + res_l.get()
        self.assertEqual(abs(self.ah * self.bh - res_m).max(), 0, "Major matches")
        self.assertGreater(abs(self.ah.astype(numpy.float64) * self.bh - res_m).max(), 0, "Exact mismatches")
        self.assertEqual(abs(self.ah.astype(numpy.float64) * self.bh - res).max(), 0, "Exact matches")

    def test_dw_plus_fp(self):
        test_kernel = ElementwiseKernel(self.ctx,
                    "float *ah, float *al, float *b, float *res_h, float *res_l",
                    "float2 tmp = dw_plus_fp((float2)(ah[i], al[i]),b[i]); res_h[i]=tmp.s0; res_l[i]=tmp.s1;",
                    preamble=self.doubleword)
        ah_g = pyopencl.array.to_device(self.queue, self.ah)
        al_g = pyopencl.array.to_device(self.queue, self.al)
        b_g = pyopencl.array.to_device(self.queue, self.bh)
        res_l = pyopencl.array.empty_like(b_g)
        res_h = pyopencl.array.empty_like(b_g)
        test_kernel(ah_g, al_g, b_g, res_h, res_l)
        res_m = res_h.get()
        res = res_h.get().astype(numpy.float64) + res_l.get()
        self.assertLess(abs(self.a + self.bh - res_m).max(), EPS32, "Major matches")
        self.assertGreater(abs(self.a + self.bh - res_m).max(), EPS64, "Exact mismatches")
        self.assertLess(abs(self.ah.astype(numpy.float64) + self.al + self.bh - res).max(), 2 * EPS32 ** 2, "Exact matches")

    def test_dw_plus_dw(self):
        test_kernel = ElementwiseKernel(self.ctx,
                    "float *ah, float *al, float *bh, float *bl, float *res_h, float *res_l",
                    "float2 tmp = dw_plus_dw((float2)(ah[i], al[i]),(float2)(bh[i], bl[i])); res_h[i]=tmp.s0; res_l[i]=tmp.s1;",
                    preamble=self.doubleword)
        ah_g = pyopencl.array.to_device(self.queue, self.ah)
        al_g = pyopencl.array.to_device(self.queue, self.al)
        bh_g = pyopencl.array.to_device(self.queue, self.bh)
        bl_g = pyopencl.array.to_device(self.queue, self.bl)
        res_l = pyopencl.array.empty_like(bh_g)
        res_h = pyopencl.array.empty_like(bh_g)
        test_kernel(ah_g, al_g, bh_g, bl_g, res_h, res_l)
        res_m = res_h.get()
        res = res_h.get().astype(numpy.float64) + res_l.get()
        self.assertLess(abs(self.a + self.b - res_m).max(), EPS32, "Major matches")
        self.assertGreater(abs(self.a + self.b - res_m).max(), EPS64, "Exact mismatches")
        self.assertLess(abs(self.a + self.b - res).max(), 3 * EPS32 ** 2, "Exact matches")

    def test_dw_times_fp(self):
        test_kernel = ElementwiseKernel(self.ctx,
                    "float *ah, float *al, float *b, float *res_h, float *res_l",
                    "float2 tmp = dw_times_fp((float2)(ah[i], al[i]),b[i]); res_h[i]=tmp.s0; res_l[i]=tmp.s1;",
                    preamble=self.doubleword)
        ah_g = pyopencl.array.to_device(self.queue, self.ah)
        al_g = pyopencl.array.to_device(self.queue, self.al)
        b_g = pyopencl.array.to_device(self.queue, self.bh)
        res_l = pyopencl.array.empty_like(b_g)
        res_h = pyopencl.array.empty_like(b_g)
        test_kernel(ah_g, al_g, b_g, res_h, res_l)
        res_m = res_h.get()
        res = res_h.get().astype(numpy.float64) + res_l.get()
        self.assertLess(abs(self.a * self.bh - res_m).max(), EPS32, "Major matches")
        self.assertGreater(abs(self.a * self.bh - res_m).max(), EPS64, "Exact mismatches")
        self.assertLess(abs(self.a * self.bh - res).max(), 2 * EPS32 ** 2, "Exact matches")

    def test_dw_times_dw(self):
        test_kernel = ElementwiseKernel(self.ctx,
                    "float *ah, float *al, float *bh, float *bl, float *res_h, float *res_l",
                    "float2 tmp = dw_times_dw((float2)(ah[i], al[i]),(float2)(bh[i], bl[i])); res_h[i]=tmp.s0; res_l[i]=tmp.s1;",
                    preamble=self.doubleword)
        ah_g = pyopencl.array.to_device(self.queue, self.ah)
        al_g = pyopencl.array.to_device(self.queue, self.al)
        bh_g = pyopencl.array.to_device(self.queue, self.bh)
        bl_g = pyopencl.array.to_device(self.queue, self.bl)
        res_l = pyopencl.array.empty_like(bh_g)
        res_h = pyopencl.array.empty_like(bh_g)
        test_kernel(ah_g, al_g, bh_g, bl_g, res_h, res_l)
        res_m = res_h.get()
        res = res_h.get().astype(numpy.float64) + res_l.get()
        self.assertLess(abs(self.a * self.b - res_m).max(), EPS32, "Major matches")
        self.assertGreater(abs(self.a * self.b - res_m).max(), EPS64, "Exact mismatches")
        self.assertLess(abs(self.a * self.b - res).max(), 5 * EPS32 ** 2, "Exact matches")

    def test_dw_div_fp(self):
        test_kernel = ElementwiseKernel(self.ctx,
                    "float *ah, float *al, float *b, float *res_h, float *res_l",
                    "float2 tmp = dw_div_fp((float2)(ah[i], al[i]),b[i]); res_h[i]=tmp.s0; res_l[i]=tmp.s1;",
                    preamble=self.doubleword)
        ah_g = pyopencl.array.to_device(self.queue, self.ah)
        al_g = pyopencl.array.to_device(self.queue, self.al)
        b_g = pyopencl.array.to_device(self.queue, self.bh)
        res_l = pyopencl.array.empty_like(b_g)
        res_h = pyopencl.array.empty_like(b_g)
        test_kernel(ah_g, al_g, b_g, res_h, res_l)
        res_m = res_h.get()
        res = res_h.get().astype(numpy.float64) + res_l.get()
        self.assertLess(abs(self.a / self.bh - res_m).max(), EPS32, "Major matches")
        self.assertGreater(abs(self.a / self.bh - res_m).max(), EPS64, "Exact mismatches")
        self.assertLess(abs(self.a / self.bh - res).max(), 3 * EPS32 ** 2, "Exact matches")

    def test_dw_div_dw(self):
        test_kernel = ElementwiseKernel(self.ctx,
                    "float *ah, float *al, float *bh, float *bl, float *res_h, float *res_l",
                    "float2 tmp = dw_div_dw((float2)(ah[i], al[i]),(float2)(bh[i], bl[i])); res_h[i]=tmp.s0; res_l[i]=tmp.s1;",
                    preamble=self.doubleword)
        ah_g = pyopencl.array.to_device(self.queue, self.ah)
        al_g = pyopencl.array.to_device(self.queue, self.al)
        bh_g = pyopencl.array.to_device(self.queue, self.bh)
        bl_g = pyopencl.array.to_device(self.queue, self.bl)
        res_l = pyopencl.array.empty_like(bh_g)
        res_h = pyopencl.array.empty_like(bh_g)
        test_kernel(ah_g, al_g, bh_g, bl_g, res_h, res_l)
        res_m = res_h.get()
        res = res_h.get().astype(numpy.float64) + res_l.get()
        self.assertLess(abs(self.a / self.b - res_m).max(), EPS32, "Major matches")
        self.assertGreater(abs(self.a / self.b - res_m).max(), EPS64, "Exact mismatches")
        self.assertLess(abs(self.a / self.b - res).max(), 6 * EPS32 ** 2, "Exact matches")
