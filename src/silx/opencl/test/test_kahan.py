#!/usr/bin/env python
#
#    Project: OpenCL numerical library
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2015-2021 European Synchrotron Radiation Facility, Grenoble, France
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


class TestKahan(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        cls.queue = None
        cls.ctx = None

    @staticmethod
    def dummy_sum(ary, dtype=None):
        "perform the actual sum in a dummy way "
        if dtype is None:
            dtype = ary.dtype.type
        sum_ = dtype(0)
        for i in ary:
            sum_ += i
        return sum_

    def test_kahan(self):
        # simple test
        N = 26
        data = (1 << (N - 1 - numpy.arange(N))).astype(numpy.float32)

        ref64 = numpy.sum(data, dtype=numpy.float64)
        ref32 = self.dummy_sum(data)
        if (ref64 == ref32):
            logger.warning("Kahan: invalid tests as float32 provides the same result as float64")
        # Dummy kernel to evaluate
        src = """
        kernel void summation(global float* data,
                                           int size,
                                    global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            for (int i=0; i<size; i++)
            {
                acc = kahan_sum(acc, data[i]);
            }
            result[0] = acc.s0;
            result[1] = acc.s1;
        }
        """
        prg = pyopencl.Program(self.ctx, read_cl_file("kahan.cl") + src).build(self.args)
        ones_d = pyopencl.array.to_device(self.queue, data)
        res_d = pyopencl.array.empty(self.queue, 2, numpy.float32)
        res_d.fill(0)
        evt = prg.summation(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype=numpy.float64)
        self.assertEqual(ref64, res, "test_kahan")

    def test_dot16(self):
        # simple test
        N = 16
        data = (1 << (N - 1 - numpy.arange(N))).astype(numpy.float32)

        ref64 = numpy.dot(data.astype(numpy.float64), data.astype(numpy.float64))
        ref32 = numpy.dot(data, data)
        if (ref64 == ref32):
            logger.warning("dot16: invalid tests as float32 provides the same result as float64")
        # Dummy kernel to evaluate
        src = """
        kernel void test_dot16(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float16 data16 = (float16) (data[0],data[1],data[2],data[3],data[4],
                                        data[5],data[6],data[7],data[8],data[9],
                         data[10],data[11],data[12],data[13],data[14],data[15]);
            acc = comp_dot16(data16, data16);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        kernel void test_dot8(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float8 data0 = (float8) (data[0],data[2],data[4],data[6],data[8],data[10],data[12],data[14]);
            float8 data1 = (float8) (data[1],data[3],data[5],data[7],data[9],data[11],data[13],data[15]);
            acc = comp_dot8(data0, data1);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        kernel void test_dot4(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float4 data0 = (float4) (data[0],data[4],data[8],data[12]);
            float4 data1 = (float4) (data[3],data[7],data[11],data[15]);
            acc = comp_dot4(data0, data1);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        kernel void test_dot3(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float3 data0 = (float3) (data[0],data[4],data[12]);
            float3 data1 = (float3) (data[3],data[11],data[15]);
            acc = comp_dot3(data0, data1);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        kernel void test_dot2(global float* data,
                                           int size,
                               global float* result)
        {
            float2 acc = (float2)(0.0f, 0.0f);
            float2 data0 = (float2) (data[0],data[14]);
            float2 data1 = (float2) (data[1],data[15]);
            acc = comp_dot2(data0, data1);
            result[0] = acc.s0;
            result[1] = acc.s1;
        }

        """

        prg = pyopencl.Program(self.ctx, read_cl_file("kahan.cl") + src).build(self.args)
        ones_d = pyopencl.array.to_device(self.queue, data)
        res_d = pyopencl.array.empty(self.queue, 2, numpy.float32)
        res_d.fill(0)
        evt = prg.test_dot16(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot16")

        res_d.fill(0)
        data0 = data[0::2]
        data1 = data[1::2]
        ref64 = numpy.dot(data0.astype(numpy.float64), data1.astype(numpy.float64))
        ref32 = numpy.dot(data0, data1)
        if (ref64 == ref32):
            logger.warning("dot8: invalid tests as float32 provides the same result as float64")
        evt = prg.test_dot8(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot8")

        res_d.fill(0)
        data0 = data[0::4]
        data1 = data[3::4]
        ref64 = numpy.dot(data0.astype(numpy.float64), data1.astype(numpy.float64))
        ref32 = numpy.dot(data0, data1)
        if (ref64 == ref32):
            logger.warning("dot4: invalid tests as float32 provides the same result as float64")
        evt = prg.test_dot4(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot4")

        res_d.fill(0)
        data0 = numpy.array([data[0], data[4], data[12]])
        data1 = numpy.array([data[3], data[11], data[15]])
        ref64 = numpy.dot(data0.astype(numpy.float64), data1.astype(numpy.float64))
        ref32 = numpy.dot(data0, data1)
        if (ref64 == ref32):
            logger.warning("dot3: invalid tests as float32 provides the same result as float64")
        evt = prg.test_dot3(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot3")

        res_d.fill(0)
        data0 = numpy.array([data[0], data[14]])
        data1 = numpy.array([data[1], data[15]])
        ref64 = numpy.dot(data0.astype(numpy.float64), data1.astype(numpy.float64))
        ref32 = numpy.dot(data0, data1)
        if (ref64 == ref32):
            logger.warning("dot2: invalid tests as float32 provides the same result as float64")
        evt = prg.test_dot2(self.queue, (1,), (1,), ones_d.data, numpy.int32(N), res_d.data)
        evt.wait()
        res = res_d.get().sum(dtype="float64")
        self.assertEqual(ref64, res, "test_dot2")
