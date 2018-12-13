# -*- coding: utf-8 -*-
#
#    Project: SILX
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""A module for performing basic statistical analysis (min, max, mean, std) on
large data where numpy is not very efficient.  

"""
from __future__ import absolute_import, print_function, with_statement, division


__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "13/12/2018"
__copyright__ = "2012-2017, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import logging
import numpy
from collections import OrderedDict, namedtuple
from math import sqrt

from .common import pyopencl
from .processing import EventDescription, OpenclProcessing, BufferDescription
from .utils import concatenate_cl_kernel

if pyopencl:
    mf = pyopencl.mem_flags
    from pyopencl.reduction import ReductionKernel
    try:
        from pyopencl import cltypes
    except:
        v = pyopencl.array.vec()
        float8 = v.float8
        old_style = numpy.dtype([("s0", numpy.float32), ("s1", numpy.float32),
                                 ("s2", numpy.float32), ("s3", numpy.float32),
                                 ("s4", numpy.float32), ("s5", numpy.float32),
                                 ("s6", numpy.float32), ("s7", numpy.float32)])
    else:
        float8 = cltypes.float8
        old_style = False

else:
    raise ImportError("pyopencl is not installed")
logger = logging.getLogger(__name__)

StatResults = namedtuple("StatResults", ["min", "max", "cnt", "sum", "mean", "var", "std"])
zero8 = "(float8)(FLT_MAX, -FLT_MAX, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)"
#                    min      max    cnt  cnt_e  sum   sum_e  var  var_e


class Statistics(OpenclProcessing):
    """A class for doing statistical analysis using OpenCL"""
    buffers = [
               BufferDescription("raw", 1, numpy.float32, mf.READ_ONLY),
               BufferDescription("converted", 1, numpy.float32, mf.READ_WRITE),
              ]
    kernel_files = ["preprocess.cl"]
    mapping = {numpy.int8: "s8_to_float",
               numpy.uint8: "u8_to_float",
               numpy.int16: "s16_to_float",
               numpy.uint16: "u16_to_float",
               numpy.uint32: "u32_to_float",
               numpy.int32: "s32_to_float"}

    def __init__(self, size=None, dtype=None, template=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False
                 ):
        """Constructor of the OpenCL statistical analysis

        :param shape: shape of the data to treat
        :param dtype: input datatype
        :param template: data template to extract size & dtype
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)
        self.size = size
        self.dtype = dtype
        if template is not None:
            self.size = template.size
            self.dtype = template.dtype

        self.buffers = [BufferDescription(i.name, i.size * self.size, i.dtype, i.flags)
                        for i in self.__class__.buffers]

        self.allocate_buffers(use_array=True)
        self.compile_kernels()
        self.set_kernel_arguments()

    def set_kernel_arguments(self):
        """Parametrize all kernel arguments
        """
        for val in self.mapping.values():
            self.cl_kernel_args[val] = OrderedDict(((i, self.cl_mem[i]) for i in ("raw", "converted")))

    def compile_kernels(self):
        "Compile the kernel"
        OpenclProcessing.compile_kernels(self, self.kernel_files, "-D NIMAGE=%i" % self.size)
        src = concatenate_cl_kernel(("kahan.cl", "statistics.cl"))
        self.reduction = ReductionKernel(self.ctx,
                                         dtype_out=float8,
                                         neutral=zero8,
                                         map_expr="map_statistics(data, i)",
                                         reduce_expr="reduce_statistics(a,b)",
                                         arguments="__global float *data",
                                         preamble=src)

    def send_buffer(self, data, dest):
        """Send a numpy array to the device, including the cast on the device if possible

        :param data: numpy array with data
        :param dest: name of the buffer as registered in the class
        """

        dest_type = numpy.dtype([i.dtype for i in self.buffers if i.name == dest][0])
        events = []
        if (data.dtype == dest_type) or (data.dtype.itemsize > dest_type.itemsize):
            copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest].data, numpy.ascontiguousarray(data, dest_type))
            events.append(EventDescription("copy H->D %s" % dest, copy_image))
        else:
            copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["raw"].data, numpy.ascontiguousarray(data))
            kernel = getattr(self.program, self.mapping[data.dtype.type])
            cast_to_float = kernel(self.queue, (self.size,), None, self.cl_mem["raw"].data, self.cl_mem[dest].data)
            events += [EventDescription("copy H->D %s" % dest, copy_image), EventDescription("cast to float", cast_to_float)]
        if self.profile:
            self.events += events
        return events

    def process(self, data):
        """Actually calculate the statics on the data

        :param data: numpy array with the image
        :return: Statistics named tuple
        """
        if data.ndim != 1:
            data = data.ravel()
        size = data.size
        assert size <= self.size, "size is OK"
        events = []
        with self.sem:
            self.send_buffer(data, "converted")
            res_d, evt = self.reduction(self.cl_mem["converted"][:self.size], queue=self.queue, return_event=True)
            events.append(EventDescription("statistical reduction", evt))
            if self.profile:
                self.events += events
            res_h = res_d.get()
        min = 1.0 * res_h["s0"]
        max = 1.0 * res_h["s1"]
        count = 1.0 * res_h["s2"] + res_h["s3"]
        sum = 1.0 * res_h["s4"] + res_h["s5"]
        M2 = 1.0 * res_h["s6"] + res_h["s7"]
        var = M2 / (count - 1.0)
        res = StatResults(min,
                          max,
                          count,
                          sum,
                          sum / count,
                          var,
                          sqrt(var))
        return res

    __call__ = process

