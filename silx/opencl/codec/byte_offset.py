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
Contains classes CBF byte offset decompression
"""

from __future__ import division, print_function, with_statement

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "13/10/2017"
__status__ = "production"

import os
import gc
import numpy
from ..common import ocl, pyopencl, kernel_workgroup_size
from ..processing import BufferDescription, EventDescription, OpenclProcessing

import logging
logger = logging.getLogger(__name__)
if not pyopencl:
    logger.warning("No PyOpenCL, no byte-offset, please see fabio")


class ByteOffsetDecompressor(OpenclProcessing):
    "Perform the byte offset decompression on the GPU"
    def __init__(self, raw_size, dec_size=None, ctx=None, devicetype="all",
                 platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """Constructor of the Byte Offset decompressor 
        
        """

        OpenclProcessing.__init__(ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)
        self.raw_size = int(raw_size)
        if not dec_size:
            dec_size = self.raw_size
        self.dec_size = int(dec_size)

        buffers = [
                    BufferDescription("raw", self.raw_size, numpy.int8, None),
                    BufferDescription("mask", self.raw_size, numpy.int32, None),
                    BufferDescription("exceptions", self.raw_size, numpy.int32, None),
                    BufferDescription("counter", 1, numpy.int32, None),
                    BufferDescription("data_simple", self.raw_size, numpy.int32, None),
                    BufferDescription("data_except", self.raw_size, numpy.int32, None),
                    BufferDescription("delta", self.raw_size, numpy.int32, None),
                    BufferDescription("exceptions", self.raw_size, numpy.int32, None),
                    BufferDescription("data_float", self.dec_size, numpy.float32, None),
                   ]
        self.allocate_buffers(buffers, as_array=True)
        self.compile_kernels(os.path.join("codec", "byte_offset"))

    def dec(self, raw, as_float=True, out=None):
        """This function actually performs the decompression by calling the kernels
        """
        events = []
        with self.sem:
            evt = self.kernels.fill_char_mem(self.queue, (self.raw_size,), None,
                                             self.cl_mem["raw"].data,
                                             numpy.int32(self.raw_size),
                                             numpy.int32(0), numpy.int32(len(raw)))
            events.append(EventDescription("memset raw buffer", evt))
            evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["raw"].data,
                                        raw,
                                        is_blocking=False)
            events.append(EventDescription("copy raw H -> D", evt))
            evt = self.kernels.fill_int_mem(self.queue, (self.raw_size,), None,
                                            self.cl_mem["mask"].data,
                                            numpy.int32(self.raw_size),
                                            numpy.int32(0))
            events.append(EventDescription("memset mask buffer", evt))
            evt = self.kernels.fill_int_mem(self.queue, (self.raw_size,), None,
                                            self.cl_mem["data_simple"].data,
                                            numpy.int32(self.raw_size),
                                            numpy.int32(0))
            events.append(EventDescription("memset data_simple", evt))
            evt = self.kernels.fill_int_mem(self.queue, (self.raw_size,), None,
                                            self.cl_mem["data_except"].data,
                                            numpy.int32(self.raw_size),
                                            numpy.int32(0))
            events.append(EventDescription("memset data_except", evt))
            evt = self.kernels.fill_int_mem(self.queue, (self.raw_size,), None,
                                            self.cl_mem["delta"].data,
                                            numpy.int32(self.raw_size),
                                            numpy.int32(0))
            events.append(EventDescription("memset delta", evt))
            evt = self.kernels.fill_int_mem(self.queue, (1,), None,
                                            self.cl_mem["counter"].data,
                                            numpy.int32(1),
                                            numpy.int32(0))
            events.append(EventDescription("memset counter", evt))
            evt = self.kernels.fill_int_mem(self.queue, (self.raw_size,), None,
                                            self.cl_mem["exceptions"].data,
                                            numpy.int32(self.raw_size),
                                            numpy.int32(0))
            events.append(EventDescription("memset exceptions", evt))

            evt = self.kernels.mark_exceptions(self.queue, (self.raw_size,), None,
                                               self.cl_mem["raw"].data,
                                               numpy.int32(self.raw_size),
                                               self.cl_mem["mask"].data,
                                               self.cl_mem["counter"].data,
                                               self.cl_mem["exceptions"].data)
            events.append(EventDescription("mark exceptions", evt))
            nb_exceptions = numpy.empty(1, dtype=numpy.int32)
            evt = pyopencl.enqueue_copy(self.queue, nb_exceptions, self.cl_mem["counter"].data,
                                        is_blocking=False)
            events.append(EventDescription("copy counter D -> H", evt))
            evt.wait()
            nbexc = nb_exceptions[0]
            evt = self.kernels.treat_exceptions(self.queue, (nbexc,), (1,),
                                                self.cl_mem["raw"].data,
                                                numpy.int32(self.raw_size),
                                                self.cl_mem["mask"].data,
                                                self.cl_mem["exceptions"].data,
                                                self.cl_mem["data_except"].data,
                                                self.cl_mem["delta"].data
                                                )
            events.append(EventDescription("treat exceptions", evt))
            evt = self.kernels.calc_size(self.queue, (self.raw_size,), None,
                                         numpy.int32(self.raw_size),
                                         self.cl_mem["mask"].data,
                                         self.cl_mem["delta"].data)
            events.append(EventDescription("calc size", evt))
            ary1_d, count_d, e = pyopencl.algorithm.copy_if(delta_d, predicate="ary[i]>0", queue=queue)
            events.append(e)
            size_out = count_d.get()
            e, ary2_d = pyopencl.array.cumsum(ary1_d, numpy.int32, queue, return_event=True)
            events.append(e)

            e = copy_values(queue, (size,) , None,
                            raw_d.data, numpy.int32(size), numpy.int32(count.get()),
                            data1_d.data, data2_d.data,
                            ary2_d.data, delta_d.data)
            events.append(e)
            e, ary3_d = pyopencl.array.cumsum(data1_d, numpy.int32, queue, return_event=True)
            events.append(e)
        return ary3_d
