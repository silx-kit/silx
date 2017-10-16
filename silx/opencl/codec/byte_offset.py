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
__date__ = "16/10/2017"
__status__ = "production"

import os
import numpy
from ..common import ocl, pyopencl
from ..processing import BufferDescription, EventDescription, OpenclProcessing
import logging
logger = logging.getLogger(__name__)
if pyopencl:
    import pyopencl.algorithm
else:
    logger.warning("No PyOpenCL, no byte-offset, please see fabio")


class ByteOffset(OpenclProcessing):
    "Perform the byte offset decompression on the GPU"
    def __init__(self, raw_size, dec_size, ctx=None, devicetype="all",
                 platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """Constructor of the Byte Offset decompressor 
        
        :param raw_size: size of the raw stream, can be (slightly) larger than the array 
        :param dec_size: size of the output array (mandatory)
        """

        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)
        self.raw_size = int(raw_size)
        self.dec_size = int(dec_size)

        buffers = [
                    BufferDescription("raw", self.raw_size, numpy.int8, None),
                    BufferDescription("mask", self.raw_size + 8, numpy.int32, None),
                    BufferDescription("exceptions", self.raw_size, numpy.int32, None),
                    BufferDescription("counter", 1, numpy.int32, None),
                    BufferDescription("data_non_compact", self.raw_size, numpy.int32, None),
                    BufferDescription("exceptions", self.raw_size, numpy.int32, None),
                    BufferDescription("data_float", self.dec_size, numpy.float32, None),
                    BufferDescription("data_int", self.dec_size, numpy.int32, None),
                   ]
        self.allocate_buffers(buffers, use_array=True)
        self.compile_kernels([os.path.join("codec", "byte_offset")])

    def dec(self, raw, as_float=False, out=None):
        """This function actually performs the decompression by calling the kernels
        """
        # TODO: handle the case where size(raw) > allocated value
        events = []
        with self.sem:
            if self.raw_size > len(raw):
                evt = self.kernels.fill_char_mem(self.queue, (self.raw_size,), None,
                                                 self.cl_mem["raw"].data,
                                                 numpy.int32(self.raw_size),
                                                 numpy.int8(0),
                                                 numpy.int32(len(raw)))
                events.append(EventDescription("memset raw buffer", evt))
            evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["raw"].data,
                                        raw,
                                        is_blocking=False)
            events.append(EventDescription("copy raw H -> D", evt))
            evt = self.kernels.fill_int_mem(self.queue, (self.raw_size,), None,
                                            self.cl_mem["mask"].data,
                                            numpy.int32(self.cl_mem["mask"].size),
                                            numpy.int32(0))
            events.append(EventDescription("memset mask buffer", evt))
            evt = self.kernels.fill_int_mem(self.queue, (self.raw_size,), None,
                                            self.cl_mem["data_non_compact"].data,
                                            numpy.int32(self.raw_size),
                                            numpy.int32(0))
            events.append(EventDescription("memset data_non_compact", evt))
            evt = self.kernels.fill_int_mem(self.queue, (1,), None,
                                            self.cl_mem["counter"].data,
                                            numpy.int32(1),
                                            numpy.int32(0))
            events.append(EventDescription("memset counter", evt))

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
                                                self.cl_mem["data_non_compact"].data
                                                )
            events.append(EventDescription("treat_exceptions", evt))
            evt = self.kernels.treat_simple(self.queue, (self.raw_size,), None,
                                            self.cl_mem["raw"].data,
                                            numpy.int32(self.raw_size),
                                            self.cl_mem["mask"].data,
                                            self.cl_mem["data_non_compact"].data)
            events.append(EventDescription("treat_simple", evt))

            evt, cumsummed_d = pyopencl.array.cumsum(self.cl_mem["data_non_compact"],
                                                     numpy.int32, self.queue,
                                                     return_event=True,
                                                     wait_for=[evt])  # synchro here
            events.append(EventDescription("cumsum", evt))

            indexes_d, count_d, evt = pyopencl.algorithm.copy_if(self.cl_mem["mask"],
                                                                 predicate="ary[i]>=0",
                                                                 queue=self.queue,
                                                                 )
            events.append(EventDescription("copy_if", evt))
            size_out = count_d.get()  # synchro here

            if out is not None:
                if out.dtype == numpy.float32:
                    copy_results = self.kernels.copy_result_float
                else:
                    copy_results = self.kernels.copy_result_int
            else:
                if as_float:
                    out = self.cl_mem["data_float"]
                    copy_results = self.kernels.copy_result_float
                else:
                    out = self.cl_mem["data_int"]
                    copy_results = self.kernels.copy_result_int
            if out.size != size_out:
                print("out size: %s expected: %s" % (out.size, size_out))
            evt = copy_results(self.queue, (size_out,), None,
                               cumsummed_d.data,
                               indexes_d.data,
                               numpy.int32(size_out),
                               out.data
                               )
            events.append(EventDescription("copy_results", evt))
            if self.profile:
                self.events += events
        return out

    __call__ = dec
