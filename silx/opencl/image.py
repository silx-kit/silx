# -*- coding: utf-8 -*-
#
#    Project: silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2017 European Synchrotron Radiation Facility, Grenoble, France
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

"""A general purpose library for manipulating 2D images in 1 or 3 colors 

"""
from __future__ import absolute_import, print_function, with_statement, division


__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "11/10/2017"
__copyright__ = "2012-2017, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import logging
import numpy
from collections import OrderedDict

from .common import pyopencl, kernel_workgroup_size
from .processing import EventDescription, OpenclProcessing, BufferDescription

if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")
logger = logging.getLogger(__name__)


class ImageProcessing(OpenclProcessing):

    kernel_files = ["cast", ]

    converter = {numpy.dtype(numpy.uint8): "u8_to_float",
                 numpy.dtype(numpy.int8): "s8_to_float",
                 numpy.dtype(numpy.uint16): "u16_to_float",
                 numpy.dtype(numpy.int16): "s16_to_float",
                 numpy.dtype(numpy.uint32): "u32_to_float",
                 numpy.dtype(numpy.int32): "s32_to_float",
                 }

    def __init__(self, shape=None, ncolors=1, template=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, memory=None, profile=False):
        """Constructor of the ImageProcessing class

        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the
                            out come of the compilation
        :param memory: minimum memory available on device
        :param profile: switch on profiling to be able to profile at the kernel
                         level, store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, memory=memory, profile=profile)
        if template is not None:
            shape = template.shape
            if len(shape) > 2:
                self.ncolors = shape[2]
                self.shape = shape[:2]
            else:
                self.ncolors = 1
                self.shape = shape
        else:
            self.ncolors = ncolors
            self.shape = shape
            assert shape is not None

        buffers = [BufferDescription("image1_d", self.shape + (self.ncolors,), numpy.float32, None),
                   BufferDescription("image2_d", self.shape + (self.ncolors,), numpy.float32, None),
                        ]
        self.allocate_buffers(buffers, use_array=True)
        kernel_files = [os.path.join("image", i) for i in self.kernel_files]
        self.compile_kernels(kernel_files,
                             compile_options="-DNB_COLOR=%i" % self.ncolors)

    def to_float(self, img, copy=True, out=None):
        """ Takes any array and convert it to a float array for ease of processing.
        
        :param img: expects a numpy array or a pyopencl.array of dim 2 or 3
        :param copy: set to False to directly re-use a pyopencl array
        :param out: provide an output buffer to store the result
        """
        assert img.shape == self.shape + (self.ncolors,)
        events = []
        with self.sem:
            if out is not None and isinstance(out, pyopencl.array.Array):
                assert out.shape == self.shape + (self.ncolors,)
                assert out.dtype == numpy.float32
                out.finish()
                out_array = out
            else:
                out_array = self.cl_mem["image2_d"]

            if isinstance(img, pyopencl.array.Array):
                if copy:
                    evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["image1_d"].data, img.data)
                    input_array = self.cl_mem["image1_d"]
                    events.append(EventDescription("copy D->D", evt))
                else:
                    img.finish()
                    input_array = img
                    evt = None
            else:
                # assume this is numpy
                if img.dtype.itemsize > 4:
                    evt = pyopencl.enqueue_copy(self.queue, out_array, numpy.ascontiguousarray(img, numpy.float32))
                    input_array = None
                    events.append(EventDescription("copy H->D", evt))
                else:
                    evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["image1_d"].data, numpy.ascontiguousarray(img))
                    input_array = self.cl_mem["image1_d"]
                    events.append(EventDescription("copy H->D", evt))

            # Cast to float:
            if (input_array is not None):
                name = self.converter[img.dtype]
                kernel = self.kernels.get_kernel(name)
                ev = kernel(self.queue, (self.shape[1], self.shape[0]), None,
                            input_array.data, out_array.data,
                            numpy.int32(self.shape[1]), numpy.int32(self.shape[0])
                            )
                events.append(EventDescription("cast %s" % name, evt))

        if self.profile:
            self.events += events

        if out is None:
            res = out_array.get()
            return res
        else:
            out_array.finish()
            return out_array
