#!/usr/bin/env python
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2022-2022  European Synchrotron Radiation Facility, Grenoble, France
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
This module provides a class for CBF byte offset compression/decompression.
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/11/2022"
__status__ = "production"


import os
import numpy
from ..common import ocl, pyopencl
from ..processing import BufferDescription, EventDescription, OpenclProcessing

import logging
logger = logging.getLogger(__name__)


class BitshuffleLz4(OpenclProcessing):
    """Perform the bitshuffle-lz4 decompression on the GPU

        See :class:`OpenclProcessing` for optional arguments description.

        :param int cmp_size:
            Size of the raw stream for decompression.
            It can be (slightly) larger than the array.
        :param int dec_size:
            Size of the decompression output array
            (mandatory for decompression)
        :param dtype: dtype of decompressed data
        """
    LZ4_BLOCK_SIZE = 8192
    
    def __init__(self, cmp_size, dec_size, dtype,
                 ctx=None, devicetype="all",
                 platformid=None, deviceid=None,
                 block_size=None, profile=False):
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)
        if self.block_size is None:
            try:
                self.block_size = self.ctx.devices[0].preferred_work_group_size_multiple
            except:
                self.block_size = self.device.max_work_group_size

        self.cmp_size = numpy.uint64(cmp_size)
        self.dec_size = numpy.uint64(dec_size)
        self.dec_dtype = numpy.dtype(dtype)
        self.num_blocks = numpy.uint32((self.dec_dtype.itemsize*self.dec_size+self.LZ4_BLOCK_SIZE-1)//self.LZ4_BLOCK_SIZE)

        buffers = [BufferDescription("nb_blocks", 1, numpy.uint32, None),
                   BufferDescription("block_position", self.num_blocks, numpy.uint64, None),
                   BufferDescription("cmp", self.cmp_size, numpy.uint8, None),
                   BufferDescription("dec", self.dec_size, self.dec_dtype, None),
        ]

        self.allocate_buffers(buffers, use_array=True)

        self.compile_kernels([os.path.join("codec", "bitshuffle_lz4")])

    def decompress(self, raw, out=None, wg=None):
        """This function actually performs the decompression by calling the kernels

        :param numpy.ndarray raw: The compressed data as a 1D numpy array of char or string
        :param pyopencl.array out: pyopencl array in which to place the result.
        :param wg: tuneable parameter with the workgroup size. 
        :return: The decompressed image as an pyopencl array.
        :rtype: pyopencl.array
        """

        events = []
        with self.sem:
            len_raw = numpy.uint64(len(raw))
            if len_raw > self.cmp_size:
                self.cmp_size = len_raw
                logger.info("increase cmp buffer size to %s", self.cmp_size)
                self.cl_mem["cmp"] = pyopencl.array.empty(self.queue, self.cmp_size, dtype=numpy.uint8)
            wg = int(wg or self.block_size)

            evt = pyopencl.enqueue_copy(self.queue, 
                                        self.cl_mem["cmp"].data,
                                        raw,
                                        is_blocking=False)
            events.append(EventDescription("copy raw H -> D", evt))
            evt = self.program.lz4_unblock(self.queue, (1,), (1,), 
                                           self.cl_mem["cmp"].data,
                                           len_raw,
                                           self.cl_mem["block_position"].data,
                                           self.num_blocks,
                                           self.cl_mem["nb_blocks"].data)
            events.append(EventDescription("LZ4 unblock", evt))

            if out is None:
                out = self.cl_mem["dec"]
            else:
                assert out.dtype == self.dec_dtype
                assert out.size == self.dec_size
             
            evt = self.program.bslz4_decompress_block(self.queue, (self.num_blocks*wg,), (wg,),
                                                      self.cl_mem["cmp"].data,
                                                      out.data,
                                                      self.cl_mem["block_position"].data,
                                                      self.cl_mem["nb_blocks"].data,
                                                      numpy.uint8(self.dec_dtype.itemsize)
                                                      )
            events.append(EventDescription("LZ4 decompress", evt))
        self.profile_multi(events)
        return out

    __call__ = decompress
