#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2018  European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import division, print_function, with_statement

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/10/2018"
__status__ = "production"


import functools
import os
import numpy
from ..common import ocl, pyopencl
from ..processing import BufferDescription, EventDescription, OpenclProcessing

import logging
logger = logging.getLogger(__name__)

if pyopencl:
    import pyopencl.version
    if pyopencl.version.VERSION < (2016, 0):
        from pyopencl.scan import GenericScanKernel, GenericDebugScanKernel
    else:
        from pyopencl.algorithm import GenericScanKernel
        from pyopencl.scan import GenericDebugScanKernel
else:
    logger.warning("No PyOpenCL, no byte-offset, please see fabio")


class ByteOffset(OpenclProcessing):
    """Perform the byte offset compression/decompression on the GPU

        See :class:`OpenclProcessing` for optional arguments description.

        :param int raw_size:
            Size of the raw stream for decompression.
            It can be (slightly) larger than the array.
        :param int dec_size:
            Size of the decompression output array
            (mandatory for decompression)
        """

    def __init__(self, raw_size=None, dec_size=None,
                 ctx=None, devicetype="all",
                 platformid=None, deviceid=None,
                 block_size=None, profile=False):
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile)
        if self.block_size is None:
            self.block_size = self.device.max_work_group_size
        wg = self.block_size

        buffers = [BufferDescription("counter", 1, numpy.int32, None)]

        if raw_size is None:
            self.raw_size = -1
            self.padded_raw_size = -1
        else:
            self.raw_size = int(raw_size)
            self.padded_raw_size = int((self.raw_size + wg - 1) & ~(wg - 1))
            buffers += [
                BufferDescription("raw", self.padded_raw_size, numpy.int8, None),
                BufferDescription("mask", self.padded_raw_size, numpy.int32, None),
                BufferDescription("values", self.padded_raw_size, numpy.int32, None),
                BufferDescription("exceptions", self.padded_raw_size, numpy.int32, None)
            ]

        if dec_size is None:
            self.dec_size = None
        else:
            self.dec_size = numpy.int32(dec_size)
            buffers += [
                BufferDescription("data_float", self.dec_size, numpy.float32, None),
                BufferDescription("data_int", self.dec_size, numpy.int32, None)
            ]

        self.allocate_buffers(buffers, use_array=True)

        self.compile_kernels([os.path.join("codec", "byte_offset")])
        self.kernels.__setattr__("scan", self._init_double_scan())
        self.kernels.__setattr__("compression_scan",
                                 self._init_compression_scan())

    def _init_double_scan(self):
        """"generates a double scan on indexes and values in one operation"""
        arguments = "__global int *value", "__global int *index"
        int2 = pyopencl.tools.get_or_register_dtype("int2")
        input_expr = "index[i]>0 ? (int2)(0, 0) : (int2)(value[i], 1)"
        scan_expr = "a+b"
        neutral = "(int2)(0,0)"
        output_statement = "value[i] = item.s0; index[i+1] = item.s1;"

        if self.block_size > 256:
            knl = GenericScanKernel(self.ctx,
                                    dtype=int2,
                                    arguments=arguments,
                                    input_expr=input_expr,
                                    scan_expr=scan_expr,
                                    neutral=neutral,
                                    output_statement=output_statement)
        else:  # MacOS on CPU
            knl = GenericDebugScanKernel(self.ctx,
                                         dtype=int2,
                                         arguments=arguments,
                                         input_expr=input_expr,
                                         scan_expr=scan_expr,
                                         neutral=neutral,
                                         output_statement=output_statement)
        return knl

    def decode(self, raw, as_float=False, out=None):
        """This function actually performs the decompression by calling the kernels

        :param numpy.ndarray raw: The compressed data as a 1D numpy array of char.
        :param bool as_float: True to decompress as float32,
                              False (default) to decompress as int32
        :param pyopencl.array out: pyopencl array in which to place the result.
        :return: The decompressed image as an pyopencl array.
        :rtype: pyopencl.array
        """
        assert self.dec_size is not None, \
            "dec_size is a mandatory ByteOffset init argument for decompression"

        events = []
        with self.sem:
            len_raw = numpy.int32(len(raw))
            if len_raw > self.padded_raw_size:
                wg = self.block_size
                self.raw_size = int(len(raw))
                self.padded_raw_size = (self.raw_size + wg - 1) & ~(wg - 1)
                logger.info("increase raw buffer size to %s", self.padded_raw_size)
                buffers = {
                           "raw": pyopencl.array.empty(self.queue, self.padded_raw_size, dtype=numpy.int8),
                           "mask": pyopencl.array.empty(self.queue, self.padded_raw_size, dtype=numpy.int32),
                           "exceptions": pyopencl.array.empty(self.queue, self.padded_raw_size, dtype=numpy.int32),
                           "values": pyopencl.array.empty(self.queue, self.padded_raw_size, dtype=numpy.int32),
                          }
                self.cl_mem.update(buffers)
            else:
                wg = self.block_size

            evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["raw"].data,
                                        raw,
                                        is_blocking=False)
            events.append(EventDescription("copy raw H -> D", evt))
            evt = self.kernels.fill_int_mem(self.queue, (self.padded_raw_size,), (wg,),
                                            self.cl_mem["mask"].data,
                                            numpy.int32(self.padded_raw_size),
                                            numpy.int32(0),
                                            numpy.int32(0))
            events.append(EventDescription("memset mask", evt))
            evt = self.kernels.fill_int_mem(self.queue, (1,), (1,),
                                            self.cl_mem["counter"].data,
                                            numpy.int32(1),
                                            numpy.int32(0),
                                            numpy.int32(0))
            events.append(EventDescription("memset counter", evt))
            evt = self.kernels.mark_exceptions(self.queue, (self.padded_raw_size,), (wg,),
                                               self.cl_mem["raw"].data,
                                               len_raw,
                                               numpy.int32(self.raw_size),
                                               self.cl_mem["mask"].data,
                                               self.cl_mem["values"].data,
                                               self.cl_mem["counter"].data,
                                               self.cl_mem["exceptions"].data)
            events.append(EventDescription("mark exceptions", evt))
            nb_exceptions = numpy.empty(1, dtype=numpy.int32)
            evt = pyopencl.enqueue_copy(self.queue, nb_exceptions, self.cl_mem["counter"].data,
                                        is_blocking=False)
            events.append(EventDescription("copy counter D -> H", evt))
            evt.wait()
            nbexc = int(nb_exceptions[0])
            if nbexc == 0:
                logger.info("nbexc %i", nbexc)
            else:
                evt = self.kernels.treat_exceptions(self.queue, (nbexc,), (1,),
                                                    self.cl_mem["raw"].data,
                                                    len_raw,
                                                    self.cl_mem["mask"].data,
                                                    self.cl_mem["exceptions"].data,
                                                    self.cl_mem["values"].data
                                                    )
                events.append(EventDescription("treat_exceptions", evt))

            #self.cl_mem["copy_values"] = self.cl_mem["values"].copy()
            #self.cl_mem["copy_mask"] = self.cl_mem["mask"].copy()
            evt = self.kernels.scan(self.cl_mem["values"],
                                    self.cl_mem["mask"],
                                    queue=self.queue,
                                    size=int(len_raw),
                                    wait_for=(evt,))
            events.append(EventDescription("double scan", evt))
            #evt.wait()
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
            evt = copy_results(self.queue, (self.padded_raw_size,), (wg,),
                               self.cl_mem["values"].data,
                               self.cl_mem["mask"].data,
                               len_raw,
                               self.dec_size,
                               out.data
                               )
            events.append(EventDescription("copy_results", evt))
            #evt.wait()
            if self.profile:
                self.events += events
        return out

    __call__ = decode

    def _init_compression_scan(self):
        """Initialize CBF compression scan kernels"""
        preamble = """
        int compressed_size(int diff) {
            int abs_diff = abs(diff);

            if (abs_diff < 128) {
                return 1;
            }
            else if (abs_diff < 32768) {
                return 3;
            }
            else {
                return 7;
            }
        }

        void write(const int index,
                   const int diff,
                   global char *output) {
            int abs_diff = abs(diff);

            if (abs_diff < 128) {
                output[index] = (char) diff;
            }
            else if (abs_diff < 32768) {
                output[index] = -128;
                output[index + 1] = (char) (diff >> 0);
                output[index + 2] = (char) (diff >> 8);
            }
            else {
                output[index] = -128;
                output[index + 1] = 0;
                output[index + 2] = -128;
                output[index + 3] = (char) (diff >> 0);
                output[index + 4] = (char) (diff >> 8);
                output[index + 5] = (char) (diff >> 16);
                output[index + 6] = (char) (diff >> 24);
            }
        }
        """
        arguments = "__global const int *data, __global char *compressed, __global int *size"
        input_expr = "compressed_size((i == 0) ? data[0] : (data[i] - data[i - 1]))"
        scan_expr = "a+b"
        neutral = "0"
        output_statement = """
        if (prev_item == 0) { // 1st thread store compressed data size
            size[0] = last_item;
        }
        write(prev_item, (i == 0) ? data[0] : (data[i] - data[i - 1]), compressed);
        """

        if self.block_size >= 64:
            knl = GenericScanKernel(self.ctx,
                                    dtype=numpy.int32,
                                    preamble=preamble,
                                    arguments=arguments,
                                    input_expr=input_expr,
                                    scan_expr=scan_expr,
                                    neutral=neutral,
                                    output_statement=output_statement)
        else:  # MacOS on CPU
            knl = GenericDebugScanKernel(self.ctx,
                                         dtype=numpy.int32,
                                         preamble=preamble,
                                         arguments=arguments,
                                         input_expr=input_expr,
                                         scan_expr=scan_expr,
                                         neutral=neutral,
                                         output_statement=output_statement)
        return knl

    def encode(self, data, out=None):
        """Compress data to CBF.

        :param data: The data to compress as a numpy array
                     (or a pyopencl Array) of int32.
        :type data: Union[numpy.ndarray, pyopencl.array.Array]
        :param pyopencl.array out:
            pyopencl array of int8 in which to store the result.
            The array should be large enough to store the compressed data.
        :return: The compressed data as a pyopencl array.
                 If out is provided, this array shares the backing buffer,
                 but has the exact size of the compressed data and the queue
                 of the ByteOffset instance.
        :rtype: pyopencl.array
        :raises ValueError: if out array is not large enough
        """

        events = []
        with self.sem:
            if isinstance(data, pyopencl.array.Array):
                d_data = data  # Uses provided array

            else:  # Copy data to device
                data = numpy.ascontiguousarray(data, dtype=numpy.int32).ravel()

                # Make sure data array exists and is large enough
                if ("data_input" not in self.cl_mem or
                        self.cl_mem["data_input"].size < data.size):
                    logger.info("increase data input buffer size to %s", data.size)
                    self.cl_mem.update({
                        "data_input": pyopencl.array.empty(self.queue,
                                                           data.size,
                                                           dtype=numpy.int32)})
                d_data = self.cl_mem["data_input"]

                evt = pyopencl.enqueue_copy(
                    self.queue, d_data.data, data, is_blocking=False)
                events.append(EventDescription("copy data H -> D", evt))

            # Make sure compressed array exists and is large enough
            compressed_size = d_data.size * 7
            if ("compressed" not in self.cl_mem or
                    self.cl_mem["compressed"].size < compressed_size):
                logger.info("increase compressed buffer size to %s", compressed_size)
                self.cl_mem.update({
                    "compressed": pyopencl.array.empty(self.queue,
                                                       compressed_size,
                                                       dtype=numpy.int8)})
            d_compressed = self.cl_mem["compressed"]
            d_size = self.cl_mem["counter"]  # Shared with decompression

            evt = self.kernels.compression_scan(d_data, d_compressed, d_size)
            events.append(EventDescription("compression scan", evt))
            byte_count = int(d_size.get()[0])

            if out is None:
                # Create out array from a sub-region of the compressed buffer
                out = pyopencl.array.Array(
                    self.queue,
                    shape=(byte_count,),
                    dtype=numpy.int8,
                    allocator=functools.partial(
                        d_compressed.base_data.get_sub_region,
                        d_compressed.offset))

            elif out.size < byte_count:
                raise ValueError(
                    "Provided output buffer is not large enough: "
                    "requires %d bytes, got %d" % (byte_count, out.size))

            else:  # out.size >= byte_count
                # Create an array with a sub-region of out and this class queue
                out = pyopencl.array.Array(
                    self.queue,
                    shape=(byte_count,),
                    dtype=numpy.int8,
                    allocator=functools.partial(out.base_data.get_sub_region,
                                                out.offset))

                evt = pyopencl.enqueue_copy(self.queue, out.data, d_compressed.data,
                                            byte_count=byte_count)
                events.append(
                    EventDescription("copy D -> D: internal -> out", evt))

            if self.profile:
                self.events += events

        return out

    def encode_to_bytes(self, data):
        """Compresses data to CBF and returns compressed data as bytes.

        Usage:

        Provided an image (`image`) stored as a numpy array of int32,
        first, create a byte offset compression/decompression object:

        >>> from silx.opencl.codec.byte_offset import ByteOffset
        >>> byte_offset_codec = ByteOffset()

        Then, compress an image into bytes:

        >>> compressed = byte_offset_codec.encode_to_bytes(image)

        :param data: The data to compress as a numpy array
                     (or a pyopencl Array) of int32.
        :type data: Union[numpy.ndarray, pyopencl.array.Array]
        :return: The compressed data as bytes.
        :rtype: bytes
        """
        compressed_array = self.encode(data)
        return compressed_array.get().tostring()
