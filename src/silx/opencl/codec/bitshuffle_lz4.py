#!/usr/bin/env python
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2022-2023  European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "12/11/2026"
__status__ = "production"


import os
import struct
import numpy
from ..common import ocl  # noqa F401 Initialize OpenCL
from ..common import pyopencl, kernel_workgroup_size
from ..processing import BufferDescription, EventDescription, OpenclProcessing
from ._unblock_bslz4 import unblock_bslz4

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

    def __init__(
        self,
        cmp_size,
        dec_size,
        dtype,
        ctx=None,
        devicetype="all",
        platformid=None,
        deviceid=None,
        block_size=None,
        profile=False,
    ):
        """Constructor of the class:

        :param cmp_size: size of the compressed data buffer (in bytes)
        :param dec_size: size of the compressed data buffer (in words)
        :param dtype: data type of one work in decompressed array

        For the other, see the doc of OpenclProcessing
        """
        OpenclProcessing.__init__(
            self,
            ctx=ctx,
            devicetype=devicetype,
            platformid=platformid,
            deviceid=deviceid,
            block_size=block_size,
            profile=profile,
        )
        if self.block_size is None:
            try:
                self.block_size = self.ctx.devices[0].preferred_work_group_size_multiple
            except Exception:
                self.block_size = self.device.max_work_group_size

        self.cmp_size = numpy.uint64(cmp_size)
        self.dec_size = numpy.uint64(dec_size)
        self.dec_dtype = numpy.dtype(dtype)
        self.num_blocks = numpy.uint32(
            (self.dec_dtype.itemsize * self.dec_size + self.LZ4_BLOCK_SIZE - 1)
            // self.LZ4_BLOCK_SIZE
        )

        buffers = [
            BufferDescription("nb_blocks", 1, numpy.uint32, None),
            BufferDescription("block_position", self.num_blocks, numpy.uint64, None),
            BufferDescription("cmp", self.cmp_size, numpy.uint8, None),
            BufferDescription("dec", self.dec_size, self.dec_dtype, None),
        ]

        self.allocate_buffers(buffers, use_array=True)

        self.compile_kernels([os.path.join("codec", "bitshuffle_lz4")])
        self.block_size = min(
            self.block_size,
            kernel_workgroup_size(self.program, "bslz4_decompress_block"),
        )

    def decompress(
        self,
        raw,
        out=None,
        wg: int = None,
        nbytes: int = None,
        force_unblock_on_device: bool = False,
        block_position=None,
    ):
        """This function actually performs the decompression by calling the kernels

        The LZ4 blocks of the chunk have to be located before being decompressed in
        parallel. This "unblock" step is performed on the host (Cython, fast) for host
        data unless `force_unblock_on_device` is set, and on the device (a single-threaded
        kernel, slow) for data already on the device. For device data whose host copy is
        available (e.g. pinned memory), the block positions can be computed on the host
        with :func:`unblock_bslz4` and passed as `block_position`.

        :param raw: The compressed data, as any contiguous buffer on the host (bytes,
                    numpy array, pinned memory...) or on the device (pyopencl Buffer or Array)
        :param pyopencl.array out: pyopencl array in which to place the result.
        :param wg: tuneable parameter with the workgroup size.
        :param int nbytes: (Optional) Number of bytes occupied by the chunk in raw.
        :param bool force_unblock_on_device: set to False for allowing Cython unblocking
        :param block_position: (Optional) start offset of every LZ4 block in `raw`,
            as returned by :func:`unblock_bslz4`: a numpy array of uint64, or a pyopencl
            array of uint64 already on the device. Skips the unblock step.
        :return: The decompressed image as an pyopencl array.
        :rtype: pyopencl.array
        """

        events = []
        with self.sem:
            if nbytes is not None:
                assert nbytes <= raw.size
                len_raw = numpy.uint64(nbytes)
            elif isinstance(raw, pyopencl.Buffer):
                len_raw = numpy.uint64(raw.size)
            else:
                len_raw = numpy.uint64(len(raw))

            unblock_on_device = True
            if isinstance(raw, (pyopencl.array.Array, pyopencl.Buffer)):
                cmp_buffer = raw.data if isinstance(raw, pyopencl.array.Array) else raw
                num_blocks = self.num_blocks
                if block_position is not None and len(block_position) > self.num_blocks:
                    # chunk larger than the allocation: read its header to size the output
                    header = numpy.empty(8, dtype=numpy.uint8)
                    pyopencl.enqueue_copy(
                        self.queue, header, cmp_buffer, is_blocking=True
                    )
                    (dest_size,) = struct.unpack(">Q", header.tobytes())
                    num_blocks = self._grow_for_chunk(dest_size)
            else:
                unblock_on_device = force_unblock_on_device
                if len_raw > self.cmp_size:
                    self.cmp_size = len_raw
                    logger.info("increase cmp buffer size to %s", self.cmp_size)
                    self.cl_mem["cmp"] = pyopencl.array.empty(
                        self.queue, self.cmp_size, dtype=numpy.uint8
                    )
                evt = pyopencl.enqueue_copy(
                    self.queue, self.cl_mem["cmp"].data, raw, is_blocking=False
                )
                events.append(EventDescription("copy raw H -> D", evt))
                cmp_buffer = self.cl_mem["cmp"].data

                (dest_size,) = struct.unpack(">Q", raw[:8])
                num_blocks = self._grow_for_chunk(dest_size)

            wg = int(wg or self.block_size)

            block_position_buffer = self.cl_mem["block_position"].data
            if block_position is not None:
                block_position_buffer = self._set_block_position(block_position, events)
            elif unblock_on_device:
                evt = self.kernels.lz4_unblock(
                    self.queue,
                    (1,),
                    (1,),
                    cmp_buffer,
                    len_raw,
                    numpy.uint32(self.dec_dtype.itemsize),
                    self.cl_mem["block_position"].data,
                    num_blocks,
                    self.cl_mem["nb_blocks"].data,
                )
                events.append(EventDescription("LZ4 unblock", evt))
            else:
                # Perform unblock using Cython
                block_position = unblock_bslz4(raw, self.dec_dtype.itemsize)
                block_position_buffer = self._set_block_position(block_position, events)

            if out is None:
                out = self.cl_mem["dec"]
            else:
                assert out.dtype == self.dec_dtype
                assert out.size == self.dec_size

            evt = self.kernels.bslz4_decompress_block(
                self.queue,
                (self.num_blocks * wg,),
                (wg,),
                cmp_buffer,
                out.data,
                block_position_buffer,
                self.cl_mem["nb_blocks"].data,
                numpy.uint8(self.dec_dtype.itemsize),
                numpy.uint64(len_raw),
            )
            events.append(EventDescription("LZ4 decompress", evt))
        self.profile_multi(events)
        return out

    __call__ = decompress

    def _grow_for_chunk(self, dest_size):
        """Prepare the buffers for a chunk of `dest_size` decompressed bytes

        The output and block-position buffers are grown if the chunk is larger than
        what was allocated.

        :param dest_size: decompressed size of the chunk, in bytes (from its header)
        :return: number of LZ4 blocks in the chunk
        """
        if dest_size == self.dec_size * self.dec_dtype.itemsize:
            return self.num_blocks
        num_blocks = numpy.uint32(
            (dest_size + self.LZ4_BLOCK_SIZE - 1) // self.LZ4_BLOCK_SIZE
        )
        if dest_size > self.dec_size * self.dec_dtype.itemsize:
            self.dec_size = numpy.uint64(dest_size // self.dec_dtype.itemsize)
            self.cl_mem["dec"] = pyopencl.array.empty(
                self.queue, self.dec_size, self.dec_dtype
            )
            self._grow_block_position(num_blocks)
        return num_blocks

    def _grow_block_position(self, num_blocks):
        """Make sure the device can hold `num_blocks` block positions

        Both the `lz4_unblock` kernel (which writes up to `num_blocks` positions) and
        the decompression kernel (launched with `num_blocks` workgroups) rely on it.
        """
        if num_blocks > self.num_blocks:
            self.num_blocks = numpy.uint32(num_blocks)
            self.cl_mem["block_position"] = pyopencl.array.empty(
                self.queue, self.num_blocks, numpy.uint64
            )

    def _set_block_position(self, block_position, events):
        """Send the LZ4 block positions to the device, skipping the unblock kernel

        :param block_position: positions as a numpy array of uint64 (as returned by
            :func:`unblock_bslz4`) or a pyopencl array of uint64 already on the device
        :param events: list of EventDescription to which profiling events are appended
        :return: the pyopencl Buffer holding the positions (`nb_blocks` is updated too)
        """
        if isinstance(block_position, pyopencl.array.Array):
            if block_position.dtype != numpy.uint64:
                raise TypeError("block_position must be an array of uint64")
            size = block_position.size
            self._grow_block_position(size)
            buffer = block_position.data
        else:
            block_position = numpy.ascontiguousarray(block_position, dtype=numpy.uint64)
            size = block_position.size
            self._grow_block_position(size)
            evt = pyopencl.enqueue_copy(
                self.queue,
                self.cl_mem["block_position"].data,
                block_position,
                is_blocking=False,
            )
            events.append(EventDescription("copy block_position H -> D", evt))
            buffer = self.cl_mem["block_position"].data
        evt = pyopencl.enqueue_copy(
            self.queue,
            self.cl_mem["nb_blocks"].data,
            numpy.array([size], numpy.uint32),
            is_blocking=False,
        )
        events.append(EventDescription("copy nb_blocks H -> D", evt))
        return buffer
