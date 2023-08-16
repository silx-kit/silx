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
This module provides a class for bitshuffle-LZ4 compression/decompression.
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/08/2023"
__status__ = "production"

import time
import os
import struct
import numpy
import json
from ..common import ocl, pyopencl, kernel_workgroup_size
from ..processing import BufferDescription, EventDescription, OpenclProcessing
import pyopencl.array as cla

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
        """Constructor of the class:
        
        :param cmp_size: size of the compressed data buffer (in bytes)
        :param dec_size: size of the compressed data buffer (in words)
        :param dtype: data type of one work in decompressed array
        
        For the other, see the doc of OpenclProcessing
        """
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
        self.block_size = min(self.block_size, kernel_workgroup_size(self.program, "bslz4_decompress_block"))

    def decompress(self, raw, out=None, wg=None, nbytes=None):
        """This function actually performs the decompression by calling the kernels
        :param numpy.ndarray raw: The compressed data as a 1D numpy array of char or string
        :param pyopencl.array out: pyopencl array in which to place the result.
        :param wg: tuneable parameter with the workgroup size. 
        :param int nbytes: (Optional) Number of bytes occupied by the chunk in raw.
        :return: The decompressed image as an pyopencl array.
        :rtype: pyopencl.array
        """

        events = []
        with self.sem:
            if nbytes is not None:
                assert nbytes <= raw.size
                len_raw = numpy.uint64(nbytes)
            elif  isinstance(raw, pyopencl.Buffer):
                len_raw = numpy.uint64(raw.size)
            else:
                len_raw = numpy.uint64(len(raw))

            if isinstance(raw, pyopencl.array.Array):
                cmp_buffer = raw.data
                num_blocks = self.num_blocks
            elif  isinstance(raw, pyopencl.Buffer):
                cmp_buffer = raw
                num_blocks = self.num_blocks
            else:
                if len_raw > self.cmp_size:
                    self.cmp_size = len_raw
                    logger.info("increase cmp buffer size to %s", self.cmp_size)
                    self.cl_mem["cmp"] = pyopencl.array.empty(self.queue, self.cmp_size, dtype=numpy.uint8)
                evt = pyopencl.enqueue_copy(self.queue,
                                            self.cl_mem["cmp"].data,
                                            raw,
                                            is_blocking=False)
                events.append(EventDescription("copy raw H -> D", evt))
                cmp_buffer = self.cl_mem["cmp"].data

                dest_size = struct.unpack(">Q", raw[:8])
                self_dest_nbyte = self.dec_size * self.dec_dtype.itemsize
                if dest_size<self_dest_nbyte:
                    num_blocks = numpy.uint32((dest_size+self.LZ4_BLOCK_SIZE-1) // self.LZ4_BLOCK_SIZE)
                elif dest_size>self_dest_nbyte:
                    num_blocks = numpy.uint32((dest_size+self.LZ4_BLOCK_SIZE-1) // self.LZ4_BLOCK_SIZE)
                    self.cl_mem["dec"] = pyopencl.array.empty(self.queue,dest_size , self.dec_dtype)
                    self.dec_size = dest_size // self.dec_dtype.itemsize
                else:
                    num_blocks = self.num_blocks

            wg = int(wg or self.block_size)

            evt = self.program.lz4_unblock(self.queue, (1,), (1,), 
                                           cmp_buffer,
                                           len_raw,
                                           self.cl_mem["block_position"].data,
                                           num_blocks,
                                           self.cl_mem["nb_blocks"].data)
            events.append(EventDescription("LZ4 unblock", evt))

            if out is None:
                out = self.cl_mem["dec"]
            else:
                assert out.dtype == self.dec_dtype
                assert out.size == self.dec_size

            evt = self.program.bslz4_decompress_block(self.queue, (self.num_blocks*wg,), (wg,),
                                                      cmp_buffer,
                                                      out.data,
                                                      self.cl_mem["block_position"].data,
                                                      self.cl_mem["nb_blocks"].data,
                                                      numpy.uint8(self.dec_dtype.itemsize)
                                                      )
            events.append(EventDescription("LZ4 decompress", evt))
        self.profile_multi(events)
        return out

    __call__ = decompress


def test_lz4_analysis(data, block_size=1024, workgroup_size=32, segments_size=None, 
                      profile=True, compaction=True):
    """Function that tests LZ4 analysis (i.e. the correctness of segments) on a dataset.
    
    :param data: some data to play with
    :paam block_size: how many items are treated by a workgroup
    :param workgroup_size: size of the workgroup
    :param segments_size: by default, data_size/4
    :param profile: tune on profiling for OpenCL
    :param compaction: set to false to retrieve the raw segment before compaction 
    :return: a set of segment containing:
             - position in the input stream
             - length of the littral section
             - length of the matching section
             - position in the output stream
    
    Prints out performance (measured from Python) in ms
    """
    t0 = time.perf_counter_ns()
    performances = {}
    if isinstance(data, bytes):
        data = numpy.frombuffer(data, "uint8")
    else:
        data = data.view("uint8")
    data_size = data.size
    num_workgroup = (data_size+block_size-1)//block_size
    if segments_size is None:
        segments_size = block_size//4
    
    segment_pos = numpy.zeros((num_workgroup,2), "int32")
    tmp_sp = numpy.arange(0,segments_size*(num_workgroup+1), segments_size)
    segment_pos[:,0] = tmp_sp[:-1]
    segment_pos[:,1] = tmp_sp[1:]
    
    
    # Opencl setup
    t1 = time.perf_counter_ns()
    ctx = pyopencl.create_some_context()
    src_file = os.path.abspath(os.path.join(os.path.abspath(__file__),"../../../resources/opencl/codec/lz4_compression.cl"))
    src = open(src_file).read()
    prg = pyopencl.Program(ctx, src).build(options=f"-DBUFFER_SIZE={block_size} -DSEGMENT_SIZE={segments_size} -DWORKGROUP_SIZE={workgroup_size}")
    t1a = time.perf_counter_ns()
    if profile:
        queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = pyopencl.CommandQueue(ctx)
    
    data_d = cla.to_device(queue, data)
    segment_posd = cla.to_device(queue, segment_pos)
    segments_d = cla.zeros(queue, (segments_size*num_workgroup,4), "int32")
    wgcnt_d = cla.to_device(queue, numpy.array([num_workgroup], "int32"))
    output_size_d = cla.zeros(queue, num_workgroup, "int32")
    
    t2 = time.perf_counter_ns()
    prg.LZ4_cmp_stage1(queue, (workgroup_size*num_workgroup,), (workgroup_size,), 
                    data_d.data, numpy.int32(data_size), 
                    segment_posd.data,
                    segments_d.data, numpy.int32(compaction), output_size_d.data, wgcnt_d.data
                    ).wait()
    t3 = time.perf_counter_ns()
    segments = segments_d.get()
    if compaction:
        final_positons = segment_posd.get()
        segments = segments[final_positons[0,0]:final_positons[0,1]]
    t4 = time.perf_counter_ns()
    if 1: #profile:
        performances["python_setup"] = (t1-t0)*1e-6
        performances["opencl_compilation"] = (t1a-t1)*1e-6
        performances["opencl_setup"] = (t2-t1a)*1e-6
        performances["opencl_run"] = (t3-t2)*1e-6
        performances["opencl_retrieve"] = (t4-t3)*1e-6
        print(json.dumps(performances, indent=2))

    if compaction:
        compacted = segments
    else:
        compacted = _repack_segments(segments)

    # Check validity: input indexes
    inp_idx = compacted[:,0]
    res = numpy.where((inp_idx[1:]-inp_idx[:-1])<=0)
    
#     if res[0].size:
    if True:
        print(f"Input position are all ascending except {res[0]}")
    # Check validity: input size
    size = segments[:,1:3].sum()
    if True:
#     if data.size != size:
        print(f"Input size matches, got {size}, expected {data.size}")
    
    # Check validity: input size (bis)
    size = compacted[-1,:-1].sum()
#     if data.size != size:
    if True:
        print(f"Input size does match the end of segments, got {size}, expected {data.size}")
    
    # Check validity: output indexes
    out_idx = compacted[:,-1]
    res = numpy.where((out_idx[1:]-out_idx[:-1])<=0)
#     if res[0].size:
    if True:
        print(f"Output position are all ascending, except {res[0]}")
    
    #check for invalid segments, those have no matches, allowd only on last segment
    match_size = compacted[:-1,2]
    res = numpy.where(match_size==0)
    if True:
#     if res[0].size:
        print(f"Found empty match at {res[0]}")
    
    # Validate that match are all constant:
    print(f"Non constant match section found at {_validate_content(data, compacted)}")
        
    return segments


def _validate_content(data, segments):
    data = data.view('uint8')
    bad = {}
    for i,s in enumerate(segments):
        if s[2] == 0: continue
        start = s[0]+s[1]
        stop = start + s[2]
        res = numpy.where(data[start:stop]-data[start])[0]
        if res.size: 
            bad[i] = res
    return bad


def _repack_segments(segments):
    "repack a set of segments to be contiguous"
    valid = numpy.where(segments.sum(axis=-1)!=0)[0]
    repacked1 = segments[valid]
    blocks = numpy.where(repacked1[:,-1]==0)[0]
    sub_tot = 0
    repacked2 = repacked1.copy()
    for start, stop in zip(blocks,numpy.concatenate((blocks[1:], [len(repacked1)]))):
        repacked2[start:stop, -1] += sub_tot
        sub_tot+=repacked1[stop-1,-1]
    repacked3 = repacked2[numpy.where(repacked2[:,1:3].sum(axis=-1)!=0)[0]]
    return repacked3
    