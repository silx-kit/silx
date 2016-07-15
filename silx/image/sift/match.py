#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Contains a class for creating a matching plan, allocating arrays, compiling kernels and other things like that
"""

from __future__ import division, print_function, with_statement

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-07-23"
__status__ = "beta"
__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""
import time, math, os, logging, sys, threading
import gc
import numpy
from .param import par
from .opencl import ocl, pyopencl
from .utils import calc_size, kernel_size, sizeof
logger = logging.getLogger("sift.match")
if pyopencl:
    from pyopencl import mem_flags as MF
else:
    logger.warning("No PyOpenCL, no sift")

class MatchPlan(object):
    """
    Plan to compare sets of SIFT keypoint

    siftp = sift.MatchPlan(devicetype="GPU")
    kp = siftp.match(kp1,kp2)

    kp is a nx132 array. the second dimension is composed of x,y, scale and angle as well as 128 floats describing the keypoint

    """
    kernels = {"matching_gpu":64,
               "matching_cpu":16,
               "memset":128, }

    dtype_kp = numpy.dtype([('x', numpy.float32),
                                ('y', numpy.float32),
                                ('scale', numpy.float32),
                                ('angle', numpy.float32),
                                ('desc', (numpy.uint8, 128))
                                ])

    def __init__(self, size=16384, devicetype="CPU", profile=False, device=None, max_workgroup_size=None, roi=None, context=None):
        """
        Constructor of the class:

        @param size: size of the input keypoint-list alocated on the GPU.
        @param devicetype: can be CPU or GPU
        @param profile: set to true to activate profiling information collection
        @param device: 2-tuple of integer, see clinfo
        @param max_workgroup_size: CPU on MacOS, limit to 1. None by default to use default ones (max=128). 
        @param roi: Region Of Interest: TODO
        @param context: Use an external context (discard devicetype and device options)
        """
        self.profile = bool(profile)
        if max_workgroup_size:
            self.max_workgroup_size = int(max_workgroup_size)
            self.kernels = {}
            for k, v in self.__class__.kernels.items():
                self.kernels[k] = min(v, self.max_workgroup_size)
        else:
            self.max_workgroup_size = None

        self.events = []
        self.kpsize = size
        self.buffers = {}
        self.programs = {}
        self.memory = None
        self.octave_max = None
        self.red_size = None
        if context:
            self.ctx = context
            device_name = self.ctx.devices[0].name.strip()
            platform_name = self.ctx.devices[0].platform.name.strip()
            platform = ocl.get_platform(platform_name)
            device = platform.get_device(device_name)
            self.device = platform.id, device.id
        else:
            if device is None:
                self.device = ocl.select_device(type=devicetype, memory=self.memory, best=True)
            else:
                self.device = device
            self.ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[self.device[0]].get_devices()[self.device[1]]])
        if profile:
            self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = pyopencl.CommandQueue(self.ctx)
#        self._calc_workgroups()
        self._compile_kernels()
        self._allocate_buffers()
        self.debug = []
        self._sem = threading.Semaphore()

        self.devicetype = ocl.platforms[self.device[0]].devices[self.device[1]].type
        if (self.devicetype == "CPU"):
            self.USE_CPU = True
            self.matching_kernel = "matching_cpu"
            if sys.platform == "darwin":
                logger.warning("MacOSX computer working on CPU: limiting workgroup size to 1 !")
                self.max_workgroup_size = 1
                self.kernels = {}
                for k, v in self.__class__.kernels.items():
                    if isinstance(v, int):
                        self.kernels[k] = 1
                    else:
                        self.kernels[k] = tuple([1 for i in v])
        else:
            self.USE_CPU = False
            self.matching_kernel = "matching_gpu"
        self.roi = None
        if roi:
            self.set_roi(roi)

    def __del__(self):
        """
        Destructor: release all buffers
        """
        self._free_kernels()
        self._free_buffers()
        self.queue = None
        self.ctx = None
        gc.collect()

    def _allocate_buffers(self):
        self.buffers[ "Kp_1" ] = pyopencl.array.empty(self.queue, (self.kpsize,), dtype=self.dtype_kp)
        self.buffers[ "Kp_2" ] = pyopencl.array.empty(self.queue, (self.kpsize,), dtype=self.dtype_kp)
#        self.buffers[ "tmp" ] = pyopencl.array.empty(self.queue, (self.kpsize,), dtype=self.dtype_kp)
        self.buffers[ "match" ] = pyopencl.array.empty(self.queue, (self.kpsize, 2), dtype=numpy.int32)
        self.buffers["cnt" ] = pyopencl.array.empty(self.queue, 1, dtype=numpy.int32)

    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        for buffer_name in self.buffers:
            if self.buffers[buffer_name] is not None:
                try:
                    del self.buffers[buffer_name]
                    self.buffers[buffer_name] = None
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s" % buffer_name)

    def _compile_kernels(self):
        """
        Call the OpenCL compiler
        """
        kernel_directory = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(kernel_directory, "matching_gpu" + ".cl")):
            while (".zip" in kernel_directory)  and (len(kernel_directory) > 4):
                kernel_directory = os.path.dirname(kernel_directory)
            kernel_directory = os.path.join(kernel_directory, "sift_kernels")

        for kernel in self.kernels:
            kernel_file = os.path.join(kernel_directory, kernel + ".cl")
            kernel_src = open(kernel_file).read()
            wg_size = self.kernels[kernel]
            try:
                program = pyopencl.Program(self.ctx, kernel_src).build('-D WORKGROUP_SIZE=%s' % wg_size)
            except pyopencl.MemoryError as error:
                raise MemoryError(error)
            except pyopencl.RuntimeError as error:
                if kernel == "keypoints":
                    logger.warning("Failed compiling kernel '%s' with workgroup size %s: %s: use low_end alternative", kernel, wg_size, error)
                    self.LOW_END = True
                else:
                    logger.error("Failed compiling kernel '%s' with workgroup size %s: %s", kernel, wg_size, error)
                    raise error
            self.programs[kernel] = program

    def _free_kernels(self):
        """
        free all kernels
        """
        self.programs = {}

    def match(self, nkp1, nkp2, raw_results=False):
        """
        calculate the matching of 2 keypoint list

        @param nkp1, nkp2: numpy 1D recarray of keypoints or equivalent GPU buffer
        @param raw_results: if true return the 2D array of indexes of matching keypoints (not the actual keypoints)

        TODO: implement the ROI ...

        """
        assert len(nkp1.shape) == 1  # Nota: nkp1.ndim is not valid for gpu_arrays
        assert len(nkp2.shape) == 1
        assert type(nkp1) in [numpy.core.records.recarray, pyopencl.array.Array]
        assert type(nkp2) in [numpy.core.records.recarray, pyopencl.array.Array]
        result = None
        with self._sem:
            if type(nkp1) == numpy.core.records.recarray:
                if nkp1.size > self.buffers[ "Kp_1" ].size:
                    logger.warning("increasing size of keypoint vector 1 to %i" % nkp1.size)
                    self.buffers[ "Kp_1" ] = pyopencl.array.empty(self.queue, (nkp1.size,), dtype=self.dtype_kp)
                kpt1_gpu = self.buffers[ "Kp_1" ]
                self._reset_buffer1()
                evt1 = pyopencl.enqueue_copy(self.queue, kpt1_gpu.data, nkp1)
                if self.profile:
                    self.events.append(("copy H->D KP_1", evt1))
            else:
                kpt1_gpu = nkp1
            if type(nkp2) == numpy.core.records.recarray:
                if nkp2.size > self.buffers[ "Kp_2" ].size:
                    logger.warning("increasing size of keypoint vector 2 to %i" % nkp2.size)
                    self.buffers[ "Kp_2" ] = pyopencl.array.empty(self.queue, (nkp2.size,), dtype=self.dtype_kp)
                kpt2_gpu = self.buffers[ "Kp_2" ]
                self._reset_buffer2()
                evt2 = pyopencl.enqueue_copy(self.queue, kpt2_gpu.data, nkp2)
                if self.profile:
                    self.events.append(("copy H->D KP_1", evt1))
            else:
                kpt2_gpu = nkp2
            if min(kpt1_gpu.size, kpt2_gpu.size) > self.buffers[ "match" ].shape[0]:
                self.kpsize = min(kpt1_gpu.size, kpt2_gpu.size)
                self.buffers[ "match" ] = pyopencl.array.empty(self.queue, (self.kpsize, 2), dtype=numpy.int32)
            self._reset_output()
            evt = self.programs[self.matching_kernel].matching(self.queue, calc_size((nkp1.size,), (self.kernels[self.matching_kernel],)), (self.kernels[self.matching_kernel],),
                                          kpt1_gpu.data,
                                          kpt2_gpu.data,
                                          self.buffers[ "match" ].data,
                                          self.buffers[ "cnt" ].data,
                                          numpy.int32(self.kpsize),
                                          numpy.float32(par.MatchRatio * par.MatchRatio),
                                          numpy.int32(nkp1.size),
                                          numpy.int32(nkp2.size))
            if self.profile:
                self.events.append(("matching", evt))
            size = self.buffers["cnt"].get()[0]
            match = numpy.empty(shape=(size, 2), dtype=numpy.int32)
            if size > 0:
                cpyD2H = pyopencl.enqueue_copy(self.queue, match, self.buffers[ "match" ].data)
            if self.profile:
                self.events.append(("copy D->H match", cpyD2H))
            if raw_results:
                result = match
            else:
                result = numpy.recarray(shape=(size, 2), dtype=self.dtype_kp)

                result[:, 0] = nkp1[match[:size, 0]]
                result[:, 1] = nkp2[match[:size, 1]]
        return result


    def _reset_buffer(self):
        """Reseet all buffers"""
        self._reset_buffer1()
        self._reset_buffer2()
        self._reset_output()

    def _reset_buffer1(self):
        ev1 = self.programs["memset"].memset_kp(self.queue, calc_size((self.buffers[ "Kp_1" ].size,), (self.kernels["memset"],)), (self.kernels["memset"],),
                                          self.buffers[ "Kp_1" ].data, numpy.float32(-1.0), numpy.uint8(0), numpy.int32(self.buffers[ "Kp_1" ].size))
        if self.profile:
            self.events.append(("memset Kp1", ev1))
    def _reset_buffer2(self):
        ev2 = self.programs["memset"].memset_kp(self.queue, calc_size((self.buffers[ "Kp_2" ].size,), (self.kernels["memset"],)), (self.kernels["memset"],),
                                          self.buffers[ "Kp_2" ].data, numpy.float32(-1.0), numpy.uint8(0), numpy.int32(self.buffers[ "Kp_2" ].size))
        if self.profile:
            self.events.append(("memset Kp2", ev2))
    def _reset_output(self):
        ev3 = self.programs["memset"].memset_int(self.queue, calc_size((self.buffers[ "match" ].size,), (self.kernels["memset"],)), (self.kernels["memset"],),
                                                 self.buffers[ "match" ].data, numpy.int32(-1), numpy.int32(self.buffers[ "match" ].size))
        ev4 = self.programs["memset"].memset_int(self.queue, (1,), (1,),
                                                 self.buffers[ "cnt" ].data, numpy.int32(0), numpy.int32(1))
        if self.profile:
            self.events += [("memset match", ev3),
                            ("memset cnt", ev4), ]
    def reset_timer(self):
        """
        Resets the profiling timers
        """
        with self._sem:
            self.events = []

    def set_roi(self, roi):
        """
        Defines the region of interest

        @param roi: region of interest as 2D numpy array with non zero where
        valid pixels are.
        """
        with self._sem:
            self.roi = numpy.ascontiguousarray(roi, numpy.int8)
            self.buffers["ROI"] = pyopencl.array.to_device(self.queue, self.roi)
    def unset_roi(self):
        """
        Unset the region of interest
        """
        with self._sem:
            self.roi = None
            self.buffers["ROI"] = None


