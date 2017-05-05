#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: S I L X project
#             https://github.com/silx-kit/silx
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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
#

"""
Common OpenCL abstract base classes for different processing
"""

from __future__ import absolute_import, print_function, division


__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "03/02/2017"
__status__ = "stable"


import os
import logging
import gc
from collections import namedtuple
import numpy
import threading
from .common import ocl, pyopencl, release_cl_buffers
from .utils import concatenate_cl_kernel


BufferDescription = namedtuple("BufferDescription", ["name", "size", "dtype", "flags"])
EventDescription = namedtuple("EventDescription", ["name", "event"])

logger = logging.getLogger("pyFAI.opencl.processing")


class OpenclProcessing(object):
    """Abstract class for different types of OpenCL processing.

    This class provides:
    * Generation of the context, queues, profiling mode
    * Additional function to allocate/free all buffers declared as static attributes of the class
    * Functions to compile kernels, cache them and clean them
    * helper functions to clone the object
    """
    # Example of how to create an output buffer of 10 floats
    buffers = [BufferDescription("output", 10, numpy.float32, None),
               ]
    # list of kernel source files to be concatenated before compilation of the program
    kernel_files = []

    def __init__(self, ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, profile=False):
        """Constructor of the abstract OpenCL processing class

        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        """
        self.sem = threading.Semaphore()
        self.profile = None
        self.events = []  # List with of EventDescription, kept for profiling
        self.cl_mem = {}  # dict with all buffer allocated
        self.cl_program = None  # The actual OpenCL program
        self.cl_kernel_args = {}  # dict with all kernel arguments
        if ctx:
            self.ctx = ctx
        else:
            self.ctx = ocl.create_context(devicetype=devicetype, platformid=platformid, deviceid=deviceid)
        device_name = self.ctx.devices[0].name.strip()
        platform_name = self.ctx.devices[0].platform.name.strip()
        platform = ocl.get_platform(platform_name)
        self.device = platform.get_device(device_name)
        self.cl_kernel_args = {}  # dict with all kernel arguments

        self.set_profiling(profile)
        self.block_size = block_size
        self.program = None

    def __del__(self):
        """Destructor: release all buffers and programs
        """
        self.free_kernels()
        self.free_buffers()
        self.queue = None
        self.ctx = None
        gc.collect()

    def allocate_buffers(self, buffers=None):
        """
        Allocate OpenCL buffers required for a specific configuration

        :param buffers: a list of BufferDescriptions, leave to None for
                        paramatrized buffers.

        Note that an OpenCL context also requires some memory, as well
        as Event and other OpenCL functionalities which cannot and are
        not taken into account here.  The memory required by a context
        varies depending on the device. Typical for GTX580 is 65Mb but
        for a 9300m is ~15Mb In addition, a GPU will always have at
        least 3-5Mb of memory in use.  Unfortunately, OpenCL does NOT
        have a built-in way to check the actual free memory on a
        device, only the total memory.
        """

        if buffers is None:
            buffers = self.buffers

        with self.sem:
            mem = {}

            # check if enough memory is available on the device
            ualloc = 0
            for buf in buffers:
                ualloc += numpy.dtype(buf.dtype).itemsize * buf.size
            logger.info("%.3fMB are needed on device which has %.3fMB",
                        ualloc / 1.0e6, self.device.memory / 1.0e6)

            if ualloc >= self.device.memory:
                raise MemoryError("Fatal error in allocate_buffers. Not enough "
                                  " device memory for buffers (%lu requested, %lu available)"
                                  % (ualloc, self.device.memory))

            # do the allocation
            try:
                for buf in buffers:
                    size = numpy.dtype(buf.dtype).itemsize * buf.size
                    mem[buf.name] = pyopencl.Buffer(self.ctx, buf.flags, size)
            except pyopencl.MemoryError as error:
                release_cl_buffers(mem)
                raise MemoryError(error)

        self.cl_mem.update(mem)

    def free_buffers(self):
        """free all device.memory allocated on the device
        """
        with self.sem:
            for key, buf in list(self.cl_mem.items()):
                if buf is not None:
                    if isinstance(buf, pyopencl.array.Array):
                        try:
                            buf.data.release()
                        except pyopencl.LogicError:
                            logger.error("Error while freeing buffer %s", key)
                    else:
                        try:
                            buf.release()
                        except pyopencl.LogicError:
                            logger.error("Error while freeing buffer %s", key)
                    self.cl_mem[key] = None

    def compile_kernels(self, kernel_files=None, compile_options=None):
        """Call the OpenCL compiler

        :param kernel_files: list of path to the kernel
            (by default use the one declared in the class)
        :param compile_options: string of compile options
        """
        # concatenate all needed source files into a single openCL module
        kernel_files = kernel_files or self.kernel_files
        kernel_src = concatenate_cl_kernel(kernel_files)

        compile_options = compile_options or ""
        logger.info("Compiling file %s with options %s", kernel_files, compile_options)
        try:
            self.program = pyopencl.Program(self.ctx, kernel_src).build(options=compile_options)
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)

    def free_kernels(self):
        """Free all kernels
        """
        for kernel in self.cl_kernel_args:
            self.cl_kernel_args[kernel] = []
        self.program = None

    def set_profiling(self, value=True):
        """Switch On/Off the profiling flag of the command queue to allow debugging

        :param value: set to True to enable profiling, or to False to disable it.
                      Without profiling, the processing is marginally faster

        Profiling information can then be retrieved with the 'log_profile' method
        """
        if bool(value) != self.profile:
            with self.sem:
                self.profile = bool(value)
                if self.profile:
                    self.queue = pyopencl.CommandQueue(self.ctx,
                        properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
                else:
                    self.queue = pyopencl.CommandQueue(self.ctx)

    def log_profile(self):
        """If we are in profiling mode, prints out all timing for every single OpenCL call
        """
        t = 0.0
        out = ["", "Profiling info for OpenCL %s" % self.__class__.__name__]
        if self.profile:
            for e in self.events:
                if "__len__" in dir(e) and len(e) >= 2:
                    et = 1e-6 * (e[1].profile.end - e[1].profile.start)
                    out.append("%50s:\t%.3fms" % (e[0], et))
                    t += et

        out.append("_" * 80)
        out.append("%50s:\t%.3fms" % ("Total execution time", t))
        logger.info(os.linesep.join(out))
        return out

# This should be implemented by concrete class
#     def __copy__(self):
#         """Shallow copy of the object
#
#         :return: copy of the object
#         """
#         return self.__class__((self._data, self._indices, self._indptr),
#                               self.size, block_size=self.BLOCK_SIZE,
#                               platformid=self.platform.id,
#                               deviceid=self.device.id,
#                               checksum=self.on_device.get("data"),
#                               profile=self.profile, empty=self.empty)
#
#     def __deepcopy__(self, memo=None):
#         """deep copy of the object
#
#         :return: deepcopy of the object
#         """
#         if memo is None:
#             memo = {}
#         new_csr = self._data.copy(), self._indices.copy(), self._indptr.copy()
#         memo[id(self._data)] = new_csr[0]
#         memo[id(self._indices)] = new_csr[1]
#         memo[id(self._indptr)] = new_csr[2]
#         new_obj = self.__class__(new_csr, self.size,
#                                  block_size=self.BLOCK_SIZE,
#                                  platformid=self.platform.id,
#                                  deviceid=self.device.id,
#                                  checksum=self.on_device.get("data"),
#                                  profile=self.profile, empty=self.empty)
#         memo[id(self)] = new_obj
#         return new_obj
