#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: S I L X project
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2017 European Synchrotron Radiation Facility, Grenoble, France
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
from platform import platform

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "24/01/2017"
__status__ = "stable"

import os
import logging
import threading
import gc

import numpy
from collections import namedtuple
BufferDescription = namedtuple("BufferDescription", ["name", "flags", "dtype", "size"])
EventDescription = namedtuple("EventDescription", ["name", "event"])

logger = logging.getLogger("silx.opencl")

from .utils import get_opencl_code, concatenate_cl_kernel

if os.environ.get("SILX_OPENCL") in ["0", "False"]:
    logger.warning("Use of OpenCL has been disables from environment variable: SILX_OPENCL=0")
    pyopencl = None
else:
    try:
        import pyopencl
    except ImportError:
        logger.warning("Unable to import pyOpenCl. Please install it from: http://pypi.python.org/pypi/pyopencl")
        pyopencl = None
        mf = None
    else:
        import pyopencl.array as array
        mf = pyopencl.mem_flags


FLOP_PER_CORE = {"GPU": 64,  # GPU, Fermi at least perform 64 flops per cycle/multicore, G80 were at 24 or 48 ...
                 "CPU": 4,  # CPU, at least intel's have 4 operation per cycle
                 "ACC": 8}  # ACC: the Xeon-phi (MIC) appears to be able to process 8 Flops per hyperthreaded-core

# Sources : https://en.wikipedia.org/wiki/CUDA
NVIDIA_FLOP_PER_CORE = {(1, 0): 24,  # Guessed !
                        (1, 1): 24,  # Measured on G98 [Quadro NVS 295]
                        (1, 2): 24,  # Guessed !
                        (1, 3): 24,  # measured on a GT285 (GT200)
                        (2, 0): 64,  # Measured on a 580 (GF110)
                        (2, 1): 96,  # Measured on Quadro2000 GF106GL
                        (3, 0): 384,  # Guessed!
                        (3, 5): 384,  # Measured on K20
                        (3, 7): 384,  # K80: Guessed!
                        (5, 0): 256,  # Maxwell 4 warps/SM 2 flops/ CU
                        (5, 2): 256,  # Titan-X
                        (5, 3): 256,  # TX1
                        (6, 0): 128,  # GP100
                        (6, 1): 128,  # GP104
                        (6, 2): 128,  # ?
                        (7, 0): 256,  # Volta ?
                        (7, 1): 256,  # Volta ?
                        }

AMD_FLOP_PER_CORE = 160  # Measured on a M7820 10 core, 700MHz 1120GFlops


class Device(object):
    """
    Simple class that contains the structure of an OpenCL device
    """
    def __init__(self, name="None", dtype=None, version=None, driver_version=None,
                 extensions="", memory=None, available=None,
                 cores=None, frequency=None, flop_core=None, idx=0, workgroup=1):
        """
        Simple container with some important data for the OpenCL device description.

        :param name: name of the device
        :param dtype: device type: CPU/GPU/ACC...
        :param version: driver version
        :param driver_version:
        :param extensions: List of opencl extensions
        :param memory: maximum memory available on the device
        :param available: is the device deactivated or not
        :param cores: number of SM/cores
        :param frequency: frequency of the device
        :param flop_cores: Flopating Point operation per core per cycle
        :param idx: index of the device within the platform
        :param workgroup: max workgroup size
        """
        self.name = name.strip()
        self.type = dtype
        self.version = version
        self.driver_version = driver_version
        self.extensions = extensions.split()
        self.memory = memory
        self.available = available
        self.cores = cores
        self.frequency = frequency
        self.id = idx
        self.max_work_group_size = workgroup
        if not flop_core:
            flop_core = FLOP_PER_CORE.get(dtype, 1)
        if cores and frequency:
            self.flops = cores * frequency * flop_core
        else:
            self.flops = flop_core

    def __repr__(self):
        return "%s" % self.name

    def pretty_print(self):
        """
        Complete device description

        :return: string
        """
        lst = ["Name\t\t:\t%s" % self.name,
               "Type\t\t:\t%s" % self.type,
               "Memory\t\t:\t%.3f MB" % (self.memory / 2.0 ** 20),
               "Cores\t\t:\t%s CU" % self.cores,
               "Frequency\t:\t%s MHz" % self.frequency,
               "Speed\t\t:\t%.3f GFLOPS" % (self.flops / 1000.),
               "Version\t\t:\t%s" % self.version,
               "Available\t:\t%s" % self.available]
        return os.linesep.join(lst)


class Platform(object):
    """
    Simple class that contains the structure of an OpenCL platform
    """
    def __init__(self, name="None", vendor="None", version=None,
        extensions=None, idx=0):
        """
        Class containing all descriptions of a platform and all devices
        description within that platform.

        :param name: platform name
        :param vendor: name of the brand/vendor
        :param version:
        :param extension: list of the extension provided by the platform to all
                          of its devices
        :param idx: index of the platform
        """
        self.name = name.strip()
        self.vendor = vendor.strip()
        self.version = version
        self.extensions = extensions.split()
        self.devices = []
        self.id = idx

    def __repr__(self):
        return "%s" % self.name

    def add_device(self, device):
        """
        Add new device to the platform

        :param device: Device instance
        """
        self.devices.append(device)

    def get_device(self, key):
        """
        Return a device according to key

        :param key: identifier for a device, either it's id (int) or it's name
        :type key: int or str
        """
        out = None
        try:
            devid = int(key)
        except ValueError:
            for a_dev in self.devices:
                if a_dev.name == key:
                    out = a_dev
        else:
            if len(self.devices) > devid > 0:
                out = self.devices[devid]
        return out


def _measure_workgroup_size(device_or_context, fast=False):
    """Mesure the maximal work group size of the given device

    :param device: instance of pyopencl.Device or pyopencl.Context or 2-tuple
                   (platformid,deviceid)
    :param fast: ask the kernel the valid value, don't probe it
    :return: maximum size for the workgroup
    """
    if isinstance(device_or_context, pyopencl.Device):
        ctx = pyopencl.Context(devices=[device_or_context])
        device = device_or_context
    elif isinstance(device_or_context, pyopencl.Context):
        ctx = device_or_context
        device = device_or_context.devices[0]
    elif isinstance(device_or_context, (tuple, list)) and len(device_or_context) == 2:
        ctx = ocl.create_context(platformid=device_or_context[0],
                                 deviceid=device_or_context[1])
        device = ctx.devices[0]
    else:
        raise RuntimeError("""given parameter device_or_context is not an
            instanciation of a device or a context""")
    shape = device.max_work_group_size
    # get the context

    assert ctx is not None
    queue = pyopencl.CommandQueue(ctx)

    max_valid_wg = 1
    data = numpy.random.random(shape).astype(numpy.float32)
    d_data = pyopencl.array.to_device(queue, data)
    d_data_1 = pyopencl.array.zeros_like(d_data) + 1

    program = pyopencl.Program(ctx, get_opencl_code("addition")).build()
    if fast:
        max_valid_wg = program.addition.get_work_group_info(pyopencl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    else:
        maxi = int(round(numpy.log2(shape)))
        for i in range(maxi + 1):
            d_res = pyopencl.array.empty_like(d_data)
            wg = 1 << i
            try:
                evt = program.addition(queue, (shape,), (wg,),
                       d_data.data, d_data_1.data, d_res.data, numpy.int32(shape))
                evt.wait()
            except Exception as error:
                logger.info("%s on device %s for WG=%s/%s" , error, device.name, wg, shape)
                program = queue = d_res = d_data_1 = d_data = None
                break
            else:
                res = d_res.get()
                good = numpy.allclose(res, data + 1)
                if good:
                    if wg > max_valid_wg:
                        max_valid_wg = wg
                else:
                    logger.warning("ArithmeticError on %s for WG=%s/%s", wg, device.name, shape)

    return max_valid_wg


class OpenCL(object):
    """
    Simple class that wraps the structure ocl_tools_extended.h

    This is a static class.
    ocl should be the only instance and shared among all python modules.
    """
    def _is_nvidia_gpu(vendor, devtype) : return (vendor == "NVIDIA Corporation") and (devtype == "GPU")
    
    platforms = []
    nb_devices = 0
    context_cache = {}  # key: 2-tuple of int, value: context
    if pyopencl:
        platform = device = pypl = devtype = extensions = pydev = None
        for idx, platform in enumerate(pyopencl.get_platforms()):
            pypl = Platform(platform.name, platform.vendor, platform.version, platform.extensions, idx)
            for idd, device in enumerate(platform.get_devices()):
                ####################################################
                # Nvidia does not report int64 atomics (we are using) ...
                # this is a hack around as any nvidia GPU with double-precision supports int64 atomics
                ####################################################
                extensions = device.extensions
                if (pypl.vendor == "NVIDIA Corporation") and ('cl_khr_fp64' in extensions):
                                extensions += ' cl_khr_int64_base_atomics cl_khr_int64_extended_atomics'
                try:
                    devtype = pyopencl.device_type.to_string(device.type).upper()
                except ValueError:
                    # pocl does not describe itself as a CPU !
                    devtype = "CPU"
                if len(devtype) > 3:
                    devtype = devtype[:3]
                if _is_nvidia_gpu(pypl.vendor, devtype) and "compute_capability_major_nv" in dir(device):
                    comput_cap = device.compute_capability_major_nv, device.compute_capability_minor_nv
                    flop_core = NVIDIA_FLOP_PER_CORE.get(comput_cap, min(NVIDIA_FLOP_PER_CORE.values()))
                elif (pypl.vendor == "Advanced Micro Devices, Inc.") and (devtype == "GPU"):
                    flop_core = AMD_FLOP_PER_CORE
                elif devtype == "CPU":
                    flop_core = FLOP_PER_CORE.get(devtype, 1)
                else:
                    flop_core = 1
                workgroup = device.max_work_group_size
                if (devtype == "CPU") and (pypl.vendor == "Apple"):
                    logger.warning("For Apple's OpenCL on CPU: Measuring actual valid max_work_goup_size.")
                    workgroup = _measure_workgroup_size(device, fast=True)
                if (devtype == "GPU") and os.environ.get("GPU") == "False":
                    # Environment variable to disable GPU devices
                    continue
                pydev = Device(device.name, devtype, device.version, device.driver_version, extensions,
                               device.global_mem_size, bool(device.available), device.max_compute_units,
                               device.max_clock_frequency, flop_core, idd, workgroup)
                pypl.add_device(pydev)
                nb_devices += 1
            platforms.append(pypl)
        del platform, device, pypl, devtype, extensions, pydev

    def __repr__(self):
        out = ["OpenCL devices:"]
        for platformid, platform in enumerate(self.platforms):
            deviceids = ["(%s,%s) %s" % (platformid, deviceid, dev.name) \
                for deviceid, dev in enumerate(platform.devices)]
            out.append("[%s] %s: " % (platformid, platform.name) + ", ".join(deviceids))
        return os.linesep.join(out)

    def get_platform(self, key):
        """
        Return a platform according

        :param key: identifier for a platform, either an Id (int) or it's name
        :type key: int or str
        """
        out = None
        try:
            platid = int(key)
        except ValueError:
            for a_plat in self.platforms:
                if a_plat.name == key:
                    out = a_plat
        else:
            if len(self.platforms) > platid > 0:
                out = self.platforms[platid]
        return out

    def select_device(self, dtype="ALL", memory=None, extensions=None, best=True, **kwargs):
        """
        Select a device based on few parameters (at the end, keep the one with most memory)

        :param type: "gpu" or "cpu" or "all" ....
        :param memory: minimum amount of memory (int)
        :param extensions: list of extensions to be present
        :param best: shall we look for the
        """
        if extensions is None:
            extensions = []
        if "type" in kwargs:
            dtype = kwargs["type"].upper()
        else:
            dtype = dtype.upper()
        if len(dtype) > 3:
            dtype = dtype[:3]
        best_found = None
        for platformid, platform in enumerate(self.platforms):
            for deviceid, device in enumerate(platform.devices):
                if (dtype in ["ALL", "DEF"]) or (device.type == dtype):
                    if (memory is None) or (memory <= device.memory):
                        found = True
                        for ext in extensions:
                            if ext not in device.extensions:
                                found = False
                        if found:
                            if not best:
                                return platformid, deviceid
                            else:
                                if not best_found:
                                    best_found = platformid, deviceid, device.flops
                                elif best_found[2] < device.flops:
                                    best_found = platformid, deviceid, device.flops
        if best_found:
            return best_found[0], best_found[1]

    def create_context(self, devicetype="ALL", useFp64=False, platformid=None,
        deviceid=None, cached=True):
        """
        Choose a device and initiate a context.

        Devicetypes can be GPU,gpu,CPU,cpu,DEF,ACC,ALL.
        Suggested are GPU,CPU.
        For each setting to work there must be such an OpenCL device and properly installed.
        E.g.: If Nvidia driver is installed, GPU will succeed but CPU will fail.
              The AMD SDK kit is required for CPU via OpenCL.
        :param devicetype: string in ["cpu","gpu", "all", "acc"]
        :param useFp64: boolean specifying if double precision will be used
        :param platformid: integer
        :param devid: integer
        :return: OpenCL context on the selected device
        """
        if (platformid is not None) and (deviceid is not None):
            platformid = int(platformid)
            deviceid = int(deviceid)
        else:
            if useFp64:
                ids = ocl.select_device(type=devicetype, extensions=["cl_khr_int64_base_atomics"])
            else:
                ids = ocl.select_device(dtype=devicetype)
            if ids:
                platformid = ids[0]
                deviceid = ids[1]
        if (platformid is not None) and (deviceid is not None):
            if (platformid, deviceid) in self.context_cache:
                ctx = self.context_cache[(platformid, deviceid)]
            else:
                ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
                if cached:
                    self.context_cache[(platformid, deviceid)] = ctx
        else:
            logger.warning("Last chance to get an OpenCL device ... probably not the one requested")
            ctx = pyopencl.create_some_context(interactive=False)
        return ctx

    def device_from_context(self, context):
        """
        Retrieves the Device from the context

        :param context: OpenCL context
        :return: instance of Device
        """
        odevice = context.devices[0]
        oplat = odevice.platform
        device_id = oplat.get_devices().index(odevice)
        platform_id = pyopencl.get_platforms().index(oplat)
        return self.platforms[platform_id].devices[device_id]

if pyopencl:
    ocl = OpenCL()
    if ocl.nb_devices == 0:
        ocl = None
else:
    ocl = None


def release_cl_buffers(cl_buffers):
    """
    :param cl_buffer: the buffer you want to release
    :type cl_buffer: dict(str, pyopencl.Buffer)

    This method release the memory of the buffers store in the dict
    """
    for key, buffer_ in cl_buffers.items():
        if buffer_ is not None:
            if isinstance(buffer_, pyopencl.array.Array):
                try:
                    buffer_.data.release()
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s", key)
            else:
                try:
                    buffer_.release()
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s", key)
            cl_buffers[key] = None
    return cl_buffers


def allocate_cl_buffers(buffers, device=None, context=None):
    """
    :param buffers: the buffers info use to create the pyopencl.Buffer
    :type buffer: list(std, flag, numpy.dtype, int)
    :return: a dict containing the instanciated pyopencl.Buffer
    :rtype: dict(str, pyopencl.Buffer)

    This method instanciate the pyopencl.Buffer from the buffers
    description.
    """
    mem = {}
    if device is None:
        device = ocl.device_from_context(context)

    # check if enough memory is available on the device
    ualloc = 0
    for _, _, dtype, size in buffers:
        ualloc += numpy.dtype(dtype).itemsize * size
    memory = device.memory
    logger.info("%.3fMB are needed on device which has %.3fMB",
                ualloc / 1.0e6, memory / 1.0e6)
    if ualloc >= memory:
        memError = "Fatal error in allocate_buffers."
        memError += "Not enough device memory for buffers"
        memError += "(%lu requested, %lu available)" % (ualloc, memory)
        raise MemoryError(memError)  # noqa

    # do the allocation
    try:
        for name, flag, dtype, size in buffers:
            mem[name] = pyopencl.Buffer(context, flag,
                                        numpy.dtype(dtype).itemsize * size)
    except pyopencl.MemoryError as error:
        release_cl_buffers(mem)
        raise MemoryError(error)

    return mem


def measure_workgroup_size(device):
    """Measure the actual size of the workgroup

    :param device: device or context or 2-tuple with indexes
    :return: the actual measured workgroup size

    if device is "all", returns a dict with all devices with their ids as keys.
    """
    if (ocl is None) or (device is None):
        return None

    if isinstance(device, tuple) and (len(device) == 2):
        # this is probably a tuple (platformid, deviceid)
        device = ocl.create_context(platformid=device[0], deviceid=device[1])

    if device == "all":
        res = {}
        for pid, platform in enumerate(ocl.platforms):
            for did, _devices in enumerate(platform.devices):
                tup = (pid, did)
                res[tup] = measure_workgroup_size(tup)
    else:
        res = _measure_workgroup_size(device)
    return res


def kernel_workgroup_size(program, kernel):
    """Extract the compile time maximum workgroup size

    :param program: OpenCL program
    :param kernel: kernel or name of the kernel
    :return: the maximum acceptable workgroup size for the given kernel
    """
    assert isinstance(program, pyopencl.Program)
    if not isinstance(kernel, pyopencl.Kernel):
        kernel_name = kernel
        assert kernel in (k.function_name for k in program.all_kernels()), "the kernel exists"
        kernel = program.__getattr__(kernel_name)

    device = program.devices[0]
    query_wg = pyopencl.kernel_work_group_info.WORK_GROUP_SIZE
    return kernel.get_work_group_info(query_wg, device)

class OpenclProcessing(object):
    """Abstract class for different types of OpenCL processing.

    This class provides:
    * Generation of the context, queues, profiling mode
    * Additional function to allocate/free all buffers declared as static
        attributes of the class
    * Functions to compile kernels, cache them and clean them
    * helper functions to clone the object
    """
    # The list of buffers to be associated to this class, this is an example
    buffers = [BufferDescription("output", None, numpy.float32, 10),
              ]
    # list of kernel source files to be concatenated before compilation of the program
    kernel_files = []

    def __init__(self, ctx=None, devicetype="all", platformid=None,
                 deviceid=None, block_size=None, profile=False):
        """Constructor of the abstract OpenCL processing class

        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by
                           clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the
                           outpcome of the compilation
        :param profile: switch on profiling to be able to profile at the kernel
                        level, store profiling elements (makes code slower)
        """
        self.sem = threading.Semaphore()
        self.profile = bool(profile)
        self.events = []  # List with of EventDescription, kept for profiling
        self.cl_mem = {}  # dict with all buffer allocated
        self.cl_program = None  # The actual OpenCL program
        self.cl_kernel_args = {}  # dict with all kernel arguments
        if ctx:
            self.ctx = ctx
        else:
            self.ctx = ocl.create_context(devicetype=devicetype,
                                          platformid=platformid,
                                          deviceid=deviceid)

        device_name = self.ctx.devices[0].name.strip()
        platform_name = self.ctx.devices[0].platform.name.strip()
        platform = ocl.get_platform(platform_name)
        self.device = platform.get_device(device_name)
        # self.device = platform.id, device.id

        if profile:
            self.queue = pyopencl.CommandQueue(
                self.ctx,
               properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = pyopencl.CommandQueue(self.ctx)

        self.block_size = block_size

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
        self.cl_mem = allocate_cl_buffers(buffers, self.device, self.ctx)

    def free_buffers(self):
        """free all memory allocated on the device
        """
        self.cl_mem = release_cl_buffers(self.cl_mem)

    def compile_kernels(self, kernel_files=None, compile_options=None):
        """Call the OpenCL compiler

        :param kernel_files: list of path to the kernel
        (by default use the one declared in the class)
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
