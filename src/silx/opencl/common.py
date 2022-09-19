#!/usr/bin/env python
#
#    Project: S I L X project
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2021 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "2012-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/09/2021"
__status__ = "stable"
__all__ = ["ocl", "pyopencl", "mf", "release_cl_buffers", "allocate_cl_buffers",
           "measure_workgroup_size", "kernel_workgroup_size"]

import os
import logging

import numpy

from .utils import get_opencl_code

logger = logging.getLogger(__name__)

if os.environ.get("SILX_OPENCL") in ["0", "False"]:
    logger.info("Use of OpenCL has been disabled from environment variable: SILX_OPENCL=0")
    pyopencl = None
else:
    try:
        import pyopencl
    except ImportError:
        logger.warning("Unable to import pyOpenCl. Please install it from: https://pypi.org/project/pyopencl")
        pyopencl = None
    else:
        try:
            pyopencl.get_platforms()
        except pyopencl.LogicError:
            logger.warning("The module pyOpenCL has been imported but can't be used here")
            pyopencl = None
        else:
            import pyopencl.array as array
            mf = pyopencl.mem_flags

if pyopencl is None:

    # Define default mem flags
    class mf(object):
        WRITE_ONLY = 1
        READ_ONLY = 1
        READ_WRITE = 1

FLOP_PER_CORE = {"GPU": 64,  # GPU, Fermi at least perform 64 flops per cycle/multicore, G80 were at 24 or 48 ...
                 "CPU": 4,  # CPU, at least intel's have 4 operation per cycle
                 "ACC": 8}  # ACC: the Xeon-phi (MIC) appears to be able to process 8 Flops per hyperthreaded-core

# Sources : https://en.wikipedia.org/wiki/CUDA
NVIDIA_FLOP_PER_CORE = {(1, 0): 24,   # Guessed !
                        (1, 1): 24,   # Measured on G98 [Quadro NVS 295]
                        (1, 2): 24,   # Guessed !
                        (1, 3): 24,   # measured on a GT285 (GT200)
                        (2, 0): 64,   # Measured on a 580 (GF110)
                        (2, 1): 96,   # Measured on Quadro2000 GF106GL
                        (3, 0): 384,  # Guessed!
                        (3, 5): 384,  # Measured on K20
                        (3, 7): 384,  # K80: Guessed!
                        (5, 0): 256,  # Maxwell 4 warps/SM 2 flops/ CU
                        (5, 2): 256,  # Titan-X
                        (5, 3): 256,  # TX1
                        (6, 0): 128,  # GP100
                        (6, 1): 128,  # GP104
                        (6, 2): 128,  # ?
                        (7, 0): 128,  # Volta  # measured on Telsa V100
                        (7, 2): 128,  # Volta  ? 
                        (7, 5): 128,  # Turing # measured on RTX 6000
                        (8, 0): 128,  # Ampere # measured on Tesla A100
                        (8, 6): 256,  # Ampere # measured on RTX A5000
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
        :param flop_core: Flopating Point operation per core per cycle
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

    def set_unavailable(self):
        """Use this method to flag a faulty device
        """
        self.available = False


class Platform(object):
    """
    Simple class that contains the structure of an OpenCL platform
    """

    def __init__(self, name="None", vendor="None", version=None, extensions=None, idx=0):
        """
        Class containing all descriptions of a platform and all devices description within that platform.

        :param name: platform name
        :param vendor: name of the brand/vendor
        :param version:
        :param extensions: list of the extension provided by the platform to all of its devices
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

    DEPRECATED since not perfectly correct !

    :param device_or_context: instance of pyopencl.Device or pyopencl.Context
                    or 2-tuple (platformid,deviceid)
    :param fast: ask the kernel the valid value, don't probe it
    :return: maximum size for the workgroup
    """
    if isinstance(device_or_context, pyopencl.Device):
        try:
            ctx = pyopencl.Context(devices=[device_or_context])
        except pyopencl._cl.LogicError as error:
            platform = device_or_context.platform
            platformid = pyopencl.get_platforms().index(platform)
            deviceid = platform.get_devices().index(device_or_context)
            ocl.platforms[platformid].devices[deviceid].set_unavailable()
            raise RuntimeError("Unable to create context on %s/%s: %s" % (platform, device_or_context, error))
        else:
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
    d_data_1 = pyopencl.array.empty_like(d_data)
    d_data_1.fill(numpy.float32(1.0))

    program = pyopencl.Program(ctx, get_opencl_code("addition")).build()
    if fast:
        max_valid_wg = program.addition.get_work_group_info(pyopencl.kernel_work_group_info.WORK_GROUP_SIZE, device)
    else:
        maxi = int(round(numpy.log2(shape)))
        for i in range(maxi + 1):
            d_res = pyopencl.array.empty_like(d_data)
            wg = 1 << i
            try:
                evt = program.addition(
                    queue, (shape,), (wg,),
                    d_data.data, d_data_1.data, d_res.data, numpy.int32(shape))
                evt.wait()
            except Exception as error:
                logger.info("%s on device %s for WG=%s/%s", error, device.name, wg, shape)
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


def _is_nvidia_gpu(vendor, devtype):
    return (vendor == "NVIDIA Corporation") and (devtype == "GPU")


class OpenCL(object):
    """
    Simple class that wraps the structure ocl_tools_extended.h

    This is a static class.
    ocl should be the only instance and shared among all python modules.
    """

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
                    if "GPU" in devtype:
                        devtype = "GPU"
                    elif "ACC" in devtype:
                        devtype = "ACC"
                    elif "CPU" in devtype:
                        devtype = "CPU"
                    else:
                        devtype = devtype[:3]
                if _is_nvidia_gpu(device.vendor, devtype) and ("compute_capability_major_nv" in dir(device)):
                    try:
                        comput_cap = device.compute_capability_major_nv, device.compute_capability_minor_nv
                    except pyopencl.LogicError:
                        flop_core = FLOP_PER_CORE["GPU"]
                    else:
                        flop_core = NVIDIA_FLOP_PER_CORE.get(comput_cap, FLOP_PER_CORE["GPU"])
                elif (pypl.vendor == "Advanced Micro Devices, Inc.") and (devtype == "GPU"):
                    flop_core = AMD_FLOP_PER_CORE
                elif devtype == "CPU":
                    flop_core = FLOP_PER_CORE.get(devtype, 1)
                else:
                    flop_core = 1
                workgroup = device.max_work_group_size
                if (devtype == "CPU") and (pypl.vendor == "Apple"):
                    logger.info("For Apple's OpenCL on CPU: Measuring actual valid max_work_goup_size.")
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
            deviceids = ["(%s,%s) %s" % (platformid, deviceid, dev.name)
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

        :param dtype: "gpu" or "cpu" or "all" ....
        :param memory: minimum amount of memory (int)
        :param extensions: list of extensions to be present
        :param best: shall we look for the
        :returns: A tuple of plateform ID and device ID, else None if nothing
            found
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
                if not device.available:
                    continue
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

        # Nothing found
        return None

    def create_context(self, devicetype="ALL", useFp64=False, platformid=None,
                       deviceid=None, cached=True, memory=None, extensions=None):
        """
        Choose a device and initiate a context.

        Devicetypes can be GPU,gpu,CPU,cpu,DEF,ACC,ALL.
        Suggested are GPU,CPU.
        For each setting to work there must be such an OpenCL device and properly installed.
        E.g.: If Nvidia driver is installed, GPU will succeed but CPU will fail.
              The AMD SDK kit is required for CPU via OpenCL.
        :param devicetype: string in ["cpu","gpu", "all", "acc"]
        :param useFp64: boolean specifying if double precision will be used: deprecated use extensions=["cl_khr_fp64"]
        :param platformid: integer
        :param deviceid: integer
        :param cached: True if we want to cache the context
        :param memory: minimum amount of memory of the device
        :param extensions: list of extensions to be present
        :return: OpenCL context on the selected device
        """
        if extensions is None:
            extensions = []
        if useFp64:
            logger.warning("Deprecation: please select your device using the extension name!, i.e. extensions=['cl_khr_fp64']")
            extensions.append('cl_khr_fp64')

        if (platformid is not None) and (deviceid is not None):
            platformid = int(platformid)
            deviceid = int(deviceid)
        elif "PYOPENCL_CTX" in os.environ:
            pyopencl_ctx = [int(i) if i.isdigit() else 0 for i in os.environ["PYOPENCL_CTX"].split(":")]
            pyopencl_ctx += [0] * (2 - len(pyopencl_ctx))  # pad with 0
            platformid, deviceid = pyopencl_ctx
        else:
            ids = ocl.select_device(type=devicetype, extensions=extensions)
            if ids:
                platformid, deviceid = ids
        ctx = None
        if (platformid is not None) and (deviceid is not None):
            if (platformid, deviceid) in self.context_cache:
                ctx = self.context_cache[(platformid, deviceid)]
            else:
                try:
                    ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
                except pyopencl._cl.LogicError as error:
                    self.platforms[platformid].devices[deviceid].set_unavailable()
                    logger.warning("Unable to create context on %s/%s: %s", platformid, deviceid, error)
                    ctx = None
                else:
                    if cached:
                        self.context_cache[(platformid, deviceid)] = ctx
        if ctx is None:
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
    :param cl_buffers: the buffer you want to release
    :type cl_buffers: dict(str, pyopencl.Buffer)

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
    :type buffers: list(std, flag, numpy.dtype, int)
    :param device: one of the context device
    :param context: opencl contextdevice
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


def allocate_texture(ctx, shape, hostbuf=None, support_1D=False):
    """
    Allocate an OpenCL image ("texture").

    :param ctx: OpenCL context
    :param shape: Shape of the image. Note that pyopencl and OpenCL < 1.2
        do not support 1D images, so 1D images are handled as 2D with one row
    :param support_1D: force the image to be 1D if the shape has only one dim
    """
    if len(shape) == 1 and not(support_1D):
        shape = (1,) + shape
    return pyopencl.Image(
        ctx,
        pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.USE_HOST_PTR,
        pyopencl.ImageFormat(
            pyopencl.channel_order.INTENSITY,
            pyopencl.channel_type.FLOAT
        ),
        hostbuf=numpy.zeros(shape[::-1], dtype=numpy.float32)
    )


def check_textures_availability(ctx):
    """
    Check whether textures are supported on the current OpenCL context.

    :param ctx: OpenCL context
    """
    try:
        dummy_texture = allocate_texture(ctx, (16, 16))
        # Need to further access some attributes (pocl)
        dummy_height = dummy_texture.height
        textures_available = True
        del dummy_texture, dummy_height
    except (pyopencl.RuntimeError, pyopencl.LogicError):
        textures_available = False
    # Nvidia Fermi GPUs (compute capability 2.X) do not support opencl read_imagef
    # There is no way to detect this until a kernel is compiled
    try:
        cc = ctx.devices[0].compute_capability_major_nv
        textures_available &= (cc >= 3)
    except (pyopencl.LogicError, AttributeError):  # probably not a Nvidia GPU
        pass
    #
    return textures_available


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


def query_kernel_info(program, kernel, what="WORK_GROUP_SIZE"):
    """Extract the compile time information from a kernel

    :param program: OpenCL program
    :param kernel: kernel or name of the kernel
    :param what: what is the query about ?
    :return: int or 3-int for the workgroup size.
    
    Possible information available are:
    * 'COMPILE_WORK_GROUP_SIZE': Returns the work-group size specified inside the kernel (__attribute__((reqd_work_gr oup_size(X, Y, Z))))
    * 'GLOBAL_WORK_SIZE': maximum global size that can be used to execute a kernel  #OCL2.1!
    * 'LOCAL_MEM_SIZE': amount of local memory in bytes being used by the kernel
    * 'PREFERRED_WORK_GROUP_SIZE_MULTIPLE': preferred multiple of workgroup size for launch. This is a performance hint.
    * 'PRIVATE_MEM_SIZE' Returns the minimum amount of private memory, in bytes, used by each workitem in the kernel
    * 'WORK_GROUP_SIZE': maximum work-group size that can be used to execute a kernel on a specific device given by device
    
    Further information on:
    https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clGetKernelWorkGroupInfo.html
    
    """
    assert isinstance(program, pyopencl.Program)
    if not isinstance(kernel, pyopencl.Kernel):
        kernel_name = kernel
        assert kernel in (k.function_name for k in program.all_kernels()), "the kernel exists"
        kernel = program.__getattr__(kernel_name)

    device = program.devices[0]
    query_wg = getattr(pyopencl.kernel_work_group_info, what)
    return kernel.get_work_group_info(query_wg, device)


def kernel_workgroup_size(program, kernel):
    """Extract the compile time maximum workgroup size

    :param program: OpenCL program
    :param kernel: kernel or name of the kernel
    :return: the maximum acceptable workgroup size for the given kernel
    """
    return query_kernel_info(program, kernel, what="WORK_GROUP_SIZE")
