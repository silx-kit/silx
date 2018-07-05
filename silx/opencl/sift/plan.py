#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
Contains a class for creating a plan, allocating arrays, compiling kernels and
other things like that...
to calculate SIFT keypoints and descriptors.


This code implements the SIFT algorithm
The SIFT algorithm belongs to the University of British Columbia. It is
protected by patent US6711293. If you are on a country where this pattent
applies (like the USA), please check if you are allowed to use it. The
University of British Columbia does not require a license for its use for
non-commercial research applications.


This algorithm is patented: U.S. Patent 6,711,293:
"Method and apparatus for identifying scale invariant features in an image and use of same for locating an object in an image",
David Lowe's patent for the SIFT algorithm,  Mar. 8, 1999. 
It is due to expire in March 2019. 
"""

from __future__ import division, print_function, with_statement

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/07/2018"
__status__ = "production"

import os
import time
import math
import logging
import gc
import numpy
from collections import OrderedDict
from .param import par
from silx.opencl import ocl, pyopencl, kernel_workgroup_size
from silx.opencl.utils import get_opencl_code, nextpower
from ..processing import OpenclProcessing, BufferDescription
from .utils import calc_size, kernel_size
logger = logging.getLogger(__name__)


class SiftPlan(OpenclProcessing):
    """This class implements a way to calculate SIFT keypoints.


    How to calculate a set of SIFT keypoint on an image::

        siftp = sift.SiftPlan(img.shape,img.dtype,devicetype="GPU")
        kp = siftp.keypoints(img)

    kp is a nx132 array. the second dimension is composed of x,y, scale and angle as well as 128 floats describing the keypoint

    This SIFT algorithm is patented: U.S. Patent 6,711,293:
    "Method and apparatus for identifying scale invariant features in an image and use of same for locating an object in an image",
    """
    kernels_max_wg_size = {"orientation_cpu": 1,
                           "orientation_gpu": 128,
                           "descriptor_gpu1": (8, 4, 4),
                           "descriptor_gpu2": (8, 8, 8),
                           "descriptor_cpu": (1,),
                           }

    converter = {numpy.dtype(numpy.uint8): "u8_to_float",
                 numpy.dtype(numpy.uint16): "u16_to_float",
                 numpy.dtype(numpy.uint32): "u32_to_float",
                 numpy.dtype(numpy.uint64): "u64_to_float",
                 numpy.dtype(numpy.int32): "s32_to_float",
                 numpy.dtype(numpy.int64): "s64_to_float",
                 # numpy.dtype(numpy.float64): "double_to_float",
                 }

    sigmaRatio = 2.0 ** (1.0 / par.Scales)
    PIX_PER_KP = 10  # pre_allocate buffers for keypoints
    dtype_kp = numpy.dtype([('x', numpy.float32),
                            ('y', numpy.float32),
                            ('scale', numpy.float32),
                            ('angle', numpy.float32),
                            ('desc', (numpy.uint8, 128))
                            ])

    def __init__(self, shape=None, dtype=None, template=None,
                 PIX_PER_KP=None, init_sigma=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, memory=None, profile=False):
        """
        Constructor of the class

        :param shape: shape of the input image
        :param dtype: data type of the input image
        :param devicetype: can be 'CPU' or 'GPU'
        :param template: extract shape and dtype from an image
        :param profile: collect timing info
        :param device: 2-tuple of integers
        :param PIX_PER_KP: number of keypoint pre-allocated: 1 for 10 pixel
        :param block_size: set to 1 under macosX on CPU
        :param context: provide an external context
        :param init_sigma: blurring width, you should have good reasons to modify 
                            the 1.6 default value...
        """
        self.kernels_max_wg_size = self.__class__.kernels_max_wg_size.copy()
        if template is not None:
            self.shape = template.shape
            self.dtype = template.dtype
        else:
            self.shape = shape
            self.dtype = numpy.dtype(dtype)
        if len(self.shape) == 3:
            self.RGB = True
            self.shape = self.shape[:2]
        elif len(self.shape) == 2:
            self.RGB = False
        else:
            raise RuntimeError("Unable to process image of shape %s" % (tuple(self.shape,)))
        if PIX_PER_KP:
            self.PIX_PER_KP = int(PIX_PER_KP)

        self.kpsize = None

        if init_sigma is None:
            init_sigma = par.InitSigma
        # no test on the values, just make sure it is a float
        self._init_sigma = float(init_sigma)
        memory = self._calc_memory(block_size)
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, profile=profile,
                                  memory=memory)
        # TODO WORKGROUP ....
        self.kernels_wg = {}
        self.scales = []  # in XY order
        self.procsize = []  # same as  procsize but with dimension in (X,Y) not (slow, fast)
        self.wgsize = []

        self.octave_max = None
        self.red_size = None
        self._calc_scales()
        self.LOW_END = 0
        self._calc_workgroups()
        self.compile_kernels()
        self._allocate_buffers()
        self.cnt = numpy.empty(1, dtype=numpy.int32)
        if "CPU" in self.device.type:
            self.USE_CPU = True
        else:
            self.USE_CPU = False

    def _calc_scales(self):
        """
        Nota scales are in XY order
        """
        shape = self.shape[-1::-1]
        self.scales = [tuple(numpy.int32(i) for i in shape)]
        min_size = 2 * par.BorderDist + 2
        while min(shape) > min_size:
            shape = tuple(numpy.int32(i // 2) for i in shape)
            self.scales.append(shape)
        self.scales.pop()
        self.octave_max = len(self.scales)

    def _calc_memory(self, block_size=None):
        """
        Estimates the memory footprint of all buffer to ensure it fits on the device
        """
        block_size = int(block_size) if block_size else 4096  # upper limit

        # Just the context + kernel takes about 75MB on the GPU
        memory = 75 * 2 ** 20
        size_of_float = numpy.dtype(numpy.float32).itemsize
        size_of_input = numpy.dtype(self.dtype).itemsize
        # raw images:
        size = self.shape[0] * self.shape[1]
        memory += size * size_of_input  # initial_image (no raw_float)
        if self.RGB:
            memory += 2 * size * (size_of_input)  # one of three was already counted
        nr_blur = par.Scales + 3  # 3 blurs and 2 tmp
        nr_dogs = par.Scales + 2
        memory += size * (nr_blur + nr_dogs) * size_of_float

        self.kpsize = int(self.shape[0] * self.shape[1] // self.PIX_PER_KP)  # Is the number of kp independant of the octave ? int64 causes problems with pyopencl
        memory += self.kpsize * size_of_float * 4 * 2  # those are array of float4 to register keypoints, we need two of them
        memory += self.kpsize * 128  # stores the descriptors: 128 unsigned chars
        memory += 4  # keypoint index Counter
        wg_float = min(block_size, numpy.sqrt(self.shape[0] * self.shape[1]))
        self.red_size = nextpower(wg_float)
        memory += 4 * 2 * self.red_size  # temporary storage for reduction

        ########################################################################
        # Calculate space for gaussian kernels
        ########################################################################
        curSigma = 1.0 if par.DoubleImSize else 0.5
        if self._init_sigma > curSigma:
            sigma = math.sqrt(self._init_sigma ** 2 - curSigma ** 2)
            size = kernel_size(sigma, True)
            logger.debug("pre-Allocating %s float for init blur" % size)
            memory += size * size_of_float
        prevSigma = self._init_sigma
        for _ in range(par.Scales + 2):
            increase = prevSigma * math.sqrt(self.sigmaRatio ** 2 - 1.0)
            size = kernel_size(increase, True)
            logger.debug("pre-Allocating %s float for blur sigma: %s" % (size, increase))
            memory += size * size_of_float
            prevSigma *= self.sigmaRatio
        # self.memory = memory
        return memory

    def _allocate_buffers(self):
        """
        All buffers are allocated here
        """
        shape = self.shape
        buffers = [BufferDescription("min", 1, numpy.float32, None),
                   BufferDescription("max", 1, numpy.float32, None),
                   BufferDescription("255", 1, numpy.float32, None),
                   BufferDescription("cnt", 1, numpy.int32, None),
                   BufferDescription("Kp_1", (self.kpsize, 4), numpy.float32, None),
                   BufferDescription("Kp_2", (self.kpsize, 4), numpy.float32, None),
                   BufferDescription("descr", (self.kpsize, 128), numpy.uint8, None),
                   BufferDescription("descriptors", (self.kpsize, 128), numpy.uint8, None),
                   BufferDescription("tmp", shape, numpy.float32, None),
                   BufferDescription("ori", shape, numpy.float32, None),
                   BufferDescription("DoGs", (par.Scales + 2, shape[0], shape[1]), numpy.float32, None),
                   BufferDescription("max_min", (self.red_size, 2), numpy.float32, None)  # temporary buffer for max/min reduction
                   ]
        if self.dtype != numpy.float32:
            if self.RGB:
                rgbshape = self.shape[0], self.shape[1], 3
                buffers.append(BufferDescription("raw", rgbshape, self.dtype, None))
            else:
                buffers.append(BufferDescription("raw", shape, self.dtype, None))
        for scale in range(par.Scales + 3):
            buffers.append(BufferDescription("scale_%i" % scale, shape, numpy.float32, None))

        self.allocate_buffers(buffers, use_array=True)

        self.cl_mem["255"].fill(255.0)
        ########################################################################
        # Allocate space for gaussian kernels
        ########################################################################
        curSigma = 1.0 if par.DoubleImSize else 0.5
        if self._init_sigma > curSigma:
            sigma = math.sqrt(self._init_sigma ** 2 - curSigma ** 2)
            self._init_gaussian(sigma)
        prevSigma = self._init_sigma

        for _ in range(par.Scales + 2):
            increase = prevSigma * math.sqrt(self.sigmaRatio ** 2 - 1.0)
            self._init_gaussian(increase)
            prevSigma *= self.sigmaRatio

    def _init_gaussian(self, sigma):
        """Create a buffer of the right size according to the width of the gaussian ...


        :param  sigma: width of the gaussian, the length of the function will be 8*sigma + 1

        Same calculation done on CPU
        x = numpy.arange(size) - (size - 1.0) / 2.0
        gaussian = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
        gaussian /= gaussian.sum(dtype=numpy.float32)
        """
        pyopencl.enqueue_barrier(self.queue).wait()
        name = "gaussian_%s" % sigma
        size = kernel_size(sigma, True)
        wg_size = nextpower(size)

        logger.info("Allocating %s float for blur sigma: %s. wg=%s max_wg=%s", size, sigma, wg_size, self.block_size)
        wg1 = self.kernels_wg["gaussian"]
        if wg1 >= wg_size:
            gaussian_gpu = pyopencl.array.empty(self.queue, size, dtype=numpy.float32)
            pyopencl.enqueue_barrier(self.queue).wait()
            kernel = self.kernels.get_kernel("gaussian")
            shm1 = pyopencl.LocalMemory(4 * wg_size)
            shm2 = pyopencl.LocalMemory(4 * wg_size)
            evt = kernel(self.queue, (wg_size,), (wg_size,),
                         gaussian_gpu.data,
                         numpy.float32(sigma),  # const        float     sigma,
                         numpy.int32(size),  # const        int     SIZE
                         shm1, shm2)  # some shared memory
            pyopencl.enqueue_barrier(self.queue).wait()
            if self.profile:
                self.events.append(("gaussian %s" % sigma, evt))
        else:
            logger.info("Workgroup size error: gaussian wg: %s < max_work_group_size: %s",
                        wg1, self.block_size)
            # common bug on OSX when running on CPU
            x = numpy.arange(size) - (size - 1.0) / 2.0
            gaus = numpy.exp(-(x / sigma) ** 2 / 2.0).astype(numpy.float32)
            gaus /= gaus.sum(dtype=numpy.float32)
            gaussian_gpu = pyopencl.array.to_device(self.queue, gaus)

        self.cl_mem[name] = gaussian_gpu
        return gaussian_gpu

    def compile_kernels(self):
        """Call the OpenCL compiler

        TODO: use the parameters to define the compile-time constants and use 
        them all in kernels.
        """
        to_compile = ["sift",
                      "convolution",
                      "preprocess",
                      "algebra",
                      "image",
                      "gaussian",
                      "reductions",
                      "memset"]
        to_compile += list(self.kernels_max_wg_size.keys())
        compile_options = "-D WORKGROUP_SIZE=%s" % self.block_size
        try:
            OpenclProcessing.compile_kernels(self,
                                             [os.path.join("sift", kernel)
                                              for kernel in to_compile],
                                             compile_options=compile_options)
        except Exception as err:
            logger.error("error while compiling sift: %s", err)
        else:
            for kn in self.kernels.get_kernels():
                res = self.check_workgroup_size(kn)
                self.kernels_wg[kn] = min(res, self.block_size)

    def _free_kernels(self):
        """free all kernels
        """
        self.programs = {}

    def _calc_workgroups(self):
        """First try to guess the best workgroup size, then calculate all global worksize

        Nota:
        The workgroup size is limited by the device, some devices report wrong size.
        The workgroup size is limited to the 2**n below then image size (hence changes with octaves)
        The second dimension of the wg size should be large, the first small: i.e. (1,64)
        The processing size should be a multiple of  workgroup size.
        """
        device = self.ctx.devices[0]
        max_work_item_sizes = device.max_work_item_sizes
        if self.block_size:
            self.block_size = min(max_work_item_sizes[0], self.block_size)
        else:
            self.block_size = max_work_item_sizes[0]
        # MacOSX driver on CPU usually reports bad workgroup size: this is addressed in ocl
        self.block_size = min(self.block_size,
                              self.device.max_work_group_size)

        for k, v in self.kernels_max_wg_size.items():
            if isinstance(v, int):
                self.kernels_wg[k] = min(v, self.block_size)
            else:  # probably a list
                prod = numpy.prod(v)
                if prod <= self.block_size:
                    self.kernels_wg[k] = prod
                # else it is not possible to run this kernel.
                # If the kernel is not present in the dict, it should not be used.

        wg_float = min(self.block_size, numpy.sqrt(self.shape[0] * self.shape[1]))
        self.red_size = nextpower(wg_float)

        # we recalculate the shapes ...
        shape = self.shape
        min_size = 2 * par.BorderDist + 2
        while min(shape) > min_size:
            wg = (min(nextpower(shape[-1]), self.block_size), 1)
            self.wgsize.append(wg)
            self.procsize.append(calc_size(shape[-1::-1], wg))
            shape = tuple(i // 2 for i in shape)

    def keypoints(self, image, mask=None):
        """Calculates the keypoints of the image

        TODO: use a temporary list with events and use a single test at the end

        :param image: ndimage of 2D (or 3D if RGB)
        :param mask: TODO: implement a mask for sieving out the keypoints
        :return: vector of keypoint (1D numpy array)
        """
        # self.reset_timer()
        with self.sem:
            total_size = 0
            keypoints = []
            descriptors = []
            assert image.shape[:2] == self.shape
            assert image.dtype in [self.dtype, numpy.float32]
            # old versions of pyopencl do not check for data contiguity
            if not(isinstance(image, pyopencl.array.Array)) and not(image.flags["C_CONTIGUOUS"]):
                image = numpy.ascontiguousarray(image)
            t0 = time.time()

            if image.dtype == numpy.float32:
                if isinstance(image, pyopencl.array.Array):
                    evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["scale_0"].data, image.data)
                else:
                    evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["scale_0"].data, image)
                if self.profile:
                    self.events.append(("copy H->D", evt))
            elif self.dtype == numpy.float64:
                # A preprocessing kernel double_to_float exists, but is commented (RUNS ONLY ON GPU WITH FP64)
                # TODO: benchmark this kernel vs the current pure CPU format conversion with numpy.float32
                #       and uncomment it if it proves faster (dubious, because of data transfer bottleneck)
                evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["scale_0"].data, image.astype(numpy.float32))
                if self.profile:
                    self.events.append(("copy H->D", evt))
            elif (len(image.shape) == 3) and (image.dtype == numpy.uint8) and (self.RGB):
                if isinstance(image, pyopencl.array.Array):
                    evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["raw"].data, image.data)
                else:
                    evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["raw"].data, image)
                if self.profile:
                    self.events.append(("copy H->D", evt))

                evt = self.kernels.get_kernel("rgb_to_float")(self.queue, self.procsize[0], self.wgsize[0],
                                                       self.cl_mem["raw"].data, self.cl_mem["scale_0"].data,
                                                       *self.scales[0])
                if self.profile:
                    self.events.append(("RGB -> float", evt))

            elif self.dtype in self.converter:
                program = self.kernels.get_kernel(self.converter[self.dtype])
                evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["raw"].data, image)
                if self.profile:
                    self.events.append(("copy H->D", evt))
                evt = program(self.queue, self.procsize[0], self.wgsize[0],
                              self.cl_mem["raw"].data, self.cl_mem["scale_0"].data, *self.scales[0])
                if self.profile:
                    self.events.append(("convert -> float", evt))
            else:
                raise RuntimeError("invalid input format error (%s)" % (str(self.dtype)))

            wg1 = self.kernels_wg["max_min_global_stage1"]
            wg2 = self.kernels_wg["max_min_global_stage2"]
            if min(wg1, wg2) < self.red_size:
                # common bug on OSX when running on CPU
                logger.info("Unable to use MinMax Reduction: stage1 wg: %s; stage2 wg: %s < max_work_group_size: %s, expected: %s",
                            wg1, wg2, self.block_size, self.red_size)
                kernel = self.kernels.get_kernel("max_min_vec16")
                k = kernel(self.queue, (1,), (1,),
                               self.cl_mem["scale_0"].data,
                               numpy.int32(self.shape[0] * self.shape[1]),
                               self.cl_mem["max"].data,
                               self.cl_mem["min"].data)
                if self.profile:
                    self.events.append(("max_min_serial", k))
                # python implementation:
                # buffer_ = self.cl_mem["scale_0"].get()
                # self.cl_mem["max"].set(numpy.array([buffer_.max()], dtype=numpy.float32))
                # self.cl_mem["min"].set(numpy.array([buffer_.min()], dtype=numpy.float32))
            else:
                kernel1 = self.kernels.get_kernel("max_min_global_stage1")
                kernel2 = self.kernels.get_kernel("max_min_global_stage2")
                # logger.debug("self.red_size: %s", self.red_size)
                shm = pyopencl.LocalMemory(self.red_size * 2 * 4)
                k1 = kernel1(self.queue, (self.red_size * self.red_size,), (self.red_size,),
                             self.cl_mem["scale_0"].data,
                             self.cl_mem["max_min"].data,
                             numpy.int32(self.shape[0] * self.shape[1]),
                             shm)
                k2 = kernel2(self.queue, (self.red_size,), (self.red_size,),
                             self.cl_mem["max_min"].data,
                             self.cl_mem["max"].data,
                             self.cl_mem["min"].data,
                             shm)

                if self.profile:
                    self.events.append(("max_min_stage1", k1))
                    self.events.append(("max_min_stage2", k2))

            evt = self.kernels.get_kernel("normalizes")(self.queue, self.procsize[0], self.wgsize[0],
                                                        self.cl_mem["scale_0"].data,
                                                        self.cl_mem["min"].data,
                                                        self.cl_mem["max"].data,
                                                        self.cl_mem["255"].data,
                                                        *self.scales[0])
            if self.profile:
                self.events.append(("normalize", evt))

            curSigma = 1.0 if par.DoubleImSize else 0.5
            octave = 0
            if self._init_sigma > curSigma:
                logger.debug("Bluring image to achieve std: %f", self._init_sigma)
                sigma = math.sqrt(self._init_sigma ** 2 - curSigma ** 2)
                self._gaussian_convolution(self.cl_mem["scale_0"], self.cl_mem["scale_0"], sigma, 0)

            for octave in range(self.octave_max):
                kp, descriptor = self._one_octave(octave)
                logger.info("in octave %i found %i kp" % (octave, kp.shape[0]))

                if len(kp):
                    # sieve out coordinates with NaNs
                    mask = numpy.where(numpy.logical_not(numpy.isnan(kp.sum(axis=-1))))
                    keypoints.append(kp[mask])
                    descriptors.append(descriptor[mask])
                    total_size += len(mask[0])

            ########################################################################
            # Merge keypoints in central memory
            ########################################################################
            output = numpy.recarray(shape=(total_size,), dtype=self.dtype_kp)
            last = 0
            for ds, desc in zip(keypoints, descriptors):
                l = ds.shape[0]
                if l > 0:
                    output[last:last + l].x = ds[:, 0]
                    output[last:last + l].y = ds[:, 1]
                    output[last:last + l].scale = ds[:, 2]
                    output[last:last + l].angle = ds[:, 3]
                    output[last:last + l].desc = desc
                    last += l
            logger.info("Execution time: %.3fms" % (1000 * (time.time() - t0)))
        return output

    __call__ = keypoints

    def _gaussian_convolution(self, input_data, output_data, sigma, octave=0):
        """
        Calculate the gaussian convolution with precalculated kernels.

        :param input_data: pyopencl array with input
        :param output_data: pyopencl array with result
        :param sigma: width of the gaussian
        :param octave: related to the size on the input images

        * Uses a temporary buffer
        * Needs gaussian kernel to be available on device

        """
        temp_data = self.cl_mem["tmp"]
        gaussian = self.cl_mem["gaussian_%s" % sigma]
        k1 = self.kernels.get_kernel("horizontal_convolution")(self.queue, self.procsize[octave], self.wgsize[octave],
                                                               input_data.data, temp_data.data, gaussian.data, numpy.int32(gaussian.size),
                                                               *self.scales[octave])
        k2 = self.kernels.get_kernel("vertical_convolution")(self.queue, self.procsize[octave], self.wgsize[octave],
                                                             temp_data.data, output_data.data, gaussian.data, numpy.int32(gaussian.size),
                                                             *self.scales[octave])

        if self.profile:
            self.events += [("Blur sigma %s octave %s" % (sigma, octave), k1),
                            ("Blur sigma %s octave %s" % (sigma, octave), k2)]

    def _one_octave(self, octave):
        """
        Does all scales within an octave

        :param octave: number of the octave
        """
        prevSigma = self._init_sigma
        logger.info("Calculating octave %i" % octave)
        wgsize = (128,)  # (max(self.wgsize[octave]),) #TODO: optimize
        kpsize32 = numpy.int32(self.kpsize)
        self._reset_keypoints()
        octsize = numpy.int32(2 ** octave)
        last_start = numpy.int32(0)
        for scale in range(par.Scales + 2):
            sigma = prevSigma * math.sqrt(self.sigmaRatio ** 2 - 1.0)
            logger.info("Octave %i scale %s blur with sigma %s" % (octave, scale, sigma))

            ########################################################################
            # Calculate gaussian blur and DoG
            ########################################################################

            self._gaussian_convolution(self.cl_mem["scale_%i" % scale], self.cl_mem["scale_%i" % (scale + 1)], sigma, octave)
            prevSigma *= self.sigmaRatio
            evt = self.kernels.get_kernel("combine")(self.queue, self.procsize[octave], self.wgsize[octave],
                                                     self.cl_mem["scale_%i" % (scale + 1)].data, numpy.float32(-1.0),
                                                     self.cl_mem["scale_%i" % (scale)].data, numpy.float32(+1.0),
                                                     self.cl_mem["DoGs"].data, numpy.int32(scale),
                                                     *self.scales[octave])
            if self.profile:
                self.events.append(("DoG %s %s" % (octave, scale), evt))
        for scale in range(1, par.Scales + 1):
            evt = self.kernels.get_kernel("local_maxmin")(self.queue, self.procsize[octave], self.wgsize[octave],
                                                          self.cl_mem["DoGs"].data,  # __global float* DOGS,
                                                          self.cl_mem["Kp_1"].data,  # __global keypoint* output,
                                                          numpy.int32(par.BorderDist),  # int border_dist,
                                                          numpy.float32(par.PeakThresh),  # float peak_thresh,
                                                          octsize,  # int octsize,
                                                          numpy.float32(par.EdgeThresh1),  # float EdgeThresh0,
                                                          numpy.float32(par.EdgeThresh),  # float EdgeThresh,
                                                          self.cl_mem["cnt"].data,  # __global int* counter,
                                                          kpsize32,  # int nb_keypoints,
                                                          numpy.int32(scale),  # int scale,
                                                          *self.scales[octave])  # int width, int height)
            if self.profile:
                self.events.append(("local_maxmin %s %s" % (octave, scale), evt))
            procsize = calc_size((self.kpsize,), wgsize)
            cp_evt = pyopencl.enqueue_copy(self.queue, self.cnt, self.cl_mem["cnt"].data)
            # TODO: modify interp_keypoint so that it reads end_keypoint from GPU memory
            evt = self.kernels.get_kernel("interp_keypoint")(self.queue, procsize, wgsize,
                                                             self.cl_mem["DoGs"].data,  # __global float* DOGS,
                                                             self.cl_mem["Kp_1"].data,  # __global keypoint* keypoints,
                                                             last_start,  # int start_keypoint,
                                                             self.cnt[0],  # int end_keypoint,
                                                             numpy.float32(par.PeakThresh),  # float peak_thresh,
                                                             numpy.float32(self._init_sigma),  # float InitSigma,
                                                             *self.scales[octave])  # int width, int height)
            if self.profile:
                self.events += [("get cnt", cp_evt),
                                ("interp_keypoint %s %s" % (octave, scale), evt)
                                ]

            newcnt = self._compact(last_start)
            evt = self.kernels.get_kernel("compute_gradient_orientation")(self.queue, self.procsize[octave], self.wgsize[octave],
                                                                          self.cl_mem["scale_%s" % (scale)].data,  # __global float* igray,
                                                                          self.cl_mem["tmp"].data,  # __global float *grad,
                                                                          self.cl_mem["ori"].data,  # __global float *ori,
                                                                          *self.scales[octave])  # int width,int height
            if self.profile:
                self.events.append(("compute_gradient_orientation %s %s" % (octave, scale), evt))

#           Orientation assignement: 1D kernel, rather heavy kernel
            if newcnt and newcnt > last_start:  # launch kernel only if neededwgsize = (128,)

                if self.USE_CPU:
                    orientation_name = "orientation_cpu"
                    scales = self.scales[octave]
                else:
                    orientation_name = "orientation_gpu"
                    scales = list(self.scales[octave]) + \
                             [pyopencl.LocalMemory(36 * 4),
                              pyopencl.LocalMemory(128 * 4),
                              pyopencl.LocalMemory(128 * 4)]
                orientation = self.kernels.get_kernel(orientation_name)
                wg = self.kernels_max_wg_size[orientation_name]
                wgsize2 = (wg,)
                procsize = (int(newcnt * wg),)
                evt = orientation(self.queue, procsize, wgsize2,
                                  self.cl_mem["Kp_1"].data,  # __global keypoint* keypoints,
                                  self.cl_mem["tmp"].data,  # __global float* grad,
                                  self.cl_mem["ori"].data,  # __global float* ori,
                                  self.cl_mem["cnt"].data,  # __global int* counter,
                                  octsize,  # int octsize,
                                  numpy.float32(par.OriSigma),  # float OriSigma, //WARNING: (1.5), it is not "InitSigma (=1.6)"
                                  kpsize32,  # int max of nb_keypoints,
                                  numpy.int32(last_start),  # int keypoints_start,
                                  newcnt,  # int keypoints_end,
                                  *scales)  # int grad_width, int grad_height)
                # newcnt = self.cl_mem["cnt"].get()[0] #do not forget to update numbers of keypoints, modified above !
                evt_cp = pyopencl.enqueue_copy(self.queue, self.cnt, self.cl_mem["cnt"].data)
                newcnt = self.cnt[0]  # do not forget to update numbers of keypoints, modified above !

                for _ in range(3):
                    # up to 3 attempts
                    if self.USE_CPU or (self.LOW_END > 1):
                        logger.info("Computing descriptors with CPU optimized kernels")
                        descriptor_name = "descriptor_cpu"
                        wg = self.kernels_max_wg_size[descriptor_name][0]
                        wgsize2 = (wg,)
                        procsize2 = (int(newcnt * wg),)
                    else:
                        if self.LOW_END:
                            logger.info("Computing descriptors with older-GPU optimized kernels")
                            descriptor_name = "descriptor_gpu1"
                            wgsize2 = self.kernels_max_wg_size[descriptor_name]
                            procsize2 = (int(newcnt * wgsize2[0]), wgsize2[1], wgsize2[2])
                            if self.kernels_wg[descriptor_name] < numpy.prod(wgsize2):
                                # will fail anyway:
                                self.LOW_END += 1
                                continue
                        else:
                            logger.info("Computing descriptors with newer-GPU optimized kernels")
                            descriptor_name = "descriptor_gpu2"
                            wgsize2 = self.kernels_max_wg_size[descriptor_name]
                            procsize2 = (int(newcnt * wgsize2[0]), wgsize2[1], wgsize2[2])
                            if self.kernels_wg[descriptor_name] < numpy.prod(wgsize2):
                                # will fail anyway:
                                self.LOW_END += 1
                                continue
                    try:
                        descriptor = self.kernels.get_kernel(descriptor_name)
                        evt2 = descriptor(self.queue, procsize2, wgsize2,
                                          self.cl_mem["Kp_1"].data,  # __global keypoint* keypoints,
                                          self.cl_mem["descriptors"].data,  # ___global unsigned char *descriptors
                                          self.cl_mem["tmp"].data,  # __global float* grad,
                                          self.cl_mem["ori"].data,  # __global float* ori,
                                          octsize,  # int octsize,
                                          numpy.int32(last_start),  # int keypoints_start,
                                          self.cl_mem["cnt"].data,  # int* keypoints_end,
                                          *self.scales[octave])  # int grad_width, int grad_height)
                        evt2.wait()
                    except pyopencl.RuntimeError as error:
                        self.LOW_END += 1
                        logger.error("Descriptor failed with %s. Switching to lower_end mode" % error)
                        continue
                    else:
                        break
                if self.profile:
                    self.events += [("%s %s %s" % (orientation_name, octave, scale), evt),
                                    ("copy cnt D->H", evt_cp),
                                    ("%s %s %s" % (descriptor_name, octave, scale), evt2)]
            evt_cp = pyopencl.enqueue_copy(self.queue, self.cnt, self.cl_mem["cnt"].data)
            last_start = self.cnt[0]
            if self.profile:
                self.events.append(("copy cnt D->H", evt_cp))

        ########################################################################
        # Rescale all images to populate all octaves
        ########################################################################
        if octave < self.octave_max - 1:
            evt = self.kernels.get_kernel("shrink")(self.queue, self.procsize[octave + 1], self.wgsize[octave + 1],
                                                    self.cl_mem["scale_%i" % (par.Scales)].data,
                                                    self.cl_mem["scale_0"].data,
                                                    numpy.int32(2), numpy.int32(2),
                                                    self.scales[octave][0], self.scales[octave][1],
                                                    *self.scales[octave + 1])
            if self.profile:
                self.events.append(("shrink %s->%s" % (self.scales[octave], self.scales[octave + 1]), evt))
        results = numpy.empty((last_start, 4), dtype=numpy.float32)
        descriptors = numpy.empty((last_start, 128), dtype=numpy.uint8)
        if last_start:
            evt = pyopencl.enqueue_copy(self.queue, results, self.cl_mem["Kp_1"].data)
            evt2 = pyopencl.enqueue_copy(self.queue, descriptors, self.cl_mem["descriptors"].data)
            if self.profile:
                self.events += [("copy D->H", evt),
                                ("copy D->H", evt2)]
        return results, descriptors

    def _compact(self, start=numpy.int32(0)):
        """
        Compact the vector of keypoints starting from start

        :param start: start compacting at this adress. Before just copy
        :type  start: numpy.int32
        """
        wgsize = self.kernels_wg["compact"],  
        cp0_evt = pyopencl.enqueue_copy(self.queue, self.cnt, self.cl_mem["cnt"].data)
        kp_counter = self.cnt[0]
        procsize = calc_size((self.kpsize,), wgsize)

        if kp_counter > 0.9 * self.kpsize:
            logger.warning("Keypoint counter overflow risk: counted %s / %s" % (kp_counter, self.kpsize))
        logger.info("Compact %s -> %s / %s" % (start, kp_counter, self.kpsize))
        self.cnt[0] = start
        cp1_evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["cnt"].data, self.cnt)
        evt = self.kernels.get_kernel("compact")(self.queue, procsize, wgsize,
                                                 self.cl_mem["Kp_1"].data,  # __global keypoint* keypoints,
                                                 self.cl_mem["Kp_2"].data,  # __global keypoint* output,
                                                 self.cl_mem["cnt"].data,  # __global int* counter,
                                                 start,  # int start,
                                                 kp_counter)  # int nbkeypoints
        cp2_evt = pyopencl.enqueue_copy(self.queue, self.cnt, self.cl_mem["cnt"].data)
        # swap keypoints:
        self.cl_mem["Kp_1"], self.cl_mem["Kp_2"] = self.cl_mem["Kp_2"], self.cl_mem["Kp_1"]
        # memset buffer Kp_2
#        self.cl_mem["Kp_2"].fill(-1, self.queue)
        mem_evt = self.kernels.get_kernel("memset_float")(self.queue, calc_size((4 * self.kpsize,), wgsize), wgsize, self.cl_mem["Kp_2"].data, numpy.float32(-1), numpy.int32(4 * self.kpsize))
        if self.profile:
            self.events += [("copy cnt D->H", cp0_evt),
                            ("copy cnt H->D", cp1_evt),
                            ("compact", evt),
                            ("copy cnt D->H", cp2_evt),
                            ("memset 2", mem_evt)
                            ]
        return self.cnt[0]

    def _reset_keypoints(self):
        """
        Todo: implement directly in OpenCL instead of relying on pyOpenCL
        """
        wg_size = self.kernels_wg["memset_float"],
        evt1 = self.kernels.get_kernel("memset_float")(self.queue, calc_size((4 * self.kpsize,), wg_size), wg_size,
                                                       self.cl_mem["Kp_1"].data,
                                                       numpy.float32(-1),
                                                       numpy.int32(4 * self.kpsize))
#        evt2 = self.kernels.get_kernel("memset"].memset_float(self.queue, calc_size((4 * self.kpsize,), wg_size), wg_size, self.cl_mem["Kp_2"].data, numpy.float32(-1), numpy.int32(4 * self.kpsize))
        evt3 = self.kernels.get_kernel("memset_int")(self.queue, (1,), (1,),
                                                     self.cl_mem["cnt"].data,
                                                     numpy.int32(0),
                                                     numpy.int32(1))
        if self.profile:
            self.events += [("memset 1", evt1), ("memset cnt", evt3)]
#        self.cl_mem["Kp_1"].fill(-1, self.queue)
#        self.cl_mem["Kp_2"].fill(-1, self.queue)
#        self.cl_mem["cnt"].fill(0, self.queue)

    def count_kp(self, output):
        """
        Print the number of keypoint per octave
        """
        kpt = 0
        for octave, data in enumerate(output):
            if output.shape[0] > 0:
                ksum = (data[:, 1] != -1.0).sum()
                kpt += ksum
                print("octave %i kp count %i/%i size %s ratio:%s" % (octave, ksum, self.kpsize, self.scales[octave], 1000.0 * ksum / self.scales[octave][1] / self.scales[octave][0]))
        print("Found total %i guess %s pixels per keypoint" % (kpt, self.shape[0] * self.shape[1] / kpt))


def demo():
    # Prepare debugging
    import scipy.misc
    if hasattr(scipy.misc, "ascent"):
        img = scipy.misc.ascent()
    else:
        img = scipy.misc.lena()

    s = SiftPlan(template=img)
    print(s.keypoints(img))

if __name__ == "__main__":
    demo()
