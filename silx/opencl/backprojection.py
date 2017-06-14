#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""Module for (filtered) backprojection on the GPU"""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["A. Mirone, P. Paleo"]
__license__ = "MIT"
__date__ = "12/06/2017"

import logging
import numpy as np
from collections import OrderedDict

from .common import pyopencl, kernel_workgroup_size
from .processing import EventDescription, OpenclProcessing, BufferDescription
from .utils import nextpower as nextpow2

if pyopencl:
    mf = pyopencl.mem_flags
    import pyopencl.array as parray
else:
    raise ImportError("pyopencl is not installed")
logger = logging.getLogger(__name__)

# put in .common ?
try:
    from pyfft.cl import Plan as pyfft_Plan
    _has_pyfft = True
except ImportError:
    _has_pyfft = False





def _sizeof(Type):
    """
    return the size (in bytes) of a scalar type, like the C behavior
    """
    if issubclass(Type, np.inexact):
        bits = np.finfo(Type).bits
    else:
        bits = np.iinfo(Type).bits
    return bits//8


def _idivup(a, b):
    """
    return the integer division, plus one if `a` is not a multiple of `b`
    """
    return (a + (b-1))//b




class Backprojection(OpenclProcessing):
    """A class for performing the backprojection using OpenCL"""
    kernel_files = ["backproj.cl"]


    def __init__(self, shape, axis_position=None, angles=None, filter_name=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 profile=False
                 ):
        """Constructor of the OpenCL (filtered) backprojection

        :param shape: shape of the sinogram. The sinogram is in the format (n_b, n_a) where
                      n_b is the number of detector bins and n_a is the number of angles.
        :param axis_position: Optional, axis position. Default is `(shape[1]-1)/2.0`.
        :param angles: Optional, a list of custom angles in radian.
        :param filter_name: Optional, name of the filter for FBP. Default is the Ram-Lak filter.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)
        self.shape = shape
        self.num_bins = np.int32(shape[1])
        self.num_projs = np.int32(shape[0])
        self.angles = angles
        self.slice = np.zeros((self.num_bins, self.num_bins), dtype=np.float32)
        self.filter_name = filter_name if filter_name else "Ram-Lak"
        if axis_position:
            self.axis_pos = np.float32(axis_position)
        else:
            self.axis_pos = np.float32((shape[1]-1.)/2)
        # TODO: add axis correction front-end
        self.axis_array = None

        self.compute_fft_plans()
        self.buffers = [
                    BufferDescription("d_slice", self.num_bins*self.num_bins, np.float32, mf.READ_WRITE),
                    BufferDescription("d_sino", self.num_projs*self.num_bins, np.float32, mf.READ_WRITE), # before transferring to texture (if available)
                    BufferDescription("d_cos", self.num_projs, np.float32, mf.READ_ONLY),
                    BufferDescription("d_sin", self.num_projs, np.float32, mf.READ_ONLY),
                    BufferDescription("d_axes", self.num_projs, np.float32, mf.READ_ONLY),
                   ]
        self.allocate_buffers()
        self.allocate_textures()
        self.compute_filter()
        if self.pyfft_plan:
            self.add_to_cl_mem({"d_filter": self.d_filter, "d_sino_z": self.d_sino_z})
        self.d_sino = self.cl_mem["d_sino"] # shorthand
        self.compute_angles()

        self.local_mem = 256*3*_sizeof(np.float32) # constant for all image sizes
        OpenclProcessing.compile_kernels(self, self.kernel_files)
        # Workgroup and ndrange sizes are always the same
        self.wg = (16, 16)
        self.ndrange = (
            _idivup(self.num_bins, 32)*self.wg[0],
            _idivup(self.num_bins, 32)*self.wg[1]
        )

    def compute_angles(self):
        if self.angles is None:
            self.angles = np.linspace(0, np.pi, self.num_projs, False)
        h_cos = np.cos(self.angles).astype(np.float32)
        h_sin = np.sin(self.angles).astype(np.float32)
        pyopencl.enqueue_copy(self.queue, self.cl_mem["d_cos"], h_cos)
        pyopencl.enqueue_copy(self.queue, self.cl_mem["d_sin"], h_sin)
        if self.axis_array:
            pyopencl.enqueue_copy(self.queue, self.cl_mem["d_axes"], self.axis_array.astype(np.float32))
        else:
            pyopencl.enqueue_copy(self.queue, self.cl_mem["d_axes"], np.ones(self.num_projs, dtype=np.float32)*self.axis_pos)


    def allocate_textures(self):
        """
        Allocate the texture for the sinogram.
        """
        self.d_sino_tex = pyopencl.Image(
                            self.ctx,
                            mf.READ_ONLY | mf.USE_HOST_PTR,
                            pyopencl.ImageFormat(
                                pyopencl.channel_order.INTENSITY,
                                pyopencl.channel_type.FLOAT
                            ),
                            #~ shape=(self.shape[1], self.shape[0]) # why is it not working ?
                            hostbuf=np.zeros(self.shape[::-1], dtype=np.float32)
                        )
        #~ self.d_sino_tex = pyopencl.image_from_array(self.ctx, np.ones(self.shape, dtype=np.float32), 1)


    def add_to_cl_mem(self, parrays):
        """
        Add pyopencl.array, which are allocated by pyopencl, to self.cl_mem.

        :param parrays: a dictionary of `pyopencl.array.Array` or `pyopencl.Buffer`
        """
        mem = self.cl_mem
        for name, parr in parrays.items():
            mem[name] = parr
        self.cl_mem.update(mem)


    def compute_fft_plans(self):
        """
        If pyfft is installed, prepare a batched 1D FFT plan for the filtering of FBP

        """
        self.fft_size = nextpow2(self.num_bins*2 -1)
        if _has_pyfft:
            logger.debug("pyfft is available. Computing FFT plans...")
            # batched 1D transform
            self.pyfft_plan = pyfft_Plan(self.fft_size, queue=self.queue)
            self.d_sino_z = parray.zeros(self.queue, (self.num_projs, self.fft_size), dtype=np.complex64)
            logger.debug("... done")
            logger.debug("Building OpenCL programs for filtering and conversion...")
            self.prg_mult = pyopencl.Program(self.ctx, """
                __kernel void mult(
                    __global float2* d_sino, __global float2* d_filter, int num_bins, int num_projs)
                {
                  int gid0 = get_global_id(0);
                  int gid1 = get_global_id(1);
                  if (gid0 < num_bins && gid1 < num_projs) {
                    // d_sino[gid1*num_bins+gid0] *= d_filter[gid0];
                    d_sino[gid1*num_bins+gid0].x *= d_filter[gid0].x;
                    d_sino[gid1*num_bins+gid0].y *= d_filter[gid0].x;
                  }
                }
            """).build()
            self.prg_cpy2d = pyopencl.Program(self.ctx, """
                __kernel void cpy2d(
                    __global float* d_sino, __global float2* d_sino_complex, int num_bins, int num_projs, int fft_size)
                {
                  int gid0 = get_global_id(0);
                  int gid1 = get_global_id(1);
                  if (gid0 < num_bins && gid1 < num_projs) {
                    d_sino[gid1*num_bins+gid0] = d_sino_complex[gid1*fft_size+gid0].x;
                  }
                }
            """).build()
            logger.debug("... done")

        else:
            logger.debug("pyfft not available, using numpy.fft")
            self.pyfft_plan = None
            # TODO: fall-back to fftw if present ?


    def compute_filter(self):
        """
        Compute the filter for FBP
        """
        if self.filter_name == "Ram-Lak":
            L = self.fft_size
            h = np.zeros(L, dtype=np.float32)
            L2 = L//2+1
            h[0] = 1/4.
            j = np.linspace(1, L2, L2//2, False)
            h[1:L2:2] = -1./(np.pi**2 * j**2)
            h[L2:] = np.copy(h[1:L2-1][::-1])
        else:
            # TODO: other filters
            raise ValueError("Filter %s is not available" % self.filter_name)
        self.filter = h
        if self.pyfft_plan:
            self.d_filter = parray.to_device(self.queue, h.astype(np.complex64))
            self.pyfft_plan.execute(self.d_filter.data)
        else:
            self.filter = np.fft.fft(h).astype(np.complex64)
            self.d_filter = None


    def _get_local_mem(self):
        return pyopencl.LocalMemory(self.local_mem)  # constant for all image sizes

    def backprojection(self, sino=None):
        """Perform the backprojection on an input sinogram

        :param sino: sinogram. If provided, it performs the plain backprojection.
        :return: backprojection of sinogram
        """
        events = []
        if sino is not None:
            assert sino.ndim == 2, "Treat only 2D images"
            assert sino.shape[0] == self.num_projs, "num_projs is OK"
            assert sino.shape[1] == self.num_bins, "num_bins is OK"
            # We can either
            #  (1) do the conversion on host, and directly send to the device texture, or
            #  (2) send to the device Buffer d_sino, make the conversion on device, and then transfer to texture
            # ------
            # (2) without device conversion
            pyopencl.enqueue_copy(self.queue, self.d_sino, np.ascontiguousarray(sino, dtype=np.float32))
            # (1)
            #~ pyopencl.enqueue_copy(
                #~ self.queue,
                #~ self.d_sino_tex,
                #~ np.ascontiguousarray(sino.astype(np.float32)),
                #~ origin=(0, 0),
                #~ region=(np.int32(sino.shape[1]), np.int32(sino.shape[0]))
            #~ )

        with self.sem:
            # Copy d_sino to texture
            ev = pyopencl.enqueue_copy(
                self.queue,
                self.d_sino_tex,
                self.d_sino,
                offset=0,
                origin=(0, 0),
                region=(np.int32(self.shape[1]), np.int32(self.shape[0]))
            )
            events.append(EventDescription("Buffer to Image d_sino", ev))
            # Prepare arguments for the kernel call
            kernel_args = (
                self.num_projs, # num of projections (int32)
                self.num_bins,  # num of bins (int32)
                self.axis_pos,  # axis position (float32)
                self.cl_mem["d_slice"], # d_slice (__global float32*)
                self.d_sino_tex, #  d_sino (__read_only image2d_t)
                np.float32(0),   # gpu_offset_x (float32)
                np.float32(0),   # gpu_offset_y (float32)
                self.cl_mem["d_cos"],  # d_cos (__global float32*)
                self.cl_mem["d_sin"],  # d_sin (__global float32*)
                self.cl_mem["d_axes"], # d_axis  (__global float32*)
                self._get_local_mem()  # shared mem (__local float32*)
            )
            # Call the kernel
            event_bpj = self.program.backproj_kernel(
                self.queue,
                self.ndrange,
                self.wg,
                *kernel_args
            )
            events.append(EventDescription("backprojection", event_bpj))

            result = np.empty((self.num_bins, self.num_bins), np.float32)
            ev = pyopencl.enqueue_copy(self.queue, result, self.cl_mem["d_slice"])
            events.append(EventDescription("copy D->H result", ev))
            ev.wait()
        if self.profile:
            self.events += events
        return result



    def filter_projections(self, sino, rescale=True):
        """
        Performs the FBP on a given sinogram.

        :param sinogram: sinogram to (filter-)backproject
        :param rescale: if True (default), the sinogram is multiplied with (pi/n_projs)
        """
        if sino.shape[0] != self.num_projs or sino.shape[1] != self.num_bins:
            raise ValueError("Expected sinogram with (projs, bins) = (%d, %d)" % (self.num_projs, self.num_bins))
        if rescale:
            sino = sino * np.pi/ self.num_projs
        # if pyfft is available, all can be done on the device
        if self.d_filter is not None:
            events = []
            # Zero-pad the sinogram.
            # TODO: this can be done on GPU with a "Memcpy2D":
            #  cl.enqueue_copy(queue, dst, src, host_origin=(0,0), buffer_origin=(0,0), region=shape, host_pitches=(sino.shape[1],), buffer_pitches=(self.fft_size,))
            # However it does not work properly, and raises an error for pyopencl < 2017.1
            sino_zeropadded = np.zeros((sino.shape[0], self.fft_size), dtype=np.complex64)
            sino_zeropadded[:, :self.num_bins] = sino
            sino_zeropadded = np.ascontiguousarray(sino_zeropadded, dtype=np.complex64)
            # send to GPU
            ev = pyopencl.enqueue_copy(self.queue, self.d_sino_z.data, sino_zeropadded)
            #~ del sino_zeropadded
            events.append(EventDescription("Send sino H->D", ev))
            # FFT (in-place)
            self.pyfft_plan.execute(self.d_sino_z.data, batch=self.num_projs)
            # Multiply (complex-wise) with the the filter
            ev = self.prg_mult.mult(self.queue, self.d_sino_z.shape[::-1], None,
                           self.d_sino_z.data,
                           self.d_filter.data,
                           np.int32(self.fft_size),
                           self.num_projs
                         )
            events.append(EventDescription("complex 2D-1D multiplication", ev))
            # Inverse FFT (in-place)
            self.pyfft_plan.execute(self.d_sino_z.data, batch=self.num_projs, inverse=True)
            # Copy the real part of d_sino_z[:, :self.num_bins] (complex64) to d_sino (float32)
            ev = self.prg_cpy2d.cpy2d(self.queue, self.shape[::-1], None,
                           self.d_sino,
                           self.d_sino_z.data,
                           self.num_bins,
                           self.num_projs,
                           np.int32(self.fft_size)
                         )
            events.append(EventDescription("conversion from complex padded sinogram to sinogram", ev))
            if self.profile:
                self.events += events
            # ------
        else: # no pyfft
            # Zero-padding of the sinogram
            sino_zeropadded = np.zeros((sino.shape[0], self.fft_size), dtype=np.complex64)
            sino_zeropadded[:, :self.num_bins] = np.copy(sino)
            # Linear convolution
            sino_f = np.fft.fft(sino, self.fft_size)
            sino_f = sino_f * self.filter
            sino_filtered = np.fft.ifft(sino_f)[:, :self.num_bins].real
            # Send the filtered sinogram to device
            sino_filtered = np.ascontiguousarray(sino_filtered, dtype=np.float32)
            pyopencl.enqueue_copy(self.queue, self.d_sino, sino_filtered)

        pyopencl.enqueue_copy(self.queue, self.slice, self.d_sino)
        return self.slice


    def filtered_backprojection(self, sino):
        """
        Compute the filtered backprojection (FBP) on a sinogram.

        :param sino: sinogram (`numpy.ndarray`) in the format (projections, bins)
        """

        self.filter_projections(sino)
        res = self.backprojection()
        return res



    __call__ = filtered_backprojection # or fbp ?












'''

class _MedFilt2d(object):
    median_filter = None

    @classmethod
    def medfilt2d(cls, ary, kernel_size=3):
        """Median filter a 2-dimensional array.

        Apply a median filter to the `input` array using a local window-size
        given by `kernel_size` (must be odd).

        :param ary: A 2-dimensional input array.
        :param kernel_size: A scalar or a list of length 2, giving the size of the
                            median filter window in each dimension.  Elements of
                            `kernel_size` should be odd.  If `kernel_size` is a scalar,
                            then this scalar is used as the size in each dimension.
                            Default is a kernel of size (3, 3).
        :return: An array the same size as input containing the median filtered
                result. always work on float32 values

        About the padding:

        * The filling mode in scipy.signal.medfilt2d is zero-padding
        * This implementation is equivalent to:
            scipy.ndimage.filters.median_filter(ary, kernel_size, mode="nearest")

        """
        image = np.atleast_2d(ary)
        shape = np.array(image.shape)
        if cls.median_filter is None:
            cls.median_filter = MedianFilter2D(image.shape, kernel_size)
        elif (np.array(cls.median_filter.shape) < shape).any():
            # enlarger the buffer size
            new_shape = np.maximum(np.array(cls.median_filter.shape), shape)
            ctx = cls.median_filter.ctx
            cls.median_filter = MedianFilter2D(new_shape, kernel_size, ctx=ctx)
        return cls.median_filter.medfilt2d(image)

medfilt2d = _MedFilt2d.medfilt2d
'''
