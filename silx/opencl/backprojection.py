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
__date__ = "26/06/2017"

import logging
import numpy as np

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
    return np.dtype(Type).itemsize


def _idivup(a, b):
    """
    return the integer division, plus one if `a` is not a multiple of `b`
    """
    return (a + (b - 1)) // b


class Backprojection(OpenclProcessing):
    """A class for performing the backprojection using OpenCL"""
    kernel_files = ["backproj.cl", "backproj_helper.cl"] if _has_pyfft else ["backproj.cl"]

    def __init__(self, sino_shape, slice_shape=None, axis_position=None, angles=None, filter_name=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 profile=False
                 ):
        """Constructor of the OpenCL (filtered) backprojection

        :param sino_shape: shape of the sinogram. The sinogram is in the format (n_b, n_a) where
                      n_b is the number of detector bins and n_a is the number of angles.
        :param slice_shape: Optional, shape of the reconstructed slice. By default, it is a
                    square slice where the dimension is the "x dimension" of the sinogram (number
                    of bins).
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
        # OS X enforces a workgroup size of 1 when the kernel has synchronization barriers
        # if sys.platform.startswith('darwin'): # assuming no discrete GPU
        #    raise NotImplementedError("Backprojection is not implemented on CPU for OS X yet")

        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)
        self.shape = sino_shape

        self.num_bins = np.int32(sino_shape[1])
        self.num_projs = np.int32(sino_shape[0])
        self.angles = angles
        if slice_shape is None:
            self.slice_shape = (self.num_bins, self.num_bins)
        else:
            self.slice_shape = slice_shape
        self.dimrec_shape = (
            _idivup(self.slice_shape[0], 32) * 32,
            _idivup(self.slice_shape[1], 32) * 32
        )
        self.slice = np.zeros(self.dimrec_shape, dtype=np.float32)
        self.filter_name = filter_name if filter_name else "Ram-Lak"
        if axis_position:
            self.axis_pos = np.float32(axis_position)
        else:
            self.axis_pos = np.float32((sino_shape[1] - 1.) / 2)
        self.axis_array = None  # TODO: add axis correction front-end
        self.is_cpu = False
        if self.device.type == "CPU":
            self.is_cpu = True

        self.compute_fft_plans()
        self.buffers = [
                       BufferDescription("d_slice", np.prod(self.dimrec_shape), np.float32, mf.READ_WRITE),
                       BufferDescription("d_sino", self.num_projs * self.num_bins, np.float32, mf.READ_WRITE),  # before transferring to texture (if available)
                       BufferDescription("d_cos", self.num_projs, np.float32, mf.READ_ONLY),
                       BufferDescription("d_sin", self.num_projs, np.float32, mf.READ_ONLY),
                       BufferDescription("d_axes", self.num_projs, np.float32, mf.READ_ONLY),
                      ]
        self.allocate_buffers()
        if not(self.is_cpu):
            self.allocate_textures()
        self.compute_filter()
        if self.pyfft_plan:
            self.add_to_cl_mem({"d_filter": self.d_filter, "d_sino_z": self.d_sino_z})
        self.d_sino = self.cl_mem["d_sino"]  # shorthand
        self.compute_angles()

        self.local_mem = 256 * 3 * _sizeof(np.float32)  # constant for all image sizes
        OpenclProcessing.compile_kernels(self, self.kernel_files)
        # check that workgroup can actually be (16, 16)
        self.check_workgroup_size()
        # Workgroup and ndrange sizes are always the same
        self.wg = (16, 16)
        self.ndrange = (
            _idivup(int(self.dimrec_shape[1]), 32) * self.wg[0],  # int(): pyopencl <= 2015.1
            _idivup(int(self.dimrec_shape[0]), 32) * self.wg[1]  # int(): pyopencl <= 2015.1
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
            pyopencl.enqueue_copy(self.queue, self.cl_mem["d_axes"], np.ones(self.num_projs, dtype=np.float32) * self.axis_pos)

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
                                        # shape=(self.shape[1], self.shape[0]) # why is it not working ?
                                        hostbuf=np.zeros(self.shape[::-1], dtype=np.float32)
                                        )
        # self.d_sino_tex = pyopencl.image_from_array(self.ctx, np.ones(self.shape, dtype=np.float32), 1)

    def compute_fft_plans(self):
        """
        If pyfft is installed, prepare a batched 1D FFT plan for the filtering of FBP

        """
        self.fft_size = nextpow2(self.num_bins * 2 - 1)
        if _has_pyfft:
            logger.debug("pyfft is available. Computing FFT plans...")
            # batched 1D transform
            self.pyfft_plan = pyfft_Plan(self.fft_size, queue=self.queue)
            self.d_sino_z = parray.zeros(self.queue, (self.num_projs, self.fft_size), dtype=np.complex64)
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
            L2 = L // 2 + 1
            h[0] = 1 / 4.
            j = np.linspace(1, L2, L2 // 2, False)
            h[1:L2:2] = -1. / (np.pi ** 2 * j ** 2)
            h[L2:] = np.copy(h[1:L2 - 1][::-1])
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

    def check_workgroup_size(self):
        kernel = self.program.all_kernels()[1]  # CPU kernel
        self.compiletime_workgroup_size = kernel_workgroup_size(self.program, kernel)

    def _get_local_mem(self):
        return pyopencl.LocalMemory(self.local_mem)  # constant for all image sizes

    def backprojection(self, sino=None):
        """Perform the backprojection on an input sinogram

        :param sino: sinogram. If provided, it performs the plain backprojection.
        :return: backprojection of sinogram
        """
        events = []
        with self.sem:

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
                '''
                pyopencl.enqueue_copy(
                    self.queue,
                    self.d_sino_tex,
                    np.ascontiguousarray(sino.astype(np.float32)),
                    origin=(0, 0),
                    region=(np.int32(sino.shape[1]), np.int32(sino.shape[0]))
                )
                '''
            # /sino is not None
            # Copy d_sino to texture, for GPU
            if not(self.is_cpu):
                ev = pyopencl.enqueue_copy(
                    self.queue,
                    self.d_sino_tex,
                    self.d_sino,
                    offset=0,
                    origin=(0, 0),
                    region=self.shape[::-1]  # (np.int32(self.shape[1]), np.int32(self.shape[0]))
                )
                events.append(EventDescription("Buffer to Image d_sino", ev))
            # Prepare arguments for the kernel call
            if self.is_cpu:
                d_sino_ref = self.d_sino
            else:
                d_sino_ref = self.d_sino_tex
            kernel_args = (
                self.num_projs,  # num of projections (int32)
                self.num_bins,  # num of bins (int32)
                self.axis_pos,  # axis position (float32)
                self.cl_mem["d_slice"],  # d_slice (__global float32*)
                d_sino_ref,  # d_sino (__read_only image2d_t or float*)
                np.float32(0),  # gpu_offset_x (float32)
                np.float32(0),  # gpu_offset_y (float32)
                self.cl_mem["d_cos"],  # d_cos (__global float32*)
                self.cl_mem["d_sin"],  # d_sin (__global float32*)
                self.cl_mem["d_axes"],  # d_axis  (__global float32*)
                self._get_local_mem()  # shared mem (__local float32*)
            )
            # Call the kernel
            if self.is_cpu:
                event_bpj = self.program.backproj_cpu_kernel(
                    self.queue,
                    self.ndrange,
                    self.wg,
                    *kernel_args
                )
            else:
                event_bpj = self.program.backproj_kernel(
                    self.queue,
                    self.ndrange,
                    self.wg,
                    *kernel_args
                )
            self.slice[:] = 0
            events.append(EventDescription("backprojection", event_bpj))
            ev = pyopencl.enqueue_copy(self.queue, self.slice, self.cl_mem["d_slice"])
            events.append(EventDescription("copy D->H result", ev))
            ev.wait()
        # /with self.sem
        if self.profile:
            self.events += events
        res = np.copy(self.slice)
        if self.dimrec_shape[0] > self.slice_shape[0] or self.dimrec_shape[1] > self.slice_shape[1]:
            res = res[:self.slice_shape[0], :self.slice_shape[1]]
        # if the slice is backprojected onto a bigger grid
        if self.slice_shape[1] > self.num_bins:
            res = res[:self.slice_shape[0], :self.slice_shape[1]]
        return res


    def filter_projections(self, sino, rescale=True):
        """
        Performs the FBP on a given sinogram.

        :param sinogram: sinogram to (filter-)backproject
        :param rescale: if True (default), the sinogram is multiplied with (pi/n_projs)
        """
        if sino.shape[0] != self.num_projs or sino.shape[1] != self.num_bins:
            raise ValueError("Expected sinogram with (projs, bins) = (%d, %d)" % (self.num_projs, self.num_bins))
        if rescale:
            sino = sino * np.pi / self.num_projs
        # if pyfft is available, all can be done on the device
        if self.d_filter is not None:
            events = []
            # Zero-pad the sinogram.
            # TODO: this can be done on GPU with a "Memcpy2D":
            #  cl.enqueue_copy(queue, dst, src, host_origin=(0,0), buffer_origin=(0,0), region=shape, host_pitches=(sino.shape[1],), buffer_pitches=(self.fft_size,))
            # However it does not work properly, and raises an error for pyopencl < 2017.1
            sino_zeropadded = np.zeros((sino.shape[0], self.fft_size), dtype=np.complex64)
            sino_zeropadded[:, :self.num_bins] = sino.astype(np.float32)
            sino_zeropadded = np.ascontiguousarray(sino_zeropadded, dtype=np.complex64)
            with self.sem:
                # send to GPU
                ev = pyopencl.enqueue_copy(self.queue, self.d_sino_z.data, sino_zeropadded)
                events.append(EventDescription("Send sino H->D", ev))
                # FFT (in-place)
                self.pyfft_plan.execute(self.d_sino_z.data, batch=self.num_projs)
                # Multiply (complex-wise) with the the filter
                ev = self.program.mult(self.queue,
                                       tuple(int(i) for i in self.d_sino_z.shape[::-1]),
                                       None,
                                       self.d_sino_z.data,
                                       self.d_filter.data,
                                       np.int32(self.fft_size),
                                       self.num_projs
                                       )
                events.append(EventDescription("complex 2D-1D multiplication", ev))
                # Inverse FFT (in-place)
                self.pyfft_plan.execute(self.d_sino_z.data, batch=self.num_projs, inverse=True)
                # Copy the real part of d_sino_z[:, :self.num_bins] (complex64) to d_sino (float32)
                ev = self.program.cpy2d(self.queue, self.shape[::-1], None,
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
        else:  # no pyfft
            # Zero-padding of the sinogram
            sino_zeropadded = np.zeros((sino.shape[0], self.fft_size), dtype=np.complex64)
            sino_zeropadded[:, :self.num_bins] = sino.astype(np.float32)
            # Linear convolution
            sino_f = np.fft.fft(sino, self.fft_size)
            sino_f = sino_f * self.filter
            sino_filtered = np.fft.ifft(sino_f)[:, :self.num_bins].real
            # Send the filtered sinogram to device
            sino_filtered = np.ascontiguousarray(sino_filtered, dtype=np.float32)
            with self.sem:
                pyopencl.enqueue_copy(self.queue, self.d_sino, sino_filtered)


    def filtered_backprojection(self, sino):
        """
        Compute the filtered backprojection (FBP) on a sinogram.

        :param sino: sinogram (`numpy.ndarray`) in the format (projections, bins)
        """

        self.filter_projections(sino)
        res = self.backprojection()
        return res

    __call__ = filtered_backprojection
