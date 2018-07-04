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
__date__ = "19/01/2018"

import logging
import numpy

from .common import pyopencl
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
# For silx v0.6 we disable the use FFT on GPU.
_has_pyfft = False


def _sizeof(Type):
    """
    return the size (in bytes) of a scalar type, like the C behavior
    """
    return numpy.dtype(Type).itemsize


def _idivup(a, b):
    """
    return the integer division, plus one if `a` is not a multiple of `b`
    """
    return (a + (b - 1)) // b


def fourier_filter(sino, filter_=None, fft_size=None):
    """Simple numpy based implementation of fourier space filter
    
    :param sino: of shape shape = (num_projs, num_bins)
    :param filter: filter function to apply in fourier space
    :fft_size: size on which perform the fft. May be larger than the sino array 
    :return: filtered sinogram
    """
    assert sino.ndim == 2
    num_projs, num_bins = sino.shape
    if fft_size is None:
        fft_size = nextpow2(num_bins * 2 - 1)
    else:
        assert fft_size >= num_bins
    if fft_size == num_bins:
        sino_zeropadded = sino.astype(numpy.float32)
    else:
        sino_zeropadded = numpy.zeros((num_projs, fft_size),
                                      dtype=numpy.complex64)
        sino_zeropadded[:, :num_bins] = sino.astype(numpy.float32)

    if filter_ is None:
        h = numpy.zeros(fft_size, dtype=numpy.float32)
        L2 = fft_size // 2 + 1
        h[0] = 1 / 4.
        j = numpy.linspace(1, L2, L2 // 2, False)
        h[1:L2:2] = -1. / (numpy.pi ** 2 * j ** 2)
        h[L2:] = numpy.copy(h[1:L2 - 1][::-1])
        filter_ = numpy.fft.fft(h).astype(numpy.complex64)

    # Linear convolution
    sino_f = numpy.fft.fft(sino, fft_size)
    sino_f = sino_f * filter_
    sino_filtered = numpy.fft.ifft(sino_f)[:, :num_bins].real
    # Send the filtered sinogram to device
    return numpy.ascontiguousarray(sino_filtered.real, dtype=numpy.float32)


class Backprojection(OpenclProcessing):
    """A class for performing the backprojection using OpenCL"""
    kernel_files = ["backproj.cl", "array_utils.cl"]
    if _has_pyfft:
        kernel_files.append("backproj_helper.cl")

    def __init__(self, sino_shape, slice_shape=None, axis_position=None,
                 angles=None, filter_name=None, ctx=None, devicetype="all",
                 platformid=None, deviceid=None, profile=False):
        """Constructor of the OpenCL (filtered) backprojection

        :param sino_shape: shape of the sinogram. The sinogram is in the format
                           (n_b, n_a) where n_b is the number of detector bins
                           and n_a is the number of angles.
        :param slice_shape: Optional, shape of the reconstructed slice. By
                            default, it is a square slice where the dimension
                            is the "x dimension" of the sinogram (number of
                            bins).
        :param axis_position: Optional, axis position. Default is
                              `(shape[1]-1)/2.0`.
        :param angles: Optional, a list of custom angles in radian.
        :param filter_name: Optional, name of the filter for FBP. Default is
                            the Ram-Lak filter.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by
                           clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param profile: switch on profiling to be able to profile at the kernel
                        level, store profiling elements (makes code slightly
                        slower)
        """
        # OS X enforces a workgroup size of 1 when the kernel has
        # synchronization barriers if sys.platform.startswith('darwin'):
        #  assuming no discrete GPU
        #    raise NotImplementedError("Backprojection is not implemented on CPU for OS X yet")

        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)
        self.shape = sino_shape

        self.num_bins = numpy.int32(sino_shape[1])
        self.num_projs = numpy.int32(sino_shape[0])
        self.angles = angles
        if slice_shape is None:
            self.slice_shape = (self.num_bins, self.num_bins)
        else:
            self.slice_shape = slice_shape
        self.dimrec_shape = (
            _idivup(self.slice_shape[0], 32) * 32,
            _idivup(self.slice_shape[1], 32) * 32
        )
        self.slice = numpy.zeros(self.dimrec_shape, dtype=numpy.float32)
        self.filter_name = filter_name if filter_name else "Ram-Lak"
        if axis_position:
            self.axis_pos = numpy.float32(axis_position)
        else:
            self.axis_pos = numpy.float32((sino_shape[1] - 1.) / 2)
        self.axis_array = None  # TODO: add axis correction front-end

        self.is_cpu = False
        if self.device.type == "CPU":
            self.is_cpu = True

        self.compute_fft_plans()
        self.buffers = [
                       BufferDescription("_d_slice", numpy.prod(self.dimrec_shape), numpy.float32, mf.READ_WRITE),
                       BufferDescription("d_sino", self.num_projs * self.num_bins, numpy.float32, mf.READ_WRITE),  # before transferring to texture (if available)
                       BufferDescription("d_cos", self.num_projs, numpy.float32, mf.READ_ONLY),
                       BufferDescription("d_sin", self.num_projs, numpy.float32, mf.READ_ONLY),
                       BufferDescription("d_axes", self.num_projs, numpy.float32, mf.READ_ONLY),
                      ]
        self.allocate_buffers()
        if not(self.is_cpu):
            self.allocate_textures()
        self.compute_filter()
        if self.pyfft_plan:
            self.add_to_cl_mem({
                "d_filter": self.d_filter,
                "d_sino_z": self.d_sino_z
            })
        self.d_sino = self.cl_mem["d_sino"]  # shorthand
        self.compute_angles()

        self.local_mem = 256 * 3 * _sizeof(numpy.float32)  # constant for all image sizes
        OpenclProcessing.compile_kernels(self, self.kernel_files)
        # check that workgroup can actually be (16, 16)
        self.compiletime_workgroup_size = self.kernels.max_workgroup_size("backproj_cpu_kernel")
        # Workgroup and ndrange sizes are always the same
        self.wg = (16, 16)
        self.ndrange = (
            _idivup(int(self.dimrec_shape[1]), 32) * self.wg[0],  # int(): pyopencl <= 2015.1
            _idivup(int(self.dimrec_shape[0]), 32) * self.wg[1]  # int(): pyopencl <= 2015.1
        )

    def compute_angles(self):
        if self.angles is None:
            self.angles = numpy.linspace(0, numpy.pi, self.num_projs, False)
        h_cos = numpy.cos(self.angles).astype(numpy.float32)
        h_sin = numpy.sin(self.angles).astype(numpy.float32)
        pyopencl.enqueue_copy(self.queue, self.cl_mem["d_cos"], h_cos)
        pyopencl.enqueue_copy(self.queue, self.cl_mem["d_sin"], h_sin)
        if self.axis_array:
            pyopencl.enqueue_copy(self.queue,
                                  self.cl_mem["d_axes"],
                                  self.axis_array.astype(numpy.float32))
        else:
            pyopencl.enqueue_copy(self.queue,
                                  self.cl_mem["d_axes"],
                                  numpy.ones(self.num_projs, dtype=numpy.float32) * self.axis_pos)

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
                                        hostbuf=numpy.zeros(self.shape[::-1], dtype=numpy.float32)
                                        )

    def compute_fft_plans(self):
        """
        If pyfft is installed, prepare a batched 1D FFT plan for the filtering
        of FBP

        """
        self.fft_size = nextpow2(self.num_bins * 2 - 1)
        if _has_pyfft:
            logger.debug("pyfft is available. Computing FFT plans...")
            # batched 1D transform
            self.pyfft_plan = pyfft_Plan(self.fft_size, queue=self.queue,
                                         wait_for_finish=True)
            self.d_sino_z = parray.zeros(self.queue,
                                         (self.num_projs, self.fft_size),
                                         dtype=numpy.complex64)
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
            h = numpy.zeros(L, dtype=numpy.float32)
            L2 = L // 2 + 1
            h[0] = 1 / 4.
            j = numpy.linspace(1, L2, L2 // 2, False)
            h[1:L2:2] = -1. / (numpy.pi ** 2 * j ** 2)
            h[L2:] = numpy.copy(h[1:L2 - 1][::-1])
        else:
            # TODO: other filters
            raise ValueError("Filter %s is not available" % self.filter_name)
        self.filter = h
        if self.pyfft_plan:
            self.d_filter = parray.to_device(self.queue, h.astype(numpy.complex64))
            self.pyfft_plan.execute(self.d_filter.data)
        else:
            self.filter = numpy.fft.fft(h).astype(numpy.complex64)
            self.d_filter = None

    def _get_local_mem(self):
        return pyopencl.LocalMemory(self.local_mem)  # constant for all image sizes

    def cpy2d_to_slice(self, dst):
        ndrange = (int(self.slice_shape[1]), int(self.slice_shape[0]))  # pyopencl < 2015.2
        slice_shape_ocl = numpy.int32(ndrange)
        wg = None
        kernel_args = (
            dst.data,
            self.cl_mem["_d_slice"],
            numpy.int32(self.slice_shape[1]),
            numpy.int32(self.dimrec_shape[1]),
            numpy.int32((0, 0)),
            numpy.int32((0, 0)),
            slice_shape_ocl
        )
        return self.kernels.cpy2d(self.queue, ndrange, wg, *kernel_args)

    def transfer_to_texture(self, sino):
        sino2 = sino
        if not(sino.flags["C_CONTIGUOUS"] and sino.dtype == numpy.float32):
            sino2 = numpy.ascontiguousarray(sino, dtype=numpy.float32)
        if self.is_cpu:
            ev = pyopencl.enqueue_copy(
                                        self.queue,
                                        self.d_sino,
                                        sino2
                                        )
            what = "transfer filtered sino H->D buffer"
        else:
            ev = pyopencl.enqueue_copy(
                                       self.queue,
                                       self.d_sino_tex,
                                       sino2,
                                       origin=(0, 0),
                                       region=self.shape[::-1]
                                       )
            what = "transfer filtered sino H->D texture"
        return EventDescription(what, ev)

    def transfer_device_to_texture(self, d_sino):
        if self.is_cpu:
            if id(self.d_sino) == id(d_sino):
                return
            ev = pyopencl.enqueue_copy(
                                       self.queue,
                                       self.d_sino,
                                       d_sino
                                       )
            what = "transfer filtered sino D->D buffer"
        else:
            ev = pyopencl.enqueue_copy(
                                       self.queue,
                                       self.d_sino_tex,
                                       d_sino,
                                       offset=0,
                                       origin=(0, 0),
                                       region=self.shape[::-1]
                                       )
            what = "transfer filtered sino D->D texture"
        return EventDescription(what, ev)

    def backprojection(self, sino=None, dst=None):
        """Perform the backprojection on an input sinogram

        :param sino: sinogram. If provided, it returns the plain backprojection.
        :param dst: destination (pyopencl.Array). If provided, the result will be written in this array.
        :return: backprojection of sinogram
        """
        events = []
        with self.sem:

            if sino is not None:  # assuming numpy.ndarray
                events.append(self.transfer_to_texture(sino))
            # Prepare arguments for the kernel call
            if self.is_cpu:
                d_sino_ref = self.d_sino
            else:
                d_sino_ref = self.d_sino_tex
            kernel_args = (
                self.num_projs,  # num of projections (int32)
                self.num_bins,  # num of bins (int32)
                self.axis_pos,  # axis position (float32)
                self.cl_mem["_d_slice"],  # d_slice (__global float32*)
                d_sino_ref,  # d_sino (__read_only image2d_t or float*)
                numpy.float32(0),  # gpu_offset_x (float32)
                numpy.float32(0),  # gpu_offset_y (float32)
                self.cl_mem["d_cos"],  # d_cos (__global float32*)
                self.cl_mem["d_sin"],  # d_sin (__global float32*)
                self.cl_mem["d_axes"],  # d_axis  (__global float32*)
                self._get_local_mem()  # shared mem (__local float32*)
            )
            # Call the kernel
            if self.is_cpu:
                kernel_to_call = self.kernels.backproj_cpu_kernel
            else:
                kernel_to_call = self.kernels.backproj_kernel
            event_bpj = kernel_to_call(
                self.queue,
                self.ndrange,
                self.wg,
                *kernel_args
            )
            if dst is None:
                self.slice[:] = 0
                events.append(EventDescription("backprojection", event_bpj))
                ev = pyopencl.enqueue_copy(self.queue, self.slice,
                                           self.cl_mem["_d_slice"])
                events.append(EventDescription("copy D->H result", ev))
                ev.wait()
                res = numpy.copy(self.slice)
                if self.dimrec_shape[0] > self.slice_shape[0] or self.dimrec_shape[1] > self.slice_shape[1]:
                    res = res[:self.slice_shape[0], :self.slice_shape[1]]
                # if the slice is backprojected onto a bigger grid
                if self.slice_shape[1] > self.num_bins:
                    res = res[:self.slice_shape[0], :self.slice_shape[1]]
            else:
                ev = self.cpy2d_to_slice(dst)
                events.append(EventDescription("copy D->D result", ev))
                ev.wait()
                res = dst

        # /with self.sem
        if self.profile:
            self.events += events

        return res

    def filter_projections(self, sino, rescale=True):
        """
        Performs the FBP on a given sinogram.

        :param sinogram: sinogram to (filter-)backproject
        :param rescale: if True (default), the sinogram is multiplied with
                        (pi/n_projs)
        """
        if sino.shape[0] != self.num_projs or sino.shape[1] != self.num_bins:
            raise ValueError("Expected sinogram with (projs, bins) = (%d, %d)" % (self.num_projs, self.num_bins))
        if rescale:
            sino = sino * numpy.pi / self.num_projs
        events = []
        # if pyfft is available, all can be done on the device
        if self.d_filter is not None:

            # Zero-pad the sinogram.
            # TODO: this can be done on GPU with a "Memcpy2D":
            #  cl.enqueue_copy(queue, dst, src, host_origin=(0,0), buffer_origin=(0,0), region=shape, host_pitches=(sino.shape[1],), buffer_pitches=(self.fft_size,))
            # However it does not work properly, and raises an error for pyopencl < 2017.1
            sino_zeropadded = numpy.zeros((sino.shape[0], self.fft_size), dtype=numpy.complex64)
            sino_zeropadded[:, :self.num_bins] = sino.astype(numpy.float32)
            sino_zeropadded = numpy.ascontiguousarray(sino_zeropadded, dtype=numpy.complex64)
            with self.sem:
                # send to GPU
                ev = pyopencl.enqueue_copy(self.queue, self.d_sino_z.data, sino_zeropadded)
                events.append(EventDescription("Send sino H->D", ev))

                # FFT (in-place)
                self.pyfft_plan.execute(self.d_sino_z.data, batch=self.num_projs)

                # Multiply (complex-wise) with the the filter
                ev = self.kernels.mult(self.queue,
                                       tuple(int(i) for i in self.d_sino_z.shape[::-1]),
                                       None,
                                       self.d_sino_z.data,
                                       self.d_filter.data,
                                       numpy.int32(self.fft_size),
                                       self.num_projs
                                       )
                events.append(EventDescription("complex 2D-1D multiplication", ev))
                # Inverse FFT (in-place)
                self.pyfft_plan.execute(self.d_sino_z.data, batch=self.num_projs, inverse=True)
                # Copy the real part of d_sino_z[:, :self.num_bins] (complex64) to d_sino (float32)
                ev = self.kernels.cpy2d_c2r(self.queue, self.shape[::-1], None,
                                            self.d_sino,
                                            self.d_sino_z.data,
                                            self.num_bins,
                                            self.num_projs,
                                            numpy.int32(self.fft_size)
                                            )
                events.append(EventDescription("conversion from complex padded sinogram to sinogram", ev))
                # debug
#                 ev.wait()
#                 h_sino = numpy.zeros(sino.shape, dtype=numpy.float32)
#                 ev = pyopencl.enqueue_copy(self.queue, h_sino, self.d_sino)
#                 ev.wait()
#                 numpy.save("/tmp/filtered_sinogram_%s.npy" % self.ctx.devices[0].platform.name.split()[0], h_sino)
                events.append(self.transfer_device_to_texture(self.d_sino))
            # ------
        else:  # no pyfft
            sino_filtered = fourier_filter(sino, filter_=self.filter, fft_size=self.fft_size)
            with self.sem:
                events.append(self.transfer_to_texture(sino_filtered))
        if self.profile:
            self.events += events

    def filtered_backprojection(self, sino):
        """
        Compute the filtered backprojection (FBP) on a sinogram.

        :param sino: sinogram (`numpy.ndarray`) in the format (projections,
                     bins)
        """

        self.filter_projections(sino)
        res = self.backprojection()
        return res

    __call__ = filtered_backprojection
