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
__date__ = "25/01/2019"

import logging
import numpy as np
from math import pi

from .common import pyopencl
from .processing import EventDescription, OpenclProcessing, BufferDescription
from ..image.tomography import compute_ramlak_filter
from ..math.fft import FFT
from ..math.fft.clfft import __have_clfft__


if pyopencl:
    mf = pyopencl.mem_flags
    import pyopencl.array as parray
else:
    raise ImportError("Please install pyopencl in order to use opencl backprojection")
logger = logging.getLogger(__name__)


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


class SinoFilter(OpenclProcessing):
    """
    A class for performing sinogram filtering on GPU using OpenCL.
    This is a convolution in the Fourier space, along one dimension:
      - In 2D: (n_a, d_x): n_a filterings (1D FFT of size d_x)
      - In 3D: (n_z, n_a, d_x): n_z*n_a filterings (1D FFT of size d_x)
    """
    kernel_files = ["array_utils.cl"]

    def __init__(self, sino_shape, ctx=None, devicetype="all",
                 platformid=None, deviceid=None, profile=False):
        """Constructor of OpenCL FFT-Convolve.

        :param shape: shape of the sinogram.
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
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)

        self.calculate_shapes(sino_shape)
        self.init_fft()
        self.allocate_memory()
        self.compute_filter()
        self.init_kernels()


    def calculate_shapes(self, sino_shape):
        self.ndim = len(sino_shape)
        if self.ndim == 2:
            n_angles, dwidth = sino_shape
        else:
            raise ValueError("Invalid sinogram number of dimensions: expected 2 dimensions")
        self.sino_shape = sino_shape
        self.n_angles = n_angles
        self.dwidth = dwidth
        self.dwidth_padded = 2*self.dwidth # TODO nextpow2 ?
        self.sino_padded_shape = (n_angles, self.dwidth_padded)
        sino_f_shape = list(self.sino_padded_shape)
        sino_f_shape[-1] = sino_f_shape[-1]//2+1
        self.sino_f_shape = tuple(sino_f_shape)


    def init_fft(self):
        if __have_clfft__:
            self.fft_backend = "opencl"
            self.fft = FFT(
                self.sino_padded_shape,
                dtype=np.float32,
                axes=(-1,),
                backend="opencl",
                ctx=self.ctx,
            )
        else:
            self.fft_backend = "numpy"
            print("""The gpyfft module was not found. The Fourier transforms will
            be done on CPU. For more performances, it is advised to install gpyfft.""")
            self.fft = FFT(
                data=np.zeros(self.sino_padded_shape, "f"),
                axes=(-1,),
                backend="numpy",
            )


    def allocate_memory(self):
        # These are already allocated by FFT() if using the opencl backend
        if self.fft_backend == "opencl":
            self.d_sino_padded = self.fft.data_in
            self.d_sino_f = self.fft.data_out
        else:
            # When using the numpy backend, arrays are not pre-allocated
            self.d_sino_padded = np.zeros(self.sino_padded_shape, "f")
            self.d_sino_f = np.zeros(self.sino_f_shape, np.complex64)
        # These are needed for rectangular memcpy in certain cases (see below).
        self.tmp_sino_device = parray.zeros(self.queue, self.sino_shape, "f")
        self.tmp_sino_host = np.zeros(self.sino_shape, "f")


    def compute_filter(self):
        filter_ = compute_ramlak_filter(self.dwidth_padded, dtype=np.float32)
        filter_ *= pi/self.n_angles # normalization
        self.filter_f = np.fft.rfft(filter_).astype(np.complex64)
        self.d_filter_f = parray.to_device(self.queue, self.filter_f)


    def init_kernels(self):
        OpenclProcessing.compile_kernels(self, self.kernel_files)
        h, w = self.d_sino_f.shape
        self.mult_kern_args = (
            self.queue,
            np.int32(self.d_sino_f.shape[::-1]),
            None,
            self.d_sino_f.data,
            self.d_filter_f.data,
            np.int32(w),
            np.int32(h)
        )


    def check_array(self, arr):
        if arr.dtype != np.float32:
            raise ValueError("Expected data type = numpy.float32")
        if arr.shape != self.sino_shape:
            raise ValueError("Expected sinogram shape %s, got %s" % (self.sino_shape, arr.shape))
        if not(isinstance(arr, np.ndarray) or isinstance(arr, parray.Array)):
            raise ValueError("Expected either numpy.ndarray or pyopencl.array.Array")


    @staticmethod
    def check_same_array_types(arr1, arr2):
        allowed_instances = [np.ndarray, parray.Array]
        for inst in allowed_instances:
            if isinstance(arr1, inst) and not(isinstance(arr2, inst)):
                raise ValueError("Arrays must be of the same type")


    def copy2d(self, dst, src, transfer_shape, dst_offset=(0, 0), src_offset=(0, 0)):
        self.kernels.cpy2d(
            self.queue,
            np.int32(transfer_shape),
            None,
            dst.data,
            src.data,
            np.int32(dst.shape[1]),
            np.int32(src.shape[1]),
            np.int32(dst_offset),
            np.int32(src_offset),
            np.int32(transfer_shape)
        )


    def copy2d_host(self, dst, src, transfer_shape, dst_offset=(0, 0), src_offset=(0, 0)):
        s = transfer_shape
        do = dst_offset
        so = src_offset
        dst[do[0]:do[0]+s[0], do[1]:do[1]+s[1]] = src[so[0]:so[0]+s[0], so[1]:so[1]+s[1]]


    def prepare_input_sino(self, sino):
        self.check_array(sino)
        self.d_sino_padded.fill(0)
        if self.fft_backend == "opencl":
            # OpenCL backend: FFT/mult/IFFT are done on device.
            if isinstance(sino, np.ndarray):
                # OpenCL backend + numpy input: copy H->D.
                # As pyopencl does not support rectangular copies, we have to
                # do a copy H->D in a temporary device buffer, and then call a
                # kernel doing the rectangular D-D copy.
                self.tmp_sino_device[:] = sino[:]
                d_sino_ref = self.tmp_sino_device
            else:
                d_sino_ref = sino
            # Rectangular copy D->D
            self.copy2d(self.d_sino_padded, d_sino_ref, self.sino_shape[::-1])
        else:
            # Numpy backend: FFT/mult/IFFT are done on host.
            if not(isinstance(sino, np.ndarray)):
                # Numpy backend + pyopencl input: need to copy D->H
                self.tmp_sino_host[:] = sino[:]
                h_sino_ref = self.tmp_sino_host
            else:
                h_sino_ref = sino
            # Rectangular copy H->H
            self.copy2d_host(self.d_sino_padded, h_sino_ref, self.sino_shape[::-1])


    def get_output_sino(self, output):
        if output is None:
            res = np.zeros(self.sino_shape, dtype=np.float32)
        else:
            res = output
        if self.fft_backend == "opencl":
            if isinstance(res, np.ndarray):
                # OpenCL backend + numpy output: copy D->H
                # As pyopencl does not support rectangular copies, we first have
                # to call a kernel doing rectangular copy D->D, then do a copy
                # D->H.
                self.copy2d(self.tmp_sino_device, self.d_sino_padded, self.sino_shape[::-1])
                res[:] = self.tmp_sino_device[:]
            else:
                self.copy2d(res, self.d_sino_padded, self.sino_shape[::-1])
        else:
            if not(isinstance(res, np.ndarray)):
                # Numpy backend + pyopencl output: rect copy H->H + copy H->D
                self.copy2d_host(self.tmp_sino_host, self.d_sino_padded, self.sino_shape[::-1])
                res[:] = self.tmp_sino_host[:]
            else:
                # Numpy backend + numpy output: rect copy H->H
                self.copy2d_host(res, self.d_sino_padded, self.sino_shape[::-1])
        return res


    def do_fft(self):
        if self.fft_backend == "opencl":
            self.fft.fft(self.d_sino_padded, output=self.d_sino_f)
        else:
            # numpy backend does not support "output=" argument,
            # and rfft always return a complex128 result.
            res = self.fft.fft(self.d_sino_padded).astype(np.complex64)
            self.d_sino_f[:] = res[:]


    def multiply_fourier(self):
        if self.fft_backend == "opencl":
            # Everything is on device. Call the multiplication kernel.
            self.kernels.inplace_complex_mul_2Dby1D(
                *self.mult_kern_args
            )
        else:
            # Everything is on host.
            self.d_sino_f *= self.filter_f


    def do_ifft(self):
        if self.fft_backend == "opencl":
            self.fft.ifft(self.d_sino_f, output=self.d_sino_padded)
        else:
            # numpy backend does not support "output=" argument,
            # and irfft always return a float64 result.
            res = self.fft.ifft(self.d_sino_f).astype("f")
            self.d_sino_padded[:] = res[:]


    def filter_sino(self, sino, output=None):
        # Handle input sinogram
        self.prepare_input_sino(sino)
        # FFT
        self.do_fft()
        # multiply with filter in the Fourier domain
        self.multiply_fourier()
        # iFFT
        self.do_ifft()
        # return
        res = self.get_output_sino(output)
        return res


    __call__ = filter_sino



class Backprojection(OpenclProcessing):
    """A class for performing the backprojection using OpenCL"""
    kernel_files = ["backproj.cl", "array_utils.cl"]

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

        self.init_geometry(sino_shape, slice_shape, angles, axis_position)
        self.allocate_memory()
        self.compute_angles()
        self.init_kernels()
        self.init_filter(filter_name)


    def init_geometry(self, sino_shape, slice_shape, angles, axis_position):
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
        if axis_position:
            self.axis_pos = np.float32(axis_position)
        else:
            self.axis_pos = np.float32((sino_shape[1] - 1.) / 2)
        self.axis_array = None  # TODO: add axis correction front-end


    def allocate_memory(self):
        # Host memory
        self.slice = np.zeros(self.dimrec_shape, dtype=np.float32)
        self.is_cpu = False
        if self.device.type == "CPU":
            self.is_cpu = True

        # Device memory
        self.buffers = [
            BufferDescription("_d_slice", self.dimrec_shape, np.float32, mf.READ_WRITE),
            BufferDescription("d_sino", self.shape, np.float32, mf.READ_WRITE),  # before transferring to texture (if available)
            BufferDescription("d_cos", (self.num_projs,), np.float32, mf.READ_ONLY),
            BufferDescription("d_sin", (self.num_projs,), np.float32, mf.READ_ONLY),
            BufferDescription("d_axes", (self.num_projs,), np.float32, mf.READ_ONLY),
        ]
        self.allocate_buffers(use_array=True)
        self.d_sino = self.cl_mem["d_sino"]  # shorthand

        # Texture memory (if relevant)
        if not(self.is_cpu):
            self.allocate_textures()

        # Local memory
        self.local_mem = 256 * 3 * _sizeof(np.float32)  # constant for all image sizes


    def compute_angles(self):
        if self.angles is None:
            self.angles = np.linspace(0, np.pi, self.num_projs, False)
        h_cos = np.cos(self.angles).astype(np.float32)
        h_sin = np.sin(self.angles).astype(np.float32)
        self.cl_mem["d_cos"][:] = h_cos[:]
        self.cl_mem["d_sin"][:] = h_sin[:]
        if self.axis_array:
            self.cl_mem["d_axes"][:] = self.axis_array.astype(np.float32)[:]
        else:
            self.cl_mem["d_axes"][:] = np.ones(self.num_projs, dtype="f") * self.axis_pos


    def init_kernels(self):
        OpenclProcessing.compile_kernels(self, self.kernel_files)
        # check that workgroup can actually be (16, 16)
        self.compiletime_workgroup_size = self.kernels.max_workgroup_size("backproj_cpu_kernel")
        # Workgroup and ndrange sizes are always the same
        self.wg = (16, 16)
        self.ndrange = (
            _idivup(int(self.dimrec_shape[1]), 32) * self.wg[0],
            _idivup(int(self.dimrec_shape[0]), 32) * self.wg[1]
        )
        # Prepare arguments for the kernel call
        if self.is_cpu:
            d_sino_ref = self.d_sino.data
        else:
            d_sino_ref = self.d_sino_tex
        self._backproj_kernel_args = (
            # num of projections (int32)
            self.num_projs,
            # num of bins (int32)
            self.num_bins,
            # axis position (float32)
            self.axis_pos,
            # d_slice (__global float32*)
            self.cl_mem["_d_slice"].data,
            # d_sino (__read_only image2d_t or float*)
            d_sino_ref,
            # gpu_offset_x (float32) # TODO custom ?
            -np.float32((self.num_bins - 1) / 2. - self.axis_pos),
            # gpu_offset_y (float32) # TODO custom ?
            -np.float32((self.num_bins - 1) / 2. - self.axis_pos),
            # d_cos (__global float32*)
            self.cl_mem["d_cos"].data,
            # d_sin (__global float32*)
            self.cl_mem["d_sin"].data,
            # d_axis  (__global float32*)
            self.cl_mem["d_axes"].data,
            # shared mem (__local float32*)
            self._get_local_mem()
        )


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
            hostbuf=np.zeros(self.shape[::-1], dtype=np.float32)
        )


    def init_filter(self, filter_name):
        self.filter_name = filter_name or "Ram-Lak"
        self.sino_filter = SinoFilter(self.shape, ctx=self.ctx)


    def _get_local_mem(self):
        return pyopencl.LocalMemory(self.local_mem)  # constant for all image sizes


    def cpy2d_to_slice(self, dst):
        ndrange = (int(self.slice_shape[1]), int(self.slice_shape[0]))
        slice_shape_ocl = np.int32(ndrange)
        wg = None
        kernel_args = (
            dst.data,
            self.cl_mem["_d_slice"].data,
            np.int32(self.slice_shape[1]),
            np.int32(self.dimrec_shape[1]),
            np.int32((0, 0)),
            np.int32((0, 0)),
            slice_shape_ocl
        )
        return self.kernels.cpy2d(self.queue, ndrange, wg, *kernel_args)


    def transfer_to_texture(self, sino):
        if isinstance(sino, parray.Array):
            return self.transfer_device_to_texture(sino)
        sino2 = sino
        if not(sino.flags["C_CONTIGUOUS"] and sino.dtype == np.float32):
            sino2 = np.ascontiguousarray(sino, dtype=np.float32)
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
                                       d_sino.data,
                                       offset=0,
                                       origin=(0, 0),
                                       region=self.shape[::-1]
                                       )
            what = "transfer filtered sino D->D texture"
        return EventDescription(what, ev)


    def backprojection(self, sino, output=None):
        """Perform the backprojection on an input sinogram

        :param sino: sinogram.
        :param output: optional, output slice.
            If provided, the result will be written in this array.
        :return: backprojection of sinogram
        """
        events = []
        with self.sem:
            events.append(self.transfer_to_texture(sino))
            # Call the backprojection kernel
            if self.is_cpu:
                kernel_to_call = self.kernels.backproj_cpu_kernel
            else:
                kernel_to_call = self.kernels.backproj_kernel
            event_bpj = kernel_to_call(
                self.queue,
                self.ndrange,
                self.wg,
                *self._backproj_kernel_args
            )
            # Return
            if output is None:
                res = self.cl_mem["_d_slice"].get()
                res = res[:self.slice_shape[0], :self.slice_shape[1]]
            else:
                res = output
                self.cpy2d_to_slice(output)

        # /with self.sem
        if self.profile:
            self.events += events

        return res


    def filtered_backprojection(self, sino, output=None):
        """
        Compute the filtered backprojection (FBP) on a sinogram.

        :param sino: sinogram (`np.ndarray`) in the format (projections,
                     bins)
        """
        # Filter
        self.sino_filter(sino, output=self.d_sino)
        # Backproject
        res = self.backprojection(self.d_sino, output=output)
        return res

    __call__ = filtered_backprojection
