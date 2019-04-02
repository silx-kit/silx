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

from .common import pyopencl
from .processing import EventDescription, OpenclProcessing, BufferDescription
from .sinofilter import SinoFilter
from .sinofilter import fourier_filter as fourier_filter_
from ..utils.deprecation import deprecated

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


class Backprojection(OpenclProcessing):
    """A class for performing the backprojection using OpenCL"""
    kernel_files = ["backproj.cl", "array_utils.cl"]

    def __init__(self, sino_shape, slice_shape=None, axis_position=None,
                 angles=None, filter_name=None, ctx=None, devicetype="all",
                 platformid=None, deviceid=None, profile=False,
                 extra_options=None):
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
        :param extra_options: Advanced extra options in the form of a dict.
            Current options are: cutoff, use_numpy_fft
        """
        # OS X enforces a workgroup size of 1 when the kernel has
        # synchronization barriers if sys.platform.startswith('darwin'):
        #  assuming no discrete GPU
        #    raise NotImplementedError("Backprojection is not implemented on CPU for OS X yet")

        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)

        self._init_geometry(sino_shape, slice_shape, angles, axis_position,
                           extra_options)
        self._allocate_memory()
        self._compute_angles()
        self._init_kernels()
        self._init_filter(filter_name)

    def _init_geometry(self, sino_shape, slice_shape, angles, axis_position,
                      extra_options):
        """Geometry Initialization

        :param sino_shape: shape of the sinogram. The sinogram is in the format
                           (n_b, n_a) where n_b is the number of detector bins
                           and n_a is the number of angles.
        :param slice_shape: shape of the reconstructed slice. By
                            default, it is a square slice where the dimension
                            is the "x dimension" of the sinogram (number of
                            bins).
        :param angles: list of projection angles in radian.
        :param axis_position: axis position
        :param dict extra_options: Advanced extra options
        """
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
        self._init_extra_options(extra_options)

    def _init_extra_options(self, extra_options):
        """Backprojection extra option initialization

        :param dict extra_options: Advanced extra options
        """
        self.extra_options = {
            "cutoff": 1.,
            "use_numpy_fft": False,
            # It is  axis_pos - (num_bins-1)/2  in PyHST
            "gpu_offset_x": 0., #self.axis_pos - (self.num_bins - 1) / 2.,
            "gpu_offset_y": 0., #self.axis_pos - (self.num_bins - 1) / 2.
        }
        if extra_options is not None:
            self.extra_options.update(extra_options)

    def _allocate_memory(self):
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
            self._allocate_textures()

        # Local memory
        self.local_mem = 256 * 3 * _sizeof(np.float32)  # constant for all image sizes

    def _compute_angles(self):
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

    def _init_kernels(self):
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
            # gpu_offset_x (float32) 
            np.float32(self.extra_options["gpu_offset_x"]),
            # gpu_offset_y (float32)
            np.float32(self.extra_options["gpu_offset_y"]),
            # d_cos (__global float32*)
            self.cl_mem["d_cos"].data,
            # d_sin (__global float32*)
            self.cl_mem["d_sin"].data,
            # d_axis  (__global float32*)
            self.cl_mem["d_axes"].data,
            # shared mem (__local float32*)
            self._get_local_mem()
        )

    def _allocate_textures(self):
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

    def _init_filter(self, filter_name):
        """Filter initialization

        :param str filter_name: filter name
        """
        self.filter_name = filter_name or "ram-lak"
        self.sino_filter = SinoFilter(
            self.shape,
            ctx=self.ctx,
            filter_name=self.filter_name,
            extra_options=self.extra_options,
        )

    def _get_local_mem(self):
        return pyopencl.LocalMemory(self.local_mem)  # constant for all image sizes

    def _cpy2d_to_slice(self, dst):
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

    def _transfer_to_texture(self, sino):
        if isinstance(sino, parray.Array):
            return self._transfer_device_to_texture(sino)
        sino2 = sino
        if not(sino.flags["C_CONTIGUOUS"] and sino.dtype == np.float32):
            sino2 = np.ascontiguousarray(sino, dtype=np.float32)
        if self.is_cpu:
            ev = pyopencl.enqueue_copy(
                                        self.queue,
                                        self.d_sino.data,
                                        sino2
                                        )
            what = "transfer filtered sino H->D buffer"
            ev.wait()
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

    def _transfer_device_to_texture(self, d_sino):
        if self.is_cpu:
            if id(self.d_sino) == id(d_sino):
                return
            ev = pyopencl.enqueue_copy(
                                       self.queue,
                                       self.d_sino.data,
                                       d_sino
                                       )
            what = "transfer filtered sino D->D buffer"
            ev.wait()
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
            events.append(self._transfer_to_texture(sino))
            # Call the backprojection kernel
            if self.is_cpu:
                kernel_to_call = self.kernels.backproj_cpu_kernel
            else:
                kernel_to_call = self.kernels.backproj_kernel
            kernel_to_call(
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
                self._cpy2d_to_slice(output)

        # /with self.sem
        if self.profile:
            self.events += events

        return res

    def filtered_backprojection(self, sino, output=None):
        """
        Compute the filtered backprojection (FBP) on a sinogram.

        :param sino: sinogram (`np.ndarray` or `pyopencl.array.Array`)
            with the shape (n_projections, n_bins)
        :param output: output (`np.ndarray` or `pyopencl.array.Array`).
            If nothing is provided, a new numpy array is returned.
        """
        # Filter
        self.sino_filter(sino, output=self.d_sino)
        # Backproject
        res = self.backprojection(self.d_sino, output=output)
        return res

    __call__ = filtered_backprojection


    # -------------------
    # - Compatibility  -
    # -------------------

    @deprecated(replacement="Backprojection.sino_filter", since_version="0.10")
    def filter_projections(self, sino, rescale=True):
        self.sino_filter(sino, output=self.d_sino)



def fourier_filter(sino, filter_=None, fft_size=None):
    return fourier_filter_(sino, filter_=filter_, fft_size=fft_size)

