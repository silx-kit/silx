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
"""Module for tomographic projector on the GPU"""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["A. Mirone, P. Paleo"]
__license__ = "MIT"
__date__ = "28/02/2018"

import logging
import numpy as np

from .common import pyopencl
from .processing import EventDescription, OpenclProcessing, BufferDescription
from .backprojection import _sizeof, _idivup

if pyopencl:
    mf = pyopencl.mem_flags
    import pyopencl.array as parray
else:
    raise ImportError("pyopencl is not installed")
logger = logging.getLogger(__name__)


class Projection(OpenclProcessing):
    """
    A class for performing a tomographic projection (Radon Transform) using
    OpenCL
    """
    kernel_files = ["proj.cl", "array_utils.cl"]
    logger.warning("Forward Projecter is untested and unsuported for now")

    def __init__(self, slice_shape, angles, axis_position=None,
                 detector_width=None, normalize=False, ctx=None,
                 devicetype="all", platformid=None, deviceid=None,
                 profile=False, extra_options=None):
        """Constructor of the OpenCL projector.

        :param slice_shape: shape of the slice: (num_rows, num_columns).
        :param angles: Either an integer number of angles, or a list of custom
                       angles values in radian.
        :param axis_position: Optional, axis position. Default is
                              `(shape[1]-1)/2.0`.
        :param detector_width: Optional, detector width in pixels.
                               If detector_width > slice_shape[1], the
                               projection data will be surrounded with zeros.
                               Using detector_width < slice_shape[1] might
                               result in a local tomography setup.
        :param normalize: Optional, normalization. If set, the sinograms are
                          multiplied by the factor pi/(2*nprojs).
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by
                           clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param profile: switch on profiling to be able to profile at the kernel
                        level, store profiling elements (makes code slightly
                        slower)
        :param extra_options: dict containing advanced options.
            Current allowed options:
        """
        # OS X enforces a workgroup size of 1 when the kernel has synchronization barriers
        # if sys.platform.startswith('darwin'): # assuming no discrete GPU
        #    raise NotImplementedError("Backprojection is not implemented on CPU for OS X yet")

        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)

        self._configure_extra_options(extra_options)
        self._init_geometry(slice_shape, axis_position, angles, detector_width)
        self.normalize = normalize
        self._configure_kernel_args()
        self._allocate_memory()
        self.compute_angles()
        self._proj_precomputations()
        self._compile_kernels()

    def _configure_extra_options(self, extra_options):
        self.extra_options = {
            "offset_x": None,
            "offset_y": None,
            "axis_corrections": None, # TODO
            "dont_use_textures": False, # TODO
        }
        extra_opts = extra_options or {}
        self.extra_options.update(extra_opts)

    def _init_geometry(self, slice_shape, axis_position, angles, detector_width):
        self.shape = slice_shape
        self.axis_pos = axis_position or (self.shape[1] - 1) / 2.
        self.dwidth = detector_width or self.shape[1]
        self.angles = angles
        if not(np.iterable(self.angles)):
            if self.angles is None:
                self.nprojs = self.shape[0]
            else:
                self.nprojs = self.angles
            self.angles = np.linspace(
                start=0,
                stop=np.pi,
                num=self.nprojs,
                endpoint=False
            ).astype(dtype=np.float32)
        else:
            self.nprojs = len(self.angles)
        self.offset_x = self.extra_options["offset_x"] or \
            -np.float32((self.shape[1] - 1) / 2. - self.axis_pos)
        self.offset_y = self.extra_options["offset_y"] or \
            -np.float32((self.shape[0] - 1) / 2. - self.axis_pos)
        self.axis_pos0 = np.float((self.shape[1] - 1) / 2.)

    def _configure_kernel_args(self):
        self.dimgrid_x = _idivup(self.dwidth, 16)
        self.dimgrid_y = _idivup(self.nprojs, 16)
        self._dimrecx = np.int32(self.dimgrid_x * 16)
        self._dimrecy = np.int32(self.dimgrid_y * 16)
        self.local_mem = 16 * 7 * _sizeof(np.float32)
        self.wg = (16, 16)
        self.ndrange = (
            int(self.dimgrid_x) * self.wg[0],
            int(self.dimgrid_y) * self.wg[1]
        )
        self.is_cpu = False
        if self.device.type == "CPU":
            self.is_cpu = True

    def _allocate_memory(self):
        # Describe memory needed
        self.buffers = [
            BufferDescription("d_sino", (self._dimrecy, self._dimrecx), np.float32, mf.READ_WRITE),
            BufferDescription("d_angles", (self._dimrecy,), np.float32, mf.READ_ONLY),
            BufferDescription("_d_beginPos", (self._dimrecy * 2,), np.int32, mf.READ_ONLY),
            BufferDescription("_d_strideJoseph", (self._dimrecy * 2,), np.int32, mf.READ_ONLY),
            BufferDescription("_d_strideLine", (self._dimrecy * 2,), np.int32, mf.READ_ONLY),
            BufferDescription("d_axis_corrections", (self.nprojs,), np.float32, mf.READ_ONLY),
        ]
        if self.is_cpu:
            self.buffers.append(
                BufferDescription("d_slice", (self.shape[1] + 2, self.shape[1] + 2), np.float32, mf.READ_WRITE),
            )
        else:
            self.d_image_tex = self.allocate_texture((self.shape[0] + 2, self.shape[1] + 2))
        # Allocate
        self.allocate_buffers(use_array=True)
        # Create shortcuts
        for buffer_desc in self.buffers:
            buffer_name = buffer_desc.name
            setattr(self, buffer_name, self.cl_mem[buffer_name])
        # Fill with zeros
        for arr in [self.d_axis_corrections]:
            arr.fill(0.)
        if self.is_cpu:
            self.d_slice.fill(0.)
        # Allocate a tmp_image on host (see transfer_to_slice)
        self.tmp_image = np.zeros((self.shape[0] + 2, self.shape[1] + 2), dtype=np.float32)

    def _compile_kernels(self):
        OpenclProcessing.compile_kernels(self, self.kernel_files)
        # check that workgroup can actually be (16, 16)
        self.compiletime_workgroup_size = self.kernels.max_workgroup_size("forward_kernel_cpu")
        # Configure projection kernel arguments
        if not(self.is_cpu):
            slice_ref = self.d_image_tex
            self._projection_kernel = self.kernels.forward_kernel
        else:
            slice_ref = self.cl_mem["d_slice"].data
            self._projection_kernel = self.kernels.forward_kernel_cpu
        self.kernel_args = (
            self.queue,
            self.ndrange,
            self.wg,
            self.d_sino.data,
            slice_ref,
            np.int32(self.shape[1]),
            np.int32(self.dwidth),
            self.d_angles.data,
            np.float32(self.axis_pos0),
            self.d_axis_corrections.data,
            self._d_beginPos.data,
            self._d_strideJoseph.data,
            self._d_strideLine.data,
            np.int32(self.nprojs),
            self._dimrecx,
            self._dimrecy,
            self.offset_x,
            self.offset_y,
            np.int32(1),  # josephnoclip, 1 by default
            np.int32(self.normalize)
        )

    def compute_angles(self):
        angles2 = np.zeros(self._dimrecy, dtype=np.float32)  # dimrecy != num_projs
        angles2[:self.nprojs] = np.copy(self.angles)
        angles2[self.nprojs:] = angles2[self.nprojs - 1]
        self.angles2 = angles2
        self.d_angles[:] = angles2[:]

    def _proj_precomputations(self):
        beginPos = np.zeros((2, self._dimrecy), dtype=np.int32)
        strideJoseph = np.zeros((2, self._dimrecy), dtype=np.int32)
        strideLine = np.zeros((2, self._dimrecy), dtype=np.int32)
        cos_angles = np.cos(self.angles2)
        sin_angles = np.sin(self.angles2)
        dimslice = self.shape[1]

        M1 = np.abs(cos_angles) > 0.70710678
        M1b = np.logical_not(M1)
        M2 = cos_angles > 0
        M2b = np.logical_not(M2)
        M3 = sin_angles > 0
        M3b = np.logical_not(M3)
        case1 = M1 * M2
        case2 = M1 * M2b
        case3 = M1b * M3
        case4 = M1b * M3b

        beginPos[:, case1] = 0
        strideJoseph[0][case1] = 1
        strideJoseph[1][case1] = 0
        strideLine[0][case1] = 0
        strideLine[1][case1] = 1

        beginPos[:, case2] = dimslice - 1
        strideJoseph[0][case2] = -1
        strideJoseph[1][case2] = 0
        strideLine[0][case2] = 0
        strideLine[1][case2] = -1

        beginPos[0][case3] = dimslice - 1
        beginPos[1][case3] = 0
        strideJoseph[0][case3] = 0
        strideJoseph[1][case3] = 1
        strideLine[0][case3] = -1
        strideLine[1][case3] = 0

        beginPos[0][case4] = 0
        beginPos[1][case4] = dimslice - 1
        strideJoseph[0][case4] = 0
        strideJoseph[1][case4] = -1
        strideLine[0][case4] = 1
        strideLine[1][case4] = 0

        pyopencl.enqueue_copy(self.queue, self._d_beginPos.data, beginPos)
        pyopencl.enqueue_copy(self.queue, self._d_strideJoseph.data, strideJoseph)
        pyopencl.enqueue_copy(self.queue, self._d_strideLine.data, strideLine)

    def _get_local_mem(self):
        return pyopencl.LocalMemory(self.local_mem)  # constant for all image sizes

    def transfer_to_slice(self, image):
        # pyopencl does not support rectangular copies, so we use this workaround
        # Another way would be to (1) transfer image to a tmp device array, and
        # (2) shift the array of (1, 1) pixel
        self.tmp_image[1:-1, 1:-1] = image.astype(np.float32)
        self.cl_mem["d_slice"][:] = self.tmp_image[:]

    def cpy2d_to_sino(self, dst):
        """
        copy a sinogram to self.d_sino which is dimrecx * dimrecy
        """
        ndrange = (int(self.dwidth), int(self.nprojs))
        sino_shape_ocl = np.int32(ndrange)
        wg = None
        kernel_args = (
            dst.data,
            self.d_sino.data,
            np.int32(self.dwidth),
            np.int32(self._dimrecx),
            np.int32((0, 0)),
            np.int32((0, 0)),
            sino_shape_ocl
        )
        return self.kernels.cpy2d(self.queue, ndrange, wg, *kernel_args)

    #~ def cpy2d_to_slice(self, src):
        #~ """
        #~ copy a Nx * Ny slice to self.d_slice which is (Nx+2)*(Ny+2)
        #~ """
        #~ ndrange = (int(self.shape[1]), int(self.shape[0]))
        #~ wg = None
        #~ slice_shape_ocl = np.int32(ndrange)
        #~ kernel_args = (
            #~ self.d_slice.data,
            #~ src,
            #~ np.int32(self.shape[1] + 2),
            #~ np.int32(self.shape[1]),
            #~ np.int32((1, 1)),
            #~ np.int32((0, 0)),
            #~ slice_shape_ocl
        #~ )
        #~ return self.kernels.cpy2d(self.queue, ndrange, wg, *kernel_args)

    def set_image(self, image):
        if not(self.is_cpu):
            self.transfer_to_texture(image, self.d_image_tex, origin=(1, 1))
        else:
            self.transfer_to_slice(image)

    def projection(self, image, output=None):
        """Perform the projection on an input image

        :param image: Image to project. Can be a numpy array or pyopencl array.
        :param output: Optional, output image. Can be a numpy or a pyopencl array.
        :return: A sinogram
        """
        events = []
        self.sem.acquire()

        self.set_image(image)

        event_pj = self._projection_kernel(*self.kernel_args)
        events.append(EventDescription("projection", event_pj))


        if output is None:
            res = self.d_sino.get()
            res = res[:self.nprojs, :self.dwidth] # copy ?
        else:
            ev = self.cpy2d_to_sino(output)
            events.append(EventDescription("copy D->D result", ev))
            ev.wait()
            res = output
        self.sem.release()
        if self.profile:
            self.events += events
        return res

    __call__ = projection
