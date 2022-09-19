#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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

__authors__ = ["A. Mirone, P. Paleo"]
__license__ = "MIT"
__date__ = "01/08/2019"

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
                 profile=False
                 ):
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
        """
        # OS X enforces a workgroup size of 1 when the kernel has synchronization barriers
        # if sys.platform.startswith('darwin'): # assuming no discrete GPU
        #    raise NotImplementedError("Backprojection is not implemented on CPU for OS X yet")

        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)
        self.shape = slice_shape
        self.axis_pos = axis_position
        self.angles = angles
        self.dwidth = detector_width
        self.normalize = normalize

        # Default values
        if self.axis_pos is None:
            self.axis_pos = (self.shape[1] - 1) / 2.
        if self.dwidth is None:
            self.dwidth = self.shape[1]
        if not(np.iterable(self.angles)):
            if self.angles is None:
                self.nprojs = self.shape[0]
            else:
                self.nprojs = self.angles
            self.angles = np.linspace(start=0,
                                      stop=np.pi,
                                      num=self.nprojs,
                                      endpoint=False).astype(dtype=np.float32)
        else:
            self.nprojs = len(self.angles)
        self.offset_x = -np.float32((self.shape[1] - 1) / 2. - self.axis_pos)  # TODO: custom
        self.offset_y = -np.float32((self.shape[0] - 1) / 2. - self.axis_pos)  # TODO: custom
        # Reset axis_pos once offset are computed
        self.axis_pos0 = np.float64((self.shape[1] - 1) / 2.)

        # Workgroup, ndrange and shared size
        self.dimgrid_x = _idivup(self.dwidth, 16)
        self.dimgrid_y = _idivup(self.nprojs, 16)
        self._dimrecx = np.int32(self.dimgrid_x * 16)
        self._dimrecy = np.int32(self.dimgrid_y * 16)
        self.local_mem = 16 * 7 * _sizeof(np.float32)
        self.wg = (16, 16)
        self.ndrange = (
            int(self.dimgrid_x) * self.wg[0],  # int(): pyopencl <= 2015.1
            int(self.dimgrid_y) * self.wg[1]  # int(): pyopencl <= 2015.1
        )

        self._use_textures = self.check_textures_availability()

        # Allocate memory
        self.buffers = [
            BufferDescription("_d_sino", self._dimrecx * self._dimrecy, np.float32, mf.READ_WRITE),
            BufferDescription("d_angles", self._dimrecy, np.float32, mf.READ_ONLY),
            BufferDescription("d_beginPos", self._dimrecy * 2, np.int32, mf.READ_ONLY),
            BufferDescription("d_strideJoseph", self._dimrecy * 2, np.int32, mf.READ_ONLY),
            BufferDescription("d_strideLine", self._dimrecy * 2, np.int32, mf.READ_ONLY),
        ]
        d_axis_corrections = parray.empty(self.queue, self.nprojs, np.float32)
        d_axis_corrections.fill(np.float32(0.0))
        self.add_to_cl_mem(
            {
                "d_axis_corrections": d_axis_corrections
            }
        )
        self._tmp_extended_img = np.zeros((self.shape[0] + 2, self.shape[1] + 2),
                                          dtype=np.float32)
        if not(self._use_textures):
            self.allocate_slice()
        else:
            self.allocate_textures()
        self.allocate_buffers()
        self._ex_sino = np.zeros((self._dimrecy, self._dimrecx),
                                 dtype=np.float32)
        if not(self._use_textures):
            self.cl_mem["d_slice"].fill(0.)
            # enqueue_fill_buffer has issues if opencl 1.2 is not present
            # ~ pyopencl.enqueue_fill_buffer(
                # ~ self.queue,
                # ~ self.cl_mem["d_slice"],
                # ~ np.float32(0),
                # ~ 0,
                # ~ self._tmp_extended_img.size * _sizeof(np.float32)
            # ~ )
        # Precomputations
        self.compute_angles()
        self.proj_precomputations()
        self.cl_mem["d_axis_corrections"].fill(0.)
        # enqueue_fill_buffer has issues if opencl 1.2 is not present
        # ~ pyopencl.enqueue_fill_buffer(
                                    # ~ self.queue,
                                    # ~ self.cl_mem["d_axis_corrections"],
                                    # ~ np.float32(0),
                                    # ~ 0,
                                    # ~ self.nprojs*_sizeof(np.float32)
                                    # ~ )
        # Shorthands
        self._d_sino = self.cl_mem["_d_sino"]

        compile_options = None
        if not(self._use_textures):
            compile_options = "-DDONT_USE_TEXTURES"
        OpenclProcessing.compile_kernels(
            self,
            self.kernel_files,
            compile_options=compile_options
        )
        # check that workgroup can actually be (16, 16)
        self.compiletime_workgroup_size = self.kernels.max_workgroup_size("forward_kernel_cpu")

    def compute_angles(self):
        angles2 = np.zeros(self._dimrecy, dtype=np.float32)  # dimrecy != num_projs
        angles2[:self.nprojs] = np.copy(self.angles)
        angles2[self.nprojs:] = angles2[self.nprojs - 1]
        self.angles2 = angles2
        pyopencl.enqueue_copy(self.queue, self.cl_mem["d_angles"], angles2)

    def allocate_slice(self):
        ary = parray.empty(self.queue, (self.shape[1] + 2, self.shape[1] + 2), np.float32)
        ary.fill(0)
        self.add_to_cl_mem({"d_slice": ary})

    def allocate_textures(self):
        self.d_image_tex = pyopencl.Image(
                self.ctx,
                mf.READ_ONLY | mf.USE_HOST_PTR,
                pyopencl.ImageFormat(
                    pyopencl.channel_order.INTENSITY,
                    pyopencl.channel_type.FLOAT
                ), hostbuf=np.ascontiguousarray(self._tmp_extended_img.T),
            )

    def transfer_to_texture(self, image):
        image2 = image
        if not(image.flags["C_CONTIGUOUS"] and image.dtype == np.float32):
            image2 = np.ascontiguousarray(image)
        if not(self._use_textures):
            # TODO: create NoneEvent
            return self.transfer_to_slice(image2)
            # ~ return pyopencl.enqueue_copy(
                        # ~ self.queue,
                        # ~ self.cl_mem["d_slice"].data,
                        # ~ image2,
                        # ~ origin=(1, 1),
                        # ~ region=image.shape[::-1]
                        # ~ )
        else:
            return pyopencl.enqueue_copy(
                       self.queue,
                       self.d_image_tex,
                       image2,
                       origin=(1, 1),
                       region=image.shape[::-1]
                   )

    def transfer_device_to_texture(self, d_image):
        if not(self._use_textures):
            # TODO this copy should not be necessary
            return self.cpy2d_to_slice(d_image)
        else:
            return pyopencl.enqueue_copy(
                self.queue,
                self.d_image_tex,
                d_image,
                offset=0,
                origin=(1, 1),
                region=(int(self.shape[1]), int(self.shape[0]))  # self.shape[::-1] # pyopencl <= 2015.2
            )

    def transfer_to_slice(self, image):
        image2 = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.float32)
        image2[1:-1, 1:-1] = image.astype(np.float32)
        self.cl_mem["d_slice"].set(image2)

    def proj_precomputations(self):
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

        beginPos[0][case1] = 0
        beginPos[1][case1] = 0
        strideJoseph[0][case1] = 1
        strideJoseph[1][case1] = 0
        strideLine[0][case1] = 0
        strideLine[1][case1] = 1

        beginPos[0][case2] = dimslice - 1
        beginPos[1][case2] = dimslice - 1
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

        # For debug purpose
        # ~ self.beginPos = beginPos
        # ~ self.strideJoseph = strideJoseph
        # ~ self.strideLine = strideLine
        #

        pyopencl.enqueue_copy(self.queue, self.cl_mem["d_beginPos"], beginPos)
        pyopencl.enqueue_copy(self.queue, self.cl_mem["d_strideJoseph"], strideJoseph)
        pyopencl.enqueue_copy(self.queue, self.cl_mem["d_strideLine"], strideLine)

    def _get_local_mem(self):
        return pyopencl.LocalMemory(self.local_mem)  # constant for all image sizes

    def cpy2d_to_sino(self, dst):
        ndrange = (int(self.dwidth), int(self.nprojs))  # pyopencl < 2015.2
        sino_shape_ocl = np.int32(ndrange)
        wg = None
        kernel_args = (
            dst.data,
            self._d_sino,
            np.int32(self.dwidth),
            np.int32(self._dimrecx),
            np.int32((0, 0)),
            np.int32((0, 0)),
            sino_shape_ocl
        )
        return self.kernels.cpy2d(self.queue, ndrange, wg, *kernel_args)

    def cpy2d_to_slice(self, src):
        """
        copy a Nx * Ny slice to self.d_slice which is (Nx+2)*(Ny+2)
        """
        ndrange = (int(self.shape[1]), int(self.shape[0]))  # self.shape[::-1] # pyopencl < 2015.2
        wg = None
        slice_shape_ocl = np.int32(ndrange)
        kernel_args = (
            self.cl_mem["d_slice"].data,
            src,
            np.int32(self.shape[1] + 2),
            np.int32(self.shape[1]),
            np.int32((1, 1)),
            np.int32((0, 0)),
            slice_shape_ocl
        )
        return self.kernels.cpy2d(self.queue, ndrange, wg, *kernel_args)

    def projection(self, image=None, dst=None):
        """Perform the projection on an input image

        :param image: Image to project
        :return: A sinogram
        """
        events = []
        with self.sem:
            if image is not None:
                assert image.ndim == 2, "Treat only 2D images"
                assert image.shape[0] == self.shape[0], "image shape is OK"
                assert image.shape[1] == self.shape[1], "image shape is OK"
                if self._use_textures:
                    self.transfer_to_texture(image)
                    slice_ref = self.d_image_tex
                else:
                    self.transfer_to_slice(image)
                    slice_ref = self.cl_mem["d_slice"].data
            else:
                if not(self._use_textures):
                    slice_ref = self.cl_mem["d_slice"].data
                else:
                    slice_ref = self.d_image_tex

            kernel_args = (
                self._d_sino,
                slice_ref,
                np.int32(self.shape[1]),
                np.int32(self.dwidth),
                self.cl_mem["d_angles"],
                np.float32(self.axis_pos0),
                self.cl_mem["d_axis_corrections"].data,  # TODO custom
                self.cl_mem["d_beginPos"],
                self.cl_mem["d_strideJoseph"],
                self.cl_mem["d_strideLine"],
                np.int32(self.nprojs),
                self._dimrecx,
                self._dimrecy,
                self.offset_x,
                self.offset_y,
                np.int32(1),  # josephnoclip, 1 by default
                np.int32(self.normalize)
            )

            # Call the kernel
            if not(self._use_textures):
                event_pj = self.kernels.forward_kernel_cpu(
                    self.queue,
                    self.ndrange,
                    self.wg,
                    *kernel_args
                )
            else:
                event_pj = self.kernels.forward_kernel(
                    self.queue,
                    self.ndrange,
                    self.wg,
                    *kernel_args
                )
            events.append(EventDescription("projection", event_pj))
            if dst is None:
                self._ex_sino[:] = 0
                ev = pyopencl.enqueue_copy(self.queue, self._ex_sino, self._d_sino)
                events.append(EventDescription("copy D->H result", ev))
                ev.wait()
                res = np.copy(self._ex_sino[:self.nprojs, :self.dwidth])
            else:
                ev = self.cpy2d_to_sino(dst)
                events.append(EventDescription("copy D->D result", ev))
                ev.wait()
                res = dst
        # /with self.sem
        if self.profile:
            self.events += events
        # ~ res = self._ex_sino
        return res

    __call__ = projection
