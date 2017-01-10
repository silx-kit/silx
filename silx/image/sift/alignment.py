#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
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
Contains classes for image alignment on a reference images.
"""

from __future__ import division, print_function, with_statement

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/09/2016"
__status__ = "production"

import gc
from threading import Semaphore
import numpy
from silx.opencl import ocl, pyopencl, kernel_workgroup_size
from .utils import calc_size, matching_correction, get_opencl_code
import logging
logger = logging.getLogger("sift.alignment")
if not pyopencl:
    logger.warning("No PyOpenCL, no sift")

from .match import MatchPlan
from .plan import SiftPlan

try:
    import feature
except ImportError:
    feature = None


def arrow_start(kplist):
#     x_ref = kplist.x
#     y_ref = kplist.y
    angle_ref = kplist.angle
    scale_ref = kplist.scale
    x_ref2 = kplist.x + scale_ref * numpy.cos(angle_ref)
    y_ref2 = kplist.y + scale_ref * numpy.sin(angle_ref)
    return x_ref2, y_ref2


def transform_pts(matrix, offset, x, y):
    nx = -offset[1] + y * matrix[1, 0] + x * matrix[1, 1]
    ny = -offset[0] + x * matrix[0, 1] + y * matrix[0, 0]
    return nx, ny


class LinearAlign(object):
    """
    Align images on a reference image based on an afine transformation (bi-linear + offset)
    """
    kernels = {"transform": 128}

    def __init__(self, image, devicetype="CPU", profile=False, device=None, max_workgroup_size=None,
                 ROI=None, extra=0, context=None, init_sigma=None):
        """
        Constructor of the class

        :param image: reference image on which other image should be aligned
        :param devicetype: Kind of preferred devce
        :param profile:collect profiling information ?
        :param device: 2-tuple of integer. see clinfo
        :param max_workgroup_size: limit the workgroup size
        :param ROI: Region of interest: to be implemented
        :param extra: extra space around the image, can be an integer, or a 2 tuple in YX convention: TODO!
        :param init_sigma: bluring width, you should have good reasons to modify the 1.6 default value...
        """
        self.profile = bool(profile)
        self.events = []
        self.program = None
        self.ref = numpy.ascontiguousarray(image, numpy.float32)
        self.buffers = {}
        self.shape = image.shape
        if len(self.shape) == 3:
            self.RGB = True
            self.shape = self.shape[:2]
        elif len(self.shape) == 2:
            self.RGB = False
        else:
            raise RuntimeError("Unable to process image of shape %s" % (tuple(self.shape,)))
        if "__len__" not in dir(extra):
            self.extra = (int(extra), int(extra))
        else:
            self.extra = extra[:2]
        self.outshape = tuple(i + 2 * j for i, j in zip(self.shape, self.extra))
        self.ROI = ROI
        if context:
            self.ctx = context
            device_name = self.ctx.devices[0].name.strip()
            platform_name = self.ctx.devices[0].platform.name.strip()
            platform = ocl.get_platform(platform_name)
            device = platform.get_device(device_name)
            self.device = platform.id, device.id
        else:
            if device is None:
                self.device = ocl.select_device(type=devicetype, best=True)
            else:
                self.device = device
            self.ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[self.device[0]].get_devices()[self.device[1]]])
        ocldevice = ocl.platforms[self.device[0]].devices[self.device[1]]
        self.devicetype = ocldevice.type

        if max_workgroup_size:
            self.max_workgroup_size = min(int(max_workgroup_size), ocldevice.max_work_group_size)
        else:
            self.max_workgroup_size = ocldevice.max_work_group_size
        self.kernels = {}
        for k, v in self.__class__.kernels.items():
            self.kernels[k] = min(v, self.max_workgroup_size)

        if self.RGB:
            target = (4, 8, 4)
            self.wg = tuple(min(t, i, self.max_workgroup_size) for t, i in zip(target, self.ctx.devices[0].max_work_item_sizes))
        else:
            target = (8, 4)
            self.wg = tuple(min(t, i, self.max_workgroup_size) for t, i in zip(target, self.ctx.devices[0].max_work_item_sizes))

        self.sift = SiftPlan(template=image, context=self.ctx, profile=self.profile,
                             max_workgroup_size=self.max_workgroup_size, init_sigma=init_sigma)
        self.ref_kp = self.sift.keypoints(image)
        if self.ROI is not None:
            kpx = numpy.round(self.ref_kp.x).astype(numpy.int32)
            kpy = numpy.round(self.ref_kp.y).astype(numpy.int32)
            masked = self.ROI[(kpy, kpx)].astype(bool)
            logger.warning("Reducing keypoint list from %i to %i because of the ROI" % (self.ref_kp.size, masked.sum()))
            self.ref_kp = self.ref_kp[masked]
        self.match = MatchPlan(context=self.ctx, profile=self.profile, max_workgroup_size=self.max_workgroup_size)
#        Allocate reference keypoints on the GPU within match context:
        self.buffers["ref_kp_gpu"] = pyopencl.array.to_device(self.match.queue, self.ref_kp)
        # TODO optimize match so that the keypoint2 can be optional
        self.fill_value = 0
#        print self.ctx.devices[0]
        if self.profile:
            self.queue = pyopencl.CommandQueue(self.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = pyopencl.CommandQueue(self.ctx)
        self._compile_kernels()
        self._allocate_buffers()
        self.sem = Semaphore()
        self.relative_transfo = None

    def __del__(self):
        """
        Destructor: release all buffers
        """
        self._free_kernels()
        self._free_buffers()
        self.queue = None
        self.ctx = None
        gc.collect()

    def _allocate_buffers(self):
        """
        All buffers are allocated here
        """
        if self.RGB:
            self.buffers["input"] = pyopencl.array.empty(self.queue, shape=self.shape + (3,), dtype=numpy.uint8)
            self.buffers["output"] = pyopencl.array.empty(self.queue, shape=self.outshape + (3,), dtype=numpy.uint8)
        else:
            self.buffers["input"] = pyopencl.array.empty(self.queue, shape=self.shape, dtype=numpy.float32)
            self.buffers["output"] = pyopencl.array.empty(self.queue, shape=self.outshape, dtype=numpy.float32)
        self.buffers["matrix"] = pyopencl.array.empty(self.queue, shape=(2, 2), dtype=numpy.float32)
        self.buffers["offset"] = pyopencl.array.empty(self.queue, shape=(1, 2), dtype=numpy.float32)

    def _free_buffers(self):
        """
        free all memory allocated on the device
        """
        for buffer_name in self.buffers:
            if self.buffers[buffer_name] is not None:
                try:
                    del self.buffers[buffer_name]
                    self.buffers[buffer_name] = None
                except pyopencl.LogicError:
                    logger.error("Error while freeing buffer %s" % buffer_name)

    def _compile_kernels(self):
        """
        Call the OpenCL compiler
        """
        for kernel in list(self.kernels.keys()):
            kernel_src = get_opencl_code(kernel)
            try:
                program = pyopencl.Program(self.ctx, kernel_src).build()
            except pyopencl.MemoryError as error:
                raise MemoryError(error)
            self.program = program
            for one_function in program.all_kernels():
                workgroup_size = kernel_workgroup_size(program, one_function)
                self.kernels[kernel+"."+one_function.function_name] = workgroup_size


    def _free_kernels(self):
        """
        free all kernels
        """
        self.program = None

    def align(self, img, shift_only=False, return_all=False, double_check=False, relative=False, orsa=False):
        """
        Align image on reference image

        :param img: numpy array containing the image to align to reference
        :param return_all: return in addition ot the image, keypoints, matching keypoints, and transformations as a dict
        :param reltive: update reference keypoints with those from current image to perform relative alignment
        :return: aligned image or all informations
        """
        logger.debug("ref_keypoints: %s" % self.ref_kp.size)
        if self.RGB:
            data = numpy.ascontiguousarray(img, numpy.uint8)
        else:
            data = numpy.ascontiguousarray(img, numpy.float32)
        with self.sem:
            cpy = pyopencl.enqueue_copy(self.queue, self.buffers["input"].data, data)
            if self.profile:
                self.events.append(("Copy H->D", cpy))
            cpy.wait()
            kp = self.sift.keypoints(self.buffers["input"])
#            print("ref %s img %s" % (self.buffers["ref_kp_gpu"].shape, kp.shape))
            logger.debug("mod image keypoints: %s" % kp.size)
            raw_matching = self.match.match(self.buffers["ref_kp_gpu"], kp, raw_results=True)
#            print(raw_matching.max(axis=0))

            matching = numpy.recarray(shape=raw_matching.shape, dtype=MatchPlan.dtype_kp)
            len_match = raw_matching.shape[0]
            if len_match == 0:
                logger.warning("No matching keypoints")
                return
            matching[:, 0] = self.ref_kp[raw_matching[:, 0]]
            matching[:, 1] = kp[raw_matching[:, 1]]

            if orsa:
                if feature:
                    matching = feature.sift_orsa(matching, self.shape, 1)
                else:
                    logger.warning("feature is not available. No ORSA filtering")

            if (len_match < 3 * 6) or (shift_only):  # 3 points per DOF
                if shift_only:
                    logger.debug("Shift Only mode: Common keypoints: %s" % len_match)
                else:
                    logger.warning("Shift Only mode: Common keypoints: %s" % len_match)
                dx = matching[:, 1].x - matching[:, 0].x
                dy = matching[:, 1].y - matching[:, 0].y
                matrix = numpy.identity(2, dtype=numpy.float32)
                offset = numpy.array([+numpy.median(dy), +numpy.median(dx)], numpy.float32)
            else:
                logger.debug("Common keypoints: %s" % len_match)

                transform_matrix = matching_correction(matching)
                offset = numpy.array([transform_matrix[5], transform_matrix[2]], dtype=numpy.float32)
                matrix = numpy.empty((2, 2), dtype=numpy.float32)
                matrix[0, 0], matrix[0, 1] = transform_matrix[4], transform_matrix[3]
                matrix[1, 0], matrix[1, 1] = transform_matrix[1], transform_matrix[0]
            if double_check and (len_match >= 3 * 6):  # and abs(matrix - numpy.identity(2)).max() > 0.1:
                logger.warning("Validating keypoints, %s,%s" % (matrix, offset))
                dx = matching[:, 1].x - matching[:, 0].x
                dy = matching[:, 1].y - matching[:, 0].y
                dangle = matching[:, 1].angle - matching[:, 0].angle
                dscale = numpy.log(matching[:, 1].scale / matching[:, 0].scale)
                distance = numpy.sqrt(dx * dx + dy * dy)
                outlayer = numpy.zeros(distance.shape, numpy.int8)
                outlayer += abs((distance - distance.mean()) / distance.std()) > 4
                outlayer += abs((dangle - dangle.mean()) / dangle.std()) > 4
                outlayer += abs((dscale - dscale.mean()) / dscale.std()) > 4
#                 print(outlayer)
                outlayersum = outlayer.sum()
                if outlayersum > 0 and not numpy.isinf(outlayersum):
                    matching2 = matching[outlayer == 0]
                    transform_matrix = matching_correction(matching2)
                    offset = numpy.array([transform_matrix[5], transform_matrix[2]], dtype=numpy.float32)
                    matrix = numpy.empty((2, 2), dtype=numpy.float32)
                    matrix[0, 0], matrix[0, 1] = transform_matrix[4], transform_matrix[3]
                    matrix[1, 0], matrix[1, 1] = transform_matrix[1], transform_matrix[0]
            if relative:  # update stable part to perform a relative alignment
                self.ref_kp = kp
                if self.ROI is not None:
                    kpx = numpy.round(self.ref_kp.x).astype(numpy.int32)
                    kpy = numpy.round(self.ref_kp.y).astype(numpy.int32)
                    masked = self.ROI[(kpy, kpx)].astype(bool)
                    logger.warning("Reducing keypoint list from %i to %i because of the ROI" % (self.ref_kp.size, masked.sum()))
                    self.ref_kp = self.ref_kp[masked]
                self.buffers["ref_kp_gpu"] = pyopencl.array.to_device(self.match.queue, self.ref_kp)
                transfo = numpy.zeros((3, 3), dtype=numpy.float64)
                transfo[:2, :2] = matrix
                transfo[0, 2] = offset[0]
                transfo[1, 2] = offset[1]
                transfo[2, 2] = 1
                if self.relative_transfo is None:
                    self.relative_transfo = transfo
                else:
                    self.relative_transfo = numpy.dot(transfo, self.relative_transfo)
                matrix = numpy.ascontiguousarray(self.relative_transfo[:2, :2], dtype=numpy.float32)
                offset = numpy.ascontiguousarray(self.relative_transfo[:2, 2], dtype=numpy.float32)
#                print(self.relative_transfo)
            cpy1 = pyopencl.enqueue_copy(self.queue, self.buffers["matrix"].data, matrix)
            cpy2 = pyopencl.enqueue_copy(self.queue, self.buffers["offset"].data, offset)
            if self.profile:
                self.events += [("Copy matrix", cpy1), ("Copy offset", cpy2)]

            if self.RGB:
                shape = (4, self.shape[1], self.shape[0])
                transform = self.program.transform_RGB
            else:
                shape = self.shape[1], self.shape[0]
                transform = self.program.transform
#             print(kernel_workgroup_size(self.program, transform), self.wg, self.ctx.devices[0].max_work_item_sizes)
            ev = transform(self.queue, calc_size(shape, self.wg), self.wg,
                           self.buffers["input"].data,
                           self.buffers["output"].data,
                           self.buffers["matrix"].data,
                           self.buffers["offset"].data,
                           numpy.int32(self.shape[1]),
                           numpy.int32(self.shape[0]),
                           numpy.int32(self.outshape[1]),
                           numpy.int32(self.outshape[0]),
                           self.sift.buffers["min"].get()[0],
                           numpy.int32(1))
            if self.profile:
                self.events += [("transform", ev)]
            result = self.buffers["output"].get()

#        print (self.buffers["offset"])
        if return_all:
#            corr = numpy.dot(matrix, numpy.vstack((matching[:, 1].y, matching[:, 1].x))).T - \
#                   offset.T - numpy.vstack((matching[:, 0].y, matching[:, 0].x)).T
            corr = numpy.dot(matrix, numpy.vstack((matching[:, 0].y, matching[:, 0].x))).T + offset.T - numpy.vstack((matching[:, 1].y, matching[:, 1].x)).T
            rms = numpy.sqrt((corr * corr).sum(axis=-1).mean())

            # Todo: calculate the RMS of deplacement and return it:
            return {"result": result, "keypoint": kp, "matching": matching, "offset": offset, "matrix": matrix, "rms": rms}
        return result
    __call__ = align

    def log_profile(self):
        """
        If we are in debugging mode, prints out all timing for every single OpenCL call
        """
        t = 0.0
#         orient = 0.0
#         descr = 0.0
        if self.profile:
            for e in self.events:
                if "__len__" in dir(e) and len(e) >= 2:
                    et = 1e-6 * (e[1].profile.end - e[1].profile.start)
                    print("%50s:\t%.3fms" % (e[0], et))
                    t += et
