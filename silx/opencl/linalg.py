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
"""Module for basic linear algebra in OpenCL"""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "01/08/2019"

import numpy as np

from .common import pyopencl
from .processing import EventDescription, OpenclProcessing

import pyopencl.array as parray
cl = pyopencl


class LinAlg(OpenclProcessing):

    kernel_files = ["linalg.cl"]

    def __init__(self, shape, do_checks=False, ctx=None, devicetype="all", platformid=None, deviceid=None, profile=False):
        """
        Create a "Linear Algebra" plan for a given image shape.

        :param shape: shape of the image (num_rows, num_columns)
        :param do_checks (optional): if True, memory and data type checks are performed when possible.
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

        self.d_gradient = parray.empty(self.queue, shape, np.complex64)
        self.d_gradient.fill(np.complex64(0.0))
        self.d_image = parray.empty(self.queue, shape, np.float32)
        self.d_image.fill(np.float32(0.0))
        self.add_to_cl_mem({
            "d_gradient": self.d_gradient,
            "d_image": self.d_image
        })

        self.wg2D = None
        self.shape = shape
        self.ndrange2D = (
            int(self.shape[1]),
            int(self.shape[0])
        )
        self.do_checks = bool(do_checks)
        OpenclProcessing.compile_kernels(self, self.kernel_files)

    @staticmethod
    def check_array(array, dtype, shape, arg_name):
        if array.shape != shape or array.dtype != dtype:
            raise ValueError("%s should be a %s array of type %s" %(arg_name, str(shape), str(dtype)))

    def get_data_references(self, src, dst, default_src_ref, default_dst_ref):
        """
        From various types of src and dst arrays,
        returns the references to the underlying data (Buffer) that will be used by the OpenCL kernels.
        # TODO documentation

        This function will make a copy host->device if the input is on host (eg. numpy array)
        """
        if dst is not None:
            if isinstance(dst, cl.array.Array):
                dst_ref = dst.data
            elif isinstance(dst, cl.Buffer):
                dst_ref = dst
            else:
                raise ValueError("dst should be either pyopencl.array.Array or pyopencl.Buffer")
        else:
            dst_ref = default_dst_ref

        if isinstance(src, cl.array.Array):
            src_ref = src.data
        elif isinstance(src, cl.Buffer):
            src_ref = src
        else:  # assuming numpy.ndarray
            evt = cl.enqueue_copy(self.queue, default_src_ref, src)
            self.events.append(EventDescription("copy H->D", evt))
            src_ref = default_src_ref
        return src_ref, dst_ref

    def gradient(self, image, dst=None, return_to_host=False):
        """
        Compute the spatial gradient of an image.
        The gradient is computed with first-order difference (not central difference).

        :param image: image to compute the gradient from. It can be either a numpy.ndarray, a pyopencl Array or Buffer.
        :param dst: optional, reference to a destination pyopencl Array or Buffer. It must be of complex64 data type.
        :param return_to_host: optional, set to True if you want the result to be transferred back to host.

        if dst is provided, it should be of type numpy.complex64 !
        """
        n_y, n_x = np.int32(self.shape)
        if self.do_checks:
            self.check_array(image, np.float32, self.shape, "image")
            if dst is not None:
                self.check_array(dst, np.complex64, self.shape, "dst")
        img_ref, grad_ref = self.get_data_references(image, dst, self.d_image.data, self.d_gradient.data)

        # Prepare the kernel call
        kernel_args = [
            img_ref,
            grad_ref,
            n_x,
            n_y
        ]
        # Call the gradient kernel
        evt = self.kernels.kern_gradient2D(
            self.queue,
            self.ndrange2D,
            self.wg2D,
            *kernel_args
        )
        self.events.append(EventDescription("gradient2D", evt))
        # TODO: should the wait be done in any case ?
        # In the case where dst=None, the wait() is mandatory since a user will be doing arithmetic on dst afterwards
        if dst is None:
            evt.wait()

        if return_to_host:
            if dst is not None:
                res_tmp = self.d_gradient.get()
            else:
                res_tmp = np.zeros(self.shape, dtype=np.complex64)
                cl.enqueue_copy(self.queue, res_tmp, grad_ref)
            res = np.zeros((2,) + self.shape, dtype=np.float32)
            res[0] = np.copy(res_tmp.real)
            res[1] = np.copy(res_tmp.imag)
            return res
        else:
            return dst

    def divergence(self, gradient, dst=None, return_to_host=False):
        """
        Compute the spatial divergence of an image.
        The divergence is designed to be the (negative) adjoint of the gradient.

        :param gradient: gradient-like array to compute the divergence from. It can be either a numpy.ndarray, a pyopencl Array or Buffer.
        :param dst: optional, reference to a destination pyopencl Array or Buffer. It must be of complex64 data type.
        :param return_to_host: optional, set to True if you want the result to be transferred back to host.

        if dst is provided, it should be of type numpy.complex64 !
        """
        n_y, n_x = np.int32(self.shape)
        # numpy.ndarray gradients are expected to be (2, n_y, n_x)
        if isinstance(gradient, np.ndarray):
            gradient2 = np.zeros(self.shape, dtype=np.complex64)
            gradient2.real = np.copy(gradient[0])
            gradient2.imag = np.copy(gradient[1])
            gradient = gradient2
        elif self.do_checks:
            self.check_array(gradient, np.complex64, self.shape, "gradient")
            if dst is not None:
                self.check_array(dst, np.float32, self.shape, "dst")
        grad_ref, img_ref = self.get_data_references(gradient, dst, self.d_gradient.data, self.d_image.data)

        # Prepare the kernel call
        kernel_args = [
            grad_ref,
            img_ref,
            n_x,
            n_y
        ]
        # Call the gradient kernel
        evt = self.kernels.kern_divergence2D(
            self.queue,
            self.ndrange2D,
            self.wg2D,
            *kernel_args
        )
        self.events.append(EventDescription("divergence2D", evt))
        # TODO: should the wait be done in any case ?
        # In the case where dst=None, the wait() is mandatory since a user will be doing arithmetic on dst afterwards
        if dst is None:
            evt.wait()

        if return_to_host:
            if dst is not None:
                res = self.d_image.get()
            else:
                res = np.zeros(self.shape, dtype=np.float32)
                cl.enqueue_copy(self.queue, res, img_ref)
            return res
        else:
            return dst
