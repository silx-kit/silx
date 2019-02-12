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
"""Module for convolution on CPU/GPU."""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "11/02/2019"

import numpy as np

from .common import pyopencl as cl
import pyopencl.array as parray
from .processing import OpenclProcessing


class Convolution(OpenCLProcessing):
    """
    A class for performing convolution on CPU/GPU with OpenCL.
    It supports:
      - 1D, 2D, 3D convolutions
      - batched 1D and 2D
    """
    kernel_files = ["convolution.cl"]#, "convolution_batched.cl"]

    def __init__(self, shape, kernel, axes=None, ctx=None,
                 devicetype="all", platformid=None, deviceid=None,
                 profile=False, extra_options=None):
        """Constructor of OpenCL Convolution.

        :param shape: shape of the array.
        :param kernel: convolution kernel (1D, 2D or 3D).
        :param axes: axes along which the convolution is performed,
            for batched convolutions.
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

        self._configure_axes(shape, kernel, axes)
        self._allocate_memory()


    def _configure_axes(self, shape, kernel, axes):
        data_ndim = len(shape)
        kernel_ndim = kernel.ndim
        if axes is None:
            axes = tuple(np.arange(kernel_ndim))
        axes_ndim = len(axes)

        if kernel_ndim > data_ndim:
            raise ValueError("Kernel dimensions cannot exceed data dimensions")
        if kernel_ndim == data_ndim:
            # "Regular" non-separable case
            tr_name = str("nonseparable_%dD" % data_ndim)
        if kernel_ndim < data_ndim:
            # Separable/batched case
            if axes_ndim > data_ndim:
                raise ValueError("Axes dimensions cannot exceed data dimensions")
            if axes_ndim == data_ndim:
                # Separable case
                allowed_axes = {
                    # 2D data, 1D kernel (separable 2D)
                    (2, 1): [(1, 0), (0, 1)],
                    # 3D data, 1D kernel (separable 3D)
                    (3, 1): [
                        (0, 1, 2),
                        (1, 2, 0),
                        (2, 0, 1),
                        (2, 1, 0),
                        (1, 0, 2),
                        (0, 2, 1)
                    ],
                }
            if axes_ndim < data_ndim:
                # Batched case
                allowed_axes = {
                    # 2D data, 1D kernel
                    (2, 1): [(0,), (1,)],
                    # 3D data, 1D kernel
                    (3, 1): [
                        # batched 1D on 3D data
                        (0,), (1,), (2,),
                        # batched separable 2D on 3D data
                        (1, 0), (0, 1), (2, 0), (0, 2), (1, 2), (2, 1),
                    ]
                    # 3D data, 2D kernel
                    (3, 2): [(0,), (1,), (2,)],
                }








"""
Wanted:
 - 1D, 2D, 3D convol => one kernel for each dimension
 - batched 2D and 3D => other kernels...
 - Use textures when possible => still other kernels
It should be possible to make one class for all these use cases

 - compose with "ImageProcessing" ?
   if template= or dtype=   in the constructor => instantiate an ImageProcessing
   and do casts under the hood

 - Gaussian convolution => class inheriting from Convolution
  (used for blurring, ex. in sift)

 - [Bonus] utility for 3D on "big" volume, with
   H<->D transfers performed under the hood, + necessary overlap

  - Input strides and output strides ? This implies a big modification in the code


data 1D
  kernel 1D
    axis=0 [not relevant]
data 2D
  kernel 1D
    axis=0, axis=1, axis=(0, 1) [2c], axis=(1, 0) [2c]
    [used for separable 2D convol]
  kernel 2D
    axis=(1,0), axis=(0,1)  [not relevant]
data 3D
  kernel 1D
    1c: axis={0, 1, 2}
    2c: axis=
        1 0
        0 1
        2 0
        0 2
        1 2
        2 1
    3c: axis=
        0 1 2
        1 2 0
        2 0 1
        2 1 0
        1 0 2
        0 2 1
    [used for separable 3D convol]





"""









