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


class ConvolutionInfos(object):
    allowed_axes = {
        "1D": [None],
        "separable_2D_1D_2D": [None, (0, 1), (1, 0)],
        "batched_1D_2D": [(0,), (1,)],
        "separable_3D_1D_3D": [
            None,
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
            (1, 0, 2),
            (0, 2, 1)
        ],
        "batched_1D_3D": [(0,), (1,), (2,)],
        "batched_separable_2D_1D_3D": [(0,), (1,), (2,)], # unsupported (?)
        "2D": [None],
        "batched_2D_3D": [(0,), (1,), (2,)],
        "separable_3D_2D_3D": [
            (1, 0),
            (0, 1),
            (2, 0),
            (0, 2),
            (1, 2),
            (2, 1),
        ],
        "3D": [None],
    }
    use_cases = {
        (1, 1): {
            "1D": {
                "name": "1D convolution on 1D data",
                "kernels": ["convol_1D_X"],
            },
        },
        (2, 2): {
            "2D": {
                "name": "2D convolution on 2D data",
                "kernels": ["convol_2D_XY"],
            },
        },
        (3, 3): {
            "3D": {
                "name": "3D convolution on 3D data",
                "kernels": ["convol_3D_XYZ"],
            },
        },
        (2, 1): {
            "separable_2D_1D_2D": {
                "name": "Separable (2D->1D) convolution on 2D data",
                "kernels": ["convol_1D_X", "convol_1D_Y"],
            },
            "batched_1D_2D": {
                "name": "Batched 1D convolution on 2D data",
                "kernels": ["convol_1D_X", "convol_1D_Y"],
            },
        },
        (3, 1): {
            "separable_3D_1D_3D": {
                "name": "Separable (3D->1D) convolution on 3D data",
                "kernels": ["convol_1D_X", "convol_1D_Y", "convol_1D_Z"],
            },
            "batched_1D_3D": {
                "name": "Batched 1D convolution on 3D data",
                "kernels": ["convol_1D_X", "convol_1D_Y", "convol_1D_Z"],
            },
            "batched_separable_2D_1D_3D": {
                "name": "Batched separable (2D->1D) convolution on 3D data",
                "kernels": ["convol_1D_X", "convol_1D_Y", "convol_1D_Z"],
            },
        },
        (3, 2): {
            "separable_3D_2D_3D": {
                "name": "Separable (3D->2D) convolution on 3D data",
                "kernels": ["convol_2D_XY", "convol_2D_XZ", "convol_2D_YZ"],
            },
            "batched_2D_3D": {
                "name": "Batched 2D convolution on 3D data",
                "kernels": ["convol_2D_XY", "convol_2D_XZ", "convol_2D_YZ"],
            },
        },
    }




class Convolution(OpenclProcessing):
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
        :param extra_options: Advanced options (dict). Current options are:
            "allocate_input_array": True,
            "allocate_output_array": True,
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)

        self._configure_extra_options(extra_options)
        self._determine_use_case(shape, kernel, axes)
        self._allocate_memory()
        self._init_kernels()


    # TODO for separable transform, "allocate_tmp_array"
    # for swapping references instead of copying data_out to data_in
    def _configure_extra_options(self, extra_options):
        self.extra_options = {
            "allocate_input_array": True,
            "allocate_output_array": True,
            "allocate_tmp_array": True,
        }
        extra_opts = extra_options or {}
        self.extra_options.update(extra_opts)


    def _get_dimensions(self, shape, kernel):
        self.shape = shape
        self.data_ndim = self._check_dimensions(shape=shape, name="Data")
        self.kernel_ndim = self._check_dimensions(arr=kernel, name="Kernel")
        Nx = shape[-1]
        if self.data_ndim >= 2:
            Ny = shape[-2]
        else:
            Ny = 1
        if self.data_ndim >= 3:
            Nz = shape[-3]
        else:
            Nz = 1
        self.Nx = np.int32(Nx)
        self.Ny = np.int32(Ny)
        self.Nz = np.int32(Nz)


    def _determine_use_case(self, shape, kernel, axes):
        """
        Determine the convolution use case from the input/kernel shape, and axes.
        """
        self._get_dimensions(shape, kernel)
        if self.kernel_ndim > self.data_ndim:
            raise ValueError("Kernel dimensions cannot exceed data dimensions")
        data_ndim = self.data_ndim
        kernel_ndim = self.kernel_ndim
        self.kernel = kernel.astype("f")

        convol_infos = ConvolutionInfos()
        k = (data_ndim, kernel_ndim)
        if k not in convol_infos.use_cases:
            raise ValueError(
                "Cannot find a use case for data ndim = %d and kernel ndim = %d"
                % (data_ndim, kernel_ndim)
            )
        possible_use_cases = convol_infos.use_cases[k]

        self.use_case_name = None
        for uc_name, uc_params in possible_use_cases.items():
            if axes in convol_infos.allowed_axes[uc_name]:
                self.use_case_name = uc_name
                self.use_case_desc = uc_params["name"]
                self.use_case_kernels = uc_params["kernels"]
        if self.use_case_name is None:
            raise ValueError(
                "Cannot find a use case for data ndim = %d, kernel ndim = %d and axes=%s"
                % (data_ndim, kernel_ndim, str(axes))
            )
        #
        if uc_name == "batched_separable_2D_1D_3D":
            raise NotImplementedError("The use case %s is not implemented" % uc_name)
        #
        self.axes = axes
        # Replace "axes=None" with an actual value (except for ND-ND)
        allowed_axes = convol_infos.allowed_axes[uc_name]
        if len(allowed_axes) > 1:
            self.axes = allowed_axes[1] # The default choice might impact perfs
        self.separable = self.use_case_name.startswith("separable")
        self.batched = self.use_case_name.startswith("batched")


    def _allocate_memory(self):
        option_array_names = {
            "allocate_input_array": "data_in",
            "allocate_output_array": "data_out",
            "allocate_tmp_array": "data_tmp",
        }
        for option_name, array_name in option_array_names.items():
            if self.extra_options[option_name]:
                value = parray.zeros(self.queue, self.shape, "f")
            else:
                value = None
            setattr(self, array_name, value)

        if isinstance(self.kernel, np.ndarray):
            self.d_kernel = parray.to_device(self.queue, self.kernel)
        else:
            if not(isinstance(self.kernel, parray.Array)):
                raise ValueError("kernel must be either numpy array or pyopencl array")
            self.d_kernel = self.kernel
        self._old_input_ref = None
        self._old_output_ref = None


    def _init_kernels(self):
        compile_options = None
        self.compile_kernels(
            kernel_files=None,
            compile_options=compile_options
        )
        self.ndrange = np.int32(self.shape)[::-1]
        self.wg = None
        self.kernel_args = {
            1: (
                self.queue,
                self.ndrange,
                self.wg,
                self.data_in.data,
                self.data_out.data,
                self.d_kernel.data,
                np.int32(self.kernel.shape[0]),
                self.Nx,
                self.Ny,
                self.Nz
            )
        }


    @staticmethod
    def _check_dimensions(arr=None, shape=None, name="", dim_min=1, dim_max=3):
        if shape is not None:
            ndim = len(shape)
        elif arr is not None:
            ndim = arr.ndim
        else:
            raise ValueError("Please provide either arr= or shape=")
        if ndim < dim_min or ndim > dim_max:
            raise ValueError("%s dimensions should be between %d and %d"
                % (name, dim_min, dim_max)
            )
        return ndim




    def _check_array(self, arr):
        # TODO allow cl.Buffer ?
        if not(isinstance(arr, parray.Array) or isinstance(arr, np.ndarray)):
            raise TypeError("Expected either pyopencl.array.Array or numpy.ndarray")
        # TODO composition with ImageProcessing/cast ?
        if arr.dtype != np.float32:
            raise TypeError("Data must be float32")
        if arr.shape != self.shape:
            raise ValueError("Expected data shape = %s" % str(self.shape))



    def _set_arrays(self, array, output=None):
        if isinstance(array, np.ndarray):
            self.data_in[:] = array[:]
        else:
            self._old_input_ref = self.data_in
            self.data_in = array
        if output is not None:
            if isinstance(output, np.ndarray):
                self.data_out.fill(0)
            else:
                self._old_output_ref = self.data_out
                self.data_out = output



    def _separable_convolution(self):
        assert len(self.axes) == len(self.use_case_kernels)
        # Separable: one kernel call per data dimension
        for i, axis in enumerate(self.axes):
            self._batched_convolution(axis)


    def _batched_convolution(self, axis):
        # Batched: one kernel call in total
        print("Doing batched %s along axis %d" % (self.use_case_kernels[axis], axis)) # DEBUG
        opencl_kernel = self.kernels.get_kernel(self.use_case_kernels[axis])
        opencl_kernel_args = self.kernel_args[self.kernel_ndim]
        opencl_kernel(*opencl_kernel_args) # TODO event


    def _recover_arrays_references(self):
        if self._old_input_ref is not None:
            self.data_in = self._old_input_ref
            self._old_input_ref = None
        if self._old_output_ref is not None:
            self.data_out = self._old_output_ref
            self._old_output_ref = None



    def _get_output(self, output):
        if output is None:
            res = self.data_out.get()
        else:
            res = output
            if isinstance(output, np.ndarray):
                output[:] = self.data_out[:]
        self._recover_arrays_references()
        return res


    # TODO
    #  - Modify kernel on the fly
    #  - Modify axes on the fly
    def convolve(self, array, output=None):
        self._check_array(array)
        self._set_arrays(array, output=output)

        if self.axes is not None:
            if self.separable:
                self._separable_convolution()
            elif self.batched:
                assert len(self.axes) == 1 #
                self._batched_convolution(self.axes[0])
            # else: ND-ND convol
        else:
            # ND-ND convol
            raise NotImplementedError()

        res = self._get_output(output)
        return res


    __call__ = convolve


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


Use case name                       Kernel name
------------------------------------------------------------------
1D convol on 1D data                  convol_1D_X
batched 1D convol on 2D data          convol_1D_X or convol_1D_Y
separable (2,1)D convol on 2D data    convol_1D_X and convol_1D_X

batched 1D convol on 3D data          convol_1D_X or convol_1D_Y or convol_1D_Z
separable (3,1) 1D convol on 3D data  convol_1D_X and convol_1D_Y and convol_1D_Z
[batched separable 2D on 3D data]     convol_1D_X and convol_1D_Y and convol_1D_Z

2D convol on 2D data                  convol_2D_XY
batched 2D convol on 3D data          convol_2D_XY or convol_2D_XZ or convol_2D_YZ
separable (3, 2)D convol on 3D data   convol_2D_XY and convol_2D_XZ and convol_2D_YZ

3D convol on 3D data                  convol_3D_XYZ



(1, 1)
(2, 1), axes in [None, (1, 0), (0, 1)] => separable (2D->1D) on 2D data
(2, 1), axes in [(0,), (1,)]   => batched 1D on 2D data

(3, 1), axes in [None, valid 3-tuple] => separable (3D->1D) on 3D data
(3, 1), axes in [1-tuple] => batched 1D on 3D
(3, 1), axes in [valid 2-tuple] => batched (along 1 axis) separable (2D->1D) [same as (2, 1, axes=None) if Nz==1]

(2, 2) => (nonseparable) 2D on 2D data
(3, 2), axes in [None, valid 2-tuple] => separable (3D->2D) on 3D data   (along y then z  or  along x then z   or   along x then y)
(3, 2), axes in [1-tuple] => batched 2D convol on 3D data

(3, 3) => (nonseparable) 3D convol

"""








