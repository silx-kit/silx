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
"""Module for OpenCL FFT"""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "15/10/2018"

import logging
import numpy as np

from .common import pyopencl
from .processing import EventDescription, OpenclProcessing, BufferDescription

import pyopencl.array as parray
logger = logging.getLogger(__name__)

cl = pyopencl


try:
    import gpyfft
    from gpyfft.fft import FFT as gpyfft_fft
except ImportError:
    msg = "Unable to import gpyfft. Please install it from: https://github.com/geggo/gpyfft"
    logger.warning(msg)
#     raise ImportError(msg)
    gpyfft = None
    gpyfft_fft = None
else:
    if not hasattr(gpyfft, "__version__"):
        msg = "Please install a more recent version of gpyfft from https://github.com/geggo/gpyfft"
        logger.warning(msg)
        gpyfft = None
        gpyfft_fft = None

#
# TODO: "output_size" argument for zero-padding.
# In the case of rfft, it has to be taken into account in compute_output_shape()
#


class FFT(OpenclProcessing):
    """A class for OpenCL FFT"""

    def __init__(self, shape, axes=None, force_complex=False, fast_math=False, double_precision=False,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 profile=False
                 ):
        """Constructor of the class OpenCL FFT.

        :param shape: shape of the input data.
        :param axes: Optional, the axes to perform the FFT on.
        :param force_complex: Optional, whether force complex-to-complex transform
                              instead of performing "rfft" when possible.
        :param fast_math: Optional, use fast math.
        :param double_precision: Optional, use double precision computation on device.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param profile: switch on profiling to be able to profile at the kernel level,
                        store profiling elements (makes code slightly slower)

        Notes
        -------

        For Real to Complex (R2C) FFT of an array of shape (N1, N2),
        gpyfft will output a (N1//2+1, N2) array by default.
        This contrasts with the usual convention which is (N1, N2//2+1),
        i.e the "fast" dimension is halved.
        See for example:
        - http://docs.nvidia.com/cuda/cufft/#multi-dimensional
        - http://www.fftw.org/doc/Multi_002dDimensional-DFTs-of-Real-Data.html
        In this class, we enforce the "usual" convention.
        """

        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)
        self.shape = shape
        if axes is None:
            # fftn
            self.axes = tuple(range(len(shape))[::-1])  # FFTW convention
        else:
            self.axes = axes
        self.fast_math = fast_math
        self.double_precision = double_precision
        self.real_fft = not(force_complex)
        self.is_cpu = (self.device.type == "CPU")
        self.output_shape = self.compute_output_shape()

        self.allocate_arrays()
        self.compute_plans()

    def compute_output_shape(self):
        if self.real_fft:
            # See "Notes" in the class docstring
            lastdim_size = self.shape[-1] // 2 + 1
            return self.shape[:-1] + (lastdim_size,)
        else:
            return self.shape

    def allocate_arrays(self):
        # R2C
        if self.real_fft:
            if self.double_precision:
                self.input_dtype = np.float64
                self.output_dtype = np.complex128
            else:
                self.input_dtype = np.float32
                self.output_dtype = np.complex64
        # C2C
        else:
            if self.double_precision:
                self.input_dtype = np.complex128
                self.output_dtype = np.complex128
            else:
                self.input_dtype = np.complex64
                self.output_dtype = np.complex64

        self.d_input = parray.zeros(self.queue, self.shape, dtype=self.input_dtype)
        self.d_output = parray.zeros(self.queue, self.output_shape, dtype=self.output_dtype)
        self.d_input_old_ref = None
        self.d_output_old_ref = None

    def compute_plans(self):
        if gpyfft_fft is None:
            raise ImportError("Unable to import gpyfft. Please install it from: https://github.com/geggo/gpyfft")
        self.plan_forward = gpyfft_fft(self.ctx, self.queue, self.d_input, self.d_output, axes=self.axes)
        # ~ self.plan_inverse = gpyfft_fft(self.ctx, self.queue, self.d_output, self.d_input, real=True)

    @staticmethod
    def _checkarray(arr, dtype):
        if not(arr.flags["C_CONTIGUOUS"] and arr.dtype == dtype):
            return np.ascontiguousarray(arr, dtype=dtype)
        else:
            return arr

    def update_input_array(self, array, dtype):
        if isinstance(array, np.ndarray):
            # TODO check size/shape ?
            array = self._checkarray(array, dtype)
            # FIXME gpyfft "update_arrays" is not implemented yet
            # assuming id(self.plan_forward.data) == id(self.d_input)
            evt = cl.enqueue_copy(self.queue, self.d_input.data, array)
            if self.profile:
                self.events.append(EventDescription("copy H->D", evt))
        elif isinstance(array, cl.array.Array):
            # No copy, use the provided parray data directly
            self.d_input_old_ref = self.d_input
            self.d_input = array
            self.plan_forward.data = array
        else:
            raise ValueError("Unsupported array type - please use either numpy.ndarray or pyopencl.array.Array")

    def update_output_array(self, array, dtype):
        if isinstance(array, cl.array.Array):
            # No copy, use the provided parray data directly
            # assuming id(self.plan_forward.output) == id(self.d_output)
            self.d_output_old_ref = self.d_output
            self.d_output = array
        else:
            raise ValueError("Please use a pyopencl.array.Array as the output keyword argument of fft()")

    def recover_array_references(self):
        if self.d_input_old_ref is not None:
            self.d_input = self.d_input_old_ref
            self.d_input_old_ref = None
            self.plan_forward.data = self.d_input
        if self.d_output_old_ref is not None:
            self.d_output = self.d_output_old_ref
            self.d_output_old_ref = None

    def fft(self, input_array, output=None):
        """
        Forward FT.

        :param input: input array, either numpy.ndarray or pyopencl.Array. pyopencl.Buffer are not supported !
        :param output: Optional, output array, which has to be a pyopencl.Array.
                       If provided, mind the output shape for rfft !
                       If not provided, the output is returned as a new numpy array.
        """

        self.update_input_array(input_array, self.input_dtype)
        if output is not None:
            self.update_output_array(output, self.output_dtype, "output")

        ev, = self.plan_forward.enqueue()
        if self.profile:
            self.events.append(EventDescription("Forward FFT", ev))

        if output is None:
            # res = self.d_output.get()
            res = np.empty(self.d_output.shape, self.d_output.dtype)
            evt = cl.enqueue_copy(self.queue, res, self.d_output.data)
            if self.profile:
                self.events.append(EventDescription("copy D->H", evt))
        else:
            res = self.d_output
        self.recover_array_references()

        return res
