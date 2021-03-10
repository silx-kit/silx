#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
import numpy as np
from pkg_resources import parse_version


def check_version(package, required_version):
    """
    Check whether a given package version is superior or equal to required_version.
    """
    try:
        ver = getattr(package, "__version__")
    except AttributeError:
        try:
            ver = getattr(package, "version")
        except Exception:
            return False
    req_v = parse_version(required_version)
    ver_v = parse_version(ver)
    return ver_v >= req_v


class BaseFFT(object):
    """
    Base class for all FFT backends.
    """
    def __init__(self, **kwargs):
        self.__get_args(**kwargs)

        if self.shape is None and self.dtype is None and self.template is None:
            raise ValueError("Please provide either (shape and dtype) or template")
        if self.template is not None:
            self.shape = self.template.shape
            self.dtype = self.template.dtype
        self.user_data = self.template
        self.data_allocated = False
        self.__calc_axes()
        self.__set_dtypes()
        self.__calc_shape()

    def __get_args(self, **kwargs):
        expected_args = {
            "shape": None,
            "dtype": None,
            "template": None,
            "shape_out": None,
            "axes": None,
            "normalize": "rescale",
        }
        for arg_name, default_val in expected_args.items():
            if arg_name not in kwargs:
                # Base class was not instantiated properly
                raise ValueError("Please provide argument %s" % arg_name)
            setattr(self, arg_name, default_val)
        for arg_name, arg_val in kwargs.items():
            setattr(self, arg_name, arg_val)

    def __set_dtypes(self):
        dtypes_mapping = {
            np.dtype("float32"): np.complex64,
            np.dtype("float64"): np.complex128,
            np.dtype("complex64"): np.complex64,
            np.dtype("complex128"): np.complex128
        }
        dp = {
            np.dtype("float32"): np.float64,
            np.dtype("complex64"): np.complex128
        }
        self.dtype_in = np.dtype(self.dtype)
        if self.dtype_in not in dtypes_mapping:
            raise ValueError("Invalid input data type: got %s" %
                self.dtype_in
            )
        self.dtype_out = dtypes_mapping[self.dtype_in]

    def __calc_shape(self):
        # TODO allow for C2C even for real input data (?)
        if self.dtype_in in [np.float32, np.float64]:
            last_dim = self.shape[-1]//2 + 1
            # FFTW convention
            self.shape_out = self.shape[:-1] + (self.shape[-1]//2 + 1,)
        else:
            self.shape_out = self.shape

    def __calc_axes(self):
        default_axes = tuple(range(len(self.shape)))
        if self.axes is None:
            self.axes = default_axes
            self.user_axes = None
        else:
            self.user_axes = self.axes
            # Handle possibly negative axes
            self.axes = tuple(np.array(default_axes)[np.array(self.user_axes)])

    def _allocate(self, shape, dtype):
        raise ValueError("This should be implemented by back-end FFT")

    def set_data(self, dst, src, shape, dtype, copy=True):
        raise ValueError("This should be implemented by back-end FFT")

    def allocate_arrays(self):
        if not(self.data_allocated):
            self.data_in = self._allocate(self.shape, self.dtype_in)
            self.data_out = self._allocate(self.shape_out, self.dtype_out)
            self.data_allocated = True

    def set_input_data(self, data, copy=True):
        if data is None:
            return self.data_in
        else:
            return self.set_data(self.data_in, data, self.shape, self.dtype_in, copy=copy, name="data_in")

    def set_output_data(self, data, copy=True):
        if data is None:
            return self.data_out
        else:
            return self.set_data(self.data_out, data, self.shape_out, self.dtype_out, copy=copy, name="data_out")

    def fft(self, array, **kwargs):
        raise ValueError("This should be implemented by back-end FFT")

    def ifft(self, array, **kwargs):
        raise ValueError("This should be implemented by back-end FFT")
