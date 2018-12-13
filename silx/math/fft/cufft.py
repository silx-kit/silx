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

from .basefft import BaseFFT
try:
    import pycuda
    import pycuda.gpuarray as gpuarray
    from skcuda.fft import Plan
    from skcuda.fft import fft as cu_fft
    from skcuda.fft import ifft as cu_ifft
    __have_cufft__ = True
except ImportError:
    __have_cufft__ = False

class CUFFT(BaseFFT):
    def __init__(
        self,
        shape=None,
        dtype=None,
        data=None,
        shape_out=None,
        axes=None,
        normalize="rescale",
        stream=None,
    ):
        """
        Initialize a cufft plan.
        Please see FFT class for parameters help.

        CUFFT-specific parameters:
        stream: pycuda.driver.Stream
          Stream with which to associate the plan. If no stream is specified,
          the default stream is used.
        """
        if not(__have_cufft__) or not(__have_cufft__):
            raise ImportError("Please install pycuda and scikit-cuda to use the CUDA back-end")

        super(CUFFT, self).__init__(
            shape=shape,
            dtype=dtype,
            data=data,
            shape_out=shape_out,
            axes=axes,
            normalize=normalize,
        )
        self.cufft_stream = stream
        self.backend = "cufft"

        self.configure_batched_transform()
        self.allocate_arrays()
        self.real_transform = np.isrealobj(self.data_in)
        self.compute_forward_plan()
        self.compute_inverse_plan()
        self.refs = {
            "data_in": self.data_in,
            "data_out": self.data_out,
        }
        self.configure_normalization()


    def _allocate(self, shape, dtype):
        return gpuarray.zeros(shape, dtype)


    # TODO support batched transform where batch is other than dimension 0
    def configure_batched_transform(self):
        self.cufft_batch_size = 1
        self.cufft_shape = self.shape
        if (self.axes is not None) and (len(self.axes) < len(self.shape)):
            # In the easiest batch mode, dimension 0 is equal to batch_size.
            # Otherwise, we have to configure istride/idist.
            dims = tuple(np.arange(len(self.shape)))
            if self.axes != dims[1:]:
                raise NotImplementedError(
                    "With the CUDA backend, batched transform can currently be performed only with axes=%s" % dims[1:]
                )
            self.cufft_batch_size = self.shape[0]
            self.cufft_shape = self.shape[1:]
            if len(self.cufft_shape) == 1:
                self.cufft_shape = self.cufft_shape[0]


    def configure_normalization(self):
        # TODO
        if self.normalize == "ortho":
            raise NotImplementedError(
                "Normalization mode 'ortho' is not implemented with CUDA backend yet."
            )
        self.cufft_scale_inverse = (self.normalize == "rescale")


    def check_array(self, array, shape, dtype, copy=True):
        if array.shape != shape:
            raise ValueError("Invalid data shape: expected %s, got %s" %
                (shape, array.shape)
            )
        if array.dtype != dtype:
            raise ValueError("Invalid data type: expected %s, got %s" %
                (dtype, array.dtype)
            )


    def set_data(self, dst, src, shape, dtype, copy=True, name=None):
        """
        dst is a device array owned by the current instance
        (either self.data_in or self.data_out).

        copy is ignored for device<-> arrays.
        """
        self.check_array(src, shape, dtype)
        if isinstance(src, np.ndarray):
            if name == "data_out":
                # Makes little sense to provide output=numpy_array
                return dst
            if not(src.flags["C_CONTIGUOUS"]):
                src = np.ascontiguousarray(src, dtype=dtype)
            dst[:] = src[:]
        elif isinstance(src, gpuarray.GPUArray):
            # No copy, use the data as self.d_input or self.d_output
            # (this prevents the use of in-place transforms, however).
            # We have to keep their old references.
            if name is None:
                # This should not happen
                raise ValueError("Please provide either copy=True or name != None")
            assert id(self.refs[name]) == id(dst) # DEBUG
            setattr(self, name, src)
            return src
        else:
            raise ValueError(
                "Invalid array type %s, expected numpy.ndarray or pycuda.gpuarray" %
                type(src)
            )
        return dst


    def recover_array_references(self):
        self.data_in = self.refs["data_in"]
        self.data_out = self.refs["data_out"]


    def compute_forward_plan(self):
        self.plan_forward = Plan(
            self.cufft_shape,
            self.dtype,
            self.dtype_out,
            batch=self.cufft_batch_size,
            stream=self.cufft_stream,
            # cufft extensible plan API is only supported after 0.5.1
            # (commit 65288d28ca0b93e1234133f8d460dc6becb65121)
            # but there is still no official 0.5.2
            #~ auto_allocate=True # cufft extensible plan API
        )


    def compute_inverse_plan(self):
        self.plan_inverse = Plan(
            self.cufft_shape, # not shape_out
            self.dtype_out,
            self.dtype,
            batch=self.cufft_batch_size,
            stream=self.cufft_stream,
            # cufft extensible plan API is only supported after 0.5.1
            # (commit 65288d28ca0b93e1234133f8d460dc6becb65121)
            # but there is still no official 0.5.2
            #~ auto_allocate=True
        )


    def copy_output_if_numpy(self, dst, src):
        if isinstance(dst, gpuarray.GPUArray):
            return
        dst[:] = src[:]


    def fft(self, array, output=None):
        """
        Perform a
        (forward) Fast Fourier Transform.

        Parameters
        ----------
        array: numpy.ndarray or pycuda.gpuarray
            Input data. Must be consistent with the current context.
        output: numpy.ndarray or pycuda.gpuarray, optional
            Output data. By default, output is a numpy.ndarray.
        """
        data_in = self.set_input_data(array, copy=False)
        data_out = self.set_output_data(output, copy=False)

        cu_fft(
            data_in,
            data_out,
            self.plan_forward,
            scale=False
        )

        if output is not None:
            self.copy_output_if_numpy(output, self.data_out)
            res = output
        else:
            res = self.data_out.get()
        self.recover_array_references()
        return res


    def ifft(self, array, output=None):
        """
        Perform a
        (inverse) Fast Fourier Transform.

        Parameters
        ----------
        array: numpy.ndarray or pycuda.gpuarray
            Input data. Must be consistent with the current context.
        output: numpy.ndarray or pycuda.gpuarray, optional
            Output data. By default, output is a numpy.ndarray.
        """
        data_in = self.set_output_data(array, copy=False)
        data_out = self.set_input_data(output, copy=False)

        cu_ifft(
            data_in,
            data_out,
            self.plan_inverse,
            scale=self.cufft_scale_inverse,
        )

        if output is not None:
            self.copy_output_if_numpy(output, self.data_in)
            res = output
        else:
            res = self.data_in.get()
        self.recover_array_references()
        return res



