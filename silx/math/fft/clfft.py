#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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

from .basefft import BaseFFT, check_version
try:
    import pyopencl as cl
    import pyopencl.array as parray
    import gpyfft
    from gpyfft.fft import FFT as cl_fft
    from ...opencl.common import ocl
    __have_clfft__ = True
except ImportError:
    __have_clfft__ = False


# Check gpyfft version
__required_gpyfft_version__ = "0.3.0"
if __have_clfft__:
    __have_clfft__ = check_version(gpyfft, __required_gpyfft_version__)


class CLFFT(BaseFFT):
    """Initialize a clfft plan.

    Please see FFT class for parameters help.

    CLFFT-specific parameters
    --------------------------

    :param pyopencl.Context ctx:
        If set to other than None, an existing pyopencl context is used.
    :param bool fast_math:
        If set to True, computations will be done with "fast math" mode,
        i.e., more speed but less accuracy.
    :param bool choose_best_device:
        Whether to automatically choose the best available OpenCL device.
    """
    def __init__(
        self,
        shape=None,
        dtype=None,
        template=None,
        shape_out=None,
        axes=None,
        normalize="rescale",
        ctx=None,
        fast_math=False,
        choose_best_device=True,
    ):
        if not(__have_clfft__) or not(__have_clfft__):
            raise ImportError("Please install pyopencl and gpyfft >= %s to use the OpenCL back-end" % __required_gpyfft_version__)

        super(CLFFT, self).__init__(
            shape=shape,
            dtype=dtype,
            template=template,
            shape_out=shape_out,
            axes=axes,
            normalize=normalize,
        )
        self.ctx = ctx
        self.choose_best_device = choose_best_device
        self.fast_math = fast_math
        self.backend = "clfft"

        self.fix_axes()
        self.init_context_queue()
        self.allocate_arrays()
        self.real_transform = np.isrealobj(self.data_in)
        self.compute_forward_plan()
        self.compute_inverse_plan()
        self.refs = {
            "data_in": self.data_in,
            "data_out": self.data_out,
        }
        # TODO
        #  Either pyopencl ElementWiseKernel, or built-in clfft callbacks
        if self.normalize != "rescale":
            raise NotImplementedError(
                "Normalization modes other than rescale are not implemented with OpenCL backend yet."
            )

    def fix_axes(self):
        """
        "Fix" axes.

        clfft does not have the same convention as FFTW/cuda/numpy.
        """
        self.axes = self.axes[::-1]

    def _allocate(self, shape, dtype):
        ary = parray.empty(self.queue, shape, dtype=dtype)
        ary.fill(0)
        return ary


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
            # working on underlying buffer is notably faster
            #~ dst[:] = src[:]
            evt = cl.enqueue_copy(self.queue, dst.data, src)
            evt.wait()
        elif isinstance(src, parray.Array):
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
                "Invalid array type %s, expected numpy.ndarray or pyopencl.array" %
                type(src)
            )
        return dst


    def recover_array_references(self):
        self.data_in = self.refs["data_in"]
        self.data_out = self.refs["data_out"]


    def init_context_queue(self):
        if self.ctx is None:
            if self.choose_best_device:
                self.ctx = ocl.create_context()
            else:
                self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)


    def compute_forward_plan(self):
        self.plan_forward = cl_fft(
            self.ctx,
            self.queue,
            self.data_in,
            out_array=self.data_out,
            axes=self.axes,
            fast_math=self.fast_math,
            real=self.real_transform,
        )


    def compute_inverse_plan(self):
        self.plan_inverse = cl_fft(
            self.ctx,
            self.queue,
            self.data_out,
            out_array=self.data_in,
            axes=self.axes,
            fast_math=self.fast_math,
            real=self.real_transform,
        )


    def update_forward_plan_arrays(self):
        self.plan_forward.data = self.data_in
        self.plan_forward.result = self.data_out


    def update_inverse_plan_arrays(self):
        self.plan_inverse.data = self.data_out
        self.plan_inverse.result = self.data_in


    def copy_output_if_numpy(self, dst, src):
        if isinstance(dst, parray.Array):
            return
        # working on underlying buffer is notably faster
        #~ dst[:] = src[:]
        evt = cl.enqueue_copy(self.queue, dst, src.data)
        evt.wait()


    def fft(self, array, output=None, do_async=False):
        """
        Perform a (forward) Fast Fourier Transform.

        :param Union[numpy.ndarray,pyopencl.array] array:
            Input data. Must be consistent with the current context.
        :param Union[numpy.ndarray,pyopencl.array] output:
            Output data. By default, output is a numpy.ndarray.
        :param bool do_async:
            Whether to perform operation in asynchronous mode.
            Default is False, meaning that we wait for transform to complete.
        """
        self.set_input_data(array, copy=False)
        self.set_output_data(output, copy=False)
        self.update_forward_plan_arrays()
        event, = self.plan_forward.enqueue()
        if not(do_async):
            event.wait()
        if output is not None:
            self.copy_output_if_numpy(output, self.data_out)
            res = output
        else:
            res = self.data_out.get()
        self.recover_array_references()
        return res


    def ifft(self, array, output=None, do_async=False):
        """
        Perform a (inverse) Fast Fourier Transform.

        :param Union[numpy.ndarray,pyopencl.array] array:
            Input data. Must be consistent with the current context.
        :param Union[numpy.ndarray,pyopencl.array] output:
            Output data. By default, output is a numpy.ndarray.
        :param bool do_async:
            Whether to perform operation in asynchronous mode.
            Default is False, meaning that we wait for transform to complete.
        """
        self.set_output_data(array, copy=False)
        self.set_input_data(output, copy=False)
        self.update_inverse_plan_arrays()
        event, = self.plan_inverse.enqueue(forward=False)
        if not(do_async):
            event.wait()
        if output is not None:
            self.copy_output_if_numpy(output, self.data_in)
            res = output
        else:
            res = self.data_in.get()
        self.recover_array_references()
        return res


    def __del__(self):
        # It seems that gpyfft underlying clFFT destructors are not called.
        # This results in the following warning:
        #   Warning:  Program terminating, but clFFT resources not freed.
        #   Please consider explicitly calling clfftTeardown( )
        del self.plan_forward
        del self.plan_inverse

