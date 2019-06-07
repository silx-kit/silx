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
    import pyfftw
    __have_fftw__ = True
except ImportError:
    __have_fftw__ = False


# Check pyfftw version
__required_pyfftw_version__ = "0.10.0"
if __have_fftw__:
    __have_fftw__ = check_version(pyfftw, __required_pyfftw_version__)


class FFTW(BaseFFT):
    """Initialize a FFTW plan.

    Please see FFT class for parameters help.

    FFTW-specific parameters
    -------------------------

    :param bool check_alignment:
        If set to True and "data" is provided, this will enforce the input data
        to be "byte aligned", which might imply extra memory usage.
    :param int num_threads:
        Number of threads for computing FFT.
    """
    def __init__(
        self,
        shape=None,
        dtype=None,
        template=None,
        shape_out=None,
        axes=None,
        normalize="rescale",
        check_alignment=False,
        num_threads=1,
    ):
        if not(__have_fftw__):
            raise ImportError("Please install pyfftw >= %s to use the FFTW back-end" % __required_pyfftw_version__)
        super(FFTW, self).__init__(
            shape=shape,
            dtype=dtype,
            template=template,
            shape_out=shape_out,
            axes=axes,
            normalize=normalize,
        )
        self.check_alignment = check_alignment
        self.num_threads = num_threads
        self.backend = "fftw"

        self.allocate_arrays()
        self.set_fftw_flags()
        self.compute_forward_plan()
        self.compute_inverse_plan()


    def set_fftw_flags(self):
        self.fftw_flags = ('FFTW_MEASURE', ) # TODO
        self.fftw_planning_timelimit = None # TODO
        self.fftw_norm_modes = {
            "rescale": {"ortho": False, "normalize": True},
            "ortho": {"ortho": True, "normalize": False},
            "none": {"ortho": False, "normalize": False},
        }
        if self.normalize not in self.fftw_norm_modes:
            raise ValueError("Unknown normalization mode %s. Possible values are %s" %
                (self.normalize, self.fftw_norm_modes.keys())
            )
        self.fftw_norm_mode = self.fftw_norm_modes[self.normalize]


    def _allocate(self, shape, dtype):
        return pyfftw.zeros_aligned(shape, dtype=dtype)


    def check_array(self, array, shape, dtype, copy=True):
        """
        Check that a given array is compatible with the FFTW plans,
        in terms of alignment and data type.

        If the provided array does not meet any of the checks, a new array
        is returned.
        """
        if array.shape != shape:
            raise ValueError("Invalid data shape: expected %s, got %s" %
                (shape, array.shape)
            )
        if array.dtype != dtype:
            raise ValueError("Invalid data type: expected %s, got %s" %
                (dtype, array.dtype)
            )
        if self.check_alignment and not(pyfftw.is_byte_aligned(array)):
            array2 = pyfftw.zeros_aligned(self.shape, dtype=self.dtype_in)
            np.copyto(array2, array)
        else:
            if copy:
                array2 = np.copy(array)
            else:
                array2 = array
        return array2


    def set_data(self, dst, src, shape, dtype, copy=True, name=None):
        dst = self.check_array(src, shape, dtype, copy=copy)
        return dst


    def compute_forward_plan(self):
        self.plan_forward = pyfftw.FFTW(
            self.data_in,
            self.data_out,
            axes=self.axes,
            direction='FFTW_FORWARD',
            flags=self.fftw_flags,
            threads=self.num_threads,
            planning_timelimit=self.fftw_planning_timelimit,
            # the following seems to be taken into account only when using __call__
            ortho=self.fftw_norm_mode["ortho"],
            normalise_idft=self.fftw_norm_mode["normalize"],
        )


    def compute_inverse_plan(self):
        self.plan_inverse = pyfftw.FFTW(
            self.data_out,
            self.data_in,
            axes=self.axes,
            direction='FFTW_BACKWARD',
            flags=self.fftw_flags,
            threads=self.num_threads,
            planning_timelimit=self.fftw_planning_timelimit,
            # the following seem to be taken into account only when using __call__
            ortho=self.fftw_norm_mode["ortho"],
            normalise_idft=self.fftw_norm_mode["normalize"],
        )


    def fft(self, array, output=None):
        """
        Perform a (forward) Fast Fourier Transform.

        :param numpy.ndarray array:
            Input data. Must be consistent with the current context.
        :param numpy.ndarray output:
            Optional output data.
        """
        data_in = self.set_input_data(array, copy=True)
        data_out = self.set_output_data(output, copy=False)
        # execute.__call__ does both update_arrays() and normalization
        self.plan_forward(
            input_array=data_in,
            output_array=data_out,
            ortho=self.fftw_norm_mode["ortho"],
        )
        assert id(self.plan_forward.output_array) == id(self.data_out) == id(data_out) # DEBUG
        return data_out


    def ifft(self, array, output=None):
        """
        Perform a (inverse) Fast Fourier Transform.

        :param numpy.ndarray array:
            Input data. Must be consistent with the current context.
        :param numpy.ndarray output:
            Optional output data.
        """
        data_in = self.set_output_data(array, copy=True)
        data_out = self.set_input_data(output, copy=False)
        # execute.__call__ does both update_arrays() and normalization
        self.plan_inverse(
            input_array=data_in,
            output_array=data_out,
            ortho=self.fftw_norm_mode["ortho"],
            normalise_idft=self.fftw_norm_mode["normalize"]
        )
        assert id(self.plan_inverse.output_array) == id(self.data_in) == id(data_out) # DEBUG
        return data_out


