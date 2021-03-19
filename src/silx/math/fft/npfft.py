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

from .basefft import BaseFFT


class NPFFT(BaseFFT):
    """Initialize a numpy plan.

    Please see FFT class for parameters help.
    """
    def __init__(
        self,
        shape=None,
        dtype=None,
        template=None,
        shape_out=None,
        axes=None,
        normalize="rescale",
    ):
        super(NPFFT, self).__init__(
            shape=shape,
            dtype=dtype,
            template=template,
            shape_out=shape_out,
            axes=axes,
            normalize=normalize,
        )
        self.backend = "numpy"
        self.real_transform = False
        if template is not None and np.isrealobj(template):
            self.real_transform = True
        # For numpy functions.
        # TODO Issue warning if user wants ifft(fft(data)) = N*data ?
        if normalize != "ortho":
            self.normalize = None
        self.set_fft_functions()
        #~ self.allocate_arrays() # not needed for this backend
        self.compute_plans()


    def set_fft_functions(self):
        # (fwd, inv) = _fft_functions[is_real][ndim]
        self._fft_functions = {
            True: {
                1: (np.fft.rfft, np.fft.irfft),
                2: (np.fft.rfft2, np.fft.irfft2),
                3: (np.fft.rfftn, np.fft.irfftn),
            },
            False: {
                1: (np.fft.fft, np.fft.ifft),
                2: (np.fft.fft2, np.fft.ifft2),
                3: (np.fft.fftn, np.fft.ifftn),
            }
        }


    def _allocate(self, shape, dtype):
        return np.zeros(self.queue, shape, dtype=dtype)


    def compute_plans(self):
        ndim = len(self.shape)
        funcs = self._fft_functions[self.real_transform][np.minimum(ndim, 3)]
        if np.version.version[:4] in ["1.8.", "1.9."]:
            # norm keyword was introduced in 1.10 and we support numpy >= 1.8
            self.numpy_args = {}
        else:
            self.numpy_args = {"norm": self.normalize}
        # Batched transform
        if (self.user_axes is not None) and len(self.user_axes) < ndim:
            funcs = self._fft_functions[self.real_transform][np.minimum(ndim-1, 3)]
            self.numpy_args["axes"] = self.user_axes
            # Special case of batched 1D transform on 2D data
            if ndim == 2:
                assert len(self.user_axes) == 1
                self.numpy_args["axis"] = self.user_axes[0]
                self.numpy_args.pop("axes")
        self.numpy_funcs = funcs


    def fft(self, array):
        """
        Perform a (forward) Fast Fourier Transform.

        :param numpy.ndarray array:
            Input data. Must be consistent with the current context.
        """
        return self.numpy_funcs[0](array, **self.numpy_args)


    def ifft(self, array):
        """
        Perform a (inverse) Fast Fourier Transform.

        :param numpy.ndarray array:
            Input data. Must be consistent with the current context.
        """
        return self.numpy_funcs[1](array, **self.numpy_args)

