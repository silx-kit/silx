#!/usr/bin/env python
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
import warnings
from pkg_resources import parse_version

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
        super().__init__(
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

        self.set_fft_norm()
        self.set_fft_functions()
        self.compute_plans()

    def set_fft_norm(self):
        # backward, forward indicates the direction in which the
        # normalisation is done. default is "backward"

        # rescale is default norm with numpy, no need of keywords
        # if normalize == "rescale":  # normalisation 1/N on ifft
        self.numpy_args_fft = {}
        self.numpy_args_ifft = {}

        if self.normalize == "ortho":  # normalization 1/sqrt(N) on both fft & ifft
            self.numpy_args_fft = {"norm": "ortho"}
            self.numpy_args_ifft = {"norm": "ortho"}

        elif self.normalize == "none":  # no normalisation on both fft & ifft
            if parse_version(np.version.version) < parse_version("1.20"):
                # "backward" & "forward" keywords were introduced in 1.20 and we support numpy >= 1.8
                warnings.warn(
                    "Numpy version %s does not allow to non-normalization. Effective normalization will be 'rescale'"
                    % (np.version.version)
                )  # default 'rescale' normalization
            else:
                self.numpy_args_fft = {"norm": "backward"}
                self.numpy_args_ifft = {"norm": "forward"}

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
            },
        }

    def _allocate(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

    def compute_plans(self):
        ndim = len(self.shape)
        funcs = self._fft_functions[self.real_transform][np.minimum(ndim, 3)]

        # Set norm
        # self.numpy_args_fft & self.numpy_args_ifft already set in set_fft_norm

        # Batched transform
        if (self.user_axes is not None) and len(self.user_axes) < ndim:
            funcs = self._fft_functions[self.real_transform][np.minimum(ndim - 1, 3)]
            self.numpy_args_fft["axes"] = self.user_axes
            self.numpy_args_ifft["axes"] = self.user_axes
            # Special case of batched 1D transform on 2D data
            if ndim == 2:
                assert len(self.user_axes) == 1
                self.numpy_args_fft["axis"] = self.user_axes[0]
                self.numpy_args_fft.pop("axes")
                self.numpy_args_ifft["axis"] = self.user_axes[0]
                self.numpy_args_ifft.pop("axes")
        self.numpy_funcs = funcs

    def fft(self, array):
        """
        Perform a (forward) Fast Fourier Transform.

        :param numpy.ndarray array:
            Input data. Must be consistent with the current context.
        """
        return self.numpy_funcs[0](array, **self.numpy_args_fft)

    def ifft(self, array):
        """
        Perform a (inverse) Fast Fourier Transform.

        :param numpy.ndarray array:
            Input data. Must be consistent with the current context.
        """
        return self.numpy_funcs[1](array, **self.numpy_args_ifft)
