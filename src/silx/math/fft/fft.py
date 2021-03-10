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
from .fftw import FFTW
from .clfft import CLFFT
from .npfft import NPFFT
from .cufft import CUFFT


def FFT(
    shape=None,
    dtype=None,
    template=None,
    shape_out=None,
    axes=None,
    normalize="rescale",
    backend="numpy",
    **kwargs
):
    """
    Initialize a FFT plan.

    :param List[int] shape:
        Shape of the input data.
    :param numpy.dtype dtype:
        Data type of the input data.
    :param numpy.ndarray template:
        Optional data, replacement for "shape" and "dtype".
        If provided, the arguments "shape" and "dtype" are ignored,
        and are instead inferred from it.
    :param List[int] shape_out:
        Optional shape of output data.
        By default, the data has the same shape as the input
        data (in case of C2C transform), or a shape with the last dimension halved
        (in case of R2C transform). If shape_out is provided, it must be greater
        or equal than the shape of input data. In this case, FFT is performed
        with zero-padding.
    :param List[int] axes:
        Axes along which FFT is computed.
          * For 2D transform: axes=(1,0)
          * For batched 1D transform of 2D image: axes=(0,)
    :param str normalize:
        Whether to normalize FFT and IFFT. Possible values are:
          * "rescale": in this case, Fourier data is divided by "N"
            before IFFT, so that (FFT(data)) = data
          * "ortho": in this case, FFT and IFFT are adjoint of eachother,
            the transform is unitary. Both FFT and IFFT are scaled with 1/sqrt(N).
          * "none": no normalizatio is done : IFFT(FFT(data)) = data*N
    :param str backend:
        FFT Backend to use. Value can be "numpy", "fftw", "opencl", "cuda".
    """
    backends = {
        "numpy": NPFFT,
        "np": NPFFT,
        "fftw": FFTW,
        "opencl": CLFFT,
        "clfft": CLFFT,
        "cuda": CUFFT,
        "cufft": CUFFT,
    }

    backend = backend.lower()
    if backend not in backends:
        raise ValueError("Unknown backend %s, available are %s" % (backend, backends))
    F = backends[backend](
        shape=shape,
        dtype=dtype,
        template=template,
        shape_out=shape_out,
        axes=axes,
        normalize=normalize,
        **kwargs
    )
    return F
