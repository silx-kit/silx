#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
"""Module for sinogram filtering on CPU/GPU."""

from __future__ import absolute_import, print_function, with_statement, division

__authors__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "07/06/2019"

import numpy as np
from math import pi


import pyopencl.array as parray
from .common import pyopencl as cl
from .processing import OpenclProcessing
from ..math.fft.clfft import CLFFT, __have_clfft__
from ..math.fft.npfft import NPFFT
from ..image.tomography import generate_powers, get_next_power, compute_fourier_filter
from ..utils.deprecation import deprecated



class SinoFilter(OpenclProcessing):
    """A class for performing sinogram filtering on GPU using OpenCL.

    This is a convolution in the Fourier space, along one dimension:

    - In 2D: (n_a, d_x): n_a filterings (1D FFT of size d_x)
    - In 3D: (n_z, n_a, d_x): n_z*n_a filterings (1D FFT of size d_x)
    """
    kernel_files = ["array_utils.cl"]
    powers = generate_powers()

    def __init__(self, sino_shape, filter_name=None, ctx=None,
                 devicetype="all", platformid=None, deviceid=None,
                 profile=False, extra_options=None):
        """Constructor of OpenCL FFT-Convolve.

        :param sino_shape: shape of the sinogram.
        :param filter_name: Name of the filter. Defaut is "ram-lak".
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by
                           clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param profile: switch on profiling to be able to profile at the kernel
                        level, store profiling elements (makes code slightly
                        slower)
        :param dict extra_options: Advanced extra options.
            Current options are: cutoff, use_numpy_fft
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  profile=profile)

        self._init_extra_options(extra_options)
        self._calculate_shapes(sino_shape)
        self._init_fft()
        self._allocate_memory()
        self._compute_filter(filter_name)
        self._init_kernels()

    def _calculate_shapes(self, sino_shape):
        """

        :param sino_shape: shape of the sinogram.
        """
        self.ndim = len(sino_shape)
        if self.ndim == 2:
            n_angles, dwidth = sino_shape
        else:
            raise ValueError("Invalid sinogram number of dimensions: "
                             "expected 2 dimensions")
        self.sino_shape = sino_shape
        self.n_angles = n_angles
        self.dwidth = dwidth
        self.dwidth_padded = get_next_power(2 * self.dwidth, powers=self.powers)
        self.sino_padded_shape = (n_angles, self.dwidth_padded)
        sino_f_shape = list(self.sino_padded_shape)
        sino_f_shape[-1] = sino_f_shape[-1] // 2 + 1
        self.sino_f_shape = tuple(sino_f_shape)

    def _init_extra_options(self, extra_options):
        """

        :param dict extra_options: Advanced extra options.
            Current options are: cutoff,
        """
        self.extra_options = {
            "cutoff": 1.,
            "use_numpy_fft": False,
        }
        if extra_options is not None:
            self.extra_options.update(extra_options)

    def _init_fft(self):
        if __have_clfft__ and not(self.extra_options["use_numpy_fft"]):
            self.fft_backend = "opencl"
            self.fft = CLFFT(
                self.sino_padded_shape,
                dtype=np.float32,
                axes=(-1,),
                ctx=self.ctx,
            )
        else:
            self.fft_backend = "numpy"
            print("The gpyfft module was not found. The Fourier transforms "
                  "will be done on CPU. For more performances, it is advised "
                  "to install gpyfft.""")
            self.fft = NPFFT(
                template=np.zeros(self.sino_padded_shape, "f"),
                axes=(-1,),
            )

    def _allocate_memory(self):
        self.d_filter_f = parray.zeros(self.queue, (self.sino_f_shape[-1],), np.complex64)
        self.is_cpu = (self.device.type == "CPU")
        # These are already allocated by FFT() if using the opencl backend
        if self.fft_backend == "opencl":
            self.d_sino_padded = self.fft.data_in
            self.d_sino_f = self.fft.data_out
        else:
            # When using the numpy backend, arrays are not pre-allocated
            self.d_sino_padded = np.zeros(self.sino_padded_shape, "f")
            self.d_sino_f = np.zeros(self.sino_f_shape, np.complex64)
        # These are needed for rectangular memcpy in certain cases (see below).
        self.tmp_sino_device = parray.zeros(self.queue, self.sino_shape, "f")
        self.tmp_sino_host = np.zeros(self.sino_shape, "f")

    def _compute_filter(self, filter_name):
        """

        :param str filter_name: filter name
        """
        self.filter_name = filter_name or "ram-lak"
        filter_f = compute_fourier_filter(
            self.dwidth_padded,
            self.filter_name,
            cutoff=self.extra_options["cutoff"],
        )[:self.dwidth_padded // 2 + 1]  # R2C
        self.set_filter(filter_f, normalize=True)

    def set_filter(self, h_filt, normalize=True):
        """
        Set a filter for sinogram filtering.

        :param h_filt: Filter. Each line of the sinogram will be filtered with
            this filter. It has to be the Real-to-Complex Fourier Transform
            of some real filter, padded to 2*sinogram_width.
        :param normalize: Whether to normalize the filter with pi/num_angles.
        """
        if h_filt.size != self.sino_f_shape[-1]:
            raise ValueError(
                """
                Invalid filter size: expected %d, got %d.
                Please check that the filter is the Fourier R2C transform of
                some real 1D filter.
                """
                % (self.sino_f_shape[-1], h_filt.size)
            )
        if not(np.iscomplexobj(h_filt)):
            print("Warning: expected a complex Fourier filter")
        self.filter_f = h_filt
        if normalize:
            self.filter_f *= pi / self.n_angles
        self.filter_f = self.filter_f.astype(np.complex64)
        self.d_filter_f[:] = self.filter_f[:]

    def _init_kernels(self):
        OpenclProcessing.compile_kernels(self, self.kernel_files)
        h, w = self.d_sino_f.shape
        self.mult_kern_args = (self.queue, (int(w), (int(h))), None,
                               self.d_sino_f.data,
                               self.d_filter_f.data,
                               np.int32(w),
                               np.int32(h))

    def check_array(self, arr):
        if arr.dtype != np.float32:
            raise ValueError("Expected data type = numpy.float32")
        if arr.shape != self.sino_shape:
            raise ValueError("Expected sinogram shape %s, got %s" %
                             (self.sino_shape, arr.shape))
        if not(isinstance(arr, np.ndarray) or isinstance(arr, parray.Array)):
            raise ValueError("Expected either numpy.ndarray or "
                             "pyopencl.array.Array")

    def copy2d(self, dst, src, transfer_shape, dst_offset=(0, 0),
               src_offset=(0, 0)):
        """

        :param dst:
        :param src:
        :param transfer_shape:
        :param dst_offset:
        :param src_offset:
        """
        shape = tuple(int(i) for i in transfer_shape[::-1])
        ev = self.kernels.cpy2d(self.queue, shape, None,
                                dst.data,
                                src.data,
                                np.int32(dst.shape[1]),
                                np.int32(src.shape[1]),
                                np.int32(dst_offset),
                                np.int32(src_offset),
                                np.int32(transfer_shape[::-1]))
        ev.wait()

    def copy2d_host(self, dst, src, transfer_shape, dst_offset=(0, 0),
                    src_offset=(0, 0)):
        """

        :param dst:
        :param src:
        :param transfer_shape:
        :param dst_offset:
        :param src_offset:
        """
        s = transfer_shape
        do = dst_offset
        so = src_offset
        dst[do[0]:do[0] + s[0], do[1]:do[1] + s[1]] = src[so[0]:so[0] + s[0], so[1]:so[1] + s[1]]

    def _prepare_input_sino(self, sino):
        """
        :param sino: sinogram
        """
        self.check_array(sino)
        self.d_sino_padded.fill(0)
        if self.fft_backend == "opencl":
            # OpenCL backend: FFT/mult/IFFT are done on device.
            if isinstance(sino, np.ndarray):
                # OpenCL backend + numpy input: copy H->D.
                # As pyopencl does not support rectangular copies, we have to
                # do a copy H->D in a temporary device buffer, and then call a
                # kernel doing the rectangular D-D copy.
                self.tmp_sino_device[:] = sino[:]
                if self.is_cpu:
                    self.tmp_sino_device.finish()
                d_sino_ref = self.tmp_sino_device
            else:
                d_sino_ref = sino
            # Rectangular copy D->D
            self.copy2d(self.d_sino_padded, d_sino_ref, self.sino_shape)
            if self.is_cpu:
                self.d_sino_padded.finish()  # should not be required here
        else:
            # Numpy backend: FFT/mult/IFFT are done on host.
            if not(isinstance(sino, np.ndarray)):
                # Numpy backend + pyopencl input: need to copy D->H
                self.tmp_sino_host[:] = sino[:]
                h_sino_ref = self.tmp_sino_host
            else:
                h_sino_ref = sino
            # Rectangular copy H->H
            self.copy2d_host(self.d_sino_padded, h_sino_ref, self.sino_shape)

    def _get_output_sino(self, output):
        """
        :param Union[numpy.dtype,None] output: sinogram output.
        :return: sinogram
        """
        if output is None:
            res = np.zeros(self.sino_shape, dtype=np.float32)
        else:
            res = output
        if self.fft_backend == "opencl":
            if isinstance(res, np.ndarray):
                # OpenCL backend + numpy output: copy D->H
                # As pyopencl does not support rectangular copies, we first have
                # to call a kernel doing rectangular copy D->D, then do a copy
                # D->H.
                self.copy2d(dst=self.tmp_sino_device,
                            src=self.d_sino_padded,
                            transfer_shape=self.sino_shape)
                if self.is_cpu:
                    self.tmp_sino_device.finish()  # should not be required here
                res[:] = self.tmp_sino_device.get()[:]
            else:
                if self.is_cpu:
                    self.d_sino_padded.finish()
                self.copy2d(res, self.d_sino_padded, self.sino_shape)
                if self.is_cpu:
                    res.finish()  # should not be required here
        else:
            if not(isinstance(res, np.ndarray)):
                # Numpy backend + pyopencl output: rect copy H->H + copy H->D
                self.copy2d_host(dst=self.tmp_sino_host,
                                 src=self.d_sino_padded,
                                 transfer_shape=self.sino_shape)
                res[:] = self.tmp_sino_host[:]
            else:
                # Numpy backend + numpy output: rect copy H->H
                self.copy2d_host(res, self.d_sino_padded, self.sino_shape)
        return res

    def _do_fft(self):
        if self.fft_backend == "opencl":
            self.fft.fft(self.d_sino_padded, output=self.d_sino_f)
            if self.is_cpu:
                self.d_sino_f.finish()
        else:
            # numpy backend does not support "output=" argument,
            # and rfft always return a complex128 result.
            res = self.fft.fft(self.d_sino_padded).astype(np.complex64)
            self.d_sino_f[:] = res[:]

    def _multiply_fourier(self):
        if self.fft_backend == "opencl":
            # Everything is on device. Call the multiplication kernel.
            ev = self.kernels.inplace_complex_mul_2Dby1D(
                *self.mult_kern_args
            )
            ev.wait()
            if self.is_cpu:
                self.d_sino_f.finish()  # should not be required here
        else:
            # Everything is on host.
            self.d_sino_f *= self.filter_f

    def _do_ifft(self):
        if self.fft_backend == "opencl":
            if self.is_cpu:
                self.d_sino_padded.fill(0)
                self.d_sino_padded.finish()
            self.fft.ifft(self.d_sino_f, output=self.d_sino_padded)
            if self.is_cpu:
                self.d_sino_padded.finish()
        else:
            # numpy backend does not support "output=" argument,
            # and irfft always return a float64 result.
            res = self.fft.ifft(self.d_sino_f).astype("f")
            self.d_sino_padded[:] = res[:]

    def filter_sino(self, sino, output=None):
        """

        :param sino: sinogram
        :param output:
        :return: filtered sinogram
        """
        # Handle input sinogram
        self._prepare_input_sino(sino)
        # FFT
        self._do_fft()
        # multiply with filter in the Fourier domain
        self._multiply_fourier()
        # iFFT
        self._do_ifft()
        # return
        res = self._get_output_sino(output)
        return res
        # ~ return output

    __call__ = filter_sino




# -------------------
# - Compatibility  -
# -------------------


def nextpow2(N):
    p = 1
    while p < N:
        p *= 2
    return p


@deprecated(replacement="Backprojection.sino_filter", since_version="0.10")
def fourier_filter(sino, filter_=None, fft_size=None):
    """Simple np based implementation of fourier space filter.
    This function is deprecated, please use silx.opencl.sinofilter.SinoFilter.

    :param sino: of shape shape = (num_projs, num_bins)
    :param filter: filter function to apply in fourier space
    :fft_size: size on which perform the fft. May be larger than the sino array
    :return: filtered sinogram
    """
    assert sino.ndim == 2
    num_projs, num_bins = sino.shape
    if fft_size is None:
        fft_size = nextpow2(num_bins * 2 - 1)
    else:
        assert fft_size >= num_bins
    if fft_size == num_bins:
        sino_zeropadded = sino.astype(np.float32)
    else:
        sino_zeropadded = np.zeros((num_projs, fft_size),
                                      dtype=np.complex64)
        sino_zeropadded[:, :num_bins] = sino.astype(np.float32)

    if filter_ is None:
        h = np.zeros(fft_size, dtype=np.float32)
        L2 = fft_size // 2 + 1
        h[0] = 1 / 4.
        j = np.linspace(1, L2, L2 // 2, False)
        h[1:L2:2] = -1. / (np.pi ** 2 * j ** 2)
        h[L2:] = np.copy(h[1:L2 - 1][::-1])
        filter_ = np.fft.fft(h).astype(np.complex64)

    # Linear convolution
    sino_f = np.fft.fft(sino, fft_size)
    sino_f = sino_f * filter_
    sino_filtered = np.fft.ifft(sino_f)[:, :num_bins].real

    return np.ascontiguousarray(sino_filtered.real, dtype=np.float32)
