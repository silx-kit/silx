#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2018-2021 European Synchrotron Radiation Facility
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
"""Test of the FFT module"""

from os import path
import logging
import numpy as np
import unittest
from pkg_resources import parse_version
import pytest
from tempfile import TemporaryDirectory
try:
    from scipy.misc import ascent
    __have_scipy = True
except ImportError:
    __have_scipy = False
from silx.utils.testutils import ParametricTestCase
from silx.math.fft.fft import FFT
from silx.math.fft.clfft import __have_clfft__
from silx.math.fft.cufft import __have_cufft__
from silx.math.fft.fftw import __have_fftw__, import_wisdom, export_wisdom, get_wisdom_file

if __have_cufft__:
    import atexit
    import pycuda.driver as cuda
    from pycuda.tools import clear_context_caches

def get_cuda_context(device_id=None, cleanup_at_exit=True):
    """
    Create or get a CUDA context.
    """
    current_ctx = cuda.Context.get_current()
    # If a context already exists, use this one
    # TODO what if the device used is different from device_id ?
    if current_ctx is not None:
        return current_ctx
    # Otherwise create a new context
    cuda.init()

    if device_id is None:
        device_id = 0
    # Use the Context obtained by retaining the device's primary context,
    # which is the one used by the CUDA runtime API (ex. scikit-cuda).
    # Unlike Context.make_context(), the newly-created context is not made current.
    context = cuda.Device(device_id).retain_primary_context()
    context.push()
    # Register a clean-up function at exit
    def _finish_up(context):
        if context is not None:
            context.pop()
            context = None
        clear_context_caches()
    if cleanup_at_exit:
        atexit.register(_finish_up, context)
    return context

logger = logging.getLogger(__name__)

class TransformInfos(object):
    def __init__(self):
        self.dimensions = [
            "1D",
            "batched_1D",
            "2D",
            "batched_2D",
            "3D",
        ]
        self.modes = {
            "R2C": np.float32,
            "R2C_double": np.float64,
            "C2C": np.complex64,
            "C2C_double": np.complex128,
        }
        self.sizes = {
            "1D": [(128,), (127,)],
            "2D": [(128, 128), (128, 127), (127, 128), (127, 127)],
            "3D": [(64, 64, 64), (64, 64, 63), (64, 63, 64), (63, 64, 64),
                 (64, 63, 63), (63, 64, 63), (63, 63, 64), (63, 63, 63)]
        }
        self.axes = {
            "1D": None,
            "batched_1D": (-1,),
            "2D": None,
            "batched_2D": (-2, -1),
            "3D": None,
        }
        self.sizes["batched_1D"] = self.sizes["2D"]
        self.sizes["batched_2D"] = self.sizes["3D"]


class Data(object):
    def __init__(self):
        self.data = ascent().astype("float32")
        self.data1d = self.data[:, 0]  # non-contiguous data
        self.data3d = np.tile(self.data[:64, :64], (64, 1, 1))
        self.data_refs = {
            1: self.data1d,
            2: self.data,
            3: self.data3d,
        }


@unittest.skipUnless(__have_scipy, "scipy is missing")
@pytest.mark.usefixtures("test_options_class_attr")
class TestFFT(ParametricTestCase):
    """Test cuda/opencl/fftw backends of FFT"""

    def setUp(self):
        self.tol = {
            np.dtype("float32"): 1e-3,
            np.dtype("float64"): 1e-9,
            np.dtype("complex64"): 1e-3,
            np.dtype("complex128"): 1e-9,
        }
        self.transform_infos = TransformInfos()
        self.test_data = Data()

    @staticmethod
    def calc_mae(arr1, arr2):
        """
        Compute the Max Absolute Error between two arrays
        """
        return np.max(np.abs(arr1 - arr2))

    @unittest.skipIf(not __have_cufft__,
                     "cuda back-end requires pycuda and scikit-cuda")
    def test_cuda(self):
        get_cuda_context()

        # Error is higher when using cuda. fast_math mode ?
        self.tol[np.dtype("float32")] *= 2

        self.__run_tests(backend="cuda")

    @unittest.skipIf(not __have_clfft__,
                     "opencl back-end requires pyopencl and gpyfft")
    def test_opencl(self):
        from silx.opencl.common import ocl
        if ocl is not None:
            self.__run_tests(backend="opencl", ctx=ocl.create_context())

    @unittest.skipIf(not __have_fftw__,
                     "fftw back-end requires pyfftw")
    def test_fftw(self):
        self.__run_tests(backend="fftw")

    def __run_tests(self, backend, **extra_args):
        """Run all tests with the given backend

        :param str backend:
        :param dict extra_args: Additional arguments to provide to FFT
        """
        for trdim in self.transform_infos.dimensions:
            for mode in self.transform_infos.modes:
                for size in self.transform_infos.sizes[trdim]:
                    with self.subTest(trdim=trdim, mode=mode, size=size):
                        self.__test(backend, trdim, mode, size, **extra_args)

    def __test(self, backend, trdim, mode, size, **extra_args):
        """Compare given backend with numpy for given conditions"""
        logger.debug("backend: %s, trdim: %s, mode: %s, size: %s",
                     backend, trdim, mode, str(size))
        if size == "3D" and self.test_options.TEST_LOW_MEM:
            self.skipTest("low mem")

        ndim = len(size)
        input_data = self.test_data.data_refs[ndim].astype(
            self.transform_infos.modes[mode])
        tol = self.tol[np.dtype(input_data.dtype)]
        if trdim == "3D":
            tol *= 10  # Error is relatively high in high dimensions
        # It seems that cuda has problems with C2D batched 1D
        if trdim == "batched_1D" and backend == "cuda" and mode == "C2C":
            tol *= 10

        # Python < 3.5 does not want to mix **extra_args with existing kwargs
        fft_args = {
            "template": input_data,
            "axes": self.transform_infos.axes[trdim],
            "backend": backend,
        }
        fft_args.update(extra_args)
        F = FFT(
            **fft_args
        )
        F_np = FFT(
            template=input_data,
            axes=self.transform_infos.axes[trdim],
            backend="numpy"
        )

        # Forward FFT
        res = F.fft(input_data)
        res_np = F_np.fft(input_data)
        mae = self.calc_mae(res, res_np)
        all_close = np.allclose(res, res_np, atol=tol, rtol=tol),
        self.assertTrue(
            all_close,
            "FFT %s:%s, MAE(%s, numpy) = %f (tol = %.2e)" % (mode, trdim, backend, mae, tol)
        )

        # Inverse FFT
        res2 = F.ifft(res)
        mae = self.calc_mae(res2, input_data)
        self.assertTrue(
            mae < tol,
            "IFFT %s:%s, MAE(%s, numpy) = %f" % (mode, trdim, backend, mae)
        )


    # Test normalizations. silx FFT has three normalization modes:
    #    - "rescale" (default). FFT is unscaled, IFFT is scaled by 1/N.
    #      This corresponds to numpy normalize=None i.e normalize="backward"
    #    - "ortho": FFT/IFFT are both scaled with 1/sqrt(N) so that FFT is unitary.
    #    - "none": Neither FFT nor IFFT are not scaled, so IFFT(FFT(array)) = N*array

    norms_backends_support = {
        "numpy": {
            "supported_normalizations": ["rescale", "ortho", "none"],
        },
        "fftw": {
            "supported_normalizations": ["rescale", "ortho", "none"],
        },
        "opencl": {
            "supported_normalizations": ["rescale"],
        },
        "cuda": {
            "supported_normalizations": ["rescale", "none"],
        }
    }

    @staticmethod
    def _compute_numpy_normalized_fft(data, axes, silx_normalization_mode):
        if silx_normalization_mode in ["rescale", "none"]:
            return np.fft.rfftn(data, axes=axes, norm=None)
        elif silx_normalization_mode == "ortho":
            return np.fft.rfftn(data, axes=axes, norm="ortho")
        else:
            raise ValueError("Unknown normalization mode %s" % silx_normalization_mode)

    @staticmethod
    def _compute_numpy_normalized_ifft(data, axes, silx_normalization_mode):
        if silx_normalization_mode == "rescale":
            return np.fft.irfftn(data, axes=axes, norm=None)
        elif silx_normalization_mode == "ortho":
            return np.fft.irfftn(data, axes=axes, norm="ortho")
        elif silx_normalization_mode == "none":
            res =  np.fft.irfftn(data, axes=axes, norm=None)
            # This assumes a FFT on all the axes, won't work on batched FFT
            N = res.size
            return res * N
        else:
            raise ValueError("Unknown normalization mode %s" % silx_normalization_mode)

    @unittest.skipIf(not __have_fftw__, "fftw back-end requires pyfftw")
    def test_norms_fftw(self):
        return self._test_norms_with_backend("fftw")

    @unittest.skipIf(
        parse_version(np.version.version) <= parse_version("1.19.5"),
        "normalization does not work for numpy <= 1.19.5"
    )
    def test_norms_numpy(self):
        return self._test_norms_with_backend("numpy")

    @unittest.skipIf(not __have_clfft__, "opencl back-end requires pyopencl and gpyfft")
    def test_norms_opencl(self):
        from silx.opencl.common import ocl
        if ocl is not None:
            return self._test_norms_with_backend("opencl")

    @unittest.skipIf(not __have_cufft__, "cuda back-end requires pycuda and scikit-cuda")
    def test_norms_cuda(self):
        get_cuda_context()
        return self._test_norms_with_backend("cuda")

    def _test_norms_with_backend(self, backend_name):
        backend_params = self.norms_backends_support[backend_name]

        data = self.test_data.data
        tol = self.tol[np.dtype(data.dtype)]

        for norm in backend_params["supported_normalizations"]:
            fft = FFT(template=data, backend=backend_name, normalize=norm)
            res = fft.fft(data)
            ref = self._compute_numpy_normalized_fft(data, fft.axes, norm)
            assert np.allclose(res, ref, atol=tol, rtol=tol), "Something wrong with %s norm=%s" % (backend_name, norm)

            res2 = fft.ifft(res)
            ref2 = self._compute_numpy_normalized_ifft(ref, fft.axes, norm)
            # unscaled IFFT yields very large values. Use a relatively high "atol"
            assert np.allclose(res2, ref2, atol=res2.max()/1e6), "Something wrong with I%s norm=%s" % (backend_name, norm)


@unittest.skipUnless(__have_scipy, "scipy is missing")
class TestNumpyFFT(ParametricTestCase):
    """
    Test the Numpy backend individually.
    """

    def setUp(self):
        transforms = {
            "1D": {
                True: (np.fft.rfft, np.fft.irfft),
                False: (np.fft.fft, np.fft.ifft),
            },
            "2D": {
                True: (np.fft.rfft2, np.fft.irfft2),
                False: (np.fft.fft2, np.fft.ifft2),
            },
            "3D": {
                True: (np.fft.rfftn, np.fft.irfftn),
                False: (np.fft.fftn, np.fft.ifftn),
            },
        }
        transforms["batched_1D"] = transforms["1D"]
        transforms["batched_2D"] = transforms["2D"]
        self.transforms = transforms
        self.transform_infos = TransformInfos()
        self.test_data = Data()

    def test(self):
        """Test the numpy backend against native fft.

        Results should be exactly the same.
        """
        for trdim in self.transform_infos.dimensions:
            for mode in self.transform_infos.modes:
                for size in self.transform_infos.sizes[trdim]:
                    with self.subTest(trdim=trdim, mode=mode, size=size):
                        self.__test(trdim, mode, size)

    def __test(self, trdim, mode, size):
        logger.debug("trdim: %s, mode: %s, size: %s", trdim, mode, str(size))
        ndim = len(size)
        input_data = self.test_data.data_refs[ndim].astype(
            self.transform_infos.modes[mode])
        np_fft, np_ifft = self.transforms[trdim][np.isrealobj(input_data)]

        F = FFT(
            template=input_data,
            axes=self.transform_infos.axes[trdim],
            backend="numpy"
        )
        # Test FFT
        res = F.fft(input_data)
        ref = np_fft(input_data)
        self.assertTrue(np.allclose(res, ref))

        # Test IFFT
        res2 = F.ifft(res)
        ref2 = np_ifft(ref)
        self.assertTrue(np.allclose(res2, ref2))


@pytest.mark.skipif(not(__have_fftw__), reason="Need fftw/pyfftw for this test")
def test_fftw_wisdom():
    """
    Test FFTW wisdom import/export mechanism
    """

    assert path.isdir(path.dirname(get_wisdom_file())) # Default: tempdir.gettempdir()

    with TemporaryDirectory(prefix="test_fftw_wisdom") as dname:
        subdir = path.join(dname, "subdir")
        get_wisdom_file(directory=subdir, create_dirs=False)
        assert not(path.isdir(subdir))
        fname = get_wisdom_file(directory=subdir, create_dirs=True)
        assert path.isdir(subdir)
        export_wisdom(fname)
        assert path.isfile(fname)
        import_wisdom(fname)

