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
"""Test of the FFT module"""

import numpy as np
import unittest
import logging
try:
    from scipy.misc import ascent
    __have_scipy = True
except ImportError:
    __have_scipy = False
from silx.utils.testutils import ParametricTestCase
from silx.math.fft.fft import FFT
from silx.math.fft.clfft import __have_clfft__
from silx.math.fft.cufft import __have_cufft__
from silx.math.fft.fftw import __have_fftw__

from silx.test.utils import test_options

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
            "1D": [(512,), (511,)],
            "2D": [(512, 512), (512, 511), (511, 512), (511, 511)],
            "3D": [(128, 128, 128), (128, 128, 127), (128, 127, 128), (127, 128, 128),
                 (128, 127, 127), (127, 128, 127), (127, 127, 128), (127, 127, 127)]
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


class TestData(object):
    def __init__(self):
        self.data = ascent().astype("float32")
        self.data1d = self.data[:, 0]  # non-contiguous data
        self.data3d = np.tile(self.data[:128, :128], (128, 1, 1))
        self.data_refs = {
            1: self.data1d,
            2: self.data,
            3: self.data3d,
        }


@unittest.skipUnless(__have_scipy, "scipy is missing")
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
        self.test_data = TestData()

    @staticmethod
    def calc_mae(arr1, arr2):
        """
        Compute the Max Absolute Error between two arrays
        """
        return np.max(np.abs(arr1 - arr2))

    @unittest.skipIf(not __have_cufft__,
                     "cuda back-end requires pycuda and scikit-cuda")
    def test_cuda(self):
        import pycuda.autoinit

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
        if size == "3D" and test_options.TEST_LOW_MEM:
            self.skipTest("low mem")

        ndim = len(size)
        input_data = self.test_data.data_refs[ndim].astype(
            self.transform_infos.modes[mode])
        tol = self.tol[np.dtype(input_data.dtype)]
        if trdim == "3D":
            tol *= 10  # Error is relatively high in high dimensions

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
        self.assertTrue(
            mae < np.abs(input_data.max()) * tol,
            "FFT %s:%s, MAE(%s, numpy) = %f" % (mode, trdim, backend, mae)
        )

        # Inverse FFT
        res2 = F.ifft(res)
        mae = self.calc_mae(res2, input_data)
        self.assertTrue(
            mae < tol,
            "IFFT %s:%s, MAE(%s, numpy) = %f" % (mode, trdim, backend, mae)
        )


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
        self.test_data = TestData()

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


def suite():
    suite = unittest.TestSuite()
    for cls in (TestNumpyFFT, TestFFT):
        suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")


