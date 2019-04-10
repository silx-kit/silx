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
"""Test of the MFFT module"""

import numpy as np
import unittest
import logging
from scipy.misc import ascent
from silx.utils.testutils import parameterize
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
        self.data1d = self.data[:, 0] # non-contiguous data
        self.data3d = np.tile(self.data[:128, :128], (128, 1, 1))
        self.data_refs = {
            1: self.data1d,
            2: self.data,
            3: self.data3d,
        }



class TestFFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFFT, cls).setUpClass()
        if __have_clfft__:
            from silx.opencl.common import ocl
            cls.Ctx = ocl.create_context()


    @classmethod
    def tearDownClass(cls):
        super(TestFFT, cls).tearDownClass()
        if __have_clfft__:
            del cls.Ctx


    def __init__(self, methodName='runTest', param=None):
        unittest.TestCase.__init__(self, methodName)
        self.param = param


    def setUp(self):
        self.tol = {
            np.dtype("float32"): 1e-3,
            np.dtype("float64"): 1e-9,
            np.dtype("complex64"): 1e-3,
            np.dtype("complex128"): 1e-9,
        }
        self.backend = self.param["backend"]
        self.trdim = self.param["trdim"]
        self.mode = self.param["mode"]
        self.size = self.param["size"]
        self.transform_infos = self.param["transform_infos"]
        self.test_data = self.param["test_data"]
        self.configure_backends()
        self.configure_extra_args()


    def tearDown(self):
        pass


    def configure_backends(self):
        self.__have_clfft__ = __have_clfft__
        self.__have_cufft__ = __have_cufft__
        self.__have_fftw__ = __have_fftw__

        if self.backend in ["cuda", "cufft"] and __have_cufft__:
            import pycuda.autoinit
            # Error is higher when using cuda. fast_math mode ?
            self.tol[np.dtype("float32")] *= 2


    def configure_extra_args(self):
        self.extra_args = {}
        if __have_clfft__ and self.backend in ["opencl", "clfft"]:
            self.extra_args["ctx"] = self.Ctx


    def check_current_backend(self):
        if self.backend in ["cuda", "cufft"] and not(self.__have_cufft__):
            return "cuda back-end requires pycuda and scikit-cuda"
        if self.backend in ["opencl", "clfft"] and not(self.__have_clfft__):
            return "opencl back-end requires pyopencl and gpyfft"
        if self.backend == "fftw" and not(self.__have_fftw__):
            return "fftw back-end requires pyfftw"
        return None


    @staticmethod
    def calc_mae(arr1, arr2):
        """
        Compute the Max Absolute Error between two arrays
        """
        return np.max(np.abs(arr1 - arr2))


    def test_fft(self):
        err = self.check_current_backend()
        if err is not None:
            self.skipTest(err)
        if self.size == "3D" and test_options.TEST_LOW_MEM:
            self.skipTest("low mem")

        ndim = len(self.size)
        input_data = self.test_data.data_refs[ndim].astype(self.transform_infos.modes[self.mode])
        tol = self.tol[np.dtype(input_data.dtype)]
        if self.trdim == "3D":
            tol *= 10 # Error is relatively high in high dimensions

        # Python < 3.5 does not want to mix **extra_args with existing kwargs
        fft_args = {
            "template": input_data,
            "axes": self.transform_infos.axes[self.trdim],
            "backend": self.backend,
        }
        fft_args.update(self.extra_args)
        F = FFT(
            **fft_args
        )
        F_np = FFT(
            template=input_data,
            axes=self.transform_infos.axes[self.trdim],
            backend="numpy"
        )

        # Forward FFT
        res = F.fft(input_data)
        res_np = F_np.fft(input_data)
        mae = self.calc_mae(res, res_np)
        self.assertTrue(
            mae < np.abs(input_data.max()) * tol,
            "FFT %s:%s, MAE(%s, numpy) = %f" % (self.mode, self.trdim, self.backend, mae)
        )

        # Inverse FFT
        res2 = F.ifft(res)
        mae = self.calc_mae(res2, input_data)
        self.assertTrue(
            mae < tol,
            "IFFT %s:%s, MAE(%s, numpy) = %f" % (self.mode, self.trdim, self.backend, mae)
        )


class TestNumpyFFT(unittest.TestCase):
    """
    Test the Numpy backend individually.
    """

    def __init__(self, methodName='runTest', param=None):
        unittest.TestCase.__init__(self, methodName)
        self.param = param

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


    def test_numpy_fft(self):
        """
        Test the numpy backend against native fft.
        Results should be exactly the same.
        """
        trinfos = self.param["transform_infos"]
        trdim = self.param["trdim"]
        ndim = len(self.param["size"])
        input_data = self.param["test_data"].data_refs[ndim].astype(trinfos.modes[self.param["mode"]])
        np_fft, np_ifft = self.transforms[trdim][np.isrealobj(input_data)]

        F = FFT(
            template=input_data,
            axes=trinfos.axes[trdim],
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


def test_numpy_backend(dimensions=None):
    testSuite = unittest.TestSuite()
    transform_infos = TransformInfos()
    test_data = TestData()
    dimensions = dimensions or transform_infos.dimensions

    for trdim in dimensions:
        logger.debug("   testing %s" % trdim)
        for mode in transform_infos.modes:
            logger.debug("   testing %s:%s" % (trdim, mode))
            for size in transform_infos.sizes[trdim]:
                logger.debug("      size: %s" % str(size))
                testcase = parameterize(
                    TestNumpyFFT,
                    param={
                        "transform_infos": transform_infos,
                        "test_data": test_data,
                        "trdim": trdim,
                        "mode": mode,
                        "size": size,
                    }
                )
                testSuite.addTest(testcase)
    return testSuite


def test_fft(backend, dimensions=None):
    testSuite = unittest.TestSuite()
    transform_infos = TransformInfos()
    test_data = TestData()
    dimensions = dimensions or transform_infos.dimensions

    logger.info("Testing backend: %s" % backend)
    for trdim in dimensions:
        logger.debug("   testing %s" % trdim)
        for mode in transform_infos.modes:
            logger.debug("   testing %s:%s" % (trdim, mode))
            for size in transform_infos.sizes[trdim]:
                logger.debug("      size: %s" % str(size))
                testcase = parameterize(
                    TestFFT,
                    param={
                        "transform_infos": transform_infos,
                        "test_data": test_data,
                        "backend": backend,
                        "trdim": trdim,
                        "mode": mode,
                        "size": size,
                    }
                )
                testSuite.addTest(testcase)
    return testSuite


def test_all():
    suite = unittest.TestSuite()

    suite.addTest(test_numpy_backend())

    suite.addTest(test_fft("fftw"))
    suite.addTest(test_fft("opencl"))
    suite.addTest(test_fft("cuda"))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest="test_all")


