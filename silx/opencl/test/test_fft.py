#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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

from __future__ import division, print_function

__authors__ = ["Jerome Kieffer, Pierre Paleo"]
__license__ = "MIT"
__copyright__ = "2013-2018 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/11/2018"

import os
from scipy.misc import ascent
import logging
import numpy as np
import unittest
from silx.opencl import ocl
from silx.utils.testutils import parameterize
if ocl:
    from silx.opencl import fft
    from silx.opencl import pyopencl

logger = logging.getLogger(__name__)

test_cases = {
    # 1D
    "1D_R2C_simple": {
        "args": {},
        "numpy_fft": np.fft.rfft,
    },
    "1D_R2C_double": {
        "args": {"double_precision": True},
        "numpy_fft": np.fft.rfft,
    },
    "1D_C2C_simple": {
        "args": {"force_complex": True},
        "numpy_fft": np.fft.fft,
    },
    "1D_C2C_double": {
        "args": {"double_precision": True, "force_complex": True},
        "numpy_fft": np.fft.fft,
    },
    # Batched 1D
    "batched_1D_R2C_simple": {
        "args": {"axes": (1,)},
        "numpy_fft": np.fft.rfft,
    },
    "batched_1D_R2C_double": {
        "args": {"axes": (1,), "double_precision": True},
        "numpy_fft": np.fft.rfft,
    },
    "batched_1D_C2C_simple": {
        "args": {"axes": (1,), "force_complex": True},
        "numpy_fft": np.fft.fft,
    },
    "batched_1D_C2C_double": {
        "args": {"axes": (1,), "double_precision": True, "force_complex": True},
        "numpy_fft": np.fft.fft,
    },
    # 2D
    "2D_R2C_simple": {
        "args": {},
        "numpy_fft": np.fft.rfft2,
    },
    "2D_R2C_double": {
        "args": {"double_precision": True},
        "numpy_fft": np.fft.rfft2,
    },
    "2D_C2C_simple": {
        "args": {"force_complex": True},
        "numpy_fft": np.fft.fft2,
    },
    "2D_C2C_double": {
        "args": {"double_precision": True, "force_complex": True},
        "numpy_fft": np.fft.fft2,
    },
    # Batched 2D
    "batched_2D_R2C_simple": {
        "args": {"axes": (2, 1)},
        "numpy_fft": np.fft.rfft2,
    },
    "batched_2D_R2C_double": {
        "args": {"axes": (2, 1), "double_precision": True},
        "numpy_fft": np.fft.rfft2,
    },
    "batched_2D_C2C_simple": {
        "args": {"axes": (2, 1), "force_complex": True},
        "numpy_fft": np.fft.fft2,
    },
    "batched_2D_C2C_double": {
        "args": {"axes": (2, 1), "double_precision": True, "force_complex": True},
        "numpy_fft": np.fft.fft2,
    },
    # 3D
    "3D_R2C_simple": {
        "args": {},
        "numpy_fft": np.fft.rfftn,
    },
    "3D_R2C_double": {
        "args": {"double_precision": True},
        "numpy_fft": np.fft.rfftn,
    },
    "3D_C2C_simple": {
        "args": {"force_complex": True},
        "numpy_fft": np.fft.fftn,
    },
    "3D_C2C_double": {
        "args": {"double_precision": True, "force_complex": True},
        "numpy_fft": np.fft.fftn,
    },
}


@unittest.skipUnless(ocl and fft.gpyfft_fft, "gpyfft is missing")
class TestFFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFFT, cls).setUpClass()
        if ocl:
            cls.ctx = ocl.create_context()
            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
                cls.queue = pyopencl.CommandQueue(
                                cls.ctx,
                                properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
            else:
                cls.PROFILE = False
                cls.queue = pyopencl.CommandQueue(cls.ctx)
            cls.max_valid_wg = 0

    @classmethod
    def tearDownClass(cls):
        super(TestFFT, cls).tearDownClass()
        logger.debug("Maximum valid workgroup size %s on device %s" % (cls.max_valid_wg, cls.ctx.devices[0]))
        cls.ctx = None
        cls.queue = None

    def __init__(self, methodName='runTest', name=None, params=None):
        unittest.TestCase.__init__(self, methodName)
        self.name = name
        self.params = params

    def setUp(self):
        self.data = ascent().astype("float32")

    def tearDown(self):
        self.data = None

    def _prepare_test(self, test_case_name):
        """
        Return information relevant for computing FFT:
        - FFT input data
        - max error resolution
        """
        name = test_case_name
        dtype = np.float32
        # data type
        if "double" in name:
            eps = 1e-9
            if "C2C" in name:
                dtype = np.complex128
                np_fft = np.fft.fft
            else:
                dtype = np.float64
        else:
            eps = 1e-3
            if "C2C" in name:
                dtype = np.complex64
            else:
                dtype = np.float32
        # data dimensions
        if "3D" in name or "batched_2D" in name:
            # Generate 3D data. A tiling gives a constant in one direction,
            # so FT should be a 2D "Dirac"
            data = np.tile(self.data[-128:, -128:], (128, 1, 1))
            # For simple precision with 3D transform, precision is even less
            if "double" not in name:
                eps = 1e-2

        elif "2D" in name or "batched_1D" in name:
            data = self.data
        else:
            data = self.data[0]
        data = data.astype(dtype)
        return data, eps

    def _compute_fft(self, data):
        F = fft.FFT(data.shape, **self.params["args"])
        res_ocl = F.fft(data)

        np_fft = self.params["numpy_fft"]
        res_np = np_fft(data)

        err_max = abs(res_ocl - res_np).max()
        logger.debug(os.linesep.join(F.log_profile(verbose=False)))
        return err_max

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_fft(self):
        logger.debug("Testing %s" % self.name)
        data, eps = self._prepare_test(self.name)
        err_max = self._compute_fft(data)
        self.assertLess(err_max, eps * np.abs(data).max(), "Results are the same")


def suite():
    testSuite = unittest.TestSuite()
    for test_name, test_params in test_cases.items():
        testSuite.addTest(parameterize(TestFFT, name=test_name, params=test_params))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
