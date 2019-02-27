#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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

"""
Test of the Convolution class.
"""

from __future__ import division, print_function

__authors__ = ["Pierre Paleo"]
__contact__ = "pierre.paleo@esrf.fr"
__license__ = "MIT"
__copyright__ = "2019 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/02/2019"

import logging
from itertools import product
import numpy as np
from math import ceil
from silx.utils.testutils import parameterize
try:
    from scipy.ndimage import convolve, convolve1d
    from scipy.misc import ascent
    scipy_convolve = convolve
    scipy_convolve1d = convolve1d
except ImportError:
    scipy_convolve = None
import unittest
from ..common import ocl
#~ from silx.opencl.common import ocl
if ocl:
    import pyopencl as cl
    import pyopencl.array
    from ..convolution import Convolution
    #~ from silx.opencl.convolution import Convolution
logger = logging.getLogger(__name__)


# TODO move elsewhere
def gaussian_kernel(sigma, cutoff=4, force_odd_size=False):
    size = int(ceil(2 * cutoff * sigma + 1))
    if force_odd_size and size % 2 == 0:
        size += 1
    x = np.arange(size) - (size - 1.0) / 2.0
    g = np.exp(-(x / sigma) ** 2 / 2.0)
    g /= g.sum()
    return g


@unittest.skipUnless(ocl and scipy_convolve, "PyOpenCl/scipy is missing")
class TestConvolution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestConvolution, cls).setUpClass()
        cls.image = np.ascontiguousarray(ascent()[:, :511], dtype="f")
        cls.data1d = cls.image[0]
        cls.data2d = cls.image
        cls.data3d = np.tile(cls.image[224:-224, 224:-224], (62, 1, 1))
        cls.kernel1d = gaussian_kernel(1.)
        cls.kernel2d = np.outer(cls.kernel1d, cls.kernel1d)
        cls.kernel3d = np.multiply.outer(cls.kernel2d, cls.kernel1d)
        cls.ctx = ocl.create_context()
        cls.tol = {
            "1D": 1e-4,
            "2D": 1e-3,
            "3D": 1e-3,
        }

    @classmethod
    def tearDownClass(cls):
        cls.data1d = cls.data2d = cls.data3d = cls.image = None
        cls.kernel1d = cls.kernel2d = cls.kernel3d = None

    @staticmethod
    def compare(arr1, arr2):
        return np.max(np.abs(arr1 - arr2))

    @staticmethod
    def print_err(conv):
        errmsg = str(
            """
            Something wrong with %s
            mode=%s, texture=%s
            """
            % (conv.use_case_desc, conv.mode, conv.use_textures)
        )
        return errmsg

    def __init__(self, methodName='runTest', param=None):
        unittest.TestCase.__init__(self, methodName)
        self.param = param
        self.mode = param["boundary_handling"]
        logger.debug(
            """
            Testing convolution with boundary_handling=%s,
            use_textures=%s, input_gpu=%s, output_gpu=%s
            """
            % (
                self.mode, param["use_textures"],
                param["input_gpu"], param["output_gpu"]
            )
        )

    def instantiate_convol(self, shape, kernel, axes=None):
        if (self.mode == "constant") and (
            not(self.param["use_textures"])
                or (self.ctx.devices[0].type == cl._cl.device_type.CPU)
            ):
                self.skipTest("mode=constant not implemented without textures")
        C = Convolution(
            shape, kernel,
            mode=self.mode,
            ctx=self.ctx,
            axes=axes,
            extra_options={"dont_use_textures": not(self.param["use_textures"])}
        )
        return C

    def get_data_and_kernel(self, test_name):
        dims = {
            "test_1D": (1, 1),
            "test_separable_2D": (2, 1),
            "test_separable_3D": (3, 1),
            "test_nonseparable_2D": (2, 2),
            "test_nonseparable_3D":  (3, 3),
        }
        dim_data = {
            1: self.data1d,
            2: self.data2d,
            3: self.data3d
        }
        dim_kernel = {
            1: self.kernel1d,
            2: self.kernel2d,
            3: self.kernel3d,
        }
        dd, kd = dims[test_name]
        return dim_data[dd], dim_kernel[kd]

    def get_reference_function(self, test_name):
        ref_func = {
            "test_1D":
                lambda x, y : scipy_convolve1d(x, y, mode=self.mode),
            "test_separable_2D":
                lambda x, y : scipy_convolve1d(
                    scipy_convolve1d(x, y, mode=self.mode, axis=1),
                    y, mode=self.mode, axis=0
                ),
            "test_separable_3D":
                lambda x, y: scipy_convolve1d(
                    scipy_convolve1d(
                        scipy_convolve1d(x, y, mode=self.mode, axis=2),
                        y, mode=self.mode, axis=1),
                    y, mode=self.mode, axis=0
                ),
            "test_nonseparable_2D":
                lambda x, y: scipy_convolve(x, y, mode=self.mode),
            "test_nonseparable_3D":
                lambda x, y : scipy_convolve(x, y, mode=self.mode),
        }
        return ref_func[test_name]

    def template_test(self, test_name):
        data, kernel = self.get_data_and_kernel(test_name)
        conv = self.instantiate_convol(data.shape, kernel)
        res = conv(data)
        ref_func = self.get_reference_function(test_name)
        ref = ref_func(data, kernel)
        metric = self.compare(res, ref)
        logger.info("%s: max error = %.2e" % (test_name, metric))
        tol = self.tol[str("%dD" % kernel.ndim)]
        self.assertLess(metric, tol, self.print_err(conv))

    def test_1D(self):
        self.template_test("test_1D")

    def test_separable_2D(self):
        self.template_test("test_separable_2D")

    def test_separable_3D(self):
        self.template_test("test_separable_3D")

    def test_nonseparable_2D(self):
        self.template_test("test_nonseparable_2D")

    def test_nonseparable_3D(self):
        self.template_test("test_nonseparable_3D")

    def test_batched_2D(self):
        """
        Test batched (nonseparable) 2D convolution on 3D data.
        In this test: batch along "z" (axis 0)
        """
        data = self.data3d
        kernel = self.kernel2d
        conv = self.instantiate_convol(data.shape, kernel, axes=(0,))
        res = conv(data) # 3D
        ref = scipy_convolve(data[0], kernel, mode=self.mode) # 2D

        std = np.std(res, axis=0)
        std_max = np.max(np.abs(std))
        self.assertLess(std_max, self.tol["2D"], self.print_err(conv))
        metric = self.compare(res[0], ref)
        logger.info("test_nonseparable_3D: max error = %.2e" % metric)
        self.assertLess(metric, self.tol["2D"], self.print_err(conv))


# TODO replace X_gpu with X_device
def test_convolution():
    boundary_handling_ = ["reflect", "nearest", "wrap", "constant"]
    use_textures_ = [True, False]
    #~ use_textures_ = [True] # DEBUG
    #~ input_gpu_ = [True, False]
    input_gpu_ = [False] # DEBUG
    #~ output_gpu_ = [True, False]
    output_gpu_ = [False] # DEBUG
    testSuite = unittest.TestSuite()

    param_vals = list(product(
        boundary_handling_,
        use_textures_,
        input_gpu_,
        output_gpu_
    ))
    for boundary_handling, use_textures, input_gpu, output_gpu in param_vals:
        testcase = parameterize(
            TestConvolution,
            param={
                "boundary_handling": boundary_handling,
                "input_gpu": input_gpu,
                "output_gpu": output_gpu,
                "use_textures": use_textures,
            }
        )
        testSuite.addTest(testcase)
    return testSuite



def suite():
    testSuite = test_convolution()
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
