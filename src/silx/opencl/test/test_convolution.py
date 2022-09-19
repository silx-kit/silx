#!/usr/bin/env python
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

__authors__ = ["Pierre Paleo"]
__contact__ = "pierre.paleo@esrf.fr"
__license__ = "MIT"
__copyright__ = "2019 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/08/2019"

import pytest
import logging
from itertools import product
import numpy as np
from silx.image.utils import gaussian_kernel

try:
    from scipy.ndimage import convolve, convolve1d
    from scipy.misc import ascent

    scipy_convolve = convolve
    scipy_convolve1d = convolve1d
except ImportError:
    scipy_convolve = None
import unittest
from ..common import ocl, check_textures_availability

if ocl:
    import pyopencl as cl
    import pyopencl.array as parray
    from silx.opencl.convolution import Convolution
logger = logging.getLogger(__name__)


class ConvolutionData:

    def __init__(self, param):
        self.param = param
        self.mode = param["boundary_handling"]
        logger.debug(
            """
            Testing convolution with boundary_handling=%s,
            use_textures=%s, input_device=%s, output_device=%s
            """
            % (
                self.mode,
                param["use_textures"],
                param["input_on_device"],
                param["output_on_device"],
            )
        )

    @classmethod
    def setUpClass(cls):
        cls.image = np.ascontiguousarray(ascent()[:, :511], dtype="f")
        cls.data1d = cls.image[0]
        cls.data2d = cls.image
        cls.data3d = np.tile(cls.image[224:-224, 224:-224], (62, 1, 1))
        cls.kernel1d = gaussian_kernel(1.0)
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

    def instantiate_convol(self, shape, kernel, axes=None):
        if self.mode == "constant":
            if not (self.param["use_textures"]) or (
                self.param["use_textures"]
                and not (check_textures_availability(self.ctx))
            ):
                pytest.skip("mode=constant not implemented without textures")
        C = Convolution(
            shape,
            kernel,
            mode=self.mode,
            ctx=self.ctx,
            axes=axes,
            extra_options={"dont_use_textures": not (self.param["use_textures"])},
        )
        return C

    def get_data_and_kernel(self, test_name):
        dims = {
            "test_1D": (1, 1),
            "test_separable_2D": (2, 1),
            "test_separable_3D": (3, 1),
            "test_nonseparable_2D": (2, 2),
            "test_nonseparable_3D": (3, 3),
        }
        dim_data = {1: self.data1d, 2: self.data2d, 3: self.data3d}
        dim_kernel = {
            1: self.kernel1d,
            2: self.kernel2d,
            3: self.kernel3d,
        }
        dd, kd = dims[test_name]
        return dim_data[dd], dim_kernel[kd]

    def get_reference_function(self, test_name):
        ref_func = {
            "test_1D": lambda x, y: scipy_convolve1d(x, y, mode=self.mode),
            "test_separable_2D": lambda x, y: scipy_convolve1d(
                scipy_convolve1d(x, y, mode=self.mode, axis=1),
                y,
                mode=self.mode,
                axis=0,
            ),
            "test_separable_3D": lambda x, y: scipy_convolve1d(
                scipy_convolve1d(
                    scipy_convolve1d(x, y, mode=self.mode, axis=2),
                    y,
                    mode=self.mode,
                    axis=1,
                ),
                y,
                mode=self.mode,
                axis=0,
            ),
            "test_nonseparable_2D": lambda x, y: scipy_convolve(x, y, mode=self.mode),
            "test_nonseparable_3D": lambda x, y: scipy_convolve(x, y, mode=self.mode),
        }
        return ref_func[test_name]

    def template_test(self, test_name):
        data, kernel = self.get_data_and_kernel(test_name)
        conv = self.instantiate_convol(data.shape, kernel)
        if self.param["input_on_device"]:
            data_ref = parray.to_device(conv.queue, data)
        else:
            data_ref = data
        if self.param["output_on_device"]:
            d_res = parray.empty_like(conv.data_out)
            d_res.fill(0)
            res = d_res
        else:
            res = None
        res = conv(data_ref, output=res)
        if self.param["output_on_device"]:
            res = res.get()
        ref_func = self.get_reference_function(test_name)
        ref = ref_func(data, kernel)
        metric = self.compare(res, ref)
        logger.info("%s: max error = %.2e" % (test_name, metric))
        tol = self.tol[str("%dD" % kernel.ndim)]
        assert metric < tol, self.print_err(conv)


def convolution_data_params():
    boundary_handlings = ["reflect", "nearest", "wrap", "constant"]
    use_textures = [True, False]
    input_on_devices = [True, False]
    output_on_devices = [True, False]
    param_vals = list(
        product(boundary_handlings, use_textures, input_on_devices, output_on_devices)
    )
    params = []
    for boundary_handling, use_texture, input_dev, output_dev in param_vals:
        param={
            "boundary_handling": boundary_handling,
            "input_on_device": input_dev,
            "output_on_device": output_dev,
            "use_textures": use_texture,
        }
        params.append(param)

    return params


@pytest.fixture(scope="module", params=convolution_data_params())
def convolution_data(request):
    """Provide a set of convolution data

    The module scope allows to test each function during a single setup of each
    convolution data
    """
    cdata = None
    try:
        cdata = ConvolutionData(request.param)
        cdata.setUpClass()
        yield cdata
    finally:
        cdata.tearDownClass()


@pytest.mark.skipif(ocl is None, reason="OpenCL is missing")
@pytest.mark.skipif(scipy_convolve is None, reason="scipy is missing")
def test_1D(convolution_data):
    convolution_data.template_test("test_1D")

@pytest.mark.skipif(ocl is None, reason="OpenCL is missing")
@pytest.mark.skipif(scipy_convolve is None, reason="scipy is missing")
def test_separable_2D(convolution_data):
    convolution_data.template_test("test_separable_2D")

@pytest.mark.skipif(ocl is None, reason="OpenCL is missing")
@pytest.mark.skipif(scipy_convolve is None, reason="scipy is missing")
def test_separable_3D(convolution_data):
    convolution_data.template_test("test_separable_3D")

@pytest.mark.skipif(ocl is None, reason="OpenCL is missing")
@pytest.mark.skipif(scipy_convolve is None, reason="scipy is missing")
def test_nonseparable_2D(convolution_data):
    convolution_data.template_test("test_nonseparable_2D")

@pytest.mark.skipif(ocl is None, reason="OpenCL is missing")
@pytest.mark.skipif(scipy_convolve is None, reason="scipy is missing")
def test_nonseparable_3D(convolution_data):
    convolution_data.template_test("test_nonseparable_3D")

@pytest.mark.skipif(ocl is None, reason="OpenCL is missing")
@pytest.mark.skipif(scipy_convolve is None, reason="scipy is missing")
def test_batched_2D(convolution_data):
    """
    Test batched (nonseparable) 2D convolution on 3D data.
    In this test: batch along "z" (axis 0)
    """
    data = convolution_data.data3d
    kernel = convolution_data.kernel2d
    conv = convolution_data.instantiate_convol(data.shape, kernel, axes=(0,))
    res = conv(data)  # 3D
    ref = scipy_convolve(data[0], kernel, mode=convolution_data.mode)  # 2D

    std = np.std(res, axis=0)
    std_max = np.max(np.abs(std))
    assert std_max < convolution_data.tol["2D"], convolution_data.print_err(conv)
    metric = convolution_data.compare(res[0], ref)
    logger.info("test_nonseparable_3D: max error = %.2e" % metric)
    assert metric < convolution_data.tol["2D"], convolution_data.print_err(conv)
