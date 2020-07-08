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
"""Test of the linalg module"""

from __future__ import division, print_function

__authors__ = ["Pierre paleo"]
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/08/2019"


import time
import logging
import numpy as np
import unittest
try:
    import mako
except ImportError:
    mako = None
from ..common import ocl
if ocl:
    import pyopencl as cl
    import pyopencl.array as parray
    from .. import linalg
from silx.test.utils import utilstest

logger = logging.getLogger(__name__)
try:
    from scipy.ndimage.filters import laplace
    _has_scipy = True
except ImportError:
    _has_scipy = False


# TODO move this function in math or image ?
def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Code from https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[tuple(slice_all)] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


# TODO move this function in math or image ?
def divergence(grad):
    '''
    Compute the divergence of a gradient
    Code from https://github.com/emmanuelle/tomo-tv/
    '''
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


@unittest.skipUnless(ocl and mako, "PyOpenCl is missing")
class TestLinAlg(unittest.TestCase):

    def setUp(self):
        if ocl is None:
            return
        self.getfiles()
        self.la = linalg.LinAlg(self.image.shape)
        self.allocate_arrays()

    def allocate_arrays(self):
        """
        Allocate various types of arrays for the tests
        """
        # numpy images
        self.grad = np.zeros(self.image.shape, dtype=np.complex64)
        self.grad2 = np.zeros((2,) + self.image.shape, dtype=np.float32)
        self.grad_ref = gradient(self.image)
        self.div_ref = divergence(self.grad_ref)
        self.image2 = np.zeros_like(self.image)
        # Device images
        self.gradient_parray = parray.empty(self.la.queue, self.image.shape, np.complex64)
        self.gradient_parray.fill(0)
        # we should be using cl.Buffer(self.la.ctx, cl.mem_flags.READ_WRITE, size=self.image.nbytes*2),
        # but platforms not suporting openCL 1.2 have a problem with enqueue_fill_buffer,
        # so we use the parray "fill" utility
        self.gradient_buffer = self.gradient_parray.data
        # Do the same for image
        self.image_parray = parray.to_device(self.la.queue, self.image)
        self.image_buffer = self.image_parray.data
        # Refs
        tmp = np.zeros(self.image.shape, dtype=np.complex64)
        tmp.real = np.copy(self.grad_ref[0])
        tmp.imag = np.copy(self.grad_ref[1])
        self.grad_ref_parray = parray.to_device(self.la.queue, tmp)
        self.grad_ref_buffer = self.grad_ref_parray.data

    def tearDown(self):
        self.image = None
        self.image2 = None
        self.grad = None
        self.grad2 = None
        self.grad_ref = None
        self.div_ref = None
        self.gradient_parray.data.release()
        self.gradient_parray = None
        self.gradient_buffer = None
        self.image_parray.data.release()
        self.image_parray = None
        self.image_buffer = None
        self.grad_ref_parray.data.release()
        self.grad_ref_parray = None
        self.grad_ref_buffer = None

    def getfiles(self):
        # load 512x512 MRI phantom - TODO include Lena or ascent once a .npz is available
        self.image = np.load(utilstest.getfile("Brain512.npz"))["data"]

    def compare(self, result, reference, abstol, name):
        errmax = np.max(np.abs(result - reference))
        logger.info("%s: Max error = %e" % (name, errmax))
        self.assertTrue(errmax < abstol, str("%s: Max error is too high" % name))

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_gradient(self):
        arrays = {
            "numpy.ndarray": self.image,
            "buffer": self.image_buffer,
            "parray": self.image_parray
        }
        for desc, image in arrays.items():
            # Test with dst on host (numpy.ndarray)
            res = self.la.gradient(image, return_to_host=True)
            self.compare(res, self.grad_ref, 1e-6, str("gradient[src=%s, dst=numpy.ndarray]" % desc))
            # Test with dst on device (pyopencl.Buffer)
            self.la.gradient(image, dst=self.gradient_buffer)
            cl.enqueue_copy(self.la.queue, self.grad, self.gradient_buffer)
            self.grad2[0] = self.grad.real
            self.grad2[1] = self.grad.imag
            self.compare(self.grad2, self.grad_ref, 1e-6, str("gradient[src=%s, dst=buffer]" % desc))
            # Test with dst on device (pyopencl.Array)
            self.la.gradient(image, dst=self.gradient_parray)
            self.grad = self.gradient_parray.get()
            self.grad2[0] = self.grad.real
            self.grad2[1] = self.grad.imag
            self.compare(self.grad2, self.grad_ref, 1e-6, str("gradient[src=%s, dst=parray]" % desc))

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_divergence(self):
        arrays = {
            "numpy.ndarray": self.grad_ref,
            "buffer": self.grad_ref_buffer,
            "parray": self.grad_ref_parray
        }
        for desc, grad in arrays.items():
            # Test with dst on host (numpy.ndarray)
            res = self.la.divergence(grad, return_to_host=True)
            self.compare(res, self.div_ref, 1e-6, str("divergence[src=%s, dst=numpy.ndarray]" % desc))
            # Test with dst on device (pyopencl.Buffer)
            self.la.divergence(grad, dst=self.image_buffer)
            cl.enqueue_copy(self.la.queue, self.image2, self.image_buffer)
            self.compare(self.image2, self.div_ref, 1e-6, str("divergence[src=%s, dst=buffer]" % desc))
            # Test with dst on device (pyopencl.Array)
            self.la.divergence(grad, dst=self.image_parray)
            self.image2 = self.image_parray.get()
            self.compare(self.image2, self.div_ref, 1e-6, str("divergence[src=%s, dst=parray]" % desc))

    @unittest.skipUnless(ocl and mako and _has_scipy, "pyopencl and/or scipy is missing")
    def test_laplacian(self):
        laplacian_ref = laplace(self.image)
        # Laplacian = div(grad)
        self.la.gradient(self.image)
        laplacian_ocl = self.la.divergence(self.la.d_gradient, return_to_host=True)
        self.compare(laplacian_ocl, laplacian_ref, 1e-6, "laplacian")


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestLinAlg("test_gradient"))
    testSuite.addTest(TestLinAlg("test_divergence"))
    testSuite.addTest(TestLinAlg("test_laplacian"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
