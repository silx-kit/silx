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
"""Test of the filtered backprojection module"""

from __future__ import division, print_function

__authors__ = ["Pierre paleo"]
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/01/2018"


import time
import logging
import numpy
import unittest
try:
    import mako
except ImportError:
    mako = None
from ..common import ocl
if ocl:
    from .. import backprojection
from silx.test.utils import utilstest

logger = logging.getLogger(__name__)


def generate_coords(img_shp, center=None):
    """
    Return two 2D arrays containing the indexes of an image.
    The zero is at the center of the image.
    """
    l_r, l_c = float(img_shp[0]), float(img_shp[1])
    R, C = numpy.mgrid[:l_r, :l_c]
    if center is None:
        center0, center1 = l_r / 2., l_c / 2.
    else:
        center0, center1 = center
    R = R + 0.5 - center0
    C = C + 0.5 - center1
    return R, C


def clip_circle(img, center=None, radius=None):
    """
    Puts zeros outside the inscribed circle of the image support.
    """
    R, C = generate_coords(img.shape, center)
    M = R * R + C * C
    res = numpy.zeros_like(img)
    if radius is None:
        radius = img.shape[0] / 2. - 1
    mask = M < radius * radius
    res[mask] = img[mask]
    return res


@unittest.skipUnless(ocl and mako, "PyOpenCl is missing")
class TestFBP(unittest.TestCase):

    def setUp(self):
        if ocl is None:
            return
        # ~ if sys.platform.startswith('darwin'):
            # ~ self.skipTest("Backprojection is not implemented on CPU for OS X yet")
        self.getfiles()
        self.fbp = backprojection.Backprojection(self.sino.shape, profile=True)
        if self.fbp.compiletime_workgroup_size < 16 * 16:
            self.skipTest("Current implementation of OpenCL backprojection is not supported on this platform yet")

    def tearDown(self):
        self.sino = None
#         self.fbp.log_profile()
        self.fbp = None

    def getfiles(self):
        # load sinogram of 512x512 MRI phantom
        self.sino = numpy.load(utilstest.getfile("sino500.npz"))["data"]
        # load reconstruction made with ASTRA FBP (with filter designed in spatial domain)
        self.reference_rec = numpy.load(utilstest.getfile("rec_astra_500.npz"))["data"]

    def measure(self):
        "Common measurement of timings"
        t1 = time.time()
        try:
            result = self.fbp.filtered_backprojection(self.sino)
        except RuntimeError as msg:
            logger.error(msg)
            return
        t2 = time.time()
        return t2 - t1, result

    def compare(self, res):
        """
        Compare a result with the reference reconstruction.
        Only the valid reconstruction zone (inscribed circle) is taken into
        account
        """
        res_clipped = clip_circle(res)
        ref_clipped = clip_circle(self.reference_rec)
        delta = abs(res_clipped - ref_clipped)
        bad = delta > 1
#         numpy.save("/tmp/bad.npy", bad.astype(int))
        logger.debug("Absolute difference: %s with %s outlier pixels out of %s", delta.max(), bad.sum(), numpy.prod(bad.shape))
        return delta.max()

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_fbp(self):
        """
        tests FBP
        """
        # Test single reconstruction
        # --------------------------
        t, res = self.measure()
        if t is None:
            logger.info("test_fp: skipped")
        else:
            logger.info("test_backproj: time = %.3fs" % t)
            err = self.compare(res)
            msg = str("Max error = %e" % err)
            logger.info(msg)
            # TODO: cannot do better than 1e0 ?
            # The plain backprojection was much better, so it must be an issue in the filtering process
            self.assertTrue(err < 1., "Max error is too high")
        # Test multiple reconstructions
        # -----------------------------
        res0 = numpy.copy(res)
        for i in range(10):
            res = self.fbp.filtered_backprojection(self.sino)
            errmax = numpy.max(numpy.abs(res - res0))
            self.assertTrue(errmax < 1.e-6, "Max error is too high")


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestFBP("test_fbp"))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
