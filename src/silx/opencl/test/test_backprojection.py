#!/usr/bin/env python
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

__authors__ = ["Pierre paleo"]
__license__ = "MIT"
__copyright__ = "2013-2017 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/01/2018"


import time
import logging
import numpy as np
import unittest
from math import pi
try:
    import mako
except ImportError:
    mako = None
from ..common import ocl
if ocl:
    from .. import backprojection
    from ...image.tomography import compute_fourier_filter
from silx.test.utils import utilstest

logger = logging.getLogger(__name__)


def generate_coords(img_shp, center=None):
    """
    Return two 2D arrays containing the indexes of an image.
    The zero is at the center of the image.
    """
    l_r, l_c = float(img_shp[0]), float(img_shp[1])
    R, C = np.mgrid[:l_r, :l_c]
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
    res = np.zeros_like(img)
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
        self.getfiles()
        self.fbp = backprojection.Backprojection(self.sino.shape, profile=True)
        if self.fbp.compiletime_workgroup_size < 16 * 16:
            self.skipTest("Current implementation of OpenCL backprojection is "
                          "not supported on this platform yet")
        # Astra does not use the same backprojector implementation.
        # Therefore, we cannot expect results to be the "same" (up to float32
        # numerical error)
        self.tol = 5e-2
        if not(self.fbp._use_textures) or self.fbp.device.type == "CPU":
            # Precision is less when using CPU
            # (either CPU textures or "manual" linear interpolation)
            self.tol *= 2

    def tearDown(self):
        self.sino = None
        # self.fbp.log_profile()
        self.fbp = None

    def getfiles(self):
        # load sinogram of 512x512 MRI phantom
        self.sino = np.load(utilstest.getfile("sino500.npz"))["data"]
        # load reconstruction made with ASTRA FBP (with filter designed in spatial domain)
        self.reference_rec = np.load(utilstest.getfile("rec_astra_500.npz"))["data"]

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
        logger.debug("Absolute difference: %s with %s outlier pixels out of %s"
                     "", delta.max(), bad.sum(), np.prod(bad.shape))
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
            self.assertTrue(err < self.tol, "Max error is too high")

        # Test multiple reconstructions
        # -----------------------------
        res0 = np.copy(res)
        for i in range(10):
            res = self.fbp.filtered_backprojection(self.sino)
            errmax = np.max(np.abs(res - res0))
            self.assertTrue(errmax < 1.e-6, "Max error is too high")

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_fbp_filters(self):
        """
        Test the different available filters of silx FBP.
        """
        avail_filters = [
            "ramlak", "shepp-logan", "cosine", "hamming",
            "hann"
        ]
        # Create a Dirac delta function at a single angle view.
        # As the filters are radially invarant:
        #   - backprojection yields an image where each line is a Dirac.
        #   - FBP yields an image where each line is the spatial filter
        # One can simply filter "dirac" without backprojecting it, but this
        # test will also ensure that backprojection behaves well.
        dirac = np.zeros_like(self.sino)
        na, dw = dirac.shape
        dirac[0, dw//2] = na / pi * 2

        for filter_name in avail_filters:
            B = backprojection.Backprojection(dirac.shape, filter_name=filter_name)
            r = B(dirac)
            # Check that radial invariance is kept
            std0 = np.max(np.abs(np.std(r, axis=0)))
            self.assertTrue(
                std0 < 5.e-6,
                "Something wrong with FBP(filter=%s)" % filter_name
            )
            # Check that the filter is retrieved
            r_f = np.fft.fft(np.fft.fftshift(r[0])).real / 2.  # filter factor
            ref_filter_f = compute_fourier_filter(dw, filter_name)
            errmax = np.max(np.abs(r_f - ref_filter_f))
            logger.info("FBP filter %s: max error=%e" % (filter_name, errmax))
            self.assertTrue(
                errmax < 1.e-3,
                "Something wrong with FBP(filter=%s)" % filter_name
            )

    @unittest.skipUnless(ocl and mako, "pyopencl is missing")
    def test_fbp_oddsize(self):
        # Generate a 513-sinogram.
        # The padded width will be nextpow(513*2).
        # silx [0.10, 0.10.1] will give 1029, which makes R2C transform fail.
        sino = np.pad(self.sino, ((0, 0), (1, 0)), mode='edge')
        B = backprojection.Backprojection(sino.shape, axis_position=self.fbp.axis_pos+1)
        res = B(sino)
        # Compare with self.reference_rec. Tolerance is high as backprojector
        # is not fully shift-invariant.
        errmax = np.max(np.abs(clip_circle(res[1:, 1:] - self.reference_rec)))
        self.assertLess(
            errmax, 1.e-1,
            "Something wrong with FBP on odd-sized sinogram"
        )
