# -*- coding: utf-8 -*-
#
#    Project: silx (originally pyFAI)
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2017  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "02/08/2016"

import unittest
import numpy
import logging
logger = logging.getLogger(__name__)
from ..bilinear import BilinearImage


class TestBilinear(unittest.TestCase):
    """basic maximum search test"""
    N = 1000

    def test_max_search_round(self):
        """test maximum search using random points: maximum is at the pixel center"""
        a = numpy.arange(100) - 40.
        b = numpy.arange(100) - 60.
        ga = numpy.exp(-a * a / 4000)
        gb = numpy.exp(-b * b / 6000)
        gg = numpy.outer(ga, gb)
        b = BilinearImage(gg)
        ok = 0
        for s in range(self.N):
            i, j = numpy.random.randint(100), numpy.random.randint(100)
            k, l = b.local_maxi((i, j))
            if abs(k - 40) > 1e-4 or abs(l - 60) > 1e-4:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, k, l)
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, k, l)
                ok += 1
        logger.debug("Success rate: %.1f", 100. * ok / self.N)
        self.assertEqual(ok, self.N, "Maximum is always found")

    def test_max_search_half(self):
        """test maximum search using random points: maximum is at a pixel edge"""
        a = numpy.arange(100) - 40.5
        b = numpy.arange(100) - 60.5
        ga = numpy.exp(-a * a / 4000)
        gb = numpy.exp(-b * b / 6000)
        gg = numpy.outer(ga, gb)
        b = BilinearImage(gg)
        ok = 0
        for s in range(self.N):
            i, j = numpy.random.randint(100), numpy.random.randint(100)
            k, l = b.local_maxi((i, j))
            if abs(k - 40.5) > 0.5 or abs(l - 60.5) > 0.5:
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, k, l)
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)", i, j, k, l)
                ok += 1
        logger.debug("Success rate: %.1f", 100. * ok / self.N)
        self.assertEqual(ok, self.N, "Maximum is always found")

    def test_map(self):
        N = 100
        y, x = numpy.ogrid[:N, :N + 10]
        img = x + y
        b = BilinearImage(img)
        x2d = numpy.zeros_like(y) + x
        y2d = numpy.zeros_like(x) + y
        res1 = b.map_coordinates((y2d, x2d))
        self.assertEqual(abs(res1 - img).max(), 0, "images are the same (corners)")

        x2d = numpy.zeros_like(y) + (x[:, :-1] + 0.5)
        y2d = numpy.zeros_like(x[:, :-1]) + y
        res1 = b.map_coordinates((y2d, x2d))
        self.assertEqual(abs(res1 - img[:, :-1] - 0.5).max(), 0, "images are the same (middle)")

        x2d = numpy.zeros_like(y[:-1, :]) + (x[:, :-1] + 0.5)
        y2d = numpy.zeros_like(x[:, :-1]) + (y[:-1, :] + 0.5)
        res1 = b.map_coordinates((y2d, x2d))
        self.assertEqual(abs(res1 - img[:-1, 1:]).max(), 0, "images are the same (center)")

    def test_profile_grad(self):
        N = 100
        img = numpy.arange(N * N).reshape(N, N)
        b = BilinearImage(img)
        res1 = b.profile_line((0, 0), (N - 1, N - 1))
        l = numpy.ceil(numpy.sqrt(2) * N)
        self.assertEqual(len(res1), l, "Profile has correct length")
        self.assertLess((res1[:-2] - res1[1:-1]).std(), 1e-3, "profile is linear (excluding last point)")

    def test_profile_gaus(self):
        N = 100
        x = numpy.arange(N) - N // 2.0
        g = numpy.exp(-x * x / (N * N))
        img = numpy.outer(g, g)
        b = BilinearImage(img)
        res_hor = b.profile_line((N // 2, 0), (N // 2, N - 1))
        res_ver = b.profile_line((0, N // 2), (N - 1, N // 2))
        self.assertEqual(len(res_hor), N, "Profile has correct length")
        self.assertEqual(len(res_ver), N, "Profile has correct length")
        self.assertLess(abs(res_hor - g).max(), 1e-5, "correct horizontal profile")
        self.assertLess(abs(res_ver - g).max(), 1e-5, "correct vertical profile")

        # Profile with linewidth=3
        expected_profile = img[:, N // 2 - 1:N // 2 + 2].mean(axis=1)
        res_hor = b.profile_line((N // 2, 0), (N // 2, N - 1), linewidth=3)
        res_ver = b.profile_line((0, N // 2), (N - 1, N // 2), linewidth=3)

        self.assertEqual(len(res_hor), N, "Profile has correct length")
        self.assertEqual(len(res_ver), N, "Profile has correct length")
        self.assertLess(abs(res_hor - expected_profile).max(), 1e-5,
                        "correct horizontal profile")
        self.assertLess(abs(res_ver - expected_profile).max(), 1e-5,
                        "correct vertical profile")


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestBilinear("test_max_search_round"))
    testsuite.addTest(TestBilinear("test_max_search_half"))
    testsuite.addTest(TestBilinear("test_map"))
    testsuite.addTest(TestBilinear("test_profile_grad"))
    testsuite.addTest(TestBilinear("test_profile_gaus"))
    return testsuite
