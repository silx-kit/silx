# -*- coding: utf-8 -*-
#
#    Project: silx (originally pyFAI)
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2016  European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "29/04/2016"

import unittest
import numpy
import logging
logger = logging.getLogger("test_bilinear")
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
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
                ok += 1
        logger.info("Success rate: %.1f" % (100. * ok / self.N))
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
                logger.warning("Wrong guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
            else:
                logger.debug("Good guess maximum (%i,%i) -> (%.1f,%.1f)" % (i, j, k, l))
                ok += 1
        logger.info("Success rate: %.1f" % (100. * ok / self.N))
        self.assertEqual(ok, self.N, "Maximum is always found")

    def test_map(self):
        N = 100
        y, x = numpy.ogrid[:N, :N + 10]
        img = x + y
        b = BilinearImage(img)
        x2d = numpy.zeros_like(y) + x
        y2d = numpy.zeros_like(x) + y
        res1 = b.map_coordinates((y2d, x2d))
        self.assertEquals(abs(res1 - img).max(), 0, "images are the same (corners)")

        x2d = numpy.zeros_like(y) + (x[:, :-1] + 0.5)
        y2d = numpy.zeros_like(x[:, :-1]) + y
        res1 = b.map_coordinates((y2d, x2d))
        self.assertEquals(abs(res1 - img[:, :-1] - 0.5).max(), 0, "images are the same (middle)")

        x2d = numpy.zeros_like(y[:-1, :]) + (x[:, :-1] + 0.5)
        y2d = numpy.zeros_like(x[:, :-1]) + (y[:-1, :] + 0.5)
        print(x2d.shape, y2d.shape)
        res1 = b.map_coordinates((y2d, x2d))
        self.assertEquals(abs(res1 - img[:-1, 1:]).max(), 0, "images are the same (center)")


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestBilinear("test_max_search_round"))
    testsuite.addTest(TestBilinear("test_max_search_half"))
    testsuite.addTest(TestBilinear("test_map"))

    return testsuite
