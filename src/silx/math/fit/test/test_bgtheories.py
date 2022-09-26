# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
# ############################################################################*/
import copy
import unittest
import numpy
import random

from silx.math.fit import bgtheories
from silx.math.fit.functions import sum_gauss


class TestBgTheories(unittest.TestCase):
    """
    """
    def setUp(self):
        self.x = numpy.arange(100)
        self.y = 10 + 0.05 * self.x + sum_gauss(self.x, 10., 45., 15.)
        # add a very narrow high amplitude peak to test strip and snip
        self.y += sum_gauss(self.x, 100., 75., 2.)
        self.narrow_peak_index = list(self.x).index(75)
        random.seed()

    def tearDown(self):
        pass

    def testTheoriesAttrs(self):
        for theory_name in bgtheories.THEORY:
            self.assertIsInstance(theory_name, str)
            self.assertTrue(hasattr(bgtheories.THEORY[theory_name],
                                    "function"))
            self.assertTrue(hasattr(bgtheories.THEORY[theory_name].function,
                                    "__call__"))
        # Ensure legacy functions are not renamed accidentally
        self.assertTrue(
                {"No Background", "Constant", "Linear", "Strip", "Snip"}.issubset(
                        set(bgtheories.THEORY)))

    def testNoBg(self):
        nobgfun = bgtheories.THEORY["No Background"].function
        self.assertTrue(numpy.array_equal(nobgfun(self.x, self.y),
                                          numpy.zeros_like(self.x)))
        # default estimate
        self.assertEqual(bgtheories.THEORY["No Background"].estimate(self.x, self.y),
                         ([], []))

    def testConstant(self):
        consfun = bgtheories.THEORY["Constant"].function
        c = random.random() * 100
        self.assertTrue(numpy.array_equal(consfun(self.x, self.y, c),
                                          c * numpy.ones_like(self.x)))
        # default estimate
        esti_par, cons = bgtheories.THEORY["Constant"].estimate(self.x, self.y)
        self.assertEqual(cons,
                         [[0, 0, 0]])
        self.assertAlmostEqual(esti_par,
                               min(self.y))

    def testLinear(self):
        linfun = bgtheories.THEORY["Linear"].function
        a = random.random() * 100
        b = random.random() * 100
        self.assertTrue(numpy.array_equal(linfun(self.x, self.y, a, b),
                                          a + b * self.x))
        # default estimate
        esti_par, cons = bgtheories.THEORY["Linear"].estimate(self.x, self.y)

        self.assertEqual(cons,
                         [[0, 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(esti_par[0], 10, places=3)
        self.assertAlmostEqual(esti_par[1], 0.05, places=3)

    def testStrip(self):
        stripfun = bgtheories.THEORY["Strip"].function
        anchors = sorted(random.sample(list(self.x), 4))
        anchors_indices = [list(self.x).index(a) for a in anchors]

        # we really want to strip away the narrow peak
        anchors_indices_copy = copy.deepcopy(anchors_indices)
        for idx in anchors_indices_copy:
            if abs(idx - self.narrow_peak_index) < 5:
                anchors_indices.remove(idx)
                anchors.remove(self.x[idx])

        width = 2
        niter = 1000
        bgtheories.THEORY["Strip"].configure(AnchorsList=anchors, AnchorsFlag=True)

        bg = stripfun(self.x, self.y, width, niter)

        # assert peak amplitude has been decreased
        self.assertLess(bg[self.narrow_peak_index],
                        self.y[self.narrow_peak_index])

        # default estimate
        for i in anchors_indices:
            self.assertEqual(bg[i], self.y[i])

        # estimated parameters are equal to the default ones in the config dict
        bgtheories.THEORY["Strip"].configure(StripWidth=7, StripIterations=8)
        esti_par, cons = bgtheories.THEORY["Strip"].estimate(self.x, self.y)
        self.assertTrue(numpy.array_equal(cons, [[3, 0, 0], [3, 0, 0]]))
        self.assertEqual(esti_par, [7, 8])

    def testSnip(self):
        snipfun = bgtheories.THEORY["Snip"].function
        anchors = sorted(random.sample(list(self.x), 4))
        anchors_indices = [list(self.x).index(a) for a in anchors]

        # we want to strip away the narrow peak, so remove nearby anchors
        anchors_indices_copy = copy.deepcopy(anchors_indices)
        for idx in anchors_indices_copy:
            if abs(idx - self.narrow_peak_index) < 5:
                anchors_indices.remove(idx)
                anchors.remove(self.x[idx])

        width = 16
        bgtheories.THEORY["Snip"].configure(AnchorsList=anchors, AnchorsFlag=True)
        bg = snipfun(self.x, self.y, width)

        # assert peak amplitude has been decreased
        self.assertLess(bg[self.narrow_peak_index],
                        self.y[self.narrow_peak_index],
                        "Snip didn't decrease the peak amplitude.")

        # anchored data must remain fixed
        for i in anchors_indices:
            self.assertEqual(bg[i], self.y[i])

        # estimated parameters are equal to the default ones in the config dict
        bgtheories.THEORY["Snip"].configure(SnipWidth=7)
        esti_par, cons = bgtheories.THEORY["Snip"].estimate(self.x, self.y)
        self.assertTrue(numpy.array_equal(cons, [[3, 0, 0]]))
        self.assertEqual(esti_par, [7])
