# coding: utf-8
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
"""
Tests for functions module
"""

import unittest
import numpy
import math

from silx.math.fit import functions

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "21/07/2016"

class Test_functions(unittest.TestCase):
    """
    Unit tests of multi-peak functions.
    """
    def setUp(self):
        self.x = numpy.arange(11)

        # height, center, sigma1, sigma2
        (h, c, s1, s2) = (7., 5., 3., 2.1)
        self.g_params = {
            "height": h,
            "center": c,
            #"sigma": s,
            "fwhm1": 2 * math.sqrt(2 * math.log(2)) * s1,
            "fwhm2": 2 * math.sqrt(2 * math.log(2)) * s2,
            "area1": h * s1 * math.sqrt(2 * math.pi)
        }
        # result of `7 * scipy.signal.gaussian(11, 3)`
        self.scipy_gaussian = numpy.array(
            [1.74546546, 2.87778603, 4.24571462, 5.60516182, 6.62171628,
             7., 6.62171628, 5.60516182, 4.24571462, 2.87778603,
             1.74546546]
        )

        # result of:
        # numpy.concatenate((7 * scipy.signal.gaussian(11, 3)[0:5],
        #                    7 * scipy.signal.gaussian(11, 2.1)[5:11]))
        self.scipy_asym_gaussian = numpy.array(
            [1.74546546, 2.87778603, 4.24571462, 5.60516182, 6.62171628,
             7., 6.24968751, 4.44773692, 2.52313452, 1.14093853, 0.41124877]
        )

    def tearDown(self):
        pass

    def testGauss(self):
        """Compare sum_gauss with scipy.signals.gaussian"""
        y = functions.sum_gauss(self.x,
                                self.g_params["height"],
                                self.g_params["center"],
                                self.g_params["fwhm1"])

        for i in range(11):
            self.assertAlmostEqual(y[i], self.scipy_gaussian[i])

    def testAGauss(self):
        """Compare sum_agauss with scipy.signals.gaussian"""
        y = functions.sum_agauss(self.x,
                                 self.g_params["area1"],
                                 self.g_params["center"],
                                 self.g_params["fwhm1"])
        for i in range(11):
            self.assertAlmostEqual(y[i], self.scipy_gaussian[i])

    def testFastAGauss(self):
        """Compare sum_fastagauss with scipy.signals.gaussian
        Limit precision to 3 decimal places."""
        y = functions.sum_fastagauss(self.x,
                                     self.g_params["area1"],
                                     self.g_params["center"],
                                     self.g_params["fwhm1"])
        for i in range(11):
            self.assertAlmostEqual(y[i], self.scipy_gaussian[i], 3)


    def testSplitGauss(self):
        """Compare sum_splitgauss with scipy.signals.gaussian"""
        y = functions.sum_splitgauss(self.x,
                                     self.g_params["height"],
                                     self.g_params["center"],
                                     self.g_params["fwhm1"],
                                     self.g_params["fwhm2"])
        for i in range(11):
            self.assertAlmostEqual(y[i], self.scipy_asym_gaussian[i])

    def testErf(self):
        """Compare erf with math.erf"""
        # scalars
        self.assertAlmostEqual(functions.erf(0.14), math.erf(0.14), places=5)
        self.assertAlmostEqual(functions.erf(0), math.erf(0), places=5)
        self.assertAlmostEqual(functions.erf(-0.74), math.erf(-0.74), places=5)

        # lists
        x = [-5, -2, -1.5, -0.6, 0, 0.1, 2, 3]
        erfx = functions.erf(x)
        for i in range(len(x)):
            self.assertAlmostEqual(erfx[i],
                                   math.erf(x[i]),
                                   places=5)

        # ndarray
        x = numpy.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        erfx = functions.erf(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.assertAlmostEqual(erfx[i, j],
                                       math.erf(x[i, j]),
                                       places=5)

    def testErfc(self):
        """Compare erf with math.erf"""
        # scalars
        self.assertAlmostEqual(functions.erfc(0.14), math.erfc(0.14), places=5)
        self.assertAlmostEqual(functions.erfc(0), math.erfc(0), places=5)
        self.assertAlmostEqual(functions.erfc(-0.74), math.erfc(-0.74), places=5)

        # lists
        x = [-5, -2, -1.5, -0.6, 0, 0.1, 2, 3]
        erfcx = functions.erfc(x)
        for i in range(len(x)):
            self.assertAlmostEqual(erfcx[i], math.erfc(x[i]), places=5)

        # ndarray
        x = numpy.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        erfcx = functions.erfc(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.assertAlmostEqual(erfcx[i, j], math.erfc(x[i, j]), places=5)

    def testAtanStepUp(self):
        """Compare atan_stepup with math.atan

        atan_stepup(x, a, b, c) = a * (0.5 + (arctan((x - b) / c) / pi))"""
        x0 = numpy.arange(100) / 6.33
        y0 = functions.atan_stepup(x0, 11.1, 22.2, 3.33)

        for x, y in zip(x0, y0):
            self.assertAlmostEqual(
                11.1 * (0.5 + math.atan((x - 22.2) / 3.33) / math.pi),
                y
            )

    def testStepUp(self):
        """sanity check for step up:

           - derivative must be largest around the step center
           - max value must be close to height parameter

        """
        x0 = numpy.arange(1000)
        center = 444
        height = 1234
        fwhm = 210
        y0 = functions.sum_stepup(x0, height, center, fwhm)

        self.assertLess(max(y0), height)
        self.assertAlmostEqual(max(y0), height, places=1)
        self.assertAlmostEqual(min(y0), 0, places=1)

        deriv0 = _numerical_derivative(functions.sum_stepup, x0, [height, center, fwhm])

        # Test center position within +- 1 sample of max derivative
        index_max_deriv = numpy.argmax(deriv0)
        self.assertLess(abs(index_max_deriv - center),
                        1)

    def testStepDown(self):
        """sanity check for step down:

           - absolute value of derivative must be largest around the step center
           - max value must be close to height parameter

        """
        x0 = numpy.arange(1000)
        center = 444
        height = 1234
        fwhm = 210
        y0 = functions.sum_stepdown(x0, height, center, fwhm)

        self.assertLess(max(y0), height)
        self.assertAlmostEqual(max(y0), height, places=1)
        self.assertAlmostEqual(min(y0), 0, places=1)

        deriv0 = _numerical_derivative(functions.sum_stepdown, x0, [height, center, fwhm])

        # Test center position within +- 1 sample of max derivative
        index_min_deriv = numpy.argmax(-deriv0)
        self.assertLess(abs(index_min_deriv - center),
                        1)

    def testSlit(self):
        """sanity check for slit:

           - absolute value of derivative must be largest around the step center
           - max value must be close to height parameter

        """
        x0 = numpy.arange(1000)
        center = 444
        height = 1234
        fwhm = 210
        beamfwhm = 30
        y0 = functions.sum_slit(x0, height, center, fwhm, beamfwhm)

        self.assertAlmostEqual(max(y0), height, places=1)
        self.assertAlmostEqual(min(y0), 0, places=1)

        deriv0 = _numerical_derivative(functions.sum_slit, x0, [height, center, fwhm, beamfwhm])

        # Test step up center  position (center - fwhm/2) within +- 1 sample of max derivative
        index_max_deriv = numpy.argmax(deriv0)
        self.assertLess(abs(index_max_deriv - (center - fwhm/2)),
                        1)
        # Test step down center position (center + fwhm/2) within +- 1 sample of min derivative
        index_min_deriv = numpy.argmin(deriv0)
        self.assertLess(abs(index_min_deriv - (center + fwhm/2)),
                        1)


def _numerical_derivative(f, x, params=[], delta_factor=0.0001):
    """Compute the numerical derivative of ``f`` for all values of ``x``.

    :param f: function
    :param x: Array of evenly spaced abscissa values
    :param params: list of additional parameters
    :return: Array of derivative values
    """
    deltax = (x[1] - x[0]) * delta_factor
    y_plus = f(x + deltax, *params)
    y_minus = f(x - deltax, *params)

    return (y_plus - y_minus) / (2 * deltax)

test_cases = (Test_functions,)

def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
