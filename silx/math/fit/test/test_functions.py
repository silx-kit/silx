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
        self.assertAlmostEqual(functions.erf(0.14), math.erf(0.14))
        self.assertAlmostEqual(functions.erf(0), math.erf(0))
        self.assertAlmostEqual(functions.erf(-0.74), math.erf(-0.74))

        # lists
        x = [-5, -2, -1.5, -0.6, 0, 0.1, 2, 3]
        erfx = functions.erf(x)
        for i in range(len(x)):
            self.assertAlmostEqual(erfx[i], math.erf(x[i]))

        # ndarray
        x = numpy.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        erfx = functions.erf(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.assertAlmostEqual(erfx[i, j], math.erf(x[i, j]))

    def testErfc(self):
        """Compare erf with math.erf"""
        # scalars
        self.assertAlmostEqual(functions.erfc(0.14), math.erfc(0.14))
        self.assertAlmostEqual(functions.erfc(0), math.erfc(0))
        self.assertAlmostEqual(functions.erfc(-0.74), math.erfc(-0.74))

        # lists
        x = [-5, -2, -1.5, -0.6, 0, 0.1, 2, 3]
        erfcx = functions.erfc(x)
        for i in range(len(x)):
            self.assertAlmostEqual(erfcx[i], math.erfc(x[i]))

        # ndarray
        x = numpy.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        erfcx = functions.erfc(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.assertAlmostEqual(erfcx[i, j], math.erfc(x[i, j]))

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
