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
Tests for fitfunctions module
"""

import unittest
import numpy
import math

from silx.math import fitfunctions

# TODO:
#     - snip1d
#     - snip2d
#     - snip3d
#     - strip

class Test_peak_search(unittest.TestCase):
    """
    Unit tests of peak_search on various types of multi-peak functions.
    """
    def setUp(self):
        self.x = numpy.arange(5000)
        # (height1, center1, fwhm1, ...)
        self.h_c_fwhm = (50, 500, 100,
                         50, 600, 80,
                         20, 2000, 100,
                         50, 2250, 110,
                         40, 3000, 99,
                         23, 4980, 80)
        # (height1, center1, fwhm1, eta1 ...)
        self.h_c_fwhm_eta = (50, 500, 100, 0.4,
                             50, 600, 80, 0.5,
                             20, 2000, 100, 0.6,
                             50, 2250, 110, 0.7,
                             40, 3000, 99, 0.8,
                             23, 4980, 80, 0.3,)
        # (height1, center1, fwhm11, fwhm21, ...)
        self.h_c_fwhm_fwhm = (50, 500, 100, 85,
                              50, 600, 80, 110,
                              20, 2000, 100, 100,
                              50, 2250, 110, 99,
                              40, 3000, 99, 110,
                              23, 4980, 80, 80,)
        # (height1, center1, fwhm11, fwhm21, eta1 ...)
        self.h_c_fwhm_fwhm_eta = (50, 500, 100, 85, 0.4,
                                  50, 600, 80, 110, 0.5,
                                  20, 2000, 100, 100, 0.6,
                                  50, 2250, 110, 99, 0.7,
                                  40, 3000, 99, 110, 0.8,
                                  23, 4980, 80, 80, 0.3,)
        # (area1, center1, fwhm1, ...)
        self.a_c_fwhm = (2550, 500, 100,
                         2000, 600, 80,
                         500, 2000, 100,
                         4000, 2250, 110,
                         2300, 3000, 99,
                         3333, 4980, 80)
        # (area1, center1, fwhm1, eta1 ...)
        self.a_c_fwhm_eta = (500, 500, 100, 0.4,
                             500, 600, 80, 0.5,
                             200, 2000, 100, 0.6,
                             500, 2250, 110, 0.7,
                             400, 3000, 99, 0.8,
                             230, 4980, 80, 0.3,)
        # (area, position, fwhm, st_area_r, st_slope_r, lt_area_r, lt_slope_r, step_height_r)
        self.hypermet_params = (1000, 500, 200, 1.2, 100, 0.3, 100, 0.05,
                                1000, 1000, 200, 1.2, 100, 0.3, 100, 0.05,
                                1000, 2000, 200, 1.2, 100, 0.3, 100, 0.05,
                                1000, 2350, 200, 1.2, 100, 0.3, 100, 0.05,
                                1000, 3000, 200, 1.2, 100, 0.3, 100, 0.05,
                                1000, 4900, 200, 1.2, 100, 0.3, 100, 0.05,)


    def tearDown(self):
        pass

    def get_peaks(self, function, params):
        """

        :param function: Multi-peak function
        :param params: Parameter for this function
        :return: list of (peak, relevance) tuples
        """
        y = function(self.x, *params)
        return fitfunctions.peak_search(y=y, fwhm=100, relevance_info=True)

    def testPeakSearch_various_functions(self):
        f_p = ((fitfunctions.sum_gauss, self.h_c_fwhm ),
               (fitfunctions.sum_lorentz, self.h_c_fwhm),
               (fitfunctions.sum_pvoigt, self.h_c_fwhm_eta),
               (fitfunctions.sum_splitgauss, self.h_c_fwhm_fwhm),
               (fitfunctions.sum_splitlorentz, self.h_c_fwhm_fwhm),
               (fitfunctions.sum_splitpvoigt, self.h_c_fwhm_fwhm_eta),
               (fitfunctions.sum_agauss, self.a_c_fwhm),
               (fitfunctions.sum_fastagauss, self.a_c_fwhm),
               (fitfunctions.sum_alorentz, self.a_c_fwhm),
               (fitfunctions.sum_apvoigt, self.a_c_fwhm_eta),
               (fitfunctions.sum_ahypermet, self.hypermet_params),
               (fitfunctions.sum_fastahypermet, self.hypermet_params),
              )
        for function, params in f_p:
            peaks = self.get_peaks(function, params)

            self.assertEqual(len(peaks), 6,
                             "Wrong number of peaks detected")

            for i in range(6):
                theoretical_peak_index = params[i*(len(params)//6) + 1]
                found_peak_index = peaks[i][0]
                self.assertLess(abs(found_peak_index - theoretical_peak_index), 25)


class Test_smooth(unittest.TestCase):
    """
    Unit tests of smoothing functions.

    Test that the difference between a synthetic curve with 5% added random
    noise and the result of smoothing that signal is less than 5%. We compare
    the sum of all samples in each curve.
    """
    def setUp(self):
        x = numpy.arange(5000)
        # (height1, center1, fwhm1, beamfwhm...)
        slit_params = (50, 500, 200, 100,
                        50, 600, 80, 30,
                        20, 2000, 150, 150,
                        50, 2250, 110, 100,
                        40, 3000, 50, 10,
                        23, 4980, 250, 20)

        self.y1 = fitfunctions.sum_slit(x, *slit_params)
        # 5% noise
        noise1 = 2 * numpy.random.random(5000) - 1
        noise1 *= 0.05
        self.y1 *= (1 + noise1)


        # (height1, center1, fwhm1...)
        step_params = (50, 500, 200,
                       50, 600, 80,
                       20, 2000, 150,
                       50, 2250, 110,
                       40, 3000, 50,
                       23, 4980, 250,)

        self.y2 = fitfunctions.sum_upstep(x, *step_params)
        # 5% noise
        noise2 = 2 * numpy.random.random(5000) - 1
        noise2 *= 0.05
        self.y2 *= (1 + noise2)

        self.y3 = fitfunctions.sum_downstep(x, *step_params)
        # 5% noise
        noise3 = 2 * numpy.random.random(5000) - 1
        noise3 *= 0.05
        self.y3 *= (1 + noise3)


    def tearDown(self):
        pass

    def testSavitskyGolay(self):
        npts = 25
        for y in [self.y1, self.y2, self.y3]:
            smoothed_y = fitfunctions.savitsky_golay(y, npoints=npts)

            # we added +-5% of random noise. The difference must be much lower
            # than 5%.
            diff = abs(sum(smoothed_y) - sum(y)) / sum(y)
            self.assertLess(diff, 0.05,
                            "Difference between data with 5%% noise and " +
                            "smoothed data is > 5%% (%f %%)" % (diff * 100))
            # Try various smoothing levels
            npts += 25


class Test_functions(unittest.TestCase):
    """
    Unit tests of peak_search on various types of multi-peak functions.
    """
    def setUp(self):
        self.x = numpy.arange(11)

        # height, center, sigma
        (h, c, s) = (7., 5., 3.)
        self.g_params = {
            "height": h,
            "center": c,
            "sigma": s,
            "fwhm": 2 * math.sqrt(2 * math.log(2)) * s,
            "area": h * s * math.sqrt(2 * math.pi)
        }
        # result of `7 * scipy.signal.gaussian(11, 3)`
        self.scipy_gaussian = numpy.array(
            [1.74546546, 2.87778603, 4.24571462, 5.60516182, 6.62171628,
             7., 6.62171628, 5.60516182, 4.24571462, 2.87778603,
             1.74546546]
        )

    def tearDown(self):
        pass

    def testGauss(self):
        """Compare sum_gauss with scipy.signals.gaussian"""
        y = fitfunctions.sum_gauss(self.x,
                                   self.g_params["height"],
                                   self.g_params["center"],
                                   self.g_params["fwhm"])

        for i in range(11):
            self.assertAlmostEqual(y[i], self.scipy_gaussian[i])

    def testAGauss(self):
        """Compare sum_agauss with scipy.signals.gaussian"""
        y = fitfunctions.sum_agauss(self.x,
                                    self.g_params["area"],
                                    self.g_params["center"],
                                    self.g_params["fwhm"])
        for i in range(11):
            self.assertAlmostEqual(y[i], self.scipy_gaussian[i])


test_cases = (Test_peak_search, Test_smooth, Test_functions)

def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
