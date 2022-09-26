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
Tests for peaks module
"""

import unittest
import numpy
import math

from silx.math.fit import functions
from silx.math.fit import peaks

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
        self.hypermet_params = (1000, 500, 200, 0.2, 100, 0.3, 100, 0.05,
                                1000, 1000, 200, 0.2, 100, 0.3, 100, 0.05,
                                1000, 2000, 200, 0.2, 100, 0.3, 100, 0.05,
                                1000, 2350, 200, 0.2, 100, 0.3, 100, 0.05,
                                1000, 3000, 200, 0.2, 100, 0.3, 100, 0.05,
                                1000, 4900, 200, 0.2, 100, 0.3, 100, 0.05,)


    def tearDown(self):
        pass

    def get_peaks(self, function, params):
        """

        :param function: Multi-peak function
        :param params: Parameter for this function
        :return: list of (peak, relevance) tuples
        """
        y = function(self.x, *params)
        return peaks.peak_search(y=y, fwhm=100, relevance_info=True)

    def testPeakSearch_various_functions(self):
        """Run peak search on a variety of synthetic functions, and
        check that result falls within +-25 samples of the actual peak
        (reasonable delta considering a fwhm of ~100 samples) and effects
        of overlapping peaks)."""
        f_p = ((functions.sum_gauss, self.h_c_fwhm ),
               (functions.sum_lorentz, self.h_c_fwhm),
               (functions.sum_pvoigt, self.h_c_fwhm_eta),
               (functions.sum_splitgauss, self.h_c_fwhm_fwhm),
               (functions.sum_splitlorentz, self.h_c_fwhm_fwhm),
               (functions.sum_splitpvoigt, self.h_c_fwhm_fwhm_eta),
               (functions.sum_agauss, self.a_c_fwhm),
               (functions.sum_fastagauss, self.a_c_fwhm),
               (functions.sum_alorentz, self.a_c_fwhm),
               (functions.sum_apvoigt, self.a_c_fwhm_eta),
               (functions.sum_ahypermet, self.hypermet_params),
               (functions.sum_fastahypermet, self.hypermet_params),)

        for function, params in f_p:
            peaks = self.get_peaks(function, params)

            self.assertEqual(len(peaks), 6,
                             "Wrong number of peaks detected")

            for i in range(6):
                theoretical_peak_index = params[i*(len(params)//6) + 1]
                found_peak_index = peaks[i][0]
                self.assertLess(abs(found_peak_index - theoretical_peak_index), 25)
