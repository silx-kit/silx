# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "04/07/2016"

import unittest

from .test_histogramnd_error import suite as test_histo_error
from .test_histogramnd_nominal import suite as test_histo_nominal
from .test_histogramnd_vs_np import suite as test_histo_vs_np
from .test_HistogramndLut_nominal import suite as test_histolut_nominal
from ..fit.test import suite as test_fit_suite
from .test_marchingcubes import suite as test_marchingcubes_suite
from ..medianfilter.test import suite as test_medianfilter_suite
from .test_combo import suite as test_combo_suite
from .test_calibration import suite as test_calibration_suite
from .test_colormap import suite as test_colormap_suite

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_histo_nominal())
    test_suite.addTest(test_histo_error())
    test_suite.addTest(test_histo_vs_np())
    test_suite.addTest(test_fit_suite())
    test_suite.addTest(test_histolut_nominal())
    test_suite.addTest(test_marchingcubes_suite())
    test_suite.addTest(test_medianfilter_suite())
    test_suite.addTest(test_combo_suite())
    test_suite.addTest(test_calibration_suite())
    test_suite.addTest(test_colormap_suite())
    return test_suite
