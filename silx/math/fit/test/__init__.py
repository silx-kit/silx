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
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "22/06/2016"

import unittest

from .test_fit import suite as test_curve_fit
from .test_functions import suite as test_fitfuns
from .test_filters import suite as test_fitfilters
from .test_peaks import suite as test_peaks
from .test_fitmanager import suite as test_fitmanager
from .test_bgtheories import suite as test_bgtheories


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_curve_fit())
    test_suite.addTest(test_fitfuns())
    test_suite.addTest(test_fitfilters())
    test_suite.addTest(test_peaks())
    test_suite.addTest(test_fitmanager())
    test_suite.addTest(test_bgtheories())
    return test_suite
