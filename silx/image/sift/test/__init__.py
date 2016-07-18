# -*- coding: utf-8 -*-
#
#    Project: silx
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
__date__ = "18/07/2016"

import unittest
from .test_gaussian import suite as  test_suite_gaussian
from .test_preproc import suite as  test_suite_preproc
from .test_reductions import suite as  test_suite_reductions
from .test_convol import suite as  test_suite_convol
from .test_algebra import suite as  test_suite_algebra
from .test_image import suite as  test_suite_image
from .test_keypoints import suite as  test_suite_keypoints
from .test_matching import suite as  test_suite_matching


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_suite_algebra())
    testSuite.addTest(test_suite_gaussian())
    testSuite.addTest(test_suite_preproc())
    testSuite.addTest(test_suite_reductions())
    testSuite.addTest(test_suite_convol())
    testSuite.addTest(test_suite_image())
    testSuite.addTest(test_suite_keypoints())
    testSuite.addTest(test_suite_matching())
    return testSuite
