#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/kif/sift_pyocl
#

"""
Test suite for all sift_pyocl
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-05-28"
__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

import time,os
import sys
import unittest
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)

from test_gaussian import test_suite_gaussian
from test_preproc import test_suite_preproc
from test_reductions import test_suite_reductions
from test_convol import test_suite_convol
from test_algebra import test_suite_algebra
from test_image import test_suite_image
from test_keypoints import test_suite_keypoints
from test_matching import test_suite_matching

def test_suite_all():
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

if __name__ == '__main__':
    mysuite = test_suite_all()
    runner = unittest.TextTestRunner()
    if not runner.run(mysuite).wasSuccessful():
        sys.exit(1)

