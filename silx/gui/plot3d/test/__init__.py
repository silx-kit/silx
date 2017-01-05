# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""plot3d test suite."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import logging
import unittest


logger = logging.getLogger(__name__)


# from ..glutils.test import suite as test_glutils_suite
from ..scene.test import suite as test_scene_suite
from ..utils.test import suite as test_utils_suite


def suite():
    test_suite = unittest.TestSuite()
    # test_suite.addTest(test_glutils_suite())
    test_suite.addTest(test_scene_suite())
    test_suite.addTest(test_utils_suite())
    return test_suite


def run_tests():
    """Run test complete test_suite"""
    runner = unittest.TextTestRunner()
    if not runner.run(suite()).wasSuccessful():
        print("Test suite failed")
        return 1
    else:
        print("Test suite succeeded")
        return 0
