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
__date__ = "09/11/2017"


import logging
import os
import unittest
from silx.test.utils import test_options


_logger = logging.getLogger(__name__)


def suite():
    test_suite = unittest.TestSuite()

    if not test_options.WITH_GL_TEST:
        # Explicitly disabled tests
        msg = "silx.gui.plot3d tests disabled: %s" % test_options.WITH_GL_TEST_REASON
        _logger.warning(msg)

        class SkipPlot3DTest(unittest.TestCase):
            def runTest(self):
                self.skipTest(test_options.WITH_GL_TEST_REASON)

        test_suite.addTest(SkipPlot3DTest())
        return test_suite

    # Import here to avoid loading modules if tests are disabled

    from ..scene import test as test_scene
    from .testGL import suite as testGLSuite
    from .testScalarFieldView import suite as testScalarFieldViewSuite

    test_suite = unittest.TestSuite()
    test_suite.addTest(testGLSuite())
    test_suite.addTest(test_scene.suite())
    test_suite.addTest(testScalarFieldViewSuite())
    return test_suite
