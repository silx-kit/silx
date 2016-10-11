# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "11/10/2016"


import logging
import os
import sys
import unittest


_logger = logging.getLogger(__name__)


if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
    # On linux and no DISPLAY available (e.g., ssh without -X)
    _logger.warning('silx.gui tests disabled (DISPLAY env. variable not set)')

    class SkipGUITest(unittest.TestCase):
        def runTest(self):
            self.skipTest(
                'silx.gui tests disabled (DISPLAY env. variable not set)')

    def suite():
        suite = unittest.TestSuite()
        suite.addTest(SkipGUITest())
        return suite

elif os.environ.get('WITH_QT_TEST', 'True') == 'False':
    # Explicitly disabled tests
    _logger.warning(
        "silx.gui tests disabled (env. variable WITH_QT_TEST=False)")

    class SkipGUITest(unittest.TestCase):
        def runTest(self):
            self.skipTest(
                "silx.gui tests disabled (env. variable WITH_QT_TEST=False)")

    def suite():
        suite = unittest.TestSuite()
        suite.addTest(SkipGUITest())
        return suite

else:
    # Import here to avoid loading QT if tests are disabled

    from ..plot.test import suite as test_plot_suite
    from ..fit.test import suite as test_fit_suite
    from ..hdf5.test import suite as test_hdf5_suite
    from ..widgets.test import suite as test_widgets_suite
    from .test_qt import suite as test_qt_suite
    from .test_console import suite as test_console_suite
    from .test_icons import suite as test_icons_suite

    def suite():
        test_suite = unittest.TestSuite()
        test_suite.addTest(test_qt_suite())
        test_suite.addTest(test_plot_suite())
        test_suite.addTest(test_fit_suite())
        test_suite.addTest(test_hdf5_suite())
        test_suite.addTest(test_widgets_suite())
        test_suite.addTest(test_console_suite())
        test_suite.addTest(test_icons_suite())
        return test_suite
