# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
__date__ = "24/04/2018"


import logging
import os
import sys
import unittest

from silx.test.utils import test_options

_logger = logging.getLogger(__name__)


def suite():

    test_suite = unittest.TestSuite()

    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
        # On Linux and no DISPLAY available (e.g., ssh without -X)
        _logger.warning('silx.gui tests disabled (DISPLAY env. variable not set)')

        class SkipGUITest(unittest.TestCase):
            def runTest(self):
                self.skipTest(
                    'silx.gui tests disabled (DISPLAY env. variable not set)')

        test_suite.addTest(SkipGUITest())
        return test_suite

    elif not test_options.WITH_QT_TEST:
        # Explicitly disabled tests
        msg = "silx.gui tests disabled: %s" % test_options.WITH_QT_TEST_REASON
        _logger.warning(msg)

        class SkipGUITest(unittest.TestCase):
            def runTest(self):
                self.skipTest(test_options.WITH_QT_TEST_REASON)

        test_suite.addTest(SkipGUITest())
        return test_suite

    # Import here to avoid loading QT if tests are disabled

    from ..plot import test as test_plot
    from ..fit import test as test_fit
    from ..hdf5 import test as test_hdf5
    from ..widgets import test as test_widgets
    from ..data import test as test_data
    from ..dialog import test as test_dialog
    from ..utils import test as test_utils

    from . import test_qt
    # Console tests disabled due to corruption of python environment
    # (see issue #538 on github)
    # from . import test_console
    from . import test_icons
    from . import test_colors

    try:
        from ..plot3d.test import suite as test_plot3d_suite

    except ImportError:
        _logger.warning(
            'silx.gui.plot3d tests disabled '
            '(PyOpenGL or QtOpenGL not installed)')

        class SkipPlot3DTest(unittest.TestCase):
            def runTest(self):
                self.skipTest('silx.gui.plot3d tests disabled '
                              '(PyOpenGL or QtOpenGL not installed)')

        test_plot3d_suite = SkipPlot3DTest

    test_suite.addTest(test_qt.suite())
    test_suite.addTest(test_plot.suite())
    test_suite.addTest(test_fit.suite())
    test_suite.addTest(test_hdf5.suite())
    test_suite.addTest(test_widgets.suite())
    # test_suite.addTest(test_console.suite())   # see issue #538 on github
    test_suite.addTest(test_icons.suite())
    test_suite.addTest(test_colors.suite())
    test_suite.addTest(test_data.suite())
    test_suite.addTest(test_plot3d_suite())
    test_suite.addTest(test_dialog.suite())
    # Run test_utils last: it interferes with OpenGLWidget through isOpenGLAvailable
    test_suite.addTest(test_utils.suite())
    return test_suite
