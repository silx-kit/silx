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
"""Basic tests for CurvesROIWidget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/05/2016"


import logging
import os.path
import unittest

import numpy

from silx.gui import qt
from silx.testutils import temp_dir
from silx.gui.testutils import TestCaseQt
from silx.gui.plot import PlotWindow, CurvesROIWidget


logging.basicConfig()
_logger = logging.getLogger(__name__)


class TestCurvesROIWidget(TestCaseQt):
    """Basic test for CurvesROIWidget"""

    def setUp(self):
        super(TestCurvesROIWidget, self).setUp()
        self.plot = PlotWindow()
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

        self.widget = CurvesROIWidget.CurvesROIDockWidget(plot=self.plot, name='TEST')
        self.widget.show()
        self.qWaitForWindowExposed(self.widget)

    def tearDown(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot

        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.widget.close()
        del self.widget

        super(TestCurvesROIWidget, self).tearDown()

    def testEmptyPlot(self):
        """Empty plot, display ROI widget"""
        pass

    def testWithCurves(self):
        """Plot with curves: test all ROI widget buttons"""
        for offset in range(2):
            self.plot.addCurve(numpy.arange(1000),
                               offset + numpy.random.random(1000),
                               legend=str(offset))

        # Add two ROI
        self.mouseClick(self.widget.roiWidget.addButton, qt.Qt.LeftButton)
        self.mouseClick(self.widget.roiWidget.addButton, qt.Qt.LeftButton)

        # Change active curve
        self.plot.setActiveCurve(str(1))

        # Delete a ROI
        self.mouseClick(self.widget.roiWidget.delButton, qt.Qt.LeftButton)

        with temp_dir() as tmpDir:
            self.tmpFile = os.path.join(tmpDir, 'test.ini')

            # Save ROIs
            self.widget.roiWidget.save(self.tmpFile)
            self.assertTrue(os.path.isfile(self.tmpFile))

            # Reset ROIs
            self.mouseClick(self.widget.roiWidget.resetButton,
                            qt.Qt.LeftButton)

            # Load ROIs
            self.widget.roiWidget.load(self.tmpFile)

            del self.tmpFile


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestCurvesROIWidget,):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
