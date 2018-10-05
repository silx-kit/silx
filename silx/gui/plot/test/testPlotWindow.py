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
"""Basic tests for PlotWindow"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "27/06/2017"


import doctest
import unittest

from silx.gui.utils.testutils import TestCaseQt, getQToolButtonFromAction

from silx.gui import qt
from silx.gui.plot import PlotWindow


# Test of the docstrings #

# Makes sure a QApplication exists
_qapp = qt.QApplication.instance() or qt.QApplication([])


def _tearDownQt(docTest):
    """Tear down to use for test from docstring.

    Checks that plt widget is displayed
    """
    _qapp.processEvents()
    for obj in docTest.globs.values():
        if isinstance(obj, PlotWindow):
            # Commented out as it takes too long
            # qWaitForWindowExposedAndActivate(obj)
            obj.setAttribute(qt.Qt.WA_DeleteOnClose)
            obj.close()
            del obj


plotWindowDocTestSuite = doctest.DocTestSuite('silx.gui.plot.PlotWindow',
                                              tearDown=_tearDownQt)
"""Test suite of tests from the module's docstrings."""


class TestPlotWindow(TestCaseQt):
    """Base class for tests of PlotWindow."""

    def setUp(self):
        super(TestPlotWindow, self).setUp()
        self.plot = PlotWindow()
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestPlotWindow, self).tearDown()

    def testActions(self):
        """Test the actions QToolButtons"""
        self.plot.setLimits(1, 100, 1, 100)

        checkList = [  # QAction, Plot state getter
            (self.plot.xAxisAutoScaleAction, self.plot.getXAxis().isAutoScale),
            (self.plot.yAxisAutoScaleAction, self.plot.getYAxis().isAutoScale),
            (self.plot.xAxisLogarithmicAction, self.plot.getXAxis()._isLogarithmic),
            (self.plot.yAxisLogarithmicAction, self.plot.getYAxis()._isLogarithmic),
            (self.plot.gridAction, self.plot.getGraphGrid),
        ]

        for action, getter in checkList:
            self.mouseMove(self.plot)
            initialState = getter()
            toolButton = getQToolButtonFromAction(action)
            self.assertIsNot(toolButton, None)
            self.mouseClick(toolButton, qt.Qt.LeftButton)
            self.assertNotEqual(getter(), initialState,
                                msg='"%s" state not changed' % action.text())

            self.mouseClick(toolButton, qt.Qt.LeftButton)
            self.assertEqual(getter(), initialState,
                             msg='"%s" state not changed' % action.text())

        # Trigger a zoom reset
        self.mouseMove(self.plot)
        resetZoomAction = self.plot.resetZoomAction
        toolButton = getQToolButtonFromAction(resetZoomAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)

    def testToolAspectRatio(self):
        self.plot.toolBar()
        self.plot.keepDataAspectRatioButton.keepDataAspectRatio()
        self.assertTrue(self.plot.isKeepDataAspectRatio())
        self.plot.keepDataAspectRatioButton.dontKeepDataAspectRatio()
        self.assertFalse(self.plot.isKeepDataAspectRatio())

    def testToolYAxisOrigin(self):
        self.plot.toolBar()
        self.plot.yAxisInvertedButton.setYAxisUpward()
        self.assertFalse(self.plot.getYAxis().isInverted())
        self.plot.yAxisInvertedButton.setYAxisDownward()
        self.assertTrue(self.plot.getYAxis().isInverted())


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(plotWindowDocTestSuite)
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotWindow))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
