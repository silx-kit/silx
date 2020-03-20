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
"""Basic tests for PlotWindow"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "27/06/2017"


import unittest
import numpy

from silx.gui.utils.testutils import TestCaseQt, getQToolButtonFromAction

from silx.gui import qt
from silx.gui.plot import PlotWindow
from silx.gui.colors import Colormap


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

    def testDockWidgets(self):
        """Test add/remove dock widgets"""
        dock1 = qt.QDockWidget('Test 1')
        dock1.setWidget(qt.QLabel('Test 1'))

        self.plot.addTabbedDockWidget(dock1)
        self.qapp.processEvents()

        self.plot.removeDockWidget(dock1)
        self.qapp.processEvents()

        dock2 = qt.QDockWidget('Test 2')
        dock2.setWidget(qt.QLabel('Test 2'))

        self.plot.addTabbedDockWidget(dock2)
        self.qapp.processEvents()

        if qt.BINDING != 'PySide2':
            # Weird bug with PySide2 later upon gc.collect() when getting the layout
            self.assertNotEqual(self.plot.layout().indexOf(dock2),
                                -1,
                                "dock2 not properly displayed")

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

    def testColormapAutoscaleCache(self):
        # Test that the min/max cache is not computed twice

        old = Colormap._computeAutoscaleRange
        self._count = 0
        def _computeAutoscaleRange(colormap, data):
            self._count = self._count + 1
            return 10, 20
        Colormap._computeAutoscaleRange = _computeAutoscaleRange
        try:
            colormap = Colormap(name='red')
            self.plot.setVisible(True)

            # Add an image
            data = numpy.arange(8**2).reshape(8, 8)
            self.plot.addImage(data, legend="foo", colormap=colormap)
            self.plot.setActiveImage("foo")

            # Use the colorbar
            self.plot.getColorBarWidget().setVisible(True)
            self.qWait(50)

            # Remove and add again the same item
            image = self.plot.getImage("foo")
            self.plot.removeImage("foo")
            self.plot.addItem(image)
            self.qWait(50)
        finally:
            Colormap._computeAutoscaleRange = old
        self.assertEqual(self._count, 1)
        del self._count

def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotWindow))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
