# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""Basic tests for PlotTools"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import numpy
import unittest

from silx.utils.testutils import ParametricTestCase, TestLogging
from silx.gui.test.utils import (
    qWaitForWindowExposedAndActivate, TestCaseQt, getQToolButtonFromAction)
from silx.gui import qt
from silx.gui.plot import Plot2D, PlotWindow, PlotTools
from .utils import PlotWidgetTestCase


# Makes sure a QApplication exists
_qapp = qt.QApplication.instance() or qt.QApplication([])


def _tearDownDocTest(docTest):
    """Tear down to use for test from docstring.

    Checks that plot widget is displayed
    """
    plot = docTest.globs['plot']
    qWaitForWindowExposedAndActivate(plot)
    plot.setAttribute(qt.Qt.WA_DeleteOnClose)
    plot.close()
    del plot

# Disable doctest because of
# "NameError: name 'numpy' is not defined"
#
# import doctest
# positionInfoTestSuite = doctest.DocTestSuite(
#     PlotTools, tearDown=_tearDownDocTest,
#     optionflags=doctest.ELLIPSIS)
# """Test suite of tests from PlotTools docstrings.
#
# Test PositionInfo and ProfileToolBar docstrings.
# """


class TestPositionInfo(PlotWidgetTestCase):
    """Tests for PositionInfo widget."""

    def _createPlot(self):
        return PlotWindow()

    def setUp(self):
        super(TestPositionInfo, self).setUp()
        self.mouseMove(self.plot, pos=(0, 0))
        self.qapp.processEvents()
        self.qWait(100)

    def tearDown(self):
        super(TestPositionInfo, self).tearDown()

    def _test(self, positionWidget, converterNames, **kwargs):
        """General test of PositionInfo.

        - Add it to a toolbar and
        - Move mouse around the center of the PlotWindow.
        """
        toolBar = qt.QToolBar()
        self.plot.addToolBar(qt.Qt.BottomToolBarArea, toolBar)

        toolBar.addWidget(positionWidget)

        converters = positionWidget.getConverters()
        self.assertEqual(len(converters), len(converterNames))
        for index, name in enumerate(converterNames):
            self.assertEqual(converters[index][0], name)

        with TestLogging(PlotTools.__name__, **kwargs):
            # Move mouse to center
            center = self.plot.size() / 2
            self.mouseMove(self.plot, pos=(center.width(), center.height()))
            # Move out
            self.mouseMove(self.plot, pos=(1, 1))

    def testDefaultConverters(self):
        """Test PositionInfo with default converters"""
        positionWidget = PlotTools.PositionInfo(plot=self.plot)
        self._test(positionWidget, ('X', 'Y'))

    def testCustomConverters(self):
        """Test PositionInfo with custom converters"""
        converters = [
            ('Coords', lambda x, y: (int(x), int(y))),
            ('Radius', lambda x, y: numpy.sqrt(x * x + y * y)),
            ('Angle', lambda x, y: numpy.degrees(numpy.arctan2(y, x)))
        ]
        positionWidget = PlotTools.PositionInfo(plot=self.plot,
                                                converters=converters)
        self._test(positionWidget, ('Coords', 'Radius', 'Angle'))

    def testFailingConverters(self):
        """Test PositionInfo with failing custom converters"""
        def raiseException(x, y):
            raise RuntimeError()

        positionWidget = PlotTools.PositionInfo(
            plot=self.plot,
            converters=[('Exception', raiseException)])
        self._test(positionWidget, ['Exception'], error=2)


class TestPixelIntensitiesHisto(TestCaseQt, ParametricTestCase):
    """Tests for ProfileToolBar widget."""

    def setUp(self):
        super(TestPixelIntensitiesHisto, self).setUp()
        self.image = numpy.random.rand(100, 100)
        self.plotImage = Plot2D()
        self.plotImage.getIntensityHistogramAction().setVisible(True)

    def tearDown(self):
        del self.plotImage
        super(TestPixelIntensitiesHisto, self).tearDown()

    def testShowAndHide(self):
        """Simple test that the plot is showing and hiding when activating the
        action"""
        self.plotImage.addImage(self.image, origin=(0, 0), legend='sino')
        self.plotImage.show()

        histoAction = self.plotImage.getIntensityHistogramAction()

        # test the pixel intensity diagram is showing
        button = getQToolButtonFromAction(histoAction)
        self.assertIsNot(button, None)
        self.mouseMove(button)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qapp.processEvents()
        self.assertTrue(histoAction.getHistogramPlotWidget().isVisible())

        # test the pixel intensity diagram is hiding
        self.qapp.setActiveWindow(self.plotImage)
        self.qapp.processEvents()
        self.mouseMove(button)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qapp.processEvents()
        self.assertFalse(histoAction.getHistogramPlotWidget().isVisible())

    def testImageFormatInput(self):
        """Test multiple type as image input"""
        typesToTest = [numpy.uint8, numpy.int8, numpy.int16, numpy.int32,
                       numpy.float32, numpy.float64]
        self.plotImage.addImage(self.image, origin=(0, 0), legend='sino')
        self.plotImage.show()
        button = getQToolButtonFromAction(
            self.plotImage.getIntensityHistogramAction())
        self.mouseMove(button)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qapp.processEvents()
        for typeToTest in typesToTest:
            with self.subTest(typeToTest=typeToTest):
                self.plotImage.addImage(self.image.astype(typeToTest),
                                        origin=(0, 0), legend='sino')


def suite():
    test_suite = unittest.TestSuite()
    # test_suite.addTest(positionInfoTestSuite)
    for testClass in (TestPositionInfo, TestPixelIntensitiesHisto):
        test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
            testClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
