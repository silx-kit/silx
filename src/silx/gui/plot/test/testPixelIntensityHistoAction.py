# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""Basic tests for PixelIntensitiesHistoAction"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/03/2018"


import numpy
import unittest

from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt, getQToolButtonFromAction
from silx.gui import qt
from silx.gui.plot import Plot2D


class TestPixelIntensitiesHisto(TestCaseQt, ParametricTestCase):
    """Tests for PixelIntensitiesHistoAction widget."""

    def setUp(self):
        super(TestPixelIntensitiesHisto, self).setUp()
        self.image = numpy.random.rand(10, 10)
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
        self.assertTrue(histoAction.getHistogramWidget().isVisible())

        # test the pixel intensity diagram is hiding
        self.qapp.setActiveWindow(self.plotImage)
        self.qapp.processEvents()
        self.mouseMove(button)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qapp.processEvents()
        self.assertFalse(histoAction.getHistogramWidget().isVisible())

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

    def testScatter(self):
        """Test that an histogram from a scatter is displayed"""
        xx = numpy.arange(10)
        yy = numpy.arange(10)
        value = numpy.sin(xx)
        self.plotImage.addScatter(xx, yy, value)
        self.plotImage.show()

        histoAction = self.plotImage.getIntensityHistogramAction()

        # test the pixel intensity diagram is showing
        button = getQToolButtonFromAction(histoAction)
        self.assertIsNot(button, None)
        self.mouseMove(button)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qapp.processEvents()

        widget = histoAction.getHistogramWidget()
        self.assertTrue(widget.isVisible())
        items = widget.getPlotWidget().getItems()
        self.assertEqual(len(items), 1)

    def testChangeItem(self):
        """Test that histogram changes it the item changes"""
        xx = numpy.arange(10)
        yy = numpy.arange(10)
        value = numpy.sin(xx)
        self.plotImage.addScatter(xx, yy, value)
        self.plotImage.show()

        histoAction = self.plotImage.getIntensityHistogramAction()

        # test the pixel intensity diagram is showing
        button = getQToolButtonFromAction(histoAction)
        self.assertIsNot(button, None)
        self.mouseMove(button)
        self.mouseClick(button, qt.Qt.LeftButton)
        self.qapp.processEvents()

        # Reach histogram from the first item
        widget = histoAction.getHistogramWidget()
        self.assertTrue(widget.isVisible())
        items = widget.getPlotWidget().getItems()
        data1 = items[0].getValueData(copy=False)

        # Set another item to the plot
        self.plotImage.addImage(self.image, origin=(0, 0), legend='sino')
        self.qapp.processEvents()
        data2 = items[0].getValueData(copy=False)

        # Histogram is not the same
        self.assertFalse(numpy.array_equal(data1, data2))
