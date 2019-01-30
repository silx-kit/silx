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
"""Basic tests for MaskToolsWidget"""

__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "17/01/2018"


import logging
import os.path
import unittest

import numpy

from silx.gui import qt
from silx.test.utils import temp_dir
from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import getQToolButtonFromAction
from silx.gui.plot import PlotWindow, ScatterMaskToolsWidget
from .utils import PlotWidgetTestCase

import fabio


_logger = logging.getLogger(__name__)


class TestScatterMaskToolsWidget(PlotWidgetTestCase, ParametricTestCase):
    """Basic test for MaskToolsWidget"""

    def _createPlot(self):
        return PlotWindow()

    def setUp(self):
        super(TestScatterMaskToolsWidget, self).setUp()
        self.widget = ScatterMaskToolsWidget.ScatterMaskToolsDockWidget(
                plot=self.plot, name='TEST')
        self.plot.addDockWidget(qt.Qt.BottomDockWidgetArea, self.widget)

        self.maskWidget = self.widget.widget()

    def tearDown(self):
        del self.maskWidget
        del self.widget
        super(TestScatterMaskToolsWidget, self).tearDown()

    def testEmptyPlot(self):
        """Empty plot, display MaskToolsDockWidget, toggle multiple masks"""
        self.maskWidget.setMultipleMasks('single')
        self.qapp.processEvents()

        self.maskWidget.setMultipleMasks('exclusive')
        self.qapp.processEvents()

    def _drag(self):
        """Drag from plot center to offset position"""
        plot = self.plot.getWidgetHandle()
        xCenter, yCenter = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        pos0 = xCenter, yCenter
        pos1 = xCenter + offset, yCenter + offset

        self.mouseMove(plot, pos=(0, 0))
        self.mouseMove(plot, pos=pos0)
        self.mouseClick(plot, qt.Qt.LeftButton, pos=pos0)
        self.mouseMove(plot, pos=(0, 0))
        self.mouseMove(plot, pos=pos1)
        self.mouseClick(plot, qt.Qt.LeftButton, pos=pos1)

    def _drawPolygon(self):
        """Draw a star polygon in the plot"""
        plot = self.plot.getWidgetHandle()
        x, y = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        star = [(x, y + offset),
                (x - offset, y - offset),
                (x + offset, y),
                (x - offset, y),
                (x + offset, y - offset),
                (x, y + offset)]  # Close polygon

        self.mouseMove(plot, pos=[0, 0])
        for pos in star:
            self.mouseMove(plot, pos=pos)
            self.qapp.processEvents()
            self.mouseClick(plot, qt.Qt.LeftButton, pos=pos)
            self.qapp.processEvents()

    def _drawPencil(self):
        """Draw a star polygon in the plot"""
        plot = self.plot.getWidgetHandle()
        x, y = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        star = [(x, y + offset),
                (x - offset, y - offset),
                (x + offset, y),
                (x - offset, y),
                (x + offset, y - offset)]

        self.mouseMove(plot, pos=[0, 0])
        self.mouseMove(plot, pos=star[0])
        self.mousePress(plot, qt.Qt.LeftButton, pos=star[0])
        for pos in star[1:]:
            self.mouseMove(plot, pos=pos)
        self.mouseRelease(
            plot, qt.Qt.LeftButton, pos=star[-1])

    def testWithAScatter(self):
        """Plot with a Scatter: test MaskToolsWidget interactions"""

        # Add and remove a scatter (this should enable/disable GUI + change mask)
        self.plot.addScatter(
                x=numpy.arange(256),
                y=numpy.arange(256),
                value=numpy.random.random(256),
                legend='test')
        self.plot._setActiveItem(kind="scatter", legend="test")
        self.qapp.processEvents()

        self.plot.remove('test', kind='scatter')
        self.qapp.processEvents()

        self.plot.addScatter(
                x=numpy.arange(1000),
                y=1000 * (numpy.arange(1000) % 20),
                value=numpy.random.random(1000),
                legend='test')
        self.plot._setActiveItem(kind="scatter", legend="test")
        self.plot.resetZoom()
        self.qapp.processEvents()

        # Test draw rectangle #
        toolButton = getQToolButtonFromAction(self.maskWidget.rectAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)

        # mask
        self.maskWidget.maskStateGroup.button(1).click()
        self.qapp.processEvents()
        self._drag()

        self.assertFalse(
            numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0)))

        # unmask same region
        self.maskWidget.maskStateGroup.button(0).click()
        self.qapp.processEvents()
        self._drag()
        self.assertTrue(
            numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0)))

        # Test draw polygon #
        toolButton = getQToolButtonFromAction(self.maskWidget.polygonAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)

        # mask
        self.maskWidget.maskStateGroup.button(1).click()
        self.qapp.processEvents()
        self._drawPolygon()
        self.assertFalse(
            numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0)))

        # unmask same region
        self.maskWidget.maskStateGroup.button(0).click()
        self.qapp.processEvents()
        self._drawPolygon()
        self.assertTrue(
            numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0)))

        # Test draw pencil #
        toolButton = getQToolButtonFromAction(self.maskWidget.pencilAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)

        self.maskWidget.pencilSpinBox.setValue(30)
        self.qapp.processEvents()

        # mask
        self.maskWidget.maskStateGroup.button(1).click()
        self.qapp.processEvents()
        self._drawPencil()
        self.assertFalse(
            numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0)))

        # unmask same region
        self.maskWidget.maskStateGroup.button(0).click()
        self.qapp.processEvents()
        self._drawPencil()
        self.assertTrue(
            numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0)))

        # Test no draw tool #
        toolButton = getQToolButtonFromAction(self.maskWidget.browseAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)

        self.plot.clear()

    def __loadSave(self, file_format):
        self.plot.addScatter(
                x=numpy.arange(256),
                y=25 * (numpy.arange(256) % 10),
                value=numpy.random.random(256),
                legend='test')
        self.plot._setActiveItem(kind="scatter", legend="test")
        self.plot.resetZoom()
        self.qapp.processEvents()

        # Draw a polygon mask
        toolButton = getQToolButtonFromAction(self.maskWidget.polygonAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)
        self._drawPolygon()

        ref_mask = self.maskWidget.getSelectionMask()
        self.assertFalse(numpy.all(numpy.equal(ref_mask, 0)))

        with temp_dir() as tmp:
            mask_filename = os.path.join(tmp, 'mask.' + file_format)
            self.maskWidget.save(mask_filename, file_format)

            self.maskWidget.resetSelectionMask()
            self.assertTrue(
                numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0)))

            self.maskWidget.load(mask_filename)
            self.assertTrue(numpy.all(numpy.equal(
                self.maskWidget.getSelectionMask(), ref_mask)))

    def testLoadSaveNpy(self):
        self.__loadSave("npy")

    def testLoadSaveCsv(self):
        self.__loadSave("csv")

    def testSigMaskChangedEmitted(self):
        self.qapp.processEvents()
        self.plot.addScatter(
                x=numpy.arange(1000),
                y=1000 * (numpy.arange(1000) % 20),
                value=numpy.ones((1000,)),
                legend='test')
        self.plot._setActiveItem(kind="scatter", legend="test")
        self.plot.resetZoom()
        self.qapp.processEvents()

        self.plot.remove('test', kind='scatter')
        self.qapp.processEvents()

        self.plot.addScatter(
                x=numpy.arange(1000),
                y=1000 * (numpy.arange(1000) % 20),
                value=numpy.random.random(1000),
                legend='test')

        l = []

        def slot():
            l.append(1)

        self.maskWidget.sigMaskChanged.connect(slot)

        # rectangle mask
        toolButton = getQToolButtonFromAction(self.maskWidget.rectAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)
        self.maskWidget.maskStateGroup.button(1).click()
        self.qapp.processEvents()
        self._drag()

        self.assertGreater(len(l), 0)


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestScatterMaskToolsWidget,):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
