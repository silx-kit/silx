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
"""Basic tests for MaskToolsWidget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "21/06/2016"


import logging
import os.path
import unittest

import numpy

from silx.gui import qt
from silx.testutils import temp_dir
from silx.gui.testutils import TestCaseQt, getQToolButtonFromAction
from silx.gui.plot import PlotWindow, MaskToolsWidget


logging.basicConfig()
_logger = logging.getLogger(__name__)


class TestMaskToolsWidget(TestCaseQt):
    """Basic test for MaskToolsWidget"""

    def setUp(self):
        super(TestMaskToolsWidget, self).setUp()
        self.plot = PlotWindow()

        self.widget = MaskToolsWidget.MaskToolsDockWidget(plot=self.plot, name='TEST')
        self.plot.addDockWidget(qt.Qt.BottomDockWidgetArea, self.widget)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

        self.maskWidget = self.widget.widget()

    def tearDown(self):
        del self.maskWidget
        del self.widget

        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot

        super(TestMaskToolsWidget, self).tearDown()

    def testEmptyPlot(self):
        """Empty plot, display MaskToolsDockWidget, toggle multiple masks"""
        self.maskWidget.setMultipleMasks('single')
        self.qapp.processEvents()

        self.maskWidget.setMultipleMasks('exclusive')
        self.qapp.processEvents()

    def _drag(self):
        """Drag from plot center to offset position"""
        plot = self.plot.centralWidget()
        xCenter, yCenter = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        pos0 = xCenter, yCenter
        pos1 = xCenter + offset, yCenter + offset

        self.mouseMove(plot, pos=pos0)
        self.mousePress(plot, qt.Qt.LeftButton, pos=pos0)
        self.mouseMove(plot, pos=pos1)
        self.mouseRelease(plot, qt.Qt.LeftButton, pos=pos1)

    def _drawPolygon(self):
        """Draw a star polygon in the plot"""
        plot = self.plot.centralWidget()
        x, y = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        star = [(x, y + offset),
                (x - offset, y - offset),
                (x + offset, y),
                (x - offset, y),
                (x + offset, y - offset)]

        for pos in star:
            self.mouseMove(plot, pos=pos)
            btn = qt.Qt.LeftButton if pos != star[-1] else qt.Qt.RightButton
            self.mouseClick(plot, btn, pos=pos)

    def _drawPencil(self):
        """Draw a star polygon in the plot"""
        plot = self.plot.centralWidget()
        x, y = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        star = [(x, y + offset),
                (x - offset, y - offset),
                (x + offset, y),
                (x - offset, y),
                (x + offset, y - offset)]

        self.mouseMove(plot, pos=star[0])
        self.mousePress(plot, qt.Qt.LeftButton, pos=star[0])
        for pos in star:
            self.mouseMove(plot, pos=pos)
        self.mouseRelease(
            plot, qt.Qt.LeftButton, pos=star[-1])

    def testWithAnImage(self):
        """Plot with an image: test MaskToolsWidget interactions"""

        # Add and remove a image (this should enable/disable GUI + change mask)
        self.plot.addImage(numpy.random.random(1024**2).reshape(1024, 1024),
                           legend='test')
        self.qapp.processEvents()

        self.plot.remove('test', kind='image')
        self.qapp.processEvents()

        self.plot.addImage(numpy.arange(1024**2).reshape(1024, 1024),
                           legend='test')
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

        self.maskWidget.pencilSpinBox.setValue(10)
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

    def testLoadSave(self):
        """Plot with an image: test MaskToolsWidget operations"""
        self.plot.addImage(numpy.arange(1024**2).reshape(1024, 1024),
                           legend='test')
        self.qapp.processEvents()

        # Draw a polygon mask
        toolButton = getQToolButtonFromAction(self.maskWidget.polygonAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)
        self._drawPolygon()

        ref_mask = self.maskWidget.getSelectionMask()
        self.assertFalse(numpy.all(numpy.equal(ref_mask, 0)))

        with temp_dir() as tmp:
            success = self.maskWidget.save(
                os.path.join(tmp, 'mask.npy'), 'npy')
            self.assertTrue(success)

            self.maskWidget.resetSelectionMask()
            self.assertTrue(
                numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0)))

            result = self.maskWidget.load(os.path.join(tmp, 'mask.npy'))
            self.assertTrue(result)
            self.assertTrue(numpy.all(numpy.equal(
                self.maskWidget.getSelectionMask(), ref_mask)))


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestMaskToolsWidget,):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
