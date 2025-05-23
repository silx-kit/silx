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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import logging
import os.path

import numpy

from silx.gui import qt
from silx.test.utils import temp_dir
from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import getQToolButtonFromAction
from silx.gui.plot import PlotWindow, MaskToolsWidget
from .utils import PlotWidgetTestCase


_logger = logging.getLogger(__name__)


class TestMaskToolsWidget(PlotWidgetTestCase, ParametricTestCase):
    """Basic test for MaskToolsWidget"""

    def _createPlot(self):
        return PlotWindow()

    def setUp(self):
        super().setUp()
        self.widget = MaskToolsWidget.MaskToolsDockWidget(plot=self.plot, name="TEST")
        self.plot.addDockWidget(qt.Qt.BottomDockWidgetArea, self.widget)
        self.maskWidget = self.widget.widget()

    def tearDown(self):
        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.widget.close()
        del self.maskWidget
        del self.widget
        super().tearDown()

    def testEmptyPlot(self):
        """Empty plot, display MaskToolsDockWidget, toggle multiple masks"""
        self.maskWidget.setMultipleMasks("single")
        self.qapp.processEvents()

        self.maskWidget.setMultipleMasks("exclusive")
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
        self.qapp.processEvents()
        self.mousePress(plot, qt.Qt.LeftButton, pos=pos0)
        self.qapp.processEvents()
        self.mouseMove(plot, pos=(pos0[0] + offset // 2, pos0[1] + offset // 2))
        self.mouseMove(plot, pos=pos1)
        self.qapp.processEvents()
        self.mouseRelease(plot, qt.Qt.LeftButton, pos=pos1)
        self.qapp.processEvents()
        self.mouseMove(plot, pos=(0, 0))

    def _drawPolygon(self):
        """Draw a star polygon in the plot"""
        plot = self.plot.getWidgetHandle()
        x, y = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        star = [
            (x, y + offset),
            (x - offset, y - offset),
            (x + offset, y),
            (x - offset, y),
            (x + offset, y - offset),
            (x, y + offset),
        ]  # Close polygon

        self.mouseMove(plot, pos=(0, 0))
        for pos in star:
            self.mouseMove(plot, pos=pos)
            self.qapp.processEvents()
            self.mousePress(plot, qt.Qt.LeftButton, pos=pos)
            self.qapp.processEvents()
            self.mouseRelease(plot, qt.Qt.LeftButton, pos=pos)
            self.qapp.processEvents()

    def _drawPencil(self):
        """Draw a star polygon in the plot"""
        plot = self.plot.getWidgetHandle()
        x, y = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        star = [
            (x, y + offset),
            (x - offset, y - offset),
            (x + offset, y),
            (x - offset, y),
            (x + offset, y - offset),
        ]

        self.mouseMove(plot, pos=(0, 0))
        for start, end in zip(star[:-1], star[1:]):
            self.mouseMove(plot, pos=start)
            self.mousePress(plot, qt.Qt.LeftButton, pos=start)
            self.qapp.processEvents()
            self.mouseMove(plot, pos=end)
            self.qapp.processEvents()
            self.mouseRelease(plot, qt.Qt.LeftButton, pos=end)
            self.qapp.processEvents()

    def _isMaskItemSync(self):
        """Check if masks from item and tools are sync or not"""
        if self.maskWidget.isItemMaskUpdated():
            return numpy.all(
                numpy.equal(
                    self.maskWidget.getSelectionMask(),
                    self.plot.getActiveImage().getMaskData(copy=False),
                )
            )
        else:
            return True

    def testWithAnImage(self):
        """Plot with an image: test MaskToolsWidget interactions"""

        # Add and remove a image (this should enable/disable GUI + change mask)
        self.plot.addImage(
            numpy.random.random(1024**2).reshape(1024, 1024), legend="test"
        )
        self.qapp.processEvents()

        self.plot.remove("test", kind="image")
        self.qapp.processEvents()

        tests = [
            ((0, 0), (1, 1)),
            ((1000, 1000), (1, 1)),
            ((0, 0), (-1, -1)),
            ((1000, 1000), (-1, -1)),
        ]

        for itemMaskUpdated in (False, True):
            for origin, scale in tests:
                with self.subTest(origin=origin, scale=scale):
                    self.maskWidget.setItemMaskUpdated(itemMaskUpdated)
                    self.plot.addImage(
                        numpy.arange(1024**2).reshape(1024, 1024),
                        legend="test",
                        origin=origin,
                        scale=scale,
                    )
                    self.qapp.processEvents()

                    self.assertEqual(
                        self.maskWidget.isItemMaskUpdated(), itemMaskUpdated
                    )

                    # Test draw rectangle #
                    toolButton = getQToolButtonFromAction(self.maskWidget.rectAction)
                    self.assertIsNot(toolButton, None)
                    self.mouseClick(toolButton, qt.Qt.LeftButton)

                    # mask
                    self.maskWidget.maskStateGroup.button(1).click()
                    self.qapp.processEvents()
                    self._drag()
                    self.assertFalse(
                        numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0))
                    )
                    self.assertTrue(self._isMaskItemSync())

                    # unmask same region
                    self.maskWidget.maskStateGroup.button(0).click()
                    self.qapp.processEvents()
                    self._drag()
                    self.assertTrue(
                        numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0))
                    )
                    self.assertTrue(self._isMaskItemSync())

                    # Test draw polygon #
                    toolButton = getQToolButtonFromAction(self.maskWidget.polygonAction)
                    self.assertIsNot(toolButton, None)
                    self.mouseClick(toolButton, qt.Qt.LeftButton)

                    # mask
                    self.maskWidget.maskStateGroup.button(1).click()
                    self.qapp.processEvents()
                    self._drawPolygon()
                    self.assertFalse(
                        numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0))
                    )
                    self.assertTrue(self._isMaskItemSync())

                    # unmask same region
                    self.maskWidget.maskStateGroup.button(0).click()
                    self.qapp.processEvents()
                    self._drawPolygon()
                    self.assertTrue(
                        numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0))
                    )
                    self.assertTrue(self._isMaskItemSync())

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
                        numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0))
                    )
                    self.assertTrue(self._isMaskItemSync())

                    # unmask same region
                    self.maskWidget.maskStateGroup.button(0).click()
                    self.qapp.processEvents()
                    self._drawPencil()
                    self.assertTrue(
                        numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0))
                    )
                    self.assertTrue(self._isMaskItemSync())

                    # Test no draw tool #
                    toolButton = getQToolButtonFromAction(self.maskWidget.browseAction)
                    self.assertIsNot(toolButton, None)
                    self.mouseClick(toolButton, qt.Qt.LeftButton)

                    self.plot.clear()

    def __loadSave(self, file_format):
        """Plot with an image: test MaskToolsWidget operations"""
        self.plot.addImage(numpy.arange(1024**2).reshape(1024, 1024), legend="test")
        self.qapp.processEvents()

        # Draw a polygon mask
        toolButton = getQToolButtonFromAction(self.maskWidget.polygonAction)
        self.assertIsNot(toolButton, None)
        self.mouseClick(toolButton, qt.Qt.LeftButton)
        self._drawPolygon()

        ref_mask = self.maskWidget.getSelectionMask()
        self.assertFalse(numpy.all(numpy.equal(ref_mask, 0)))

        with temp_dir() as tmp:
            mask_filename = os.path.join(tmp, "mask." + file_format)
            self.maskWidget.save(mask_filename, file_format)

            self.maskWidget.resetSelectionMask()
            self.assertTrue(
                numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), 0))
            )

            self.maskWidget.load(mask_filename)
            self.assertTrue(
                numpy.all(numpy.equal(self.maskWidget.getSelectionMask(), ref_mask))
            )

    def testLoadSaveNpy(self):
        self.__loadSave("npy")

    def testLoadSaveFit2D(self):
        self.__loadSave("msk")

    def testSigMaskChangedEmitted(self):
        self.plot.addImage(numpy.arange(512**2).reshape(512, 512), legend="test")
        self.plot.resetZoom()
        self.qapp.processEvents()

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
