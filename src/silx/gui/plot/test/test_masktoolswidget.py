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

import pytest
import os.path

import sys
import numpy

from ... import qt
from ...utils.testutils import getQToolButtonFromAction
from .. import PlotWindow
from ..MaskToolsWidget import MaskToolsDockWidget


@pytest.fixture
def widgets(qWidgetFactory):
    plot = qWidgetFactory(PlotWindow)
    dockWidget = qWidgetFactory(MaskToolsDockWidget, plot=plot, name="TEST")
    plot.addDockWidget(qt.Qt.BottomDockWidgetArea, dockWidget)
    yield plot, dockWidget, dockWidget.widget()


@pytest.fixture
def draw(widgets, qapp, qapp_utils):

    def _draw(shape=None):
        plot, _, _ = widgets
        plot = plot.getWidgetHandle()
        x, y = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        qapp_utils.mouseMove(plot, pos=(0, 0))
        if shape == "polygon":
            polygon = [
                (x, y + offset),
                (x - offset, y - offset),
                (x + offset, y),
                (x - offset, y),
                (x + offset, y - offset),
                (x, y + offset),
            ]  # Close polygon
            for pos in polygon:
                qapp_utils.mouseMove(plot, pos=pos)
                qapp.processEvents()
                qapp_utils.mousePress(plot, qt.Qt.LeftButton, pos=pos)
                qapp.processEvents()
                qapp_utils.mouseRelease(plot, qt.Qt.LeftButton, pos=pos)
                qapp.processEvents()
        elif shape == "pencil":
            pencil = [
                (x, y + offset),
                (x - offset, y - offset),
                (x + offset, y),
                (x - offset, y),
                (x + offset, y - offset),
            ]

            for start, end in zip(pencil[:-1], pencil[1:]):
                qapp_utils.mouseMove(plot, pos=start)
                qapp_utils.mousePress(plot, qt.Qt.LeftButton, pos=start)
                qapp.processEvents()
                qapp_utils.mouseMove(plot, pos=end)
                qapp.processEvents()
                qapp_utils.mouseRelease(plot, qt.Qt.LeftButton, pos=end)
                qapp.processEvents()
        else:
            # Drag from plot center to offset position"""
            pos0 = x, y
            pos1 = x + offset, y + offset
            qapp_utils.mouseMove(plot, pos=pos0)
            qapp.processEvents()
            qapp_utils.mousePress(plot, qt.Qt.LeftButton, pos=pos0)
            qapp.processEvents()
            qapp_utils.mouseMove(
                plot, pos=(pos0[0] + offset // 2, pos0[1] + offset // 2)
            )
            qapp_utils.mouseMove(plot, pos=pos1)
            qapp.processEvents()
            qapp_utils.mouseRelease(plot, qt.Qt.LeftButton, pos=pos1)
            qapp.processEvents()
            qapp_utils.mouseMove(plot, pos=(0, 0))

    return _draw


def testEmptyPlot(qapp, widgets):
    """Empty plot, display MaskToolsDockWidget, toggle multiple masks"""
    _, _, maskWidget = widgets
    maskWidget.setMultipleMasks("single")
    qapp.processEvents()

    maskWidget.setMultipleMasks("exclusive")
    qapp.processEvents()


def testSigMaskChangedEmitted(widgets, qapp, qapp_utils, draw):
    plot, _, maskWidget = widgets
    plot.addImage(numpy.arange(512**2).reshape(512, 512), legend="test")
    plot.resetZoom()
    qapp.processEvents()

    lst = []

    def slot():
        lst.append(1)

    maskWidget.sigMaskChanged.connect(slot)

    # rectangle mask
    toolButton = getQToolButtonFromAction(maskWidget.rectAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)
    maskWidget.maskStateGroup.button(1).click()
    qapp.processEvents()
    draw()

    assert len(lst) > 0


@pytest.mark.skipif(
    sys.platform.startswith("win") and os.environ["CI"] == "true",
    reason="This test is flaky in the CI when running on Windows",
)
def testWithAnImage(widgets, qapp, qapp_utils, draw):
    """Plot with an image: test MaskToolsWidget interactions"""
    plot, _, maskWidget = widgets

    # Add and remove a image (this should enable/disable GUI + change mask)
    plot.addImage(numpy.random.random(1024**2).reshape(1024, 1024), legend="test")
    qapp.processEvents()

    plot.remove("test", kind="image")
    qapp.processEvents()

    tests = [
        ((0, 0), (1, 1)),
        ((1000, 1000), (1, 1)),
        ((0, 0), (-1, -1)),
        ((1000, 1000), (-1, -1)),
    ]

    def _isMaskItemSync():
        """Check if masks from item and tools are sync or not"""
        if maskWidget.isItemMaskUpdated():
            return numpy.all(
                numpy.equal(
                    maskWidget.getSelectionMask(),
                    plot.getActiveImage().getMaskData(copy=False),
                )
            )
        else:
            return True

    for itemMaskUpdated in (False, True):
        for origin, scale in tests:
            maskWidget.setItemMaskUpdated(itemMaskUpdated)
            plot.addImage(
                numpy.arange(1024**2).reshape(1024, 1024),
                legend="test",
                origin=origin,
                scale=scale,
            )
            qapp.processEvents()

            assert maskWidget.isItemMaskUpdated() == itemMaskUpdated

            # Test draw rectangle #
            toolButton = getQToolButtonFromAction(maskWidget.rectAction)
            assert toolButton is not None
            qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

            # mask
            maskWidget.maskStateGroup.button(1).click()
            qapp.processEvents()
            draw()
            assert not numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
            assert _isMaskItemSync()

            # unmask same region
            maskWidget.maskStateGroup.button(0).click()
            qapp.processEvents()
            draw()
            assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
            assert _isMaskItemSync()

            # Test draw polygon
            toolButton = getQToolButtonFromAction(maskWidget.polygonAction)
            assert toolButton is not None
            qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

            # mask
            maskWidget.maskStateGroup.button(1).click()
            qapp.processEvents()
            draw("polygon")
            assert not numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
            assert _isMaskItemSync()

            # unmask same region
            maskWidget.maskStateGroup.button(0).click()
            qapp.processEvents()
            draw("polygon")
            assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
            assert _isMaskItemSync()

            # Test draw pencil
            toolButton = getQToolButtonFromAction(maskWidget.pencilAction)
            assert toolButton is not None
            qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

            maskWidget.pencilSpinBox.setValue(30)
            qapp.processEvents()

            # mask
            maskWidget.maskStateGroup.button(1).click()
            qapp.processEvents()
            draw("pencil")
            assert not numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
            assert _isMaskItemSync()

            # unmask same region
            maskWidget.maskStateGroup.button(0).click()
            qapp.processEvents()
            draw("pencil")
            assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))
            assert _isMaskItemSync()

            # Test no draw tool #
            toolButton = getQToolButtonFromAction(maskWidget.browseAction)
            assert toolButton is not None
            qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)

            plot.clear()


@pytest.mark.parametrize("file_format", ("npy", "msk"))
def testLoadSave(widgets, qapp, qapp_utils, tmp_path, draw, file_format):
    plot, _, maskWidget = widgets
    plot.addImage(numpy.arange(1024**2).reshape(1024, 1024), legend="test")
    qapp.processEvents()

    # Draw a polygon mask
    toolButton = getQToolButtonFromAction(maskWidget.polygonAction)
    assert toolButton is not None
    qapp_utils.mouseClick(toolButton, qt.Qt.LeftButton)
    draw("polygon")

    ref_mask = maskWidget.getSelectionMask()
    assert not numpy.all(numpy.equal(ref_mask, 0))

    mask_filename = os.path.join(tmp_path, "mask." + file_format)
    maskWidget.save(mask_filename, file_format)

    maskWidget.resetSelectionMask()
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), 0))

    maskWidget.load(mask_filename)
    assert numpy.all(numpy.equal(maskWidget.getSelectionMask(), ref_mask))
