# /*##########################################################################
#
# Copyright (c) 2026 European Synchrotron Radiation Facility
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
"""Tests of DynamicColormapAction"""

import numpy
import pytest

from silx.gui.plot import items
from silx.gui.plot.actions.image import DynamicColormapAction


def _outlines(plot):
    return [item for item in plot.getItems() if isinstance(item, items.Shape)]


@pytest.fixture
def plotWithImage(plotWidget, qapp):
    # Left half in [0, 1), right half in [100, 101)
    data = numpy.zeros((100, 100), dtype=numpy.float32)
    data[:, :50] = numpy.random.random((100, 50))
    data[:, 50:] = 100 + numpy.random.random((100, 50))
    plotWidget.addImage(data, legend="image", origin=(-50, 10), scale=(0.5, 2))
    plotWidget.resetZoom()
    qapp.processEvents()
    yield plotWidget


def _moveMouse(plot, x, y):
    xPixel, yPixel = plot.dataToPixel(x, y)
    plot.onMouseMove(xPixel, yPixel)


def testDynamicColormapAction(plotWithImage, qapp):
    plot = plotWithImage
    image = plot.getImage("image")
    colormap = image.getColormap()
    initialRange = colormap.getVRange()

    action = DynamicColormapAction(plot)
    action.setScreenRegionSize(1)
    action.setImageRegionSize(10)

    # Unchecked: no update
    _moveMouse(plot, -40, 110)  # Left half of the image
    assert colormap.getVRange() == initialRange
    assert _outlines(plot) == []

    action.trigger()
    assert action.isChecked()

    _moveMouse(plot, -40, 110)  # Left half of the image
    vmin, vmax = colormap.getVRange()
    assert 0 <= vmin < vmax < 1
    assert len(_outlines(plot)) == 1
    assert _outlines(plot)[0].isVisible()

    _moveMouse(plot, -10, 110)  # Right half of the image
    vmin, vmax = colormap.getVRange()
    assert 100 <= vmin < vmax < 101
    assert len(_outlines(plot)) == 1

    # Unchecked: range is kept and outline is removed
    action.trigger()
    assert not action.isChecked()
    assert _outlines(plot) == []
    _moveMouse(plot, -40, 110)
    assert colormap.getVRange() == (vmin, vmax)
