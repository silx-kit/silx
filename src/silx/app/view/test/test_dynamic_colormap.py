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
"""Test of the dynamic colormap integration in silx view"""

import numpy
import pytest

from silx.app.view.ApplicationContext import ApplicationContext
from silx.app.view.DataPanel import DataPanel
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.data.DataViews import PLOT2D_MODE
from silx.gui.plot import items
from silx.gui.plot.actions.image import DynamicColormapAction


class _Parent(qt.QWidget):
    def saveSettings(self, settings):
        pass


@pytest.fixture
def context(qWidgetFactory, tmp_path):
    parent = qWidgetFactory(_Parent)
    settings = qt.QSettings(str(tmp_path / "silx.ini"), qt.QSettings.IniFormat)
    yield ApplicationContext(parent, settings)


def _setImage(panel, qapp_utils, data):
    panel.setData(data)
    panel.findChild(DataViewerFrame).setDisplayMode(PLOT2D_MODE)
    qapp_utils.qWait(100)


def _visibleDynamicColormapAction(panel):
    for action in panel.findChildren(DynamicColormapAction):
        if action.plot.isVisible():
            return action
    return None


def _moveMouse(qapp_utils, plot, x, y):
    # Make sure the plot is rendered, picking requires it
    plot.replot()
    qapp_utils.qWait(100)
    xPixel, yPixel = plot.dataToPixel(x, y)
    plot.onMouseMove(xPixel, yPixel)


def testDynamicColormapResetOnDataChanged(context, qWidgetFactory, qapp_utils):
    panel = qWidgetFactory(DataPanel, context=context)
    panel.resize(600, 500)
    colormap = context.getColormap(None)
    assert colormap.getVRange() == (None, None)

    data = numpy.random.random((100, 100))
    _setImage(panel, qapp_utils, data)

    action = _visibleDynamicColormapAction(panel)
    assert action is not None
    assert action.shortcut() == qt.QKeySequence(qt.Qt.Key_W)

    action.trigger()
    plot = action.plot
    plot.resetZoom()
    _moveMouse(qapp_utils, plot, 50, 50)
    assert colormap.getVRange() != (None, None)

    # Settings are saved without the dynamic range
    context.saveSettings()
    settings = context.getSettings()
    settings.beginGroup("colormap")
    saved = Colormap()
    saved.restoreState(settings.value("default"))
    settings.endGroup()
    assert saved.getVRange() == (None, None)

    # Range is kept when the action is unchecked...
    action.trigger()
    dynamicRange = colormap.getVRange()
    assert dynamicRange != (None, None)

    # ...and restored when the data changes
    _setImage(panel, qapp_utils, numpy.random.random((50, 50)))
    assert colormap.getVRange() == (None, None)


def testDynamicColormapDeactivatedOnDataChanged(context, qWidgetFactory, qapp_utils):
    panel = qWidgetFactory(DataPanel, context=context)
    panel.resize(600, 500)
    colormap = context.getColormap(None)
    colormap.setVRange(1, 2)  # User-defined range

    _setImage(panel, qapp_utils, numpy.random.random((100, 100)))
    action = _visibleDynamicColormapAction(panel)
    plot = action.plot
    plot.resetZoom()

    for _ in range(2):  # Twice to check activating again works
        action.trigger()
        assert action.isChecked()
        _moveMouse(qapp_utils, plot, 50, 50)
        assert colormap.getVRange() != (1, 2)
        assert any(isinstance(item, items.Shape) for item in plot.getItems())

        _setImage(panel, qapp_utils, numpy.random.random((100, 100)))
        assert not action.isChecked()
        assert colormap.getVRange() == (1, 2)
        assert not any(isinstance(item, items.Shape) for item in plot.getItems())

        # Mouse moves no longer update the colormap
        _moveMouse(qapp_utils, plot, 50, 50)
        assert colormap.getVRange() == (1, 2)
