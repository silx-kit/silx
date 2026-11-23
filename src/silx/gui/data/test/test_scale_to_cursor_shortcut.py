"""Test of the colormap dialog scale to cursor shortcut in the data views"""

import numpy

from silx.gui import qt
from silx.gui.data.DataViewer import DataViewer
from silx.gui.data.DataViews import PLOT2D_MODE
from silx.gui.data._utils import addScaleToCursorShortcut
from silx.gui.dialog.ColormapDialog import ColormapDialog
from silx.gui.plot import Plot2D, PlotWidget
from silx.gui.utils.testutils import QTest


def _pressW(qapp_utils, plot):
    plot.activateWindow()
    plot.getWidgetHandle().setFocus()
    qapp_utils.qWait(50)
    QTest.keyClick(plot.getWidgetHandle(), qt.Qt.Key_W)
    qapp_utils.qWait(50)


def testSharedDialog(qWidgetFactory, qapp_utils):
    """The shortcut targets the plot with the focus"""
    dialog = ColormapDialog()
    plots = []
    for _ in range(2):
        plot = qWidgetFactory(Plot2D)
        plot.getColormapAction().setColormapDialog(dialog)
        addScaleToCursorShortcut(plot)
        plot.addImage(numpy.random.random((10, 10)))
        plots.append(plot)
    action = dialog.getScaleToCursorAction()

    for plot in plots:
        _pressW(qapp_utils, plot)
        assert action.isChecked()
        assert dialog._getItem() is plot.getActiveImage()
        _pressW(qapp_utils, plot)
        assert not action.isChecked()

    dialog.deleteLater()


def testDataViewer(qWidgetFactory, qapp_utils):
    viewer = qWidgetFactory(DataViewer)
    viewer.setData(numpy.random.random((10, 10)))
    viewer.setDisplayMode(PLOT2D_MODE)
    qapp_utils.qWait(100)
    plot = [p for p in viewer.findChildren(PlotWidget) if p.isVisible()][0]

    _pressW(qapp_utils, plot)
    action = plot.getColormapAction().getColormapDialog().getScaleToCursorAction()
    assert action.isChecked()
    action.setChecked(False)
