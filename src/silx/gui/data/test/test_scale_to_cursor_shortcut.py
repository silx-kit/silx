"""Test of the colormap dialog scale to cursor shortcut"""

import numpy

from silx.gui.data._utils import _ScaleToCursorShortcut
from silx.gui.dialog.ColormapDialog import ColormapDialog
from silx.gui.plot import Plot2D


def testShortcutTargetsItsPlot(qWidgetFactory):
    """With a shared dialog, the shortcut targets its own plot"""
    dialog = ColormapDialog()
    action = dialog.getScaleToCursorAction()
    for _ in range(2):
        plot = qWidgetFactory(Plot2D)
        plot.getColormapAction().setColormapDialog(dialog)
        plot.addImage(numpy.random.random((10, 10)))
        shortcut = _ScaleToCursorShortcut(plot)

        shortcut.trigger()
        assert action.isChecked()
        assert dialog._getItem() is plot.getActiveImage()
        shortcut.trigger()
        assert not action.isChecked()
    dialog.deleteLater()
