# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""A :class:`.PlotWidget` with additionnal toolbars.

The :class:`PlotWindow` is a subclass of :class:`.PlotWidget`.
It provides the plot API fully defined in :class:`.Plot`.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "07/03/2016"


from . import PlotWidget
from .PlotActions import *  # noqa


class PlotWindow(PlotWidget):
    """Qt Widget providing a 1D/2D plot area and additional tools.

    This widget includes the following QAction as attributes:

    - resetZoomAction: Reset zoom
    - xAxisAutoScaleAction: Toggle X axis autoscale
    - yAxisAutoScaleAction: Toggle Y axis autoscale
    - xAxisLogarithmicAction: Toggle X axis log scale
    - yAxisLogarithmicAction: Toggle Y axis log scale
    - gridAction: Toggle plot grid
    - curveStyleAction: Change curve line and markers style
    - colormapAction: Open a colormap dialog to change active image
      and default colormap.
    - keepDataAspectRatioAction: Toggle keep aspect ratio
    - yAxisInvertedAction: Toggle Y Axis direction
    - copyAction: Copy plot snapshot to clipboard
    - saveAction: Save plot
    - printAction: Print plot

    Initialiser parameters:

    :param parent: The parent of this widget or None.
    :param backend: The backend to use for the plot.
                    The default is to use matplotlib.
    :type backend: str or :class:`BackendBase.BackendBase`
    :param bool resetzoom: Toggle visibility of reset zoom action.
    :param bool autoScale: Toggle visibility of axes autoscale actions.
    :param bool logScale: Toggle visibility of axes log scale actions.
    :param bool grid: Toggle visibility of grid mode action.
    :param bool curveStyle: Toggle visibility of curve style action.
    :param bool colormap: Toggle visibility of colormap action.
    :param bool aspectRatio: Toggle visibility of aspect ration action.
    :param bool yInverted: Toggle visibility of Y axis direction action.
    :param bool copy: Toggle visibility if copy action.
    :param bool save: Toggle visibility of save action.
    :param bool print_: Toggle visibility of print action.
    :param bool autoreplot: Toggle autoreplot mode (Default: True).
    """

    def __init__(self, parent=None, backend=None,
                 resetzoom=True, autoScale=True, logScale=True, grid=True,
                 curveStyle=True, colormap=True,
                 aspectRatio=True, yInverted=True,
                 copy=True, save=True, print_=True,
                 autoreplot=True):
        super(PlotWindow, self).__init__(
            parent=parent, backend=backend, autoreplot=autoreplot)

        # Init actions
        self.group = qt.QActionGroup(self)
        self.group.setExclusive(False)

        self.resetZoomAction = self.group.addAction(ResetZoomAction(self))
        self.resetZoomAction.setVisible(resetzoom)

        self.xAxisAutoScaleAction = self.group.addAction(
            XAxisAutoScaleAction(self))
        self.xAxisAutoScaleAction.setVisible(autoScale)

        self.yAxisAutoScaleAction = self.group.addAction(
            YAxisAutoScaleAction(self))
        self.yAxisAutoScaleAction.setVisible(autoScale)

        self.xAxisLogarithmicAction = self.group.addAction(
            XAxisLogarithmicAction(self))
        self.xAxisLogarithmicAction.setVisible(logScale)

        self.yAxisLogarithmicAction = self.group.addAction(
            YAxisLogarithmicAction(self))
        self.yAxisLogarithmicAction.setVisible(logScale)

        self.gridAction = self.group.addAction(
            GridAction(self, gridMode='both'))
        self.gridAction.setVisible(grid)

        self.curveStyleAction = self.group.addAction(CurveStyleAction(self))
        self.curveStyleAction.setVisible(curveStyle)

        self.colormapAction = self.group.addAction(ColormapAction(self))
        self.colormapAction.setVisible(colormap)

        self.keepDataAspectRatioAction = self.group.addAction(
            KeepAspectRatioAction(self))
        self.keepDataAspectRatioAction.setVisible(aspectRatio)

        self.yAxisInvertedAction = self.group.addAction(
            YAxisInvertedAction(self))
        self.yAxisInvertedAction.setVisible(yInverted)

        self._separator = qt.QAction('separator', self)
        self._separator.setSeparator(True)
        self.group.addAction(self._separator)

        self.copyAction = self.group.addAction(CopyAction(self))
        self.copyAction.setVisible(copy)

        self.saveAction = self.group.addAction(SaveAction(self))
        self.saveAction.setVisible(save)

        self.printAction = self.group.addAction(PrintAction(self))
        self.printAction.setVisible(print_)

        self._toolBar = self.toolBar(parent=self)
        self.addToolBar(self._toolBar)
        self._menu = self.menu()
        self.menuBar().addMenu(self._menu)

    def toolBar(self, title='Plot', parent=None):
        """Return a QToolBar from the QAction of the PlotWindow.

        :param str title: The title of the QMenu
        :param parent: See :class:`QToolBar`
        """
        toolbar = qt.QToolBar(title, parent)
        for action in self.group.actions():
            toolbar.addAction(action)
        return toolbar

    def menu(self, title='Plot', parent=None):
        """Return a QMenu from the QAction of the PlotWindow.

        :param str title: The title of the QMenu
        :param parent: See :class:`QMenu`
        """
        menu = qt.QMenu(title, parent)
        for action in self.group.actions():
            menu.addAction(action)
        return menu
