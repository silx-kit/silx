# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
"""A widget dedicated to display scatter plots

It is based on a :class:`~silx.gui.plot.PlotWidget` with additional tools
for scatter plots.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "14/06/2018"


import logging
import weakref

import numpy

from . import items
from . import PlotWidget
from . import tools
from .actions import histogram as actions_histogram
from .tools.profile import ScatterProfileToolBar
from .ColorBar import ColorBarWidget
from .ScatterMaskToolsWidget import ScatterMaskToolsWidget

from ..widgets.BoxLayoutDockWidget import BoxLayoutDockWidget
from .. import qt, icons
from ...utils.proxy import docstring
from ...utils.weakref import WeakMethodProxy


_logger = logging.getLogger(__name__)


class ScatterView(qt.QMainWindow):
    """Main window with a PlotWidget and tools specific for scatter plots.

    :param parent: The parent of this widget
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`~silx.gui.plot.PlotWidget` for the list of supported backend.
    :type backend: Union[str,~silx.gui.plot.backends.BackendBase.BackendBase]
    """

    _SCATTER_LEGEND = ' '
    """Legend used for the scatter item"""

    def __init__(self, parent=None, backend=None):
        super(ScatterView, self).__init__(parent=parent)
        if parent is not None:
            # behave as a widget
            self.setWindowFlags(qt.Qt.Widget)
        else:
            self.setWindowTitle('ScatterView')

        # Create plot widget
        plot = PlotWidget(parent=self, backend=backend)
        self._plot = weakref.ref(plot)

        # Add an empty scatter
        self.__createEmptyScatter()

        # Create colorbar widget with white background
        self._colorbar = ColorBarWidget(parent=self, plot=plot)
        self._colorbar.setAutoFillBackground(True)
        palette = self._colorbar.palette()
        palette.setColor(qt.QPalette.Background, qt.Qt.white)
        palette.setColor(qt.QPalette.Window, qt.Qt.white)
        self._colorbar.setPalette(palette)

        # Create PositionInfo widget
        self.__lastPickingPos = None
        self.__pickingCache = None
        self._positionInfo = tools.PositionInfo(
            plot=plot,
            converters=(('X', WeakMethodProxy(self._getPickedX)),
                        ('Y', WeakMethodProxy(self._getPickedY)),
                        ('Data', WeakMethodProxy(self._getPickedValue)),
                        ('Index', WeakMethodProxy(self._getPickedIndex))))

        # Combine plot, position info and colorbar into central widget
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.addWidget(plot, 0, 0)
        gridLayout.addWidget(self._colorbar, 0, 1)
        gridLayout.addWidget(self._positionInfo, 1, 0, 1, -1)
        gridLayout.setRowStretch(0, 1)
        gridLayout.setColumnStretch(0, 1)
        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)

        # Create mask tool dock widget
        self._maskToolsWidget = ScatterMaskToolsWidget(parent=self, plot=plot)
        self._maskDock = BoxLayoutDockWidget()
        self._maskDock.setWindowTitle('Scatter Mask')
        self._maskDock.setWidget(self._maskToolsWidget)
        self._maskDock.setVisible(False)
        self.addDockWidget(qt.Qt.BottomDockWidgetArea, self._maskDock)

        self._maskAction = self._maskDock.toggleViewAction()
        self._maskAction.setIcon(icons.getQIcon('image-mask'))
        self._maskAction.setToolTip("Display/hide mask tools")

        self._intensityHistoAction = actions_histogram.PixelIntensitiesHistoAction(plot=plot, parent=self)

        # Create toolbars
        self._interactiveModeToolBar = tools.InteractiveModeToolBar(
            parent=self, plot=plot)

        self._scatterToolBar = tools.ScatterToolBar(
            parent=self, plot=plot)
        self._scatterToolBar.addAction(self._maskAction)
        self._scatterToolBar.addAction(self._intensityHistoAction)

        self._profileToolBar = ScatterProfileToolBar(parent=self, plot=plot)

        self._outputToolBar = tools.OutputToolBar(parent=self, plot=plot)

        # Activate shortcuts in PlotWindow widget:
        for toolbar in (self._interactiveModeToolBar,
                        self._scatterToolBar,
                        self._profileToolBar,
                        self._outputToolBar):
            self.addToolBar(toolbar)
            for action in toolbar.actions():
                self.addAction(action)


    def __createEmptyScatter(self):
        """Create an empty scatter item that is used to display the data

        :rtype: ~silx.gui.plot.items.Scatter
        """
        plot = self.getPlotWidget()
        plot.addScatter(x=(), y=(), value=(), legend=self._SCATTER_LEGEND)
        scatter = plot._getItem(
            kind='scatter', legend=self._SCATTER_LEGEND)
        # Profile is not selectable,
        # so it does not interfere with profile interaction
        scatter._setSelectable(False)
        return scatter

    def _pickScatterData(self, x, y):
        """Get data and index and value of top most scatter plot at position (x, y)

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The data index and value at that point or None
        """
        pickingPos = x, y
        if self.__lastPickingPos != pickingPos:
            self.__pickingCache = None
            self.__lastPickingPos = pickingPos

            plot = self.getPlotWidget()
            if plot is not None:
                pixelPos = plot.dataToPixel(x, y)
                if pixelPos is not None:
                    # Start from top-most item
                    result = plot._pickTopMost(
                        pixelPos[0], pixelPos[1],
                        lambda item: isinstance(item, items.Scatter))
                    if result is not None:
                        item = result.getItem()
                        if item.getVisualization() is items.Scatter.Visualization.BINNED_STATISTIC:
                            # Get highest index of closest points
                            selected = result.getIndices(copy=False)[::-1]
                            dataIndex = selected[numpy.argmin(
                                (item.getXData(copy=False)[selected] - x)**2 +
                                (item.getYData(copy=False)[selected] - y)**2)]
                        else:
                            # Get last index
                            # with matplotlib it should be the top-most point
                            dataIndex = result.getIndices(copy=False)[-1]
                        self.__pickingCache = (
                            dataIndex,
                            item.getXData(copy=False)[dataIndex],
                            item.getYData(copy=False)[dataIndex],
                            item.getValueData(copy=False)[dataIndex])

        return self.__pickingCache

    def _getPickedIndex(self, x, y):
        """Get data index of top most scatter plot at position (x, y)

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The data index at that point or '-'
        """
        picking = self._pickScatterData(x, y)
        return '-' if picking is None else picking[0]

    def _getPickedX(self, x, y):
        """Returns X position snapped to scatter plot when close enough

        :param float x:
        :param float y:
        :rtype: float
        """
        picking = self._pickScatterData(x, y)
        return x if picking is None else picking[1]

    def _getPickedY(self, x, y):
        """Returns Y position snapped to scatter plot when close enough

        :param float x:
        :param float y:
        :rtype: float
        """
        picking = self._pickScatterData(x, y)
        return y if picking is None else picking[2]

    def _getPickedValue(self, x, y):
        """Get data value of top most scatter plot at position (x, y)

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The data value at that point or '-'
        """
        picking = self._pickScatterData(x, y)
        return '-' if picking is None else picking[3]

    def _mouseInPlotArea(self, x, y):
        """Clip mouse coordinates to plot area coordinates

        :param float x: X position in pixels
        :param float y: Y position in pixels
        :return: (x, y) in data coordinates
        """
        plot = self.getPlotWidget()
        left, top, width, height = plot.getPlotBoundsInPixels()
        xPlot = numpy.clip(x, left, left + width - 1)
        yPlot = numpy.clip(y, top, top + height - 1)
        return xPlot, yPlot

    def getPlotWidget(self):
        """Returns the :class:`~silx.gui.plot.PlotWidget` this window is based on.

        :rtype: ~silx.gui.plot.PlotWidget
        """
        return self._plot()

    def getPositionInfoWidget(self):
        """Returns the widget display mouse coordinates information.

        :rtype: ~silx.gui.plot.tools.PositionInfo
        """
        return self._positionInfo

    def getMaskToolsWidget(self):
        """Returns the widget controlling mask drawing

        :rtype: ~silx.gui.plot.ScatterMaskToolsWidget
        """
        return self._maskToolsWidget

    def getInteractiveModeToolBar(self):
        """Returns QToolBar controlling interactive mode.

        :rtype: ~silx.gui.plot.tools.InteractiveModeToolBar
        """
        return self._interactiveModeToolBar

    def getScatterToolBar(self):
        """Returns QToolBar providing scatter plot tools.

        :rtype: ~silx.gui.plot.tools.ScatterToolBar
        """
        return self._scatterToolBar

    def getScatterProfileToolBar(self):
        """Returns QToolBar providing scatter profile tools.

        :rtype: ~silx.gui.plot.tools.profile.ScatterProfileToolBar
        """
        return self._profileToolBar

    def getOutputToolBar(self):
        """Returns QToolBar containing save, copy and print actions

        :rtype: ~silx.gui.plot.tools.OutputToolBar
        """
        return self._outputToolBar

    def setColormap(self, colormap=None):
        """Set the colormap for the displayed scatter and the
        default plot colormap.

        :param ~silx.gui.colors.Colormap colormap:
            The description of the colormap.
        """
        self.getScatterItem().setColormap(colormap)
        # Resilient to call to PlotWidget API (e.g., clear)
        self.getPlotWidget().setDefaultColormap(colormap)

    def getColormap(self):
        """Return the colormap object in use.

        :return: Colormap currently in use
        :rtype: ~silx.gui.colors.Colormap
        """
        return self.getScatterItem().getColormap()

    # Control displayed scatter plot

    def setData(self, x, y, value, xerror=None, yerror=None, alpha=None, copy=True):
        """Set the data of the scatter plot.

        To reset the scatter plot, set x, y and value to None.

        :param Union[numpy.ndarray,None] x: X coordinates.
        :param Union[numpy.ndarray,None] y: Y coordinates.
        :param Union[numpy.ndarray,None] value:
            The data corresponding to the value of the data points.
        :param xerror: Values with the uncertainties on the x values.
            If it is an array, it can either be a 1D array of
            same length as the data or a 2D array with 2 rows
            of same length as the data: row 0 for positive errors,
            row 1 for negative errors.
        :type xerror: A float, or a numpy.ndarray of float32.

        :param yerror: Values with the uncertainties on the y values
        :type yerror: A float, or a numpy.ndarray of float32. See xerror.
        :param alpha: Values with the transparency (between 0 and 1)
        :type alpha: A float, or a numpy.ndarray of float32 
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        """
        x = () if x is None else x
        y = () if y is None else y
        value = () if value is None else value

        self.getScatterItem().setData(
            x=x, y=y, value=value, xerror=xerror, yerror=yerror, alpha=alpha, copy=copy)

    @docstring(items.Scatter)
    def getData(self, *args, **kwargs):
        return self.getScatterItem().getData(*args, **kwargs)

    def getScatterItem(self):
        """Returns the plot item displaying the scatter data.

        This allows to set the style of the displayed scatter.

        :rtype: ~silx.gui.plot.items.Scatter
        """
        plot = self.getPlotWidget()
        scatter = plot._getItem(kind='scatter', legend=self._SCATTER_LEGEND)
        if scatter is None:  # Resilient to call to PlotWidget API (e.g., clear)
            scatter = self.__createEmptyScatter()
        return scatter

    # Convenient proxies

    @docstring(PlotWidget)
    def getXAxis(self, *args, **kwargs):
        return self.getPlotWidget().getXAxis(*args, **kwargs)

    @docstring(PlotWidget)
    def getYAxis(self, *args, **kwargs):
        return self.getPlotWidget().getYAxis(*args, **kwargs)

    @docstring(PlotWidget)
    def setGraphTitle(self, *args, **kwargs):
        return self.getPlotWidget().setGraphTitle(*args, **kwargs)

    @docstring(PlotWidget)
    def getGraphTitle(self, *args, **kwargs):
        return self.getPlotWidget().getGraphTitle(*args, **kwargs)

    @docstring(PlotWidget)
    def resetZoom(self, *args, **kwargs):
        return self.getPlotWidget().resetZoom(*args, **kwargs)

    @docstring(ScatterMaskToolsWidget)
    def getSelectionMask(self, *args, **kwargs):
        return self.getMaskToolsWidget().getSelectionMask(*args, **kwargs)

    @docstring(ScatterMaskToolsWidget)
    def setSelectionMask(self, *args, **kwargs):
        return self.getMaskToolsWidget().setSelectionMask(*args, **kwargs)
