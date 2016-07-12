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

import collections
import logging

from . import PlotWidget
from .PlotActions import *  # noqa
from .PlotTools import PositionInfo, ProfileToolBar
from .LegendSelector import LegendsDockWidget
from .CurvesROIWidget import CurvesROIDockWidget
from .MaskToolsWidget import MaskToolsDockWidget
try:
    from ..console import IPythonDockWidget
except ImportError:
    IPythonDockWidget = None

from .. import qt


_logger = logging.getLogger(__name__)


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
    :param bool copy: Toggle visibility of copy action.
    :param bool save: Toggle visibility of save action.
    :param bool print_: Toggle visibility of print action.
    :param bool control: True to display an Options button with a sub-menu
                         to show legends, toggle crosshair and pan with arrows.
                         (Default: False)
    :param position: True to display widget with (x, y) mouse position
                     (Default: False).
                     It also supports a list of (name, funct(x, y)->value)
                     to customize the displayed values.
                     See :class:`silx.gui.plot.PlotTools.PositionInfo`.
    :param bool roi: Toggle visibilty of ROI action.
    """

    def __init__(self, parent=None, backend=None,
                 resetzoom=True, autoScale=True, logScale=True, grid=True,
                 curveStyle=True, colormap=True,
                 aspectRatio=True, yInverted=True,
                 copy=True, save=True, print_=True,
                 control=False, position=False, roi=True, mask=True):
        super(PlotWindow, self).__init__(parent=parent, backend=backend)
        if parent is None:
            self.setWindowTitle('PlotWindow')

        self._dockWidgets = []

        # Init actions
        self.group = qt.QActionGroup(self)
        self.group.setExclusive(False)

        self.resetZoomAction = self.group.addAction(ResetZoomAction(self))
        self.resetZoomAction.setVisible(resetzoom)

        self.zoomInAction = ZoomInAction(self)
        self.addAction(self.zoomInAction)

        self.zoomOutAction = ZoomOutAction(self)
        self.addAction(self.zoomOutAction)

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

        self.group.addAction(self.roiAction)
        self.roiAction.setVisible(roi)

        self.group.addAction(self.maskAction)
        self.maskAction.setVisible(mask)

        self._separator = qt.QAction('separator', self)
        self._separator.setSeparator(True)
        self.group.addAction(self._separator)

        self.copyAction = self.group.addAction(CopyAction(self))
        self.copyAction.setVisible(copy)

        self.saveAction = self.group.addAction(SaveAction(self))
        self.saveAction.setVisible(save)

        self.printAction = self.group.addAction(PrintAction(self))
        self.printAction.setVisible(print_)

        if control or position:
            hbox = qt.QHBoxLayout()
            hbox.setSpacing(0)
            hbox.setContentsMargins(0, 0, 0, 0)

            if control:
                self.controlButton = qt.QPushButton("Options")
                self.controlButton.setAutoDefault(False)
                self.controlButton.clicked.connect(self._controlButtonClicked)

                hbox.addWidget(self.controlButton)

            if position:  # Add PositionInfo widget to the bottom of the plot
                if isinstance(position, collections.Iterable):
                    # Use position as a set of converters
                    converters = position
                else:
                    converters = None
                self.positionWidget = PositionInfo(self, converters=converters)
                self.positionWidget.autoSnapToActiveCurve = True

                hbox.addWidget(self.positionWidget)

            hbox.addStretch(1)
            bottomBar = qt.QWidget()
            bottomBar.setLayout(hbox)

            layout = qt.QVBoxLayout()
            layout.setSpacing(0)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.getWidgetHandle())
            layout.addWidget(bottomBar)

            centralWidget = qt.QWidget()
            centralWidget.setLayout(layout)
            self.setCentralWidget(centralWidget)

        self.addToolBar(self.toolBar())

    @property
    def legendsDockWidget(self):
        """DockWidget with Legend panel (lazy-loaded)."""
        if not hasattr(self, '_legendsDockWidget'):
            self._legendsDockWidget = LegendsDockWidget(self)
            self._legendsDockWidget.hide()
            self._introduceNewDockWidget(self._legendsDockWidget)
        return self._legendsDockWidget

    @property
    def curvesROIDockWidget(self):
        """DockWidget with curves' ROI panel (lazy-loaded)."""
        if not hasattr(self, '_curvesROIDockWidget'):
            self._curvesROIDockWidget = CurvesROIDockWidget(
                self, name='Regions Of Interest')
            self._curvesROIDockWidget.hide()
            self._introduceNewDockWidget(self._curvesROIDockWidget)
        return self._curvesROIDockWidget

    @property
    def roiAction(self):
        """QAction toggling curve ROI dock widget"""
        return self.curvesROIDockWidget.toggleViewAction()

    @property
    def maskToolsDockWidget(self):
        """DockWidget with image mask panel (lazy-loaded)."""
        if not hasattr(self, '_maskToolsDockWidget'):
            self._maskToolsDockWidget = MaskToolsDockWidget(self, name='Mask')
            self._maskToolsDockWidget.hide()
            self._introduceNewDockWidget(self._maskToolsDockWidget)
        return self._maskToolsDockWidget

    @property
    def maskAction(self):
        """QAction toggling image mask dock widget"""
        return self.maskToolsDockWidget.toggleViewAction()

    def getSelectionMask(self):
        """Return the current mask handled by :attr:`maskToolsDockWidget`.

        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return self.maskToolsDockWidget.getSelectionMask()

    def setSelectionMask(self, mask):
        """Set the mask handled by :attr`maskToolsDockWidget`.

        If the provided mask has not the same dimension as the 'active'
        image, it will by cropped or padded.

        :param mask: The array to use for the mask.
        :type mask: numpy.ndarray of uint8 of dimension 2, C-contiguous.
                    Array of other types are converted.
        :return: True if success, False if failed
        """
        return bool(self.maskToolsDockWidget.setSelectionMask(mask))

    @property
    def consoleDockWidget(self):
        """DockWidget with IPython console (lazy-loaded)."""
        if not hasattr(self, '_consoleDockWidget'):
            vars = {"plt": self}
            banner = "The variable 'plt' is available. Use the 'whos' "
            banner += "and 'help(plt)' commands for more information.\n\n"
            if IPythonDockWidget is not None:
                self._consoleDockWidget = IPythonDockWidget(
                    available_vars=vars,
                    custom_banner=banner,
                    parent=self)
                self._consoleDockWidget.hide()
                self._introduceNewDockWidget(self._consoleDockWidget)
            else:
                self._consoleDockWidget = None
        return self._consoleDockWidget

    @property
    def crosshairAction(self):
        """Action toggling crosshair cursor mode (lazy-loaded)."""
        if not hasattr(self, '_crosshairAction'):
            self._crosshairAction = CrosshairAction(self, color='red')
        return self._crosshairAction

    @property
    def panWithArrowKeysAction(self):
        """Action toggling pan with arrow keys (lazy-loaded)."""
        if not hasattr(self, '_panWithArrowKeysAction'):
            self._panWithArrowKeysAction = PanWithArrowKeysAction(self)
        return self._panWithArrowKeysAction

    def toolBar(self, title='Plot', parent=None):
        """Return a QToolBar from the QAction of the PlotWindow.

        :param str title: The title of the QMenu
        :param parent: See :class:`QToolBar`
        """
        if not hasattr(self, '_toolbar'):
            self._toolbar = qt.QToolBar(title, parent)
            for action in self.group.actions():
                self._toolbar.addAction(action)
        return self._toolbar

    def menu(self, title='Plot', parent=None):
        """Return a QMenu from the QAction of the PlotWindow.

        :param str title: The title of the QMenu
        :param parent: See :class:`QMenu`
        """
        menu = qt.QMenu(title, parent)
        for action in self.group.actions():
            menu.addAction(action)
        return menu

    def _controlButtonClicked(self):
        """Display Options button sub-menu."""
        controlMenu = qt.QMenu()
        controlMenu.addAction(self.legendsDockWidget.toggleViewAction())
        controlMenu.addAction(self.roiAction)
        controlMenu.addAction(self.maskAction)
        if self.consoleDockWidget is not None:
            controlMenu.addAction(self.consoleDockWidget.toggleViewAction())
        else:
            disabledConsoleAction = controlMenu.addAction('Console')
            disabledConsoleAction.setCheckable(True)
            disabledConsoleAction.setEnabled(False)

        controlMenu.addSeparator()
        controlMenu.addAction(self.crosshairAction)
        controlMenu.addAction(self.panWithArrowKeysAction)
        controlMenu.exec_(self.cursor().pos())

    def _introduceNewDockWidget(self, dock_widget):
        """Maintain a list of dock widgets, in the order in which they are
        added. Tabify them as soon as there are more than one of them.

        :param dock_widget: Instance of :class:`QDockWidget` to be added.
        """
        if dock_widget not in self._dockWidgets:
            self._dockWidgets.append(dock_widget)
        if len(self._dockWidgets) == 1:
            # The first created dock widget must be added to a Widget area
            width = self.centralWidget().width()
            height = self.centralWidget().height()
            if width > (2.0 * height) and width > 1000:
                area = qt.Qt.RightDockWidgetArea
            else:
                area = qt.Qt.BottomDockWidgetArea
            self.addDockWidget(area, dock_widget)
        else:
            # Other dock widgets are added as tabs to the same widget area
            self.tabifyDockWidget(self._dockWidgets[0],
                                  dock_widget)


class Plot1D(PlotWindow):
    """PlotWindow with tools specific for curves.

    :param parent: The parent of this widget
    """

    def __init__(self, parent=None):
        super(Plot1D, self).__init__(parent=parent, backend=None,
                                     resetzoom=True, autoScale=True,
                                     logScale=True, grid=True,
                                     curveStyle=True, colormap=False,
                                     aspectRatio=False, yInverted=False,
                                     copy=True, save=True, print_=True,
                                     control=True, position=True,
                                     roi=True, mask=False)
        if parent is None:
            self.setWindowTitle('Plot1D')
        self.setGraphXLabel('X')
        self.setGraphYLabel('Y')


class Plot2D(PlotWindow):
    """PlotWindow with a toolbar specific for images.

    :param parent: The parent of this widget
    """

    def __init__(self, parent=None):
        # List of information to display at the bottom of the plot
        posInfo = [
            ('X', lambda x, y: x),
            ('Y', lambda x, y: y),
            ('Data', self._getActiveImageValue)]

        super(Plot2D, self).__init__(parent=parent, backend=None,
                                     resetzoom=True, autoScale=False,
                                     logScale=False, grid=False,
                                     curveStyle=False, colormap=True,
                                     aspectRatio=True, yInverted=True,
                                     copy=True, save=True, print_=True,
                                     control=False, position=posInfo,
                                     roi=False, mask=True)
        if parent is None:
            self.setWindowTitle('Plot2D')
        self.setGraphXLabel('Columns')
        self.setGraphYLabel('Rows')

        self.profile = ProfileToolBar(self)
        """"Profile tools attached to this plot.

        See :class:`silx.gui.plot.PlotTools.ProfileToolBar`
        """

        self.addToolBar(self.profile)

    def _getActiveImageValue(self, x, y):
        """Get value of active image at position (x, y)

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The value at that point or '-'
        """
        image = self.getActiveImage()
        if image is not None:
            data, params = image[0], image[4]
            ox, oy = params['origin']
            sx, sy = params['scale']
            if (y - oy) >= 0 and (x - ox) >= 0:
                # Test positive before cast otherwisr issue with int(-0.5) = 0
                row = int((y - oy) / sy)
                col = int((x - ox) / sx)
                if (row < data.shape[0] and col < data.shape[1]):
                    return data[row, col]
        return '-'


def plot1D(x_or_y=None, y=None, title='', xlabel='X', ylabel='Y'):
    """Plot curves in a dedicated widget.

    Examples:

    The following examples must run with a Qt QApplication initialized.

    First import :func:`plot1D` function:

    >>> from silx.gui.plot import plot1D
    >>> import numpy

    Plot a single curve given some values:

    >>> values = numpy.random.random(100)
    >>> plot_1curve = plot1D(values, title='Random data')

    Plot a single curve given the x and y values:

    >>> angles = numpy.linspace(0, numpy.pi, 100)
    >>> sin_a = numpy.sin(angles)
    >>> plot_sinus = plot1D(angles, sin_a,
    ...                     xlabel='angle (radian)', ylabel='sin(a)')

    Plot many curves by giving a 2D array:

    >>> curves = numpy.random.random(10 * 100).reshape(10, 100)
    >>> plot_curves = plot1D(curves)

    Plot many curves sharing the same x values:

    >>> angles = numpy.linspace(0, numpy.pi, 100)
    >>> values = (numpy.sin(angles), numpy.cos(angles))
    >>> plot = plot1D(angles, values)

    :param x_or_y: x values or y values if y is not provided
    :param y: y values (x_or_y) must be provided
    :param str title: The title of the Plot widget
    :param str xlabel: The label of the X axis
    :param str ylabel: The label of the Y axis
    """
    plot = Plot1D()
    plot.setGraphTitle(title)
    plot.setGraphXLabel(xlabel)
    plot.setGraphYLabel(ylabel)

    # Handle x_or_y and y arguments
    if x_or_y is None and y is not None:
        # Only y is provided, reorder arguments
        x_or_y, y = y, None

    if x_or_y is not None:
        x_or_y = numpy.array(x_or_y, copy=False)

        if y is None:  # x_or_y is y and no x provided, create x values
            y = x_or_y
            x_or_y = numpy.arange(x_or_y.shape[-1], dtype=numpy.float32)

        y = numpy.array(y, copy=False)
        y = y.reshape(-1, y.shape[-1])  # Make it 2D array

        if x_or_y.ndim == 1:
            for index, ycurve in enumerate(y):
                plot.addCurve(x_or_y, ycurve, legend=('curve_%d' % index))

        else:
            # Make x a 2D array as well
            x_or_y = x_or_y.reshape(-1, x_or_y.shape[-1])
            if x_or_y.shape[0] != y.shape[0]:
                raise ValueError(
                    'Not the same dimensions for x and y (%d != %d)' %
                    (x_or_y.shape[0], y.shape[0]))
            for index, (xcurve, ycurve) in enumerate(zip(x_or_y, y)):
                plot.addCurve(xcurve, ycurve, legend=('curve_%d' % index))

    plot.show()
    return plot


def plot2D(data=None, cmap=None, norm='linear',
           vmin=None, vmax=None,
           aspect=False,
           origin=(0., 0.), scale=(1., 1.),
           title='', xlabel='X', ylabel='Y'):
    """Plot an image in a dedicated widget.

    Example to plot an image.
    This example must run with a Qt QApplication initialized.

    >>> from silx.gui.plot import plot2D
    >>> import numpy

    >>> data = numpy.random.random(1024 * 1024).reshape(1024, 1024)
    >>> plot = plot2D(data, title='Random data')

    :param data: data to plot as an image
    :type data: numpy.ndarray-like with 2 dimensions
    :param str cmap: The name of the colormap to use for the plot.
    :param str norm: The normalization of the colormap:
                     'linear' (default) or 'log'
    :param float vmin: The value to use for the min of the colormap
    :param float vmax: The value to use for the max of the colormap
    :param bool aspect: True to keep aspect ratio (Default: False)
    :param origin: (ox, oy) The origin of the image in the plot
    :type origin: 2-tuple of floats
    :param scale: (sx, sy) The scale of the image in the plot
                  (i.e., the size of the image's pixel in plot coordinates)
    :type scale: 2-tuple of floats
    :param str title: The title of the Plot widget
    :param str xlabel: The label of the X axis
    :param str ylabel: The label of the Y axis
    """
    plot = Plot2D()
    plot.setGraphTitle(title)
    plot.setGraphXLabel(xlabel)
    plot.setGraphYLabel(ylabel)

    # Update default colormap with input parameters
    colormap = plot.getDefaultColormap()
    if cmap is not None:
        colormap['name'] = cmap
    colormap['normalization'] = norm
    if vmin is not None:
        colormap['vmin'] = vmin
    if vmax is not None:
        colormap['vmax'] = vmax
    if vmin is not None and vmax is not None:
        colormap['autoscale'] = False
    plot.setDefaultColormap(colormap)

    plot.setKeepDataAspectRatio(aspect)

    if data is not None:
        plot.addImage(data, origin=origin, scale=scale)

    plot.show()
    return plot
