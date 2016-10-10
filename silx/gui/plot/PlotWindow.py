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
from silx.gui.plot import PlotActions, PlotToolButtons
from silx.gui import icons
"""A :class:`.PlotWidget` with additionnal toolbars.

The :class:`PlotWindow` is a subclass of :class:`.PlotWidget`.
It provides the plot API fully defined in :class:`.Plot`.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "10/10/2016"

import collections
import logging

from . import PlotWidget
from .PlotActions import *  # noqa
from . import PlotToolButtons
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

    This widgets inherits from :class:`.PlotWidget` and provides its plot API.

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
    - keepDataAspectRatioButton: QToolButton to set keep data aspect ratio.
    - keepDataAspectRatioAction: Action associated to keepDataAspectRatioButton.
      Use this to change the visibility of keepDataAspectRatioButton in the
      toolbar (See :meth:`QToolBar.addWidget` documentation).
    - yAxisInvertedButton: QToolButton to set Y Axis direction.
    - yAxisInvertedAction: Action associated to yAxisInvertedButton.
      Use this to change the visibility yAxisInvertedButton in the toolbar.
      (See :meth:`QToolBar.addWidget` documentation).
    - copyAction: Copy plot snapshot to clipboard
    - saveAction: Save plot
    - printAction: Print plot
    - fitAction: Fit selected curve

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
    :param bool aspectRatio: Toggle visibility of aspect ratio button.
    :param bool yInverted: Toggle visibility of Y axis direction button.
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
    :param bool mask: Toggle visibilty of mask action.
    :param bool fit: Toggle visibilty of fit action.
    """

    def __init__(self, parent=None, backend=None,
                 resetzoom=True, autoScale=True, logScale=True, grid=True,
                 curveStyle=True, colormap=True,
                 aspectRatio=True, yInverted=True,
                 copy=True, save=True, print_=True,
                 control=False, position=False,
                 roi=True, mask=True, fit=False):
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

        self.keepDataAspectRatioButton = PlotToolButtons.AspectToolButton(
            parent=self, plot=self)
        self.keepDataAspectRatioButton.setVisible(aspectRatio)

        self.yAxisInvertedButton = PlotToolButtons.YAxisOriginToolButton(
            parent=self, plot=self)
        self.yAxisInvertedButton.setVisible(yInverted)

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

        self.fitAction = self.group.addAction(FitAction(self))
        self.fitAction.setVisible(fit)

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
                self.positionWidget = PositionInfo(
                    plot=self, converters=converters)
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

        # Creating the toolbar also create actions for toolbuttons
        self._toolbar = self._createToolBar(title='Plot', parent=None)
        self.addToolBar(self._toolbar)

    @property
    def legendsDockWidget(self):
        """DockWidget with Legend panel (lazy-loaded)."""
        if not hasattr(self, '_legendsDockWidget'):
            self._legendsDockWidget = LegendsDockWidget(plot=self)
            self._legendsDockWidget.hide()
            self._introduceNewDockWidget(self._legendsDockWidget)
        return self._legendsDockWidget

    @property
    def curvesROIDockWidget(self):
        """DockWidget with curves' ROI panel (lazy-loaded)."""
        if not hasattr(self, '_curvesROIDockWidget'):
            self._curvesROIDockWidget = CurvesROIDockWidget(
                plot=self, name='Regions Of Interest')
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
            self._maskToolsDockWidget = MaskToolsDockWidget(
                plot=self, name='Mask')
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
            available_vars = {"plt": self}
            banner = "The variable 'plt' is available. Use the 'whos' "
            banner += "and 'help(plt)' commands for more information.\n\n"
            if IPythonDockWidget is not None:
                self._consoleDockWidget = IPythonDockWidget(
                    available_vars=available_vars,
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

    def _createToolBar(self, title, parent):
        """Create a QToolBar from the QAction of the PlotWindow.

        :param str title: The title of the QMenu
        :param qt.QWidget parent: See :class:`QToolBar`
        """
        toolbar = qt.QToolBar(title, parent)

        # Order widgets with actions
        objects = self.group.actions()

        # Add push buttons to list
        index = objects.index(self.colormapAction)
        objects.insert(index + 1, self.keepDataAspectRatioButton)
        objects.insert(index + 2, self.yAxisInvertedButton)

        for obj in objects:
            if isinstance(obj, qt.QAction):
                toolbar.addAction(obj)
            else:
                # Add action for toolbutton in order to allow changing
                # visibility (see doc QToolBar.addWidget doc)
                if obj is self.keepDataAspectRatioButton:
                    self.keepDataAspectRatioAction = toolbar.addWidget(obj)
                elif obj is self.yAxisInvertedButton:
                    self.yAxisInvertedAction = toolbar.addWidget(obj)
                else:
                    raise RuntimeError()
        return toolbar

    def toolBar(self):
        """Return a QToolBar from the QAction of the PlotWindow.
        """
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

    This widgets provides the plot API of :class:`.PlotWidget`.

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
                                     roi=True, mask=False, fit=True)
        if parent is None:
            self.setWindowTitle('Plot1D')
        self.setGraphXLabel('X')
        self.setGraphYLabel('Y')


class Plot2D(PlotWindow):
    """PlotWindow with a toolbar specific for images.

    This widgets provides the plot API of :class:`.PlotWidget`.

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

        self.profile = ProfileToolBar(plot=self)
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
