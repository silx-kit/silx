# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
"""A :class:`.PlotWidget` with additional toolbars.

The :class:`PlotWindow` is a subclass of :class:`.PlotWidget`.
It provides the plot API fully defined in :class:`.Plot`.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "10/01/2017"

import collections
import logging

from . import PlotWidget
from . import PlotActions
from . import PlotToolButtons
from .PlotTools import PositionInfo
from .Profile import ProfileToolBar
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

        # lazy loaded dock widgets
        self._legendsDockWidget = None
        self._curvesROIDockWidget = None
        self._maskToolsDockWidget = None

        # Init actions
        self.group = qt.QActionGroup(self)
        self.group.setExclusive(False)

        self.resetZoomAction = self.group.addAction(PlotActions.ResetZoomAction(self))
        self.resetZoomAction.setVisible(resetzoom)

        self.zoomInAction = PlotActions.ZoomInAction(self)
        self.addAction(self.zoomInAction)

        self.zoomOutAction = PlotActions.ZoomOutAction(self)
        self.addAction(self.zoomOutAction)

        self.xAxisAutoScaleAction = self.group.addAction(
            PlotActions.XAxisAutoScaleAction(self))
        self.xAxisAutoScaleAction.setVisible(autoScale)

        self.yAxisAutoScaleAction = self.group.addAction(
            PlotActions.YAxisAutoScaleAction(self))
        self.yAxisAutoScaleAction.setVisible(autoScale)

        self.xAxisLogarithmicAction = self.group.addAction(
            PlotActions.XAxisLogarithmicAction(self))
        self.xAxisLogarithmicAction.setVisible(logScale)

        self.yAxisLogarithmicAction = self.group.addAction(
            PlotActions.YAxisLogarithmicAction(self))
        self.yAxisLogarithmicAction.setVisible(logScale)

        self.gridAction = self.group.addAction(
            PlotActions.GridAction(self, gridMode='both'))
        self.gridAction.setVisible(grid)

        self.curveStyleAction = self.group.addAction(PlotActions.CurveStyleAction(self))
        self.curveStyleAction.setVisible(curveStyle)

        self.colormapAction = self.group.addAction(PlotActions.ColormapAction(self))
        self.colormapAction.setVisible(colormap)

        self.keepDataAspectRatioButton = PlotToolButtons.AspectToolButton(
            parent=self, plot=self)
        self.keepDataAspectRatioButton.setVisible(aspectRatio)

        self.yAxisInvertedButton = PlotToolButtons.YAxisOriginToolButton(
            parent=self, plot=self)
        self.yAxisInvertedButton.setVisible(yInverted)

        self.group.addAction(self.getRoiAction())
        self.getRoiAction().setVisible(roi)

        self.group.addAction(self.getMaskAction())
        self.getMaskAction().setVisible(mask)

        self._intensityHistoAction = self.group.addAction(
            PlotActions.PixelIntensitiesHistoAction(self))
        self._intensityHistoAction.setVisible(False)

        self._separator = qt.QAction('separator', self)
        self._separator.setSeparator(True)
        self.group.addAction(self._separator)

        self.copyAction = self.group.addAction(PlotActions.CopyAction(self))
        self.copyAction.setVisible(copy)

        self.saveAction = self.group.addAction(PlotActions.SaveAction(self))
        self.saveAction.setVisible(save)

        self.printAction = self.group.addAction(PlotActions.PrintAction(self))
        self.printAction.setVisible(print_)

        self.fitAction = self.group.addAction(PlotActions.FitAction(self))
        self.fitAction.setVisible(fit)

        # lazy loaded actions needed by the controlButton menu
        self.consoleAction = None
        self.panWithArrowKeysAction = None
        self.crosshairAction = None

        if control or position:
            hbox = qt.QHBoxLayout()
            hbox.setContentsMargins(0, 0, 0, 0)

            if control:
                self.controlButton = qt.QToolButton()
                self.controlButton.setText("Options")
                self.controlButton.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
                self.controlButton.setAutoRaise(True)
                self.controlButton.setPopupMode(qt.QToolButton.InstantPopup)
                menu = qt.QMenu(self)
                menu.aboutToShow.connect(self._customControlButtonMenu)
                self.controlButton.setMenu(menu)

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

    def getSelectionMask(self):
        """Return the current mask handled by :attr:`maskToolsDockWidget`.

        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return self.getMaskToolsDockWidget().getSelectionMask()

    def setSelectionMask(self, mask):
        """Set the mask handled by :attr:`maskToolsDockWidget`.

        If the provided mask has not the same dimension as the 'active'
        image, it will by cropped or padded.

        :param mask: The array to use for the mask.
        :type mask: numpy.ndarray of uint8 of dimension 2, C-contiguous.
                    Array of other types are converted.
        :return: True if success, False if failed
        """
        return bool(self.getMaskToolsDockWidget().setSelectionMask(mask))

    def _toggleConsoleVisibility(self, is_checked=False):
        """Create IPythonDockWidget if needed,
        show it or hide it."""
        # create widget if needed (first call)
        if not hasattr(self, '_consoleDockWidget'):
            available_vars = {"plt": self}
            banner = "The variable 'plt' is available. Use the 'whos' "
            banner += "and 'help(plt)' commands for more information.\n\n"
            self._consoleDockWidget = IPythonDockWidget(
                available_vars=available_vars,
                custom_banner=banner,
                parent=self)
            self._introduceNewDockWidget(self._consoleDockWidget)
            self._consoleDockWidget.visibilityChanged.connect(
                    self.getConsoleAction().setChecked)

        self._consoleDockWidget.setVisible(is_checked)

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

    def _customControlButtonMenu(self):
        """Display Options button sub-menu."""
        controlMenu = self.controlButton.menu()
        controlMenu.clear()
        controlMenu.addAction(self.getLegendsDockWidget().toggleViewAction())
        controlMenu.addAction(self.getRoiAction())
        controlMenu.addAction(self.getMaskAction())
        controlMenu.addAction(self.getConsoleAction())

        controlMenu.addSeparator()
        controlMenu.addAction(self.getCrosshairAction())
        controlMenu.addAction(self.getPanWithArrowKeysAction())

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

    # getters for dock widgets
    def getLegendsDockWidget(self):
        """DockWidget with Legend panel"""
        if self._legendsDockWidget is None:
            self._legendsDockWidget = LegendsDockWidget(plot=self)
            self._legendsDockWidget.hide()
            self._introduceNewDockWidget(self._legendsDockWidget)
        return self._legendsDockWidget

    def getCurvesRoiDockWidget(self):
        """DockWidget with curves' ROI panel (lazy-loaded)."""
        if self._curvesROIDockWidget is None:
            self._curvesROIDockWidget = CurvesROIDockWidget(
                plot=self, name='Regions Of Interest')
            self._curvesROIDockWidget.hide()
            self._introduceNewDockWidget(self._curvesROIDockWidget)
        return self._curvesROIDockWidget

    def getMaskToolsDockWidget(self):
        """DockWidget with image mask panel (lazy-loaded)."""
        if self._maskToolsDockWidget is None:
            self._maskToolsDockWidget = MaskToolsDockWidget(
                plot=self, name='Mask')
            self._maskToolsDockWidget.hide()
            self._introduceNewDockWidget(self._maskToolsDockWidget)
        return self._maskToolsDockWidget

    # getters for actions
    def getConsoleAction(self):
        """QAction handling the IPython console activation.

        By default, it is connected to a method that initializes the
        console widget the first time the user clicks the "Console" menu
        button. The following clicks, after initialization is done,
        will toggle the visibility of the console widget.

        :rtype: QAction
        """
        if self.consoleAction is None:
            self.consoleAction = qt.QAction('Console', self)
            self.consoleAction.setCheckable(True)
            if IPythonDockWidget is not None:
                self.consoleAction.toggled.connect(self._toggleConsoleVisibility)
            else:
                self.consoleAction.setEnabled(False)
        return self.consoleAction

    def getCrosshairAction(self):
        """Action toggling crosshair cursor mode.

        :rtype: PlotActions.PlotAction
        """
        if self.crosshairAction is None:
            self.crosshairAction = PlotActions.CrosshairAction(self, color='red')
        return self.crosshairAction

    def getMaskAction(self):
        """QAction toggling image mask dock widget

        :rtype: QAction
        """
        return self.getMaskToolsDockWidget().toggleViewAction()

    def getPanWithArrowKeysAction(self):
        """Action toggling pan with arrow keys.

        :rtype: PlotActions.PlotAction
        """
        if self.panWithArrowKeysAction is None:
            self.panWithArrowKeysAction = PlotActions.PanWithArrowKeysAction(self)
        return self.panWithArrowKeysAction

    def getRoiAction(self):
        """QAction toggling curve ROI dock widget

        :rtype: QAction
        """
        return self.getCurvesRoiDockWidget().toggleViewAction()

    def getResetZoomAction(self):
        """Action resetting the zoom

        :rtype: PlotActions.PlotAction
        """
        return self.resetZoomAction

    def getZoomInAction(self):
        """Action to zoom in

        :rtype: PlotActions.PlotAction
        """
        return self.zoomInAction

    def getZoomOutAction(self):
        """Action to zoom out

        :rtype: PlotActions.PlotAction
        """
        return self.zoomOutAction

    def getXAxisAutoScaleAction(self):
        """Action to toggle the X axis autoscale on zoom reset

        :rtype: PlotActions.PlotAction
        """
        return self.xAxisAutoScaleAction

    def getYAxisAutoScaleAction(self):
        """Action to toggle the Y axis autoscale on zoom reset

        :rtype: PlotActions.PlotAction
        """
        return self.yAxisAutoScaleAction

    def getXAxisLogarithmicAction(self):
        """Action to toggle logarithmic X axis

        :rtype: PlotActions.PlotAction
        """
        return self.xAxisLogarithmicAction

    def getYAxisLogarithmicAction(self):
        """Action to toggle logarithmic Y axis

        :rtype: PlotActions.PlotAction
        """
        return self.yAxisLogarithmicAction

    def getGridAction(self):
        """Action to toggle the grid visibility in the plot

        :rtype: PlotActions.PlotAction
        """
        return self.gridAction

    def getCurveStyleAction(self):
        """Action to change curve line and markers styles

        :rtype: PlotActions.PlotAction
        """
        return self.curveStyleAction

    def getColormapAction(self):
        """Action open a colormap dialog to change active image
        and default colormap.

        :rtype: PlotActions.PlotAction
        """
        return self.colormapAction

    def getKeepDataAspectRatioButton(self):
        """Button to toggle aspect ratio preservation

        :rtype: PlotToolButtons.AspectToolButton
        """
        return self.keepDataAspectRatioButton

    def getKeepDataAspectRatioAction(self):
        """Action associated to keepDataAspectRatioButton.
        Use this to change the visibility of keepDataAspectRatioButton in the
        toolbar (See :meth:`QToolBar.addWidget` documentation).

        :rtype: PlotActions.PlotAction
        """
        return self.keepDataAspectRatioButton

    def getYAxisInvertedButton(self):
        """Button to switch the Y axis orientation

        :rtype: PlotToolButtons.YAxisOriginToolButton
        """
        return self.yAxisInvertedButton

    def getYAxisInvertedAction(self):
        """Action associated to yAxisInvertedButton.
        Use this to change the visibility yAxisInvertedButton in the toolbar.
        (See :meth:`QToolBar.addWidget` documentation).

        :rtype: PlotActions.PlotAction
        """
        return self.yAxisInvertedAction

    def getIntensityHistogramAction(self):
        """Action toggling the histogram intensity Plot widget

        :rtype: PlotActions.PlotAction
        """
        return self._intensityHistoAction

    def getCopyAction(self):
        """Action to copy plot snapshot to clipboard

        :rtype: PlotActions.PlotAction
        """
        return self.copyAction

    def getSaveAction(self):
        """Action to save plot

        :rtype: PlotActions.PlotAction
        """
        return self.saveAction

    def getPrintAction(self):
        """Action to print plot

        :rtype: PlotActions.PlotAction
        """
        return self.printAction

    def getFitAction(self):
        """Action to fit selected curve

        :rtype: PlotActions.PlotAction
        """
        return self.fitAction


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
            ('Data', self._getImageValue)]

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

        self.addToolBar(self.profile)

    def _getImageValue(self, x, y):
        """Get value of top most image at position (x, y)

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The value at that point or '-'
        """
        value = '-'
        valueZ = - float('inf')

        for image in self.getAllImages():
            data, params = image[0], image[4]
            if params['z'] >= valueZ:  # This image is over the previous one
                ox, oy = params['origin']
                sx, sy = params['scale']
                row, col = (y - oy) / sy, (x - ox) / sx
                if row >= 0 and col >= 0:
                    # Test positive before cast otherwise issue with int(-0.5) = 0
                    row, col = int(row), int(col)
                    if (row < data.shape[0] and col < data.shape[1]):
                        value = data[row, col]
                        valueZ = params['z']
        return value

    def getProfileToolbar(self):
        """Profile tools attached to this plot

        See :class:`silx.gui.plot.Profile.ProfileToolBar`
        """
        return self.profile

    def getProfileWindow(self):
        """Plot window used to display profile curve.

        :return: :class:`Plot1D`
        """
        return self.profile.profileWindow
