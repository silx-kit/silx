# /*##########################################################################
#
# Copyright (c) 2004-2022 European Synchrotron Radiation Facility
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
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "12/04/2019"

try:
    from collections import abc
except ImportError:  # Python2 support
    import collections as abc
import logging
import weakref

import silx
from silx.utils.weakref import WeakMethodProxy
from silx.utils.deprecation import deprecated
from silx.utils.proxy import docstring

from . import PlotWidget
from . import actions
from . import items
from .actions import medfilt as actions_medfilt
from .actions import fit as actions_fit
from .actions import control as actions_control
from .actions import histogram as actions_histogram
from . import PlotToolButtons
from . import tools
from .Profile import ProfileToolBar
from .LegendSelector import LegendsDockWidget
from .CurvesROIWidget import CurvesROIDockWidget
from .MaskToolsWidget import MaskToolsDockWidget
from .StatsWidget import BasicStatsWidget
from .ColorBar import ColorBarWidget
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
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
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
                     See :class:`~silx.gui.plot.tools.PositionInfo`.
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
        self._consoleDockWidget = None
        self._statsDockWidget = None

        # Create color bar, hidden by default for backward compatibility
        self._colorbar = ColorBarWidget(parent=self, plot=self)

        # Init actions
        self.group = qt.QActionGroup(self)
        self.group.setExclusive(False)

        self.resetZoomAction = self.group.addAction(
            actions.control.ResetZoomAction(self, parent=self))
        self.resetZoomAction.setVisible(resetzoom)
        self.addAction(self.resetZoomAction)

        self.zoomInAction = actions.control.ZoomInAction(self, parent=self)
        self.addAction(self.zoomInAction)

        self.zoomOutAction = actions.control.ZoomOutAction(self, parent=self)
        self.addAction(self.zoomOutAction)

        self.xAxisAutoScaleAction = self.group.addAction(
            actions.control.XAxisAutoScaleAction(self, parent=self))
        self.xAxisAutoScaleAction.setVisible(autoScale)
        self.addAction(self.xAxisAutoScaleAction)

        self.yAxisAutoScaleAction = self.group.addAction(
            actions.control.YAxisAutoScaleAction(self, parent=self))
        self.yAxisAutoScaleAction.setVisible(autoScale)
        self.addAction(self.yAxisAutoScaleAction)

        self.xAxisLogarithmicAction = self.group.addAction(
            actions.control.XAxisLogarithmicAction(self, parent=self))
        self.xAxisLogarithmicAction.setVisible(logScale)
        self.addAction(self.xAxisLogarithmicAction)

        self.yAxisLogarithmicAction = self.group.addAction(
            actions.control.YAxisLogarithmicAction(self, parent=self))
        self.yAxisLogarithmicAction.setVisible(logScale)
        self.addAction(self.yAxisLogarithmicAction)

        self.gridAction = self.group.addAction(
            actions.control.GridAction(self, gridMode='both', parent=self))
        self.gridAction.setVisible(grid)
        self.addAction(self.gridAction)

        self.curveStyleAction = self.group.addAction(
            actions.control.CurveStyleAction(self, parent=self))
        self.curveStyleAction.setVisible(curveStyle)
        self.addAction(self.curveStyleAction)

        self.colormapAction = self.group.addAction(
            actions.control.ColormapAction(self, parent=self))
        self.colormapAction.setVisible(colormap)
        self.addAction(self.colormapAction)

        self.colorbarAction = self.group.addAction(
            actions_control.ColorBarAction(self, parent=self))
        self.colorbarAction.setVisible(False)
        self.addAction(self.colorbarAction)
        self._colorbar.setVisible(False)

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
            actions_histogram.PixelIntensitiesHistoAction(self, parent=self))
        self._intensityHistoAction.setVisible(False)

        self._medianFilter2DAction = self.group.addAction(
            actions_medfilt.MedianFilter2DAction(self, parent=self))
        self._medianFilter2DAction.setVisible(False)

        self._medianFilter1DAction = self.group.addAction(
            actions_medfilt.MedianFilter1DAction(self, parent=self))
        self._medianFilter1DAction.setVisible(False)

        self.fitAction = self.group.addAction(actions_fit.FitAction(self, parent=self))
        self.fitAction.setVisible(fit)
        self.addAction(self.fitAction)

        # lazy loaded actions needed by the controlButton menu
        self._consoleAction = None
        self._panWithArrowKeysAction = None
        self._crosshairAction = None

        # Make colorbar background white
        self._colorbar.setAutoFillBackground(True)
        self._sigAxesVisibilityChanged.connect(self._updateColorBarBackground)
        self._updateColorBarBackground()

        if control:  # Create control button only if requested
            self.controlButton = qt.QToolButton()
            self.controlButton.setText("Options")
            self.controlButton.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
            self.controlButton.setAutoRaise(True)
            self.controlButton.setPopupMode(qt.QToolButton.InstantPopup)
            menu = qt.QMenu(self)
            menu.aboutToShow.connect(self._customControlButtonMenu)
            self.controlButton.setMenu(menu)

        self._positionWidget = None
        if position:  # Add PositionInfo widget to the bottom of the plot
            if isinstance(position, abc.Iterable):
                # Use position as a set of converters
                converters = position
            else:
                converters = None
            self._positionWidget = tools.PositionInfo(
                plot=self, converters=converters)
            # Set a snapping mode that is consistent with legacy one
            self._positionWidget.setSnappingMode(
                tools.PositionInfo.SNAPPING_CROSSHAIR |
                tools.PositionInfo.SNAPPING_ACTIVE_ONLY |
                tools.PositionInfo.SNAPPING_SYMBOLS_ONLY |
                tools.PositionInfo.SNAPPING_CURVE |
                tools.PositionInfo.SNAPPING_SCATTER)

        self.__setCentralWidget()

        # Creating the toolbar also create actions for toolbuttons
        self._interactiveModeToolBar = tools.InteractiveModeToolBar(
            parent=self, plot=self)
        self.addToolBar(self._interactiveModeToolBar)

        self._toolbar = self._createToolBar(title='Plot', parent=self)
        self.addToolBar(self._toolbar)

        self._outputToolBar = tools.OutputToolBar(parent=self, plot=self)
        self._outputToolBar.getCopyAction().setVisible(copy)
        self._outputToolBar.getSaveAction().setVisible(save)
        self._outputToolBar.getPrintAction().setVisible(print_)
        self.addToolBar(self._outputToolBar)

        # Activate shortcuts in PlotWindow widget:
        for toolbar in (self._interactiveModeToolBar, self._outputToolBar):
            for action in toolbar.actions():
                self.addAction(action)

    def __setCentralWidget(self):
        """Set central widget to host plot backend, colorbar, and bottom bar"""
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.addWidget(self.getWidgetHandle(), 0, 0)
        gridLayout.addWidget(self._colorbar, 0, 1)
        gridLayout.setRowStretch(0, 1)
        gridLayout.setColumnStretch(0, 1)
        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(gridLayout)

        if hasattr(self, "controlButton") or self._positionWidget is not None:
            hbox = qt.QHBoxLayout()
            hbox.setContentsMargins(0, 0, 0, 0)

            if hasattr(self, "controlButton"):
                hbox.addWidget(self.controlButton)

            if self._positionWidget is not None:
                hbox.addWidget(self._positionWidget)

            hbox.addStretch(1)
            bottomBar = qt.QWidget(centralWidget)
            bottomBar.setLayout(hbox)

            gridLayout.addWidget(bottomBar, 1, 0, 1, -1)

        self.setCentralWidget(centralWidget)

    @docstring(PlotWidget)
    def setBackend(self, backend):
        super(PlotWindow, self).setBackend(backend)
        self.__setCentralWidget()  # Recreate PlotWindow's central widget

    @docstring(PlotWidget)
    def setBackgroundColor(self, color):
        super(PlotWindow, self).setBackgroundColor(color)
        self._updateColorBarBackground()

    @docstring(PlotWidget)
    def setDataBackgroundColor(self, color):
        super(PlotWindow, self).setDataBackgroundColor(color)
        self._updateColorBarBackground()

    @docstring(PlotWidget)
    def setForegroundColor(self, color):
        super(PlotWindow, self).setForegroundColor(color)
        self._updateColorBarBackground()

    def _updateColorBarBackground(self):
        """Update the colorbar background according to the state of the plot"""
        if self.isAxesDisplayed():
            color = self.getBackgroundColor()
        else:
            color = self.getDataBackgroundColor()
            if not color.isValid():
                # If no color defined, use the background one
                color = self.getBackgroundColor()

        foreground = self.getForegroundColor()

        palette = self._colorbar.palette()
        palette.setColor(qt.QPalette.Window, color)
        palette.setColor(qt.QPalette.WindowText, foreground)
        palette.setColor(qt.QPalette.Text, foreground)
        self._colorbar.setPalette(palette)

    def getInteractiveModeToolBar(self):
        """Returns QToolBar controlling interactive mode.

        :rtype: QToolBar
        """
        return self._interactiveModeToolBar

    def getOutputToolBar(self):
        """Returns QToolBar containing save, copy and print actions

        :rtype: QToolBar
        """
        return self._outputToolBar

    @property
    @deprecated(replacement="getPositionInfoWidget()", since_version="0.8.0")
    def positionWidget(self):
        return self.getPositionInfoWidget()

    def getPositionInfoWidget(self):
        """Returns the widget displaying current cursor position information

        :rtype: ~silx.gui.plot.tools.PositionInfo
        """
        return self._positionWidget

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

    def _toggleConsoleVisibility(self, isChecked=False):
        """Create IPythonDockWidget if needed,
        show it or hide it."""
        # create widget if needed (first call)
        if self._consoleDockWidget is None:
            available_vars = {"plt": weakref.proxy(self)}
            banner = "The variable 'plt' is available. Use the 'whos' "
            banner += "and 'help(plt)' commands for more information.\n\n"
            self._consoleDockWidget = IPythonDockWidget(
                available_vars=available_vars,
                custom_banner=banner,
                parent=self)
            self.addTabbedDockWidget(self._consoleDockWidget)
            self._consoleDockWidget.toggleViewAction().toggled.connect(
                self._consoleDockWidgetToggled)

        self._consoleDockWidget.setVisible(isChecked)

    def _consoleVisibilityTriggered(self, isChecked):
        if isChecked and self.isVisible():
            self._consoleDockWidget.show()
            self._consoleDockWidget.raise_()

    def _consoleDockWidgetToggled(self, isChecked):
        if self.isVisible():
            self.getConsoleAction().setChecked(isChecked)

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
        controlMenu.addAction(self.getStatsAction())
        controlMenu.addAction(self.getMaskAction())
        controlMenu.addAction(self.getConsoleAction())

        controlMenu.addSeparator()
        controlMenu.addAction(self.getCrosshairAction())
        controlMenu.addAction(self.getPanWithArrowKeysAction())

    def addTabbedDockWidget(self, dock_widget):
        """Add a dock widget as a new tab if there are already dock widgets
        in the plot. When the first tab is added, the area is chosen
        depending on the plot geometry:
        if the window is much wider than it is high, the right dock area
        is used, else the bottom dock area is used.

        :param dock_widget: Instance of :class:`QDockWidget` to be added.
        """
        if dock_widget not in self._dockWidgets:
            self._dockWidgets.append(dock_widget)
        if len(self._dockWidgets) == 1:
            # The first created dock widget must be added to a Widget area
            width = self.centralWidget().width()
            height = self.centralWidget().height()
            if width > (1.25 * height):
                area = qt.Qt.RightDockWidgetArea
            else:
                area = qt.Qt.BottomDockWidgetArea
            self.addDockWidget(area, dock_widget)
        else:
            # Other dock widgets are added as tabs to the same widget area
            self.tabifyDockWidget(self._dockWidgets[0],
                                  dock_widget)

    def removeDockWidget(self, dockwidget):
        """Removes the *dockwidget* from the main window layout and hides it.

        Note that the *dockwidget* is *not* deleted.

        :param QDockWidget dockwidget:
        """
        if dockwidget in self._dockWidgets:
            self._dockWidgets.remove(dockwidget)
        super(PlotWindow, self).removeDockWidget(dockwidget)

    def _handleFirstDockWidgetShow(self, visible):
        """Handle QDockWidget.visibilityChanged

        It calls :meth:`addTabbedDockWidget` for the `sender` widget.
        This allows to call `addTabbedDockWidget` lazily.

        It disconnect itself from the signal once done.

        :param bool visible:
        """
        if visible:
            dockWidget = self.sender()
            dockWidget.visibilityChanged.disconnect(
                self._handleFirstDockWidgetShow)
            self.addTabbedDockWidget(dockWidget)

    def _handleDockWidgetViewActionTriggered(self, checked):
        if checked:
            action = self.sender()
            if action is None:
                return
            dockWidget = action.parent()
            if dockWidget is None:
                return
            dockWidget.show()  # Show needed here for raise to have an effect
            dockWidget.raise_()

    def getColorBarWidget(self):
        """Returns the embedded :class:`ColorBarWidget` widget.

        :rtype: ColorBarWidget
        """
        return self._colorbar

    # getters for dock widgets

    def getLegendsDockWidget(self):
        """DockWidget with Legend panel"""
        if self._legendsDockWidget is None:
            self._legendsDockWidget = LegendsDockWidget(plot=self)
            self._legendsDockWidget.hide()
            self._legendsDockWidget.toggleViewAction().triggered.connect(
                self._handleDockWidgetViewActionTriggered)
            self._legendsDockWidget.visibilityChanged.connect(
                self._handleFirstDockWidgetShow)
        return self._legendsDockWidget

    def getCurvesRoiDockWidget(self):
        # Undocumented for a "soft deprecation" in version 0.7.0
        # (still used internally for lazy loading)
        if self._curvesROIDockWidget is None:
            self._curvesROIDockWidget = CurvesROIDockWidget(
                plot=self, name='Regions Of Interest')
            self._curvesROIDockWidget.hide()
            self._curvesROIDockWidget.toggleViewAction().triggered.connect(
                self._handleDockWidgetViewActionTriggered)
            self._curvesROIDockWidget.visibilityChanged.connect(
                self._handleFirstDockWidgetShow)
        return self._curvesROIDockWidget

    def getCurvesRoiWidget(self):
        """Return the :class:`CurvesROIWidget`.

        :class:`silx.gui.plot.CurvesROIWidget.CurvesROIWidget` offers a getter
        and a setter for the ROI data:

            - :meth:`CurvesROIWidget.getRois`
            - :meth:`CurvesROIWidget.setRois`
        """
        return self.getCurvesRoiDockWidget().roiWidget

    def getMaskToolsDockWidget(self):
        """DockWidget with image mask panel (lazy-loaded)."""
        if self._maskToolsDockWidget is None:
            self._maskToolsDockWidget = MaskToolsDockWidget(
                plot=self, name='Mask')
            self._maskToolsDockWidget.hide()
            self._maskToolsDockWidget.toggleViewAction().triggered.connect(
                self._handleDockWidgetViewActionTriggered)
            self._maskToolsDockWidget.visibilityChanged.connect(
                self._handleFirstDockWidgetShow)
        return self._maskToolsDockWidget

    def getStatsWidget(self):
        """Returns a BasicStatsWidget connected to this plot

        :rtype: BasicStatsWidget
        """
        if self._statsDockWidget is None:
            self._statsDockWidget = qt.QDockWidget()
            self._statsDockWidget.setWindowTitle("Curves stats")
            self._statsDockWidget.layout().setContentsMargins(0, 0, 0, 0)
            statsWidget = BasicStatsWidget(parent=self, plot=self)
            self._statsDockWidget.setWidget(statsWidget)
            self._statsDockWidget.hide()
            self._statsDockWidget.toggleViewAction().triggered.connect(
                self._handleDockWidgetViewActionTriggered)
            self._statsDockWidget.visibilityChanged.connect(
                self._handleFirstDockWidgetShow)
        return self._statsDockWidget.widget()

    # getters for actions
    @property
    @deprecated(replacement="getInteractiveModeToolBar().getZoomModeAction()",
                since_version="0.8.0")
    def zoomModeAction(self):
        return self.getInteractiveModeToolBar().getZoomModeAction()

    @property
    @deprecated(replacement="getInteractiveModeToolBar().getPanModeAction()",
                since_version="0.8.0")
    def panModeAction(self):
        return self.getInteractiveModeToolBar().getPanModeAction()

    def getConsoleAction(self):
        """QAction handling the IPython console activation.

        By default, it is connected to a method that initializes the
        console widget the first time the user clicks the "Console" menu
        button. The following clicks, after initialization is done,
        will toggle the visibility of the console widget.

        :rtype: QAction
        """
        if self._consoleAction is None:
            self._consoleAction = qt.QAction('Console', self)
            self._consoleAction.setCheckable(True)
            if IPythonDockWidget is not None:
                self._consoleAction.toggled.connect(self._toggleConsoleVisibility)
                self._consoleAction.triggered.connect(self._consoleVisibilityTriggered)

            else:
                self._consoleAction.setEnabled(False)
        return self._consoleAction

    def getCrosshairAction(self):
        """Action toggling crosshair cursor mode.

        :rtype: actions.PlotAction
        """
        if self._crosshairAction is None:
            self._crosshairAction = actions.control.CrosshairAction(self, color='red')
        return self._crosshairAction

    def getMaskAction(self):
        """QAction toggling image mask dock widget

        :rtype: QAction
        """
        return self.getMaskToolsDockWidget().toggleViewAction()

    def getPanWithArrowKeysAction(self):
        """Action toggling pan with arrow keys.

        :rtype: actions.PlotAction
        """
        if self._panWithArrowKeysAction is None:
            self._panWithArrowKeysAction = actions.control.PanWithArrowKeysAction(self)
        return self._panWithArrowKeysAction

    def getStatsAction(self):
        return self.getStatsWidget().parent().toggleViewAction()

    def getRoiAction(self):
        """QAction toggling curve ROI dock widget

        :rtype: QAction
        """
        return self.getCurvesRoiDockWidget().toggleViewAction()

    def getResetZoomAction(self):
        """Action resetting the zoom

        :rtype: actions.PlotAction
        """
        return self.resetZoomAction

    def getZoomInAction(self):
        """Action to zoom in

        :rtype: actions.PlotAction
        """
        return self.zoomInAction

    def getZoomOutAction(self):
        """Action to zoom out

        :rtype: actions.PlotAction
        """
        return self.zoomOutAction

    def getXAxisAutoScaleAction(self):
        """Action to toggle the X axis autoscale on zoom reset

        :rtype: actions.PlotAction
        """
        return self.xAxisAutoScaleAction

    def getYAxisAutoScaleAction(self):
        """Action to toggle the Y axis autoscale on zoom reset

        :rtype: actions.PlotAction
        """
        return self.yAxisAutoScaleAction

    def getXAxisLogarithmicAction(self):
        """Action to toggle logarithmic X axis

        :rtype: actions.PlotAction
        """
        return self.xAxisLogarithmicAction

    def getYAxisLogarithmicAction(self):
        """Action to toggle logarithmic Y axis

        :rtype: actions.PlotAction
        """
        return self.yAxisLogarithmicAction

    def getGridAction(self):
        """Action to toggle the grid visibility in the plot

        :rtype: actions.PlotAction
        """
        return self.gridAction

    def getCurveStyleAction(self):
        """Action to change curve line and markers styles

        :rtype: actions.PlotAction
        """
        return self.curveStyleAction

    def getColormapAction(self):
        """Action open a colormap dialog to change active image
        and default colormap.

        :rtype: actions.PlotAction
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

        :rtype: actions.PlotAction
        """
        return self.keepDataAspectRatioAction

    def getYAxisInvertedButton(self):
        """Button to switch the Y axis orientation

        :rtype: PlotToolButtons.YAxisOriginToolButton
        """
        return self.yAxisInvertedButton

    def getYAxisInvertedAction(self):
        """Action associated to yAxisInvertedButton.
        Use this to change the visibility yAxisInvertedButton in the toolbar.
        (See :meth:`QToolBar.addWidget` documentation).

        :rtype: actions.PlotAction
        """
        return self.yAxisInvertedAction

    def getIntensityHistogramAction(self):
        """Action toggling the histogram intensity Plot widget

        :rtype: actions.PlotAction
        """
        return self._intensityHistoAction

    def getCopyAction(self):
        """Action to copy plot snapshot to clipboard

        :rtype: actions.PlotAction
        """
        return self.getOutputToolBar().getCopyAction()

    def getSaveAction(self):
        """Action to save plot

        :rtype: actions.PlotAction
        """
        return self.getOutputToolBar().getSaveAction()

    def getPrintAction(self):
        """Action to print plot

        :rtype: actions.PlotAction
        """
        return self.getOutputToolBar().getPrintAction()

    def getFitAction(self):
        """Action to fit selected curve

        :rtype: actions.PlotAction
        """
        return self.fitAction

    def getMedianFilter1DAction(self):
        """Action toggling the 1D median filter

        :rtype: actions.PlotAction
        """
        return self._medianFilter1DAction

    def getMedianFilter2DAction(self):
        """Action toggling the 2D median filter

        :rtype: actions.PlotAction
        """
        return self._medianFilter2DAction

    def getColorBarAction(self):
        """Action toggling the colorbar show/hide action

        .. warning:: to show/hide the plot colorbar call directly the ColorBar
            widget using getColorBarWidget()

        :rtype: actions.PlotAction
        """
        return self.colorbarAction


class Plot1D(PlotWindow):
    """PlotWindow with tools specific for curves.

    This widgets provides the plot API of :class:`.PlotWidget`.

    :param parent: The parent of this widget
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    def __init__(self, parent=None, backend=None):
        super(Plot1D, self).__init__(parent=parent, backend=backend,
                                     resetzoom=True, autoScale=True,
                                     logScale=True, grid=True,
                                     curveStyle=True, colormap=False,
                                     aspectRatio=False, yInverted=False,
                                     copy=True, save=True, print_=True,
                                     control=True, position=True,
                                     roi=True, mask=False, fit=True)
        if parent is None:
            self.setWindowTitle('Plot1D')
        self.getXAxis().setLabel('X')
        self.getYAxis().setLabel('Y')
        action = self.getFitAction()
        action.setXRangeUpdatedOnZoom(True)
        action.setFittedItemUpdatedFromActiveCurve(True)


class Plot2D(PlotWindow):
    """PlotWindow with a toolbar specific for images.

    This widgets provides the plot API of :~:`.PlotWidget`.

    :param parent: The parent of this widget
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    def __init__(self, parent=None, backend=None):
        # List of information to display at the bottom of the plot
        posInfo = [
            ('X', lambda x, y: x),
            ('Y', lambda x, y: y),
            ('Data', WeakMethodProxy(self._getImageValue)),
            ('Dims', WeakMethodProxy(self._getImageDims)),
        ]

        super(Plot2D, self).__init__(parent=parent, backend=backend,
                                     resetzoom=True, autoScale=False,
                                     logScale=False, grid=False,
                                     curveStyle=False, colormap=True,
                                     aspectRatio=True, yInverted=True,
                                     copy=True, save=True, print_=True,
                                     control=False, position=posInfo,
                                     roi=False, mask=True)
        if parent is None:
            self.setWindowTitle('Plot2D')
        self.getXAxis().setLabel('Columns')
        self.getYAxis().setLabel('Rows')

        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == 'downward':
            self.getYAxis().setInverted(True)

        self.profile = ProfileToolBar(plot=self)
        self.addToolBar(self.profile)

        self.colorbarAction.setVisible(True)
        self.getColorBarWidget().setVisible(True)

        # Put colorbar action after colormap action
        actions = self.toolBar().actions()
        for action in actions:
            if action is self.getColormapAction():
                break

        self.sigActiveImageChanged.connect(self.__activeImageChanged)

    def __activeImageChanged(self, previous, legend):
        """Handle change of active image

        :param Union[str,None] previous: Legend of previous active image
        :param Union[str,None] legend: Legend of current active image
        """
        if previous is not None:
            item = self.getImage(previous)
            if item is not None:
                item.sigItemChanged.disconnect(self.__imageChanged)

        if legend is not None:
            item = self.getImage(legend)
            item.sigItemChanged.connect(self.__imageChanged)

        positionInfo = self.getPositionInfoWidget()
        if positionInfo is not None:
            positionInfo.updateInfo()

    def __imageChanged(self, event):
        """Handle update of active image item

        :param event: Type of changed event
        """
        if event == items.ItemChangedType.DATA:
            positionInfo = self.getPositionInfoWidget()
            if positionInfo is not None:
                positionInfo.updateInfo()

    def _getImageValue(self, x, y):
        """Get status bar value of top most image at position (x, y)

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The value at that point or '-'
        """
        pickedMask = None
        for picked in self.pickItems(
                *self.dataToPixel(x, y, check=False),
                lambda item: isinstance(item, items.ImageBase)):
            if isinstance(picked.getItem(), items.MaskImageData):
                if pickedMask is None:  # Use top-most if many masks
                    pickedMask = picked
            else:
                image = picked.getItem()

                indices = picked.getIndices(copy=False)
                if indices is not None:
                    row, col = indices[0][0], indices[1][0]
                    value = image.getData(copy=False)[row, col]

                    if pickedMask is not None:  # Check if masked
                        maskItem = pickedMask.getItem()
                        indices = pickedMask.getIndices()
                        row, col = indices[0][0], indices[1][0]
                        if maskItem.getData(copy=False)[row, col] != 0:
                            return value, "Masked"
                    return value

        return '-'  # No image picked

    def _getImageDims(self, *args):
        activeImage = self.getActiveImage()
        if (activeImage is not None and
                    activeImage.getData(copy=False) is not None):
            dims = activeImage.getData(copy=False).shape[1::-1]
            return 'x'.join(str(dim) for dim in dims)
        else:
            return '-'

    def getProfileToolbar(self):
        """Profile tools attached to this plot

        See :class:`silx.gui.plot.Profile.ProfileToolBar`
        """
        return self.profile

    @deprecated(replacement="getProfilePlot", since_version="0.5.0")
    def getProfileWindow(self):
        return self.getProfilePlot()

    def getProfilePlot(self):
        """Return plot window used to display profile curve.

        :return: :class:`Plot1D`
        """
        return self.profile.getProfilePlot()
