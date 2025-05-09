# /*##########################################################################
#
# Copyright (c) 2004-2023 European Synchrotron Radiation Facility
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
"""
:mod:`silx.gui.plot.actions.control` provides a set of QAction relative to control
of a :class:`.PlotWidget`.

The following QAction are available:

- :class:`ColormapAction`
- :class:`CrosshairAction`
- :class:`CurveStyleAction`
- :class:`GridAction`
- :class:`KeepAspectRatioAction`
- :class:`PanWithArrowKeysAction`
- :class:`ResetZoomAction`
- :class:`ShowAxisAction`
- :class:`XAxisLogarithmicAction`
- :class:`XAxisAutoScaleAction`
- :class:`YAxisInvertedAction`
- :class:`YAxisLogarithmicAction`
- :class:`YAxisAutoScaleAction`
- :class:`ZoomBackAction`
- :class:`ZoomInAction`
- :class:`ZoomOutAction`
"""

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "27/11/2020"

from . import PlotAction
import logging
from silx.gui.plot import items
from silx.gui.plot._utils import applyZoomToPlot as _applyZoomToPlot
from silx.gui import qt
from silx.gui import icons
from silx.utils.deprecation import deprecated

_logger = logging.getLogger(__name__)


class ResetZoomAction(PlotAction):
    """QAction controlling reset zoom on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="zoom-original",
            text="Reset Zoom",
            tooltip="Auto-scale the graph",
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent,
        )
        self._autoscaleChanged(True)
        plot.getXAxis().sigAutoScaleChanged.connect(self._autoscaleChanged)
        plot.getYAxis().sigAutoScaleChanged.connect(self._autoscaleChanged)

    def _autoscaleChanged(self, enabled):
        xAxis = self.plot.getXAxis()
        yAxis = self.plot.getYAxis()
        self.setEnabled(xAxis.isAutoScale() or yAxis.isAutoScale())

        if xAxis.isAutoScale() and yAxis.isAutoScale():
            tooltip = "Auto-scale the graph"
        elif xAxis.isAutoScale():  # And not Y axis
            tooltip = "Auto-scale the x-axis of the graph only"
        elif yAxis.isAutoScale():  # And not X axis
            tooltip = "Auto-scale the y-axis of the graph only"
        else:  # no axis in autoscale
            tooltip = "Auto-scale the graph"
        self.setToolTip(tooltip)

    def _actionTriggered(self, checked=False):
        self.plot.resetZoom()


class ZoomBackAction(PlotAction):
    """QAction performing a zoom-back in :class:`.PlotWidget` limits history.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="zoom-back",
            text="Zoom Back",
            tooltip="Zoom back the plot",
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent,
        )
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def _actionTriggered(self, checked=False):
        self.plot.getLimitsHistory().pop()


class ZoomInAction(PlotAction):
    """QAction performing a zoom-in on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="zoom-in",
            text="Zoom In",
            tooltip="Zoom in the plot",
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent,
        )
        self.setShortcut(qt.QKeySequence.ZoomIn)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def _actionTriggered(self, checked=False):
        _applyZoomToPlot(self.plot, 1.1)


class ZoomOutAction(PlotAction):
    """QAction performing a zoom-out on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="zoom-out",
            text="Zoom Out",
            tooltip="Zoom out the plot",
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent,
        )
        self.setShortcut(qt.QKeySequence.ZoomOut)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def _actionTriggered(self, checked=False):
        _applyZoomToPlot(self.plot, 1.0 / 1.1)


class XAxisAutoScaleAction(PlotAction):
    """QAction controlling X axis autoscale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="plot-xauto",
            text="X Autoscale",
            tooltip="Enable x-axis auto-scale when checked.\n"
            "If unchecked, x-axis does not change when reseting zoom.",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.setChecked(plot.getXAxis().isAutoScale())
        plot.getXAxis().sigAutoScaleChanged.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.getXAxis().setAutoScale(checked)
        if checked:
            self.plot.resetZoom()


class YAxisAutoScaleAction(PlotAction):
    """QAction controlling Y axis autoscale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="plot-yauto",
            text="Y Autoscale",
            tooltip="Enable y-axis auto-scale when checked.\n"
            "If unchecked, y-axis does not change when reseting zoom.",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.setChecked(plot.getYAxis().isAutoScale())
        plot.getYAxis().sigAutoScaleChanged.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.getYAxis().setAutoScale(checked)
        if checked:
            self.plot.resetZoom()


class XAxisLogarithmicAction(PlotAction):
    """QAction controlling X axis log scale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="plot-xlog",
            text="X Log. scale",
            tooltip="Logarithmic x-axis when checked",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.axis = plot.getXAxis()
        self.setChecked(self.axis.getScale() == self.axis.LOGARITHMIC)
        self.axis.sigScaleChanged.connect(self._setCheckedIfLogScale)

    def _setCheckedIfLogScale(self, scale):
        self.setChecked(scale == self.axis.LOGARITHMIC)

    def _actionTriggered(self, checked=False):
        scale = self.axis.LOGARITHMIC if checked else self.axis.LINEAR
        self.axis.setScale(scale)


class YAxisLogarithmicAction(PlotAction):
    """QAction controlling Y axis log scale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="plot-ylog",
            text="Y Log. scale",
            tooltip="Logarithmic y-axis when checked",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.axis = plot.getYAxis()
        self.setChecked(self.axis.getScale() == self.axis.LOGARITHMIC)
        self.axis.sigScaleChanged.connect(self._setCheckedIfLogScale)

    def _setCheckedIfLogScale(self, scale):
        self.setChecked(scale == self.axis.LOGARITHMIC)

    def _actionTriggered(self, checked=False):
        scale = self.axis.LOGARITHMIC if checked else self.axis.LINEAR
        self.axis.setScale(scale)


class GridAction(PlotAction):
    """QAction controlling grid mode on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param str gridMode: The grid mode to use in 'both', 'major'.
                         See :meth:`.PlotWidget.setGraphGrid`
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, gridMode="both", parent=None):
        assert gridMode in ("both", "major")
        self._gridMode = gridMode

        super().__init__(
            plot,
            icon="plot-grid",
            text="Grid",
            tooltip="Toggle grid (on/off)",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.setChecked(plot.getGraphGrid() is not None)
        plot.sigSetGraphGrid.connect(self._gridChanged)

    def _gridChanged(self, which):
        """Slot listening for PlotWidget grid mode change."""
        self.setChecked(which != "None")

    def _actionTriggered(self, checked=False):
        self.plot.setGraphGrid(self._gridMode if checked else None)


class CurveStyleAction(PlotAction):
    """QAction controlling curve style on a :class:`.PlotWidget`.

    It changes the default line and markers style which updates all
    curves on the plot.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="plot-toggle-points",
            text="Curve style",
            tooltip="Change curve line and markers style",
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent,
        )

    def _actionTriggered(self, checked=False):
        currentState = (self.plot.isDefaultPlotLines(), self.plot.isDefaultPlotPoints())

        if currentState == (False, False):
            newState = True, False
        else:
            # line only, line and symbol, symbol only
            states = (True, False), (True, True), (False, True)
            newState = states[(states.index(currentState) + 1) % 3]

        self.plot.setDefaultPlotLines(newState[0])
        self.plot.setDefaultPlotPoints(newState[1])


class ColormapAction(PlotAction):
    """QAction opening a ColormapDialog to update the colormap.

    Both the active image colormap and the default colormap are updated.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        self._dialog = None  # To store an instance of ColormapDialog
        super().__init__(
            plot,
            icon="colormap",
            text="Colormap",
            tooltip="Change colormap",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.plot.sigActiveImageChanged.connect(self._updateColormap)
        self.plot.sigActiveScatterChanged.connect(self._updateColormap)

    def setColormapDialog(self, dialog):
        """Set a specific colormap dialog instead of using the default one."""
        assert dialog is not None
        if self._dialog is not None:
            self._dialog.visibleChanged.disconnect(self._dialogVisibleChanged)

        self._dialog = dialog
        self._dialog.visibleChanged.connect(
            self._dialogVisibleChanged, qt.Qt.UniqueConnection
        )
        self.setChecked(self._dialog.isVisible())

    @deprecated(replacement="setColormapDialog", since_version="2.0")
    def setColorDialog(self, colorDialog):
        self.setColormapDialog(colorDialog)

    def getColormapDialog(self):
        if self._dialog is None:
            self._dialog = self._createDialog(self.plot)
            self._dialog.visibleChanged.connect(self._dialogVisibleChanged)
        return self._dialog

    @staticmethod
    def _createDialog(parent):
        """Create the dialog if not already existing

        :parent QWidget parent: Parent of the new colormap
        :rtype: ColormapDialog
        """
        from silx.gui.dialog.ColormapDialog import ColormapDialog

        dialog = ColormapDialog(parent=parent)
        dialog.setModal(False)
        return dialog

    def _actionTriggered(self, checked=False):
        """Create a cmap dialog and update active image and default cmap."""
        dialog = self.getColormapDialog()
        # Run the dialog listening to colormap change
        if checked is True:
            self._updateColormap()
            dialog.show()
        else:
            dialog.hide()

    def _dialogVisibleChanged(self, isVisible):
        self.setChecked(isVisible)

    def _updateColormap(self):
        if self._dialog is None:
            return
        image = self.plot.getActiveImage()

        if isinstance(image, items.ColormapMixIn):
            # Set dialog from active image
            colormap = image.getColormap()
            # Set histogram and range if any
            self._dialog.setItem(image)

        else:
            # No active image or active image is RGBA,
            # Check for active scatter plot
            scatter = self.plot.getActiveScatter()
            if scatter is not None:
                colormap = scatter.getColormap()
                self._dialog.setItem(scatter)

            else:
                # No active data image nor scatter,
                # set dialog from default info
                colormap = self.plot.getDefaultColormap()
                # Reset histogram and range if any
                self._dialog.setData(None)

        self._dialog.setColormap(colormap)


class ColorBarAction(PlotAction):
    """QAction opening the ColorBarWidget of the specified plot.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        self._dialog = None  # To store an instance of ColorBar
        super().__init__(
            plot,
            icon="colorbar",
            text="Colorbar",
            tooltip="Show/Hide the colorbar",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        colorBarWidget = self.plot.getColorBarWidget()
        old = self.blockSignals(True)
        self.setChecked(colorBarWidget.isVisibleTo(self.plot))
        self.blockSignals(old)
        colorBarWidget.sigVisibleChanged.connect(self._widgetVisibleChanged)

    def _widgetVisibleChanged(self, isVisible):
        """Callback when the colorbar `visible` property change."""
        if self.isChecked() == isVisible:
            return
        self.setChecked(isVisible)

    def _actionTriggered(self, checked=False):
        """Create a cmap dialog and update active image and default cmap."""
        colorBarWidget = self.plot.getColorBarWidget()
        if not colorBarWidget.isHidden() == checked:
            return
        self.plot.getColorBarWidget().setVisible(checked)


class KeepAspectRatioAction(PlotAction):
    """QAction controlling aspect ratio on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        # Uses two images for checked/unchecked states
        self._states = {
            False: (icons.getQIcon("shape-circle-solid"), "Keep data aspect ratio"),
            True: (
                icons.getQIcon("shape-ellipse-solid"),
                "Do no keep data aspect ratio",
            ),
        }

        icon, tooltip = self._states[plot.isKeepDataAspectRatio()]
        super().__init__(
            plot,
            icon=icon,
            text="Toggle keep aspect ratio",
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent,
        )
        plot.sigSetKeepDataAspectRatio.connect(self._keepDataAspectRatioChanged)

    def _keepDataAspectRatioChanged(self, aspectRatio):
        """Handle Plot set keep aspect ratio signal"""
        icon, tooltip = self._states[aspectRatio]
        self.setIcon(icon)
        self.setToolTip(tooltip)

    def _actionTriggered(self, checked=False):
        # This will trigger _keepDataAspectRatioChanged
        self.plot.setKeepDataAspectRatio(not self.plot.isKeepDataAspectRatio())


class YAxisInvertedAction(PlotAction):
    """QAction controlling Y orientation on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        # Uses two images for checked/unchecked states
        self._states = {
            False: (icons.getQIcon("plot-ydown"), "Orient Y axis downward"),
            True: (icons.getQIcon("plot-yup"), "Orient Y axis upward"),
        }

        icon, tooltip = self._states[plot.getYAxis().isInverted()]
        super().__init__(
            plot,
            icon=icon,
            text="Invert Y Axis",
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent,
        )
        plot.getYAxis().sigInvertedChanged.connect(self._yAxisInvertedChanged)

    def _yAxisInvertedChanged(self, inverted):
        """Handle Plot set y axis inverted signal"""
        icon, tooltip = self._states[inverted]
        self.setIcon(icon)
        self.setToolTip(tooltip)

    def _actionTriggered(self, checked=False):
        # This will trigger _yAxisInvertedChanged
        yAxis = self.plot.getYAxis()
        yAxis.setInverted(not yAxis.isInverted())


class CrosshairAction(PlotAction):
    """QAction toggling crosshair cursor on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param str color: Color to use to draw the crosshair
    :param int linewidth: Width of the crosshair cursor
    :param str linestyle: Style of line. See :meth:`.Plot.setGraphCursor`
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, color="black", linewidth=1, linestyle="-", parent=None):
        self.color = color
        """Color used to draw the crosshair (str)."""

        self.linewidth = linewidth
        """Width of the crosshair cursor (int)."""

        self.linestyle = linestyle
        """Style of line of the cursor (str)."""

        super().__init__(
            plot,
            icon="crosshair",
            text="Crosshair Cursor",
            tooltip="Enable crosshair cursor when checked",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.setChecked(plot.getGraphCursor() is not None)
        plot.sigSetGraphCursor.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setGraphCursor(
            checked,
            color=self.color,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
        )


class PanWithArrowKeysAction(PlotAction):
    """QAction toggling pan with arrow keys on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="arrow-keys",
            text="Pan with arrow keys",
            tooltip="Enable pan with arrow keys when checked",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.setChecked(plot.isPanWithArrowKeys())
        plot.sigSetPanWithArrowKeys.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setPanWithArrowKeys(checked)


class ShowAxisAction(PlotAction):
    """QAction controlling axis visibility on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        tooltip = "Show plot axis when checked, otherwise hide them"
        PlotAction.__init__(
            self,
            plot,
            icon="axis",
            text="show axis",
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.setChecked(self.plot.isAxesDisplayed())
        plot._sigAxesVisibilityChanged.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setAxesDisplayed(checked)


class ClosePolygonInteractionAction(PlotAction):
    """QAction controlling closure of a polygon in draw interaction mode
    if the :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        tooltip = "Close the current polygon drawn"
        PlotAction.__init__(
            self,
            plot,
            icon="add-shape-polygon",
            text="Close the polygon",
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.plot.sigInteractiveModeChanged.connect(self._modeChanged)
        self._modeChanged(None)

    def _modeChanged(self, source):
        mode = self.plot.getInteractiveMode()
        enabled = "shape" in mode and mode["shape"] == "polygon"
        self.setEnabled(enabled)

    def _actionTriggered(self, checked=False):
        self.plot.interaction()._validate()


class OpenGLAction(PlotAction):
    """QAction controlling rendering of a :class:`.PlotWidget`.

    For now it can enable or not the OpenGL backend.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        # Uses two images for checked/unchecked states
        self._states = {
            "opengl": (
                icons.getQIcon("backend-opengl"),
                "OpenGL rendering (fast)\nClick to disable OpenGL",
            ),
            "matplotlib": (
                icons.getQIcon("backend-opengl"),
                "Matplotlib rendering (safe)\nClick to enable OpenGL",
            ),
            "unknown": (icons.getQIcon("backend-opengl"), "Custom rendering"),
        }

        name = self._getBackendName(plot)
        icon, tooltip = self._states[name]
        super().__init__(
            plot,
            icon=icon,
            text="Enable/disable OpenGL rendering",
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        plot.sigBackendChanged.connect(self._backendUpdated)

    def _backendUpdated(self):
        name = self._getBackendName(self.plot)
        icon, tooltip = self._states[name]
        self.setIcon(icon)
        self.setToolTip(tooltip)
        self.setChecked(name == "opengl")

    def _getBackendName(self, plot):
        backend = plot.getBackend()
        name = type(backend).__name__.lower()
        if "opengl" in name:
            return "opengl"
        elif "matplotlib" in name:
            return "matplotlib"
        else:
            return "unknown"

    def _actionTriggered(self, checked=False):
        plot = self.plot
        name = self._getBackendName(self.plot)
        if name != "opengl":
            from silx.gui.utils import glutils

            result = glutils.isOpenGLAvailable()
            if not result:
                qt.QMessageBox.critical(
                    plot, "OpenGL rendering is not available", result.error
                )
                return
            plot.setBackend("opengl")
        else:
            plot.setBackend("matplotlib")
