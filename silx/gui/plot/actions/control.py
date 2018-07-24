# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
- :class:`XAxisLogarithmicAction`
- :class:`XAxisAutoScaleAction`
- :class:`YAxisInvertedAction`
- :class:`YAxisLogarithmicAction`
- :class:`YAxisAutoScaleAction`
- :class:`ZoomBackAction`
- :class:`ZoomInAction`
- :class:`ZoomOutAction`
- :class:'ShowAxisAction'
"""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "24/04/2018"

from . import PlotAction
import logging
from silx.gui.plot import items
from silx.gui.plot._utils import applyZoomToPlot as _applyZoomToPlot
from silx.gui import qt
from silx.gui import icons

_logger = logging.getLogger(__name__)


class ResetZoomAction(PlotAction):
    """QAction controlling reset zoom on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(ResetZoomAction, self).__init__(
            plot, icon='zoom-original', text='Reset Zoom',
            tooltip='Auto-scale the graph',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)
        self._autoscaleChanged(True)
        plot.getXAxis().sigAutoScaleChanged.connect(self._autoscaleChanged)
        plot.getYAxis().sigAutoScaleChanged.connect(self._autoscaleChanged)

    def _autoscaleChanged(self, enabled):
        xAxis = self.plot.getXAxis()
        yAxis = self.plot.getYAxis()
        self.setEnabled(xAxis.isAutoScale() or yAxis.isAutoScale())

        if xAxis.isAutoScale() and yAxis.isAutoScale():
            tooltip = 'Auto-scale the graph'
        elif xAxis.isAutoScale():  # And not Y axis
            tooltip = 'Auto-scale the x-axis of the graph only'
        elif yAxis.isAutoScale():  # And not X axis
            tooltip = 'Auto-scale the y-axis of the graph only'
        else:  # no axis in autoscale
            tooltip = 'Auto-scale the graph'
        self.setToolTip(tooltip)

    def _actionTriggered(self, checked=False):
        self.plot.resetZoom()


class ZoomBackAction(PlotAction):
    """QAction performing a zoom-back in :class:`.PlotWidget` limits history.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(ZoomBackAction, self).__init__(
            plot, icon='zoom-back', text='Zoom Back',
            tooltip='Zoom back the plot',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def _actionTriggered(self, checked=False):
        self.plot.getLimitsHistory().pop()


class ZoomInAction(PlotAction):
    """QAction performing a zoom-in on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(ZoomInAction, self).__init__(
            plot, icon='zoom-in', text='Zoom In',
            tooltip='Zoom in the plot',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)
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
        super(ZoomOutAction, self).__init__(
            plot, icon='zoom-out', text='Zoom Out',
            tooltip='Zoom out the plot',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.ZoomOut)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def _actionTriggered(self, checked=False):
        _applyZoomToPlot(self.plot, 1. / 1.1)


class XAxisAutoScaleAction(PlotAction):
    """QAction controlling X axis autoscale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(XAxisAutoScaleAction, self).__init__(
            plot, icon='plot-xauto', text='X Autoscale',
            tooltip='Enable x-axis auto-scale when checked.\n'
                    'If unchecked, x-axis does not change when reseting zoom.',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
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
        super(YAxisAutoScaleAction, self).__init__(
            plot, icon='plot-yauto', text='Y Autoscale',
            tooltip='Enable y-axis auto-scale when checked.\n'
                    'If unchecked, y-axis does not change when reseting zoom.',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
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
        super(XAxisLogarithmicAction, self).__init__(
            plot, icon='plot-xlog', text='X Log. scale',
            tooltip='Logarithmic x-axis when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
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
        super(YAxisLogarithmicAction, self).__init__(
            plot, icon='plot-ylog', text='Y Log. scale',
            tooltip='Logarithmic y-axis when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
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

    def __init__(self, plot, gridMode='both', parent=None):
        assert gridMode in ('both', 'major')
        self._gridMode = gridMode

        super(GridAction, self).__init__(
            plot, icon='plot-grid', text='Grid',
            tooltip='Toggle grid (on/off)',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.getGraphGrid() is not None)
        plot.sigSetGraphGrid.connect(self._gridChanged)

    def _gridChanged(self, which):
        """Slot listening for PlotWidget grid mode change."""
        self.setChecked(which != 'None')

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
        super(CurveStyleAction, self).__init__(
            plot, icon='plot-toggle-points', text='Curve style',
            tooltip='Change curve line and markers style',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)

    def _actionTriggered(self, checked=False):
        currentState = (self.plot.isDefaultPlotLines(),
                        self.plot.isDefaultPlotPoints())

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
        super(ColormapAction, self).__init__(
            plot, icon='colormap', text='Colormap',
            tooltip="Change colormap",
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.plot.sigActiveImageChanged.connect(self._updateColormap)
        self.plot.sigActiveScatterChanged.connect(self._updateColormap)

    def setColorDialog(self, colorDialog):
        """Set a specific color dialog instead of using the default dialog."""
        assert(colorDialog is not None)
        assert(self._dialog is None)
        self._dialog = colorDialog
        self._dialog.visibleChanged.connect(self._dialogVisibleChanged)
        self.setChecked(self._dialog.isVisible())

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
        if self._dialog is None:
            self._dialog = self._createDialog(self.plot)
            self._dialog.visibleChanged.connect(self._dialogVisibleChanged)

        # Run the dialog listening to colormap change
        if checked is True:
            self._dialog.show()
            self._updateColormap()
        else:
            self._dialog.hide()

    def _dialogVisibleChanged(self, isVisible):
        self.setChecked(isVisible)

    def _updateColormap(self):
        if self._dialog is None:
            return
        image = self.plot.getActiveImage()

        if isinstance(image, items.ImageComplexData):
            # Specific init for complex images
            colormap = image.getColormap()

            mode = image.getVisualizationMode()
            if mode in (items.ImageComplexData.Mode.AMPLITUDE_PHASE,
                        items.ImageComplexData.Mode.LOG10_AMPLITUDE_PHASE):
                data = image.getData(
                    copy=False, mode=items.ImageComplexData.Mode.PHASE)
            else:
                data = image.getData(copy=False)

            # Set histogram and range if any
            self._dialog.setData(data)

        elif isinstance(image, items.ColormapMixIn):
            # Set dialog from active image
            colormap = image.getColormap()
            data = image.getData(copy=False)
            # Set histogram and range if any
            self._dialog.setData(data)

        else:
            # No active image or active image is RGBA,
            # Check for active scatter plot
            scatter = self.plot._getActiveItem(kind='scatter')
            if scatter is not None:
                colormap = scatter.getColormap()
                data = scatter.getValueData(copy=False)
                self._dialog.setData(data)

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
        super(ColorBarAction, self).__init__(
            plot, icon='colorbar', text='Colorbar',
            tooltip="Show/Hide the colorbar",
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
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
            False: (icons.getQIcon('shape-circle-solid'),
                    "Keep data aspect ratio"),
            True: (icons.getQIcon('shape-ellipse-solid'),
                   "Do no keep data aspect ratio")
        }

        icon, tooltip = self._states[plot.isKeepDataAspectRatio()]
        super(KeepAspectRatioAction, self).__init__(
            plot,
            icon=icon,
            text='Toggle keep aspect ratio',
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent)
        plot.sigSetKeepDataAspectRatio.connect(
            self._keepDataAspectRatioChanged)

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
            False: (icons.getQIcon('plot-ydown'),
                    "Orient Y axis downward"),
            True: (icons.getQIcon('plot-yup'),
                   "Orient Y axis upward"),
        }

        icon, tooltip = self._states[plot.getYAxis().isInverted()]
        super(YAxisInvertedAction, self).__init__(
            plot,
            icon=icon,
            text='Invert Y Axis',
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent)
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

    def __init__(self, plot, color='black', linewidth=1, linestyle='-',
                 parent=None):
        self.color = color
        """Color used to draw the crosshair (str)."""

        self.linewidth = linewidth
        """Width of the crosshair cursor (int)."""

        self.linestyle = linestyle
        """Style of line of the cursor (str)."""

        super(CrosshairAction, self).__init__(
            plot, icon='crosshair', text='Crosshair Cursor',
            tooltip='Enable crosshair cursor when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.getGraphCursor() is not None)
        plot.sigSetGraphCursor.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setGraphCursor(checked,
                                 color=self.color,
                                 linestyle=self.linestyle,
                                 linewidth=self.linewidth)


class PanWithArrowKeysAction(PlotAction):
    """QAction toggling pan with arrow keys on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):

        super(PanWithArrowKeysAction, self).__init__(
            plot, icon='arrow-keys', text='Pan with arrow keys',
            tooltip='Enable pan with arrow keys when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
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
        tooltip = 'Show plot axis when checked, otherwise hide them'
        PlotAction.__init__(self,
                            plot,
                            icon='axis',
                            text='show axis',
                            tooltip=tooltip,
                            triggered=self._actionTriggered,
                            checkable=True,
                            parent=parent)
        self.setChecked(self.plot._backend.isAxesDisplayed())
        plot._sigAxesVisibilityChanged.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setAxesDisplayed(checked)

