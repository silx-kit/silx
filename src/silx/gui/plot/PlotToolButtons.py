# /*##########################################################################
#
# Copyright (c) 2004-2020 European Synchrotron Radiation Facility
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
"""This module provides a set of QToolButton to use with
:class:`~silx.gui.plot.PlotWidget`.

The following QToolButton are available:

- :class:`.AspectToolButton`
- :class:`.YAxisOriginToolButton`
- :class:`.ProfileToolButton`
- :class:`.RulerToolButton`
- :class:`.SymbolToolButton`

"""

__authors__ = ["V. Valls", "H. Payno"]
__license__ = "MIT"
__date__ = "27/06/2017"


import functools
import logging

from ... import config
from ...utils.deprecation import deprecated_warning
from .. import icons, qt
from ._utils.axis_orientation_states import X_AXIS_STATE, Y_AXIS_STATE, AxisState
from .items import Scatter, SymbolMixIn
from .items.axis import XAxis, YAxis, YRightAxis
from .PlotWidget import PlotWidget
from .tools.PlotToolButton import PlotToolButton
import qtawesome

_logger = logging.getLogger(__name__)

ICON_YSCALE = "mdi6.arrow-up-down"
ICON_XSCALE = "mdi.arrow-left-right"
ICON_LINEAR = "ph.line-segment-bold"
ICON_LOG = "mdi.math-log"
ICON_ASINH = "mdi6.math-sin"
ICONS_OPTIONS = {
    "options": [
        {"offset": (-0.25, 0.0), "scale_factor": 0.7},
        {"offset": (0.25, 0.0), "scale_factor": 0.7},
    ],
}


class AspectToolButton(PlotToolButton):
    """Tool button to switch keep aspect ratio of a plot"""

    STATE = None
    """Lazy loaded states used to feed AspectToolButton"""

    def __init__(self, parent=None, plot=None):
        if self.STATE is None:
            self.STATE = {}
            # dont keep ratio
            self.STATE[False, "icon"] = icons.getQIcon("shape-ellipse-solid")
            self.STATE[False, "state"] = "Aspect ratio is not kept"
            self.STATE[False, "action"] = "Do no keep data aspect ratio"
            # keep ratio
            self.STATE[True, "icon"] = icons.getQIcon("shape-circle-solid")
            self.STATE[True, "state"] = "Aspect ratio is kept"
            self.STATE[True, "action"] = "Keep data aspect ratio"

        super().__init__(parent=parent, plot=plot)

        keepAction = self._createAction(True)
        keepAction.triggered.connect(self.keepDataAspectRatio)
        keepAction.setIconVisibleInMenu(True)

        dontKeepAction = self._createAction(False)
        dontKeepAction.triggered.connect(self.dontKeepDataAspectRatio)
        dontKeepAction.setIconVisibleInMenu(True)

        menu = qt.QMenu(self)
        menu.addAction(keepAction)
        menu.addAction(dontKeepAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)

    def _createAction(self, keepAspectRatio):
        icon = self.STATE[keepAspectRatio, "icon"]
        text = self.STATE[keepAspectRatio, "action"]
        return qt.QAction(icon, text, self)

    def _connectPlot(self, plot):
        plot.sigSetKeepDataAspectRatio.connect(self._keepDataAspectRatioChanged)
        self._keepDataAspectRatioChanged(plot.isKeepDataAspectRatio())

    def _disconnectPlot(self, plot):
        plot.sigSetKeepDataAspectRatio.disconnect(self._keepDataAspectRatioChanged)

    def keepDataAspectRatio(self):
        """Configure the plot to keep the aspect ratio"""
        plot = self.plot()
        if plot is not None:
            # This will trigger _keepDataAspectRatioChanged
            plot.setKeepDataAspectRatio(True)

    def dontKeepDataAspectRatio(self):
        """Configure the plot to not keep the aspect ratio"""
        plot = self.plot()
        if plot is not None:
            # This will trigger _keepDataAspectRatioChanged
            plot.setKeepDataAspectRatio(False)

    def _keepDataAspectRatioChanged(self, aspectRatio):
        """Handle Plot set keep aspect ratio signal"""
        icon, toolTip = (
            self.STATE[aspectRatio, "icon"],
            self.STATE[aspectRatio, "state"],
        )
        self.setIcon(icon)
        self.setToolTip(toolTip)


class _AxisOriginToolButton(PlotToolButton):
    """Tool button to switch the axis orientation of a plot."""

    def __init__(
        self, parent: qt.QWidget | None = None, plot: PlotWidget | None = None
    ):
        super().__init__(parent=parent, plot=plot)

        disableInversionAction = self._createAction(False)
        disableInversionAction.triggered.connect(self._disableInversion)
        disableInversionAction.setIconVisibleInMenu(True)

        enableInversionAction = self._createAction(True)
        enableInversionAction.triggered.connect(self._enableInversion)
        enableInversionAction.setIconVisibleInMenu(True)

        menu = qt.QMenu(self)
        menu.addAction(disableInversionAction)
        menu.addAction(enableInversionAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)

    def _getAxis(self, plot: PlotWidget):
        raise NotImplementedError()

    def _getState(self, inverted: bool) -> AxisState:
        raise NotImplementedError()

    def _createAction(self, inverted: bool) -> qt.QAction:
        state = self._getState(inverted)
        return qt.QAction(icons.getQIcon(state["icon"]), state["action"], self)

    def _connectPlot(self, plot: PlotWidget):
        axis = self._getAxis(plot)
        axis.sigInvertedChanged.connect(self._axisInvertedChanged)
        self._axisInvertedChanged(axis.isInverted())

    def _disconnectPlot(self, plot: PlotWidget):
        self._getAxis(plot).sigInvertedChanged.disconnect(self._axisInvertedChanged)

    def setAxisInverted(self, inverted: bool):
        """Invert the axis"""
        plot = self.plot()
        if plot is None:
            return
        axis = self._getAxis(plot)
        if axis is not None:
            # This will trigger _axisInvertedChanged
            axis.setInverted(inverted)

    def _enableInversion(self):
        # Wrapper function to avoid mixed python/qt circular reference
        # https://github.com/silx-kit/silx/pull/4425#discussion_r2378901544
        self.setAxisInverted(True)

    def _disableInversion(self):
        # Wrapper function to avoid mixed python/qt circular reference
        # https://github.com/silx-kit/silx/pull/4425#discussion_r2378901544
        self.setAxisInverted(False)

    def _axisInvertedChanged(self, inverted: bool):
        state = self._getState(inverted)
        self.setIcon(icons.getQIcon(state["icon"]))
        self.setToolTip(state["state"])


class XAxisOriginToolButton(_AxisOriginToolButton):
    def _getAxis(self, plot: PlotWidget) -> XAxis:
        return plot.getXAxis()

    def _getState(self, inverted: bool) -> AxisState:
        return X_AXIS_STATE[inverted]


class YAxisOriginToolButton(_AxisOriginToolButton):
    def _getAxis(self, plot: PlotWidget) -> YAxis | YRightAxis:
        return plot.getYAxis()

    def _getState(self, inverted: bool) -> AxisState:
        return Y_AXIS_STATE[inverted]

    def setYAxisUpward(self):
        deprecated_warning(
            "Method",
            name="setYAxisUpward",
            since_version="3.0.0",
            replacement="setAxisInverted(False)",
        )
        self.setAxisInverted(False)

    def setYAxisDownward(self):
        deprecated_warning(
            "Method",
            name="setYAxisDownward",
            since_version="3.0.0",
            replacement="setAxisInverted(True)",
        )
        self.setAxisInverted(True)


class YAxisScaleToolButton(PlotToolButton):
    """Tool button to choose the Y axis scale in the plot."""

    STATE = None
    """Lazy loaded states used to feed YAxisScaleToolButton"""

    def __init__(self, parent=None, plot=None):
        if self.STATE is None:
            self.STATE = {}
            # linear
            self.STATE["linear", "icon"] = qtawesome.icon(
                ICON_YSCALE, ICON_LINEAR, **ICONS_OPTIONS
            )
            self.STATE["linear", "state"] = "Y-axis scale is linear"
            self.STATE["linear", "action"] = "Linear Y-axis"
            # logarithmic
            self.STATE["log", "icon"] = qtawesome.icon(
                ICON_YSCALE, ICON_LOG, **ICONS_OPTIONS
            )
            self.STATE["log", "state"] = "Y-axis scale is logarithmic"
            self.STATE["log", "action"] = "Logarithmic Y-axis"
            # arcsinh
            self.STATE["asinh", "icon"] = qtawesome.icon(
                ICON_YSCALE, ICON_ASINH, **ICONS_OPTIONS
            )
            self.STATE["asinh", "state"] = "Y-axis scale is arcsinh"
            self.STATE["asinh", "action"] = "Arcsinh Y-axis"

        super().__init__(parent=parent, plot=plot)

        ylinearaction = self._createAction("linear")
        ylinearaction.triggered.connect(self._setYAxisScaleLinear)
        ylinearaction.setIconVisibleInMenu(True)

        ylogAction = self._createAction("log")
        ylogAction.triggered.connect(self.setYAxisScaleLog)
        ylogAction.setIconVisibleInMenu(True)

        yasinhAction = self._createAction("asinh")
        yasinhAction.triggered.connect(self.setYAxisScaleAsinh)
        yasinhAction.setIconVisibleInMenu(True)

        menu = qt.QMenu(self)
        menu.addAction(ylinearaction)
        menu.addAction(ylogAction)
        menu.addAction(yasinhAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)

    def _createAction(self, state):
        icon = self.STATE[state, "icon"]
        text = self.STATE[state, "action"]
        return qt.QAction(icon, text, self)

    def _connectPlot(self, plot):
        yAxis = plot.getYAxis()
        yAxis.sigScaleChanged.connect(self._yAxisScaleChanged)
        self._yAxisScaleChanged(yAxis.getScale())

    def _disconnectPlot(self, plot):
        plot.getYAxis().sigScaleChanged.disconnect(self._yAxisScaleChanged)

    def _setYAxisScaleLinear(self):
        """Set Y-scale to Linear"""
        self.plot().getYAxis().setScale(scale=YAxis.LINEAR)

    def _setYAxisScaleLog(self):
        """Set Y-scale to Log"""
        self.plot().getYAxis().setScale(scale=YAxis.LOGARITHMIC)

    def _setYAxisScaleAsinh(self):
        """Set Y-scale to Arcsinh"""
        self.plot().getYAxis().setScale(scale=YAxis.ARCSINH)

    def _yAxisScaleChanged(self, scale_state):
        """Handle Plot set y scale"""
        icon, toolTip = (
            self.STATE[scale_state, "icon"],
            self.STATE[scale_state, "state"],
        )
        self.setIcon(icon)
        self.setToolTip(toolTip)


class XAxisScaleToolButton(PlotToolButton):
    """Tool button to choose the X axis scale in the plot."""

    STATE = None
    """Lazy loaded states used to feed XAxisScaleToolButton"""

    def __init__(self, parent=None, plot=None):
        if self.STATE is None:
            self.STATE = {}
            # linear
            self.STATE["linear", "icon"] = qtawesome.icon(
                ICON_XSCALE, ICON_LINEAR, **ICONS_OPTIONS
            )
            self.STATE["linear", "state"] = "X-axis scale is linear"
            self.STATE["linear", "action"] = "Linear X-axis"
            # logarithmic
            self.STATE["log", "icon"] = qtawesome.icon(
                ICON_XSCALE, ICON_LOG, **ICONS_OPTIONS
            )
            self.STATE["log", "state"] = "X-axis scale is logarithmic"
            self.STATE["log", "action"] = "Logarithmic X-axis"
            # arcsinh
            self.STATE["asinh", "icon"] = qtawesome.icon(
                ICON_XSCALE, ICON_ASINH, **ICONS_OPTIONS
            )
            self.STATE["asinh", "state"] = "X-axis scale is arcsinh"
            self.STATE["asinh", "action"] = "Arcsinh X-axis"

        super().__init__(parent=parent, plot=plot)

        xlinearaction = self._createAction(XAxis.LINEAR)
        xlinearaction.triggered.connect(self._setXAxisScaleLinear)
        xlinearaction.setIconVisibleInMenu(True)

        xlogAction = self._createAction(XAxis.LOGARITHMIC)
        xlogAction.triggered.connect(self._setXAxisScaleLog)
        xlogAction.setIconVisibleInMenu(True)

        xasinhAction = self._createAction(XAxis.ARCSINH)
        xasinhAction.triggered.connect(self.setXAxisScaleAsinh)
        xasinhAction.setIconVisibleInMenu(True)

        menu = qt.QMenu(self)
        menu.addAction(xlinearaction)
        menu.addAction(xlogAction)
        menu.addAction(xasinhAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)

    def _createAction(self, state):
        icon = self.STATE[state, "icon"]
        text = self.STATE[state, "action"]
        return qt.QAction(icon, text, self)

    def _connectPlot(self, plot):
        xAxis = plot.getXAxis()
        xAxis.sigScaleChanged.connect(self._xAxisScaleChanged)
        self._xAxisScaleChanged(xAxis.getScale())

    def _disconnectPlot(self, plot):
        plot.getYAxis().sigScaleChanged.disconnect(self._xAxisScaleChanged)

    def _setXAxisScaleLinear(self):
        """Set X-scale to Linear"""
        self.plot().getXAxis().setScale(scale=XAxis.LINEAR)

    def _setXAxisScaleLog(self):
        """Set X-scale to Log"""
        self.plot().getXAxis().setScale(scale=XAxis.LOGARITHMIC)

    def _setXAxisScaleAsinh(self):
        """Set X-scale to Arcsinh"""
        self.plot().getXAxis().setScale(scale=XAxis.ARCSINH)

    def _xAxisScaleChanged(self, scale_state):
        """Handle Plot set y scale"""
        icon, toolTip = (
            self.STATE[scale_state, "icon"],
            self.STATE[scale_state, "state"],
        )
        self.setIcon(icon)
        self.setToolTip(toolTip)


class ProfileOptionToolButton(PlotToolButton):
    """Button to define option on the profile"""

    sigMethodChanged = qt.Signal(str)

    def __init__(self, parent=None, plot=None):
        PlotToolButton.__init__(self, parent=parent, plot=plot)

        self.STATE = {}
        # is down
        self.STATE["sum", "icon"] = icons.getQIcon("math-sigma")
        self.STATE["sum", "state"] = "Compute profile sum"
        self.STATE["sum", "action"] = "Compute profile sum"
        # keep ration
        self.STATE["mean", "icon"] = icons.getQIcon("math-mean")
        self.STATE["mean", "state"] = "Compute profile mean"
        self.STATE["mean", "action"] = "Compute profile mean"

        self.sumAction = self._createAction("sum")
        self.sumAction.triggered.connect(self.setSum)
        self.sumAction.setIconVisibleInMenu(True)
        self.sumAction.setCheckable(True)
        self.sumAction.setChecked(True)

        self.meanAction = self._createAction("mean")
        self.meanAction.triggered.connect(self.setMean)
        self.meanAction.setIconVisibleInMenu(True)
        self.meanAction.setCheckable(True)

        menu = qt.QMenu(self)
        menu.addAction(self.sumAction)
        menu.addAction(self.meanAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)
        self._method = "mean"
        self._update()

    def _createAction(self, method):
        icon = self.STATE[method, "icon"]
        text = self.STATE[method, "action"]
        return qt.QAction(icon, text, self)

    def setSum(self):
        self.setMethod("sum")

    def _update(self):
        icon = self.STATE[self._method, "icon"]
        toolTip = self.STATE[self._method, "state"]
        self.setIcon(icon)
        self.setToolTip(toolTip)
        self.sumAction.setChecked(self._method == "sum")
        self.meanAction.setChecked(self._method == "mean")

    def setMean(self):
        self.setMethod("mean")

    def setMethod(self, method):
        """Set the method to use.

        :param str method: Either 'sum' or 'mean'
        """
        if method != self._method:
            if method in ("sum", "mean"):
                self._method = method
                self.sigMethodChanged.emit(self._method)
                self._update()
            else:
                _logger.warning("Unsupported method '%s'. Setting ignored.", method)

    def getMethod(self):
        """Returns the current method in use (See :meth:`setMethod`).

        :rtype: str
        """
        return self._method


class ProfileToolButton(PlotToolButton):
    """Button used in Profile3DToolbar to switch between 2D profile
    and 1D profile."""

    STATE = None
    """Lazy loaded states used to feed ProfileToolButton"""

    sigDimensionChanged = qt.Signal(int)

    def __init__(self, parent=None, plot=None):
        if self.STATE is None:
            self.STATE = {
                (1, "icon"): icons.getQIcon("profile1D"),
                (1, "state"): "1D profile is computed on visible image",
                (1, "action"): "1D profile on visible image",
                (2, "icon"): icons.getQIcon("profile2D"),
                (
                    2,
                    "state",
                ): "2D profile is computed, one 1D profile for each image in the stack",
                (2, "action"): "2D profile on image stack",
            }
            # Compute 1D profile
            # Compute 2D profile

        super().__init__(parent=parent, plot=plot)

        self._dimension = 1

        profile1DAction = self._createAction(1)
        profile1DAction.triggered.connect(self.computeProfileIn1D)
        profile1DAction.setIconVisibleInMenu(True)
        profile1DAction.setCheckable(True)
        profile1DAction.setChecked(True)
        self._profile1DAction = profile1DAction

        profile2DAction = self._createAction(2)
        profile2DAction.triggered.connect(self.computeProfileIn2D)
        profile2DAction.setIconVisibleInMenu(True)
        profile2DAction.setCheckable(True)
        self._profile2DAction = profile2DAction

        menu = qt.QMenu(self)
        menu.addAction(profile1DAction)
        menu.addAction(profile2DAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)
        menu.setTitle("Select profile dimension")
        self.computeProfileIn1D()

    def _createAction(self, profileDimension):
        icon = self.STATE[profileDimension, "icon"]
        text = self.STATE[profileDimension, "action"]
        return qt.QAction(icon, text, self)

    def _profileDimensionChanged(self, profileDimension):
        """Update icon in toolbar, emit number of dimensions for profile"""
        self.setIcon(self.STATE[profileDimension, "icon"])
        self.setToolTip(self.STATE[profileDimension, "state"])
        self._dimension = profileDimension
        self.sigDimensionChanged.emit(profileDimension)
        self._profile1DAction.setChecked(profileDimension == 1)
        self._profile2DAction.setChecked(profileDimension == 2)

    def computeProfileIn1D(self):
        self._profileDimensionChanged(1)

    def computeProfileIn2D(self):
        self._profileDimensionChanged(2)

    def setDimension(self, dimension):
        """Set the selected dimension"""
        assert dimension in [1, 2]
        if self._dimension == dimension:
            return
        if dimension == 1:
            self.computeProfileIn1D()
        elif dimension == 2:
            self.computeProfileIn2D()
        else:
            _logger.warning("Unsupported dimension '%s'. Setting ignored.", dimension)

    def getDimension(self):
        """Get the selected dimension.

        :rtype: int (1 or 2)
        """
        return self._dimension


class _SymbolToolButtonBase(PlotToolButton):
    """Base class for PlotToolButton setting marker and size.

    :param parent: See QWidget
    :param plot: The `~silx.gui.plot.PlotWidget` to control
    """

    def __init__(self, parent=None, plot=None):
        super().__init__(parent=parent, plot=plot)

    def _addSizeSliderToMenu(self, menu):
        """Add a slider to set size to the given menu

        :param QMenu menu:
        """
        slider = qt.QSlider(qt.Qt.Horizontal)
        slider.setRange(1, 20)
        slider.setValue(int(config.DEFAULT_PLOT_SYMBOL_SIZE))
        slider.setTracking(False)
        slider.valueChanged.connect(self._sizeChanged)
        widgetAction = qt.QWidgetAction(menu)
        widgetAction.setDefaultWidget(slider)
        menu.addAction(widgetAction)

    def _addSymbolsToMenu(self, menu):
        """Add symbols to the given menu

        :param QMenu menu:
        """
        for marker, name in zip(
            SymbolMixIn.getSupportedSymbols(), SymbolMixIn.getSupportedSymbolNames()
        ):
            action = qt.QAction(name, menu)
            action.setCheckable(False)
            action.triggered.connect(functools.partial(self._markerChanged, marker))
            menu.addAction(action)

    def _sizeChanged(self, value):
        """Manage slider value changed

        :param int value: Marker size
        """
        plot = self.plot()
        if plot is None:
            return

        for item in plot.getItems():
            if isinstance(item, SymbolMixIn) and item.isSingleSymbolSize():
                item.setSymbolSize(value)

    def _markerChanged(self, marker):
        """Manage change of marker.

        :param str marker: Letter describing the marker
        """
        plot = self.plot()
        if plot is None:
            return

        for item in plot.getItems():
            if isinstance(item, SymbolMixIn):
                item.setSymbol(marker)


class SymbolToolButton(_SymbolToolButtonBase):
    """A tool button with a drop-down menu to control symbol size and marker.

    :param parent: See QWidget
    :param plot: The `~silx.gui.plot.PlotWidget` to control
    """

    def __init__(self, parent=None, plot=None):
        super().__init__(parent=parent, plot=plot)

        self.setToolTip("Set symbol size and marker")
        self.setIcon(icons.getQIcon("plot-symbols"))

        menu = qt.QMenu(self)
        self._addSizeSliderToMenu(menu)
        menu.addSeparator()
        self._addSymbolsToMenu(menu)

        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)


class ScatterVisualizationToolButton(_SymbolToolButtonBase):
    """QToolButton to select the visualization mode of scatter plot

    :param parent: See QWidget
    :param plot: The `~silx.gui.plot.PlotWidget` to control
    """

    def __init__(self, parent=None, plot=None):
        super().__init__(parent=parent, plot=plot)

        self.setToolTip("Set scatter visualization mode, symbol marker and size")
        self.setIcon(icons.getQIcon("eye"))

        menu = qt.QMenu(self)

        # Add visualization modes

        for mode in Scatter.supportedVisualizations():
            if mode is not Scatter.Visualization.BINNED_STATISTIC:
                name = mode.value.capitalize()
                action = qt.QAction(name, menu)
                action.setCheckable(False)
                action.triggered.connect(
                    functools.partial(self._visualizationChanged, mode, None)
                )
                menu.addAction(action)

        if Scatter.Visualization.BINNED_STATISTIC in Scatter.supportedVisualizations():
            reductions = Scatter.supportedVisualizationParameterValues(
                Scatter.VisualizationParameter.BINNED_STATISTIC_FUNCTION
            )
            if reductions:
                submenu = menu.addMenu("Binned Statistic")
                for reduction in reductions:
                    name = reduction.capitalize()
                    action = qt.QAction(name, menu)
                    action.setCheckable(False)
                    action.triggered.connect(
                        functools.partial(
                            self._visualizationChanged,
                            Scatter.Visualization.BINNED_STATISTIC,
                            {
                                Scatter.VisualizationParameter.BINNED_STATISTIC_FUNCTION: reduction
                            },
                        )
                    )
                    submenu.addAction(action)

                submenu.addSeparator()
                binsmenu = submenu.addMenu("N Bins")

                slider = qt.QSlider(qt.Qt.Horizontal)
                slider.setRange(10, 1000)
                slider.setValue(100)
                slider.setTracking(False)
                slider.valueChanged.connect(self._binningChanged)
                widgetAction = qt.QWidgetAction(binsmenu)
                widgetAction.setDefaultWidget(slider)
                binsmenu.addAction(widgetAction)

        menu.addSeparator()

        submenu = menu.addMenu(icons.getQIcon("plot-symbols"), "Symbol")
        self._addSymbolsToMenu(submenu)

        submenu = menu.addMenu(icons.getQIcon("plot-symbols"), "Symbol Size")
        self._addSizeSliderToMenu(submenu)

        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)

    def _visualizationChanged(self, mode, parameters=None):
        """Handle change of visualization mode.

        :param ScatterVisualizationMixIn.Visualization mode:
            The visualization mode to use for scatter
        :param Union[dict,None] parameters:
            Dict of VisualizationParameter: parameter_value to set
            with the visualization.
        """
        plot = self.plot()
        if plot is None:
            return

        for item in plot.getItems():
            if isinstance(item, Scatter):
                if parameters:
                    for parameter, value in parameters.items():
                        item.setVisualizationParameter(parameter, value)
                item.setVisualization(mode)

    def _binningChanged(self, value):
        """Handle change of binning.

        :param int value: The number of bin on each dimension.
        """
        plot = self.plot()
        if plot is None:
            return

        for item in plot.getItems():
            if isinstance(item, Scatter):
                item.setVisualizationParameter(
                    Scatter.VisualizationParameter.BINNED_STATISTIC_SHAPE,
                    (value, value),
                )
                item.setVisualization(Scatter.Visualization.BINNED_STATISTIC)
