# /*##########################################################################
#
# Copyright (c) 2017-2023 European Synchrotron Radiation Facility
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
"""This module provides the class for axes of the :class:`PlotWidget`."""

from __future__ import annotations

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "22/11/2018"

import datetime as dt
import enum
import typing

import dateutil.tz

from ....utils.proxy import docstring
from ... import qt
from .. import _utils


class TickMode(enum.Enum):
    """Determines if ticks are regular number or datetimes."""

    DEFAULT = 0  # Ticks are regular numbers
    TIME_SERIES = 1  # Ticks are datetime objects


AxisScaleType = typing.Literal["linear", "log"]


class Axis(qt.QObject):
    """This class describes and controls a plot axis.

    Note: This is an abstract class.
    """

    # States are half-stored on the backend of the plot, and half-stored on this
    # object.
    # TODO It would be good to store all the states of an axis in this object.
    #      i.e. vmin and vmax

    LINEAR = "linear"
    """Constant defining a linear scale"""

    LOGARITHMIC = "log"
    """Constant defining a logarithmic scale"""

    _SCALES = {LINEAR, LOGARITHMIC}

    sigInvertedChanged = qt.Signal(bool)
    """Signal emitted when axis orientation has changed"""

    sigScaleChanged = qt.Signal(str)
    """Signal emitted when axis scale has changed"""

    _sigLogarithmicChanged = qt.Signal(bool)
    """Signal emitted when axis scale has changed to or from logarithmic"""

    sigAutoScaleChanged = qt.Signal(bool)
    """Signal emitted when axis autoscale has changed"""

    sigLimitsChanged = qt.Signal(float, float)
    """Signal emitted when axis limits have changed"""

    def __init__(self, plot):
        """Constructor

        :param silx.gui.plot.PlotWidget.PlotWidget plot: Parent plot of this
            axis
        """
        qt.QObject.__init__(self, parent=plot)
        self._scale = self.LINEAR
        self._isAutoScale = True
        # Store default labels provided to setGraph[X|Y]Label
        self._defaultLabel = ""
        # Store currently displayed labels
        # Current label can differ from input one with active curve handling
        self._currentLabel = ""

    def _getPlot(self):
        """Returns the PlotWidget this Axis belongs to.

        :rtype: PlotWidget
        """
        plot = self.parent()
        if plot is None:
            raise RuntimeError("Axis no longer attached to a PlotWidget")
        return plot

    def _getBackend(self):
        """Returns the backend

        :rtype: BackendBase
        """
        return self._getPlot()._backend

    def getLimits(self) -> tuple[float, float]:
        """Get the limits of this axis.

        :return: Minimum and maximum values of this axis as tuple
        """
        return self._internalGetLimits()

    def setLimits(self, vmin: float, vmax: float):
        """Set this axis limits.

        :param vmin: minimum axis value
        :param vmax: maximum axis value
        """
        vmin, vmax = self._checkLimits(vmin, vmax)
        if self.getLimits() == (vmin, vmax):
            return

        self._internalSetLimits(vmin, vmax)
        self._getPlot()._setDirtyPlot()

        self._emitLimitsChanged()

    def _emitLimitsChanged(self):
        """Emit axis sigLimitsChanged and PlotWidget limitsChanged event"""
        vmin, vmax = self.getLimits()
        self.sigLimitsChanged.emit(vmin, vmax)
        self._getPlot()._notifyLimitsChanged(emitSignal=False)

    def _checkLimits(self, vmin: float, vmax: float) -> tuple[float, float]:
        """Makes sure axis range is not empty and within supported range.

        :param vmin: Min axis value
        :param vmax: Max axis value
        :return: (min, max) making sure min < max
        """
        return _utils.checkAxisLimits(
            vmin, vmax, isLog=self._isLogarithmic(), name=self._defaultLabel
        )

    def _getDataRange(self) -> tuple[float, float] | None:
        """Returns the range of data items over this axis as (vmin, vmax)"""
        raise NotImplementedError()

    def isInverted(self) -> bool:
        """Return True if the axis is inverted (top to bottom for the y-axis),
        False otherwise. It is always False for the X axis.

        :rtype: bool
        """
        return False

    def setInverted(self, isInverted: bool):
        """Set the axis orientation.

        This is only available for the Y axis.

        :param flag: True for Y axis going from top to bottom,
                     False for Y axis going from bottom to top
        """
        if isInverted == self.isInverted():
            return
        raise NotImplementedError()

    def isVisible(self) -> bool:
        """Returns whether the axis is displayed or not"""
        return True

    def getLabel(self) -> str:
        """Return the current displayed label of this axis.

        :param str axis: The Y axis for which to get the label (left or right)
        """
        return self._currentLabel

    def setLabel(self, label: str):
        """Set the label displayed on the plot for this axis.

        The provided label can be temporarily replaced by the label of the
        active curve if any.

        :param label: The axis label
        """
        self._defaultLabel = label
        self._setCurrentLabel(label)
        self._getPlot()._setDirtyPlot()

    def _setCurrentLabel(self, label):
        """Define the label currently displayed.

        If the label is None or empty the default label is used.

        :param str label: Currently displayed label
        """
        if label is None or label == "":
            label = self._defaultLabel
        if label is None:
            label = ""
        self._currentLabel = label
        self._internalSetCurrentLabel(label)

    def getScale(self) -> AxisScaleType:
        """Return the name of the scale used by this axis."""
        return self._scale

    def setScale(self, scale: AxisScaleType):
        """Set the scale to be used by this axis.

        :param scale: Name of the scale ("log", or "linear")
        """
        assert scale in self._SCALES
        if self._scale == scale:
            return

        # For the backward compatibility signal
        emitLog = self._scale == self.LOGARITHMIC or scale == self.LOGARITHMIC

        self._scale = scale

        vmin, vmax = self.getLimits()

        # TODO hackish way of forcing update of curves and images
        plot = self._getPlot()
        for item in plot.getItems():
            item._updated()
        plot._invalidateDataRange()

        if scale == self.LOGARITHMIC:
            self._internalSetLogarithmic(True)
            if vmin <= 0:
                dataRange = self._getDataRange()
                if dataRange is None:
                    self.setLimits(1.0, 100.0)
                else:
                    if vmax > 0 and dataRange[0] < vmax:
                        self.setLimits(dataRange[0], vmax)
                    else:
                        self.setLimits(*dataRange)
        elif scale == self.LINEAR:
            self._internalSetLogarithmic(False)
        else:
            raise ValueError("Scale %s unsupported" % scale)

        self.sigScaleChanged.emit(self._scale)
        if emitLog:
            self._sigLogarithmicChanged.emit(self._scale == self.LOGARITHMIC)

    def _isLogarithmic(self) -> bool:
        """Return True if this axis scale is logarithmic, False if linear."""
        return self._scale == self.LOGARITHMIC

    def _setLogarithmic(self, flag: bool):
        """Set the scale of this axes (either linear or logarithmic).

        :param flag: True to use a logarithmic scale, False for linear.
        """
        flag = bool(flag)
        self.setScale(self.LOGARITHMIC if flag else self.LINEAR)

    def getTimeZone(self) -> dt.tzinfo | None:
        """Sets tzinfo that is used if this axis plots date times.

        None means the datetimes are interpreted as local time.
        """
        raise NotImplementedError()

    def setTimeZone(self, tz) -> dt.tzinfo | typing.Literal["UTC"] | None:
        """Sets tzinfo that is used if this axis' tickMode is TIME_SERIES

        The tz must be a descendant of the datetime.tzinfo class, "UTC" or None.
        Use None to let the datetimes be interpreted as local time.
        Use the string "UTC" to let the date datetimes be in UTC time.

        :param tz: A timezone, "UTC" or None.
        """
        raise NotImplementedError()

    def getTickMode(self):
        """Determines if axis ticks are number or datetimes.

        :rtype: TickMode enum.
        """
        raise NotImplementedError()

    def setTickMode(self, tickMode):
        """Determines if axis ticks are number or datetimes.

        :param TickMode tickMode: tick mode enum.
        """
        raise NotImplementedError()

    def isAutoScale(self):
        """Return True if axis is automatically adjusting its limits.

        :rtype: bool
        """
        return self._isAutoScale

    def setAutoScale(self, flag=True):
        """Set the axis limits adjusting behavior of :meth:`resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        self._isAutoScale = bool(flag)
        self.sigAutoScaleChanged.emit(self._isAutoScale)

    def _setLimitsConstraints(self, minPos=None, maxPos=None):
        raise NotImplementedError()

    def setLimitsConstraints(self, minPos=None, maxPos=None):
        """
        Set a constraint on the position of the axes.

        :param float minPos: Minimum allowed axis value.
        :param float maxPos: Maximum allowed axis value.
        :return: True if the constaints was updated
        :rtype: bool
        """
        updated = self._setLimitsConstraints(minPos, maxPos)
        if updated:
            plot = self._getPlot()
            xMin, xMax = plot.getXAxis().getLimits()
            yMin, yMax = plot.getYAxis().getLimits()
            y2Min, y2Max = plot.getYAxis("right").getLimits()
            plot.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)
        return updated

    def _setRangeConstraints(self, minRange=None, maxRange=None):
        raise NotImplementedError()

    def setRangeConstraints(self, minRange=None, maxRange=None):
        """
        Set a constraint on the position of the axes.

        :param float minRange: Minimum allowed left-to-right span across the
            view
        :param float maxRange: Maximum allowed left-to-right span across the
            view
        :return: True if the constaints was updated
        :rtype: bool
        """
        updated = self._setRangeConstraints(minRange, maxRange)
        if updated:
            plot = self._getPlot()
            xMin, xMax = plot.getXAxis().getLimits()
            yMin, yMax = plot.getYAxis().getLimits()
            y2Min, y2Max = plot.getYAxis("right").getLimits()
            plot.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)
        return updated


class XAxis(Axis):
    """Axis class defining primitives for the X axis"""

    # TODO With some changes on the backend, it will be able to remove all this
    #      specialised implementations (prefixel by '_internal')

    @docstring(Axis)
    def getTimeZone(self):
        return self._getBackend().getXAxisTimeZone()

    @docstring(Axis)
    def setTimeZone(self, tz):
        if isinstance(tz, str) and tz.upper() == "UTC":
            tz = dateutil.tz.tzutc()
        elif not (tz is None or isinstance(tz, dt.tzinfo)):
            raise TypeError("tz must be a dt.tzinfo object, None or 'UTC'.")

        self._getBackend().setXAxisTimeZone(tz)
        self._getPlot()._setDirtyPlot()

    def getTickMode(self) -> TickMode:
        if self._getBackend().isXAxisTimeSeries():
            return TickMode.TIME_SERIES
        else:
            return TickMode.DEFAULT

    def setTickMode(self, tickMode: TickMode):
        if tickMode == TickMode.DEFAULT:
            self._getBackend().setXAxisTimeSeries(False)
        elif tickMode == TickMode.TIME_SERIES:
            self._getBackend().setXAxisTimeSeries(True)
        else:
            raise ValueError(f"Unexpected TickMode: {tickMode}")

    def _internalSetCurrentLabel(self, label):
        self._getBackend().setGraphXLabel(label)

    def _internalGetLimits(self):
        return self._getBackend().getGraphXLimits()

    def _internalSetLimits(self, xmin, xmax):
        self._getBackend().setGraphXLimits(xmin, xmax)

    def _internalSetLogarithmic(self, flag):
        self._getBackend().setXAxisLogarithmic(flag)

    def _setLimitsConstraints(self, minPos=None, maxPos=None):
        constrains = self._getPlot()._getViewConstraints()
        updated = constrains.update(xMin=minPos, xMax=maxPos)
        return updated

    def _setRangeConstraints(self, minRange=None, maxRange=None):
        constrains = self._getPlot()._getViewConstraints()
        updated = constrains.update(minXRange=minRange, maxXRange=maxRange)
        return updated

    @docstring(Axis)
    def _getDataRange(self) -> tuple[float, float] | None:
        ranges = self._getPlot().getDataRange()
        return ranges.x


class YAxis(Axis):
    """Axis class defining primitives for the Y axis"""

    # TODO With some changes on the backend, it will be able to remove all this
    #      specialised implementations (prefixel by '_internal')

    def _internalSetCurrentLabel(self, label):
        self._getBackend().setGraphYLabel(label, axis="left")

    def _internalGetLimits(self):
        return self._getBackend().getGraphYLimits(axis="left")

    def _internalSetLimits(self, ymin, ymax):
        self._getBackend().setGraphYLimits(ymin, ymax, axis="left")

    def _internalSetLogarithmic(self, flag):
        self._getBackend().setYAxisLogarithmic(flag)

    def setInverted(self, flag: bool = True):
        """Set the axis orientation.

        This is only available for the Y axis.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        flag = bool(flag)
        if self.isInverted() == flag:
            return
        self._getBackend().setYAxisInverted(flag)
        self._getPlot()._setDirtyPlot()
        self.sigInvertedChanged.emit(flag)

    def isInverted(self) -> bool:
        """Return True if the axis is inverted (top to bottom for the y-axis),
        False otherwise. It is always False for the X axis.

        :rtype: bool
        """
        return self._getBackend().isYAxisInverted()

    def _setLimitsConstraints(self, minPos=None, maxPos=None):
        constrains = self._getPlot()._getViewConstraints()
        updated = constrains.update(yMin=minPos, yMax=maxPos)
        return updated

    def _setRangeConstraints(self, minRange=None, maxRange=None):
        constrains = self._getPlot()._getViewConstraints()
        updated = constrains.update(minYRange=minRange, maxYRange=maxRange)
        return updated

    @docstring(Axis)
    def _getDataRange(self) -> tuple[float, float] | None:
        ranges = self._getPlot().getDataRange()
        return ranges.y


class YRightAxis(Axis):
    """Proxy axis for the secondary Y axes. It manages it own label and limit
    but share the some state like scale and direction with the main axis."""

    # TODO With some changes on the backend, it will be able to remove all this
    #      specialised implementations (prefixel by '_internal')

    def __init__(self, plot, mainAxis):
        """Constructor

        :param silx.gui.plot.PlotWidget.PlotWidget plot: Parent plot of this
            axis
        :param Axis mainAxis: Axis which sharing state with this axis
        """
        Axis.__init__(self, plot)
        self.__mainAxis = mainAxis
        self.__mainAxis.sigInvertedChanged.connect(self.sigInvertedChanged.emit)
        self.__mainAxis.sigScaleChanged.connect(self.sigScaleChanged.emit)
        self.__mainAxis._sigLogarithmicChanged.connect(self._sigLogarithmicChanged.emit)
        self.__mainAxis.sigAutoScaleChanged.connect(self.sigAutoScaleChanged.emit)

    def _internalSetCurrentLabel(self, label):
        self._getBackend().setGraphYLabel(label, axis="right")

    def _internalGetLimits(self) -> tuple[float, float]:
        return self._getBackend().getGraphYLimits(axis="right")

    def _internalSetLimits(self, ymin, ymax):
        self._getBackend().setGraphYLimits(ymin, ymax, axis="right")

    def setInverted(self, flag: bool = True):
        """Set the Y axis orientation.

        :param flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        return self.__mainAxis.setInverted(flag)

    def isInverted(self) -> bool:
        """Return True if Y axis goes from top to bottom, False otherwise."""
        return self.__mainAxis.isInverted()

    def isVisible(self) -> bool:
        """Returns whether the axis is displayed or not"""
        return self._getBackend().isYRightAxisVisible()

    def getScale(self) -> AxisScaleType:
        """Return the name of the scale used by this axis."""
        return self.__mainAxis.getScale()

    def setScale(self, scale: AxisScaleType):
        """Set the scale to be used by this axis.

        :param scale: Name of the scale ("log", or "linear")
        """
        self.__mainAxis.setScale(scale)

    def _isLogarithmic(self) -> bool:
        """Return True if Y axis scale is logarithmic, False if linear."""
        return self.__mainAxis._isLogarithmic()

    def _setLogarithmic(self, flag: bool):
        """Set the Y axes scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        return self.__mainAxis._setLogarithmic(flag)

    def isAutoScale(self) -> bool:
        """Return True if Y axes are automatically adjusting its limits."""
        return self.__mainAxis.isAutoScale()

    def setAutoScale(self, flag: bool = True):
        """Set the Y axis limits adjusting behavior of :meth:`PlotWidget.resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        return self.__mainAxis.setAutoScale(flag)

    @docstring(Axis)
    def _getDataRange(self) -> tuple[float, float] | None:
        ranges = self._getPlot().getDataRange()
        return ranges.y2
