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
import time
import enum
from typing import Optional
import numpy
import weakref
import collections

import dateutil.tz

from ....utils.proxy import docstring
from ... import qt
from .. import _utils


class TickMode(enum.Enum):
    """Determines if ticks are regular number or datetimes."""

    DEFAULT = 0  # Ticks are regular numbers
    TIME_SERIES = 1  # Ticks are datetime objects


def _bezier_interpolation(t):
    """Returns the value of a specific Bezier transfer function to compute
    acceleration then desceleration.

    The input is supposed to be between [0..1], but the result is clamped.
    The result for an input smaller 0.0 is 0.0 and the result for an input
    highter than 1.0 is 1.0
    """
    if t < 0:
        return 0
    elif t < .5:
        return 4 * t ** 3
    elif t < 1.0:
        return (t - 1) * (2 * t - 2) ** 2 + 1
    else:
        return 1.0


class View(object):

    INTERPOLATION_DURATION = 500
    """Duration of the interpolation animation, in millisecond"""

    REFRESH_PERIOD = 50
    """Period used to referesh the animation"""

    def __init__(self, plot):
        self.__plot = weakref.ref(plot)
        self.__isAnimated = False
        self.__computedRange = None
        self.__targetedDataRange = None
        self.__timer = qt.QTimer(plot)
        self.__timer.timeout.connect(self.__tick)

    def getPlot(self):
        return self.__plot()

    def setAnimated(self, isAnimated):
        isAnimated = bool(isAnimated)
        if self.__isAnimated == isAnimated:
            return
        self.__isAnimated = isAnimated
        if isAnimated and self.__computedRange is not None:
            self.__interruptAnimation()

    def __interruptAnimation(self):
        self.__computedRange = None
        self.__startTime = None
        self.__timer.stop()
        plot = self.getPlot()
        plot.resetZoom()

    def dataRangeToArray(self, dataRange):
        normalized = numpy.zeros(6)
        normalized[0:2] = dataRange.x
        normalized[2:4] = dataRange.y
        normalized[4:6] = dataRange.yright
        return normalized

    RangeType = collections.namedtuple("RangeType", ["x", "y", "yright"])

    def arrayToRange(self, data):
        vmin = data[0] if not numpy.isnan(data[0]) else None
        vmax = data[1] if not numpy.isnan(data[1]) else None
        return vmin, vmax

    def arrayToDataRange(self, array):
        return self.RangeType(
            self.arrayToRange(array[0:2]),
            self.arrayToRange(array[2:4]),
            self.arrayToRange(array[4:6]),
        )

    def __updateSmoothing(self, newRange):
        self.__startTime = time.time()
        if self.__computedRange is None:
            self.__start = self.dataRangeToArray(self.__targetedDataRange)
            self.__stop = self.dataRangeToArray(newRange)
            self.__timer.start(self.REFRESH_PERIOD)
        else:
            self.__start = self.dataRangeToArray(self.__computedRange)
            self.__stop = self.dataRangeToArray(newRange)

        # Be conservative during the animation
        nanstop = numpy.isnan(self.__stop)
        nanstart = numpy.isnan(self.__start)
        self.__stop[nanstop] = self.__start[nanstop]
        self.__start[nanstart] = self.__stop[nanstart]

    def __tick(self):
        coef = ((time.time() - self.__startTime) * 1000) / self.INTERPOLATION_DURATION
        if coef >= 1.0:
            self.__interruptAnimation()
            return

        # Can be cached
        coef = _bezier_interpolation(coef)
        dataRange = self.__start * (1 - coef) + self.__stop * coef
        self.__computedRange = self.arrayToDataRange(dataRange)
        plot = self.getPlot()
        plot.resetZoom()

    def updateDataRange(self, dataRange):
        """
        Returns this PlotWidget's data range.

        :return: a namedtuple with the following members:
                x, y (left y axis), yright. Each member is a tuple (min, max)
                or None if no data is associated with the axis.
        :rtype: namedtuple
        """
        if not self.__isAnimated:
            return dataRange

        if self.__targetedDataRange is None:
            # Initial value
            self.__targetedDataRange = dataRange
        if self.__targetedDataRange == dataRange:
            if self.__computedRange is not None:
                return self.__computedRange
            else:
                return dataRange
        else:
            if dataRange.x is None and dataRange.y is None and dataRange.yright is None:
                return dataRange
            self.__updateSmoothing(dataRange)
            tmp = self.__targetedDataRange
            self.__targetedDataRange = dataRange
            return tmp


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

    _SCALES = set([LINEAR, LOGARITHMIC])

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

    def getLimits(self):
        """Get the limits of this axis.

        :return: Minimum and maximum values of this axis as tuple
        """
        return self._internalGetLimits()

    def setLimits(self, vmin, vmax):
        """Set this axis limits.

        :param float vmin: minimum axis value
        :param float vmax: maximum axis value
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

    def _checkLimits(self, vmin, vmax):
        """Makes sure axis range is not empty and within supported range.

        :param float vmin: Min axis value
        :param float vmax: Max axis value
        :return: (min, max) making sure min < max
        :rtype: 2-tuple of float
        """
        return _utils.checkAxisLimits(
            vmin, vmax, isLog=self._isLogarithmic(), name=self._defaultLabel
        )

    def _getDataRange(self) -> Optional[tuple[float, float]]:
        """Returns the range of data items over this axis as (vmin, vmax)"""
        raise NotImplementedError()

    def isInverted(self):
        """Return True if the axis is inverted (top to bottom for the y-axis),
        False otherwise. It is always False for the X axis.

        :rtype: bool
        """
        return False

    def setInverted(self, isInverted):
        """Set the axis orientation.

        This is only available for the Y axis.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        if isInverted == self.isInverted():
            return
        raise NotImplementedError()

    def isVisible(self) -> bool:
        """Returns whether the axis is displayed or not"""
        return True

    def getLabel(self):
        """Return the current displayed label of this axis.

        :param str axis: The Y axis for which to get the label (left or right)
        :rtype: str
        """
        return self._currentLabel

    def setLabel(self, label):
        """Set the label displayed on the plot for this axis.

        The provided label can be temporarily replaced by the label of the
        active curve if any.

        :param str label: The axis label
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

    def getScale(self):
        """Return the name of the scale used by this axis.

        :rtype: str
        """
        return self._scale

    def setScale(self, scale):
        """Set the scale to be used by this axis.

        :param str scale: Name of the scale ("log", or "linear")
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

    def _isLogarithmic(self):
        """Return True if this axis scale is logarithmic, False if linear.

        :rtype: bool
        """
        return self._scale == self.LOGARITHMIC

    def _setLogarithmic(self, flag):
        """Set the scale of this axes (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        flag = bool(flag)
        self.setScale(self.LOGARITHMIC if flag else self.LINEAR)

    def getTimeZone(self):
        """Sets tzinfo that is used if this axis plots date times.

        None means the datetimes are interpreted as local time.

        :rtype: datetime.tzinfo of None.
        """
        raise NotImplementedError()

    def setTimeZone(self, tz):
        """Sets tzinfo that is used if this axis' tickMode is TIME_SERIES

        The tz must be a descendant of the datetime.tzinfo class, "UTC" or None.
        Use None to let the datetimes be interpreted as local time.
        Use the string "UTC" to let the date datetimes be in UTC time.

        :param tz: datetime.tzinfo, "UTC" or None.
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

    def getTimeZone(self):
        return self._getBackend().getXAxisTimeZone()

    def setTimeZone(self, tz):
        if isinstance(tz, str) and tz.upper() == "UTC":
            tz = dateutil.tz.tzutc()
        elif not (tz is None or isinstance(tz, dt.tzinfo)):
            raise TypeError("tz must be a dt.tzinfo object, None or 'UTC'.")

        self._getBackend().setXAxisTimeZone(tz)
        self._getPlot()._setDirtyPlot()

    def getTickMode(self):
        if self._getBackend().isXAxisTimeSeries():
            return TickMode.TIME_SERIES
        else:
            return TickMode.DEFAULT

    def setTickMode(self, tickMode):
        if tickMode == TickMode.DEFAULT:
            self._getBackend().setXAxisTimeSeries(False)
        elif tickMode == TickMode.TIME_SERIES:
            self._getBackend().setXAxisTimeSeries(True)
        else:
            raise ValueError("Unexpected TickMode: {}".format(tickMode))

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
    def _getDataRange(self) -> Optional[tuple[float, float]]:
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

    def setInverted(self, flag=True):
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

    def isInverted(self):
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
    def _getDataRange(self) -> Optional[tuple[float, float]]:
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

    def _internalGetLimits(self):
        return self._getBackend().getGraphYLimits(axis="right")

    def _internalSetLimits(self, ymin, ymax):
        self._getBackend().setGraphYLimits(ymin, ymax, axis="right")

    def setInverted(self, flag=True):
        """Set the Y axis orientation.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        return self.__mainAxis.setInverted(flag)

    def isInverted(self):
        """Return True if Y axis goes from top to bottom, False otherwise."""
        return self.__mainAxis.isInverted()

    def isVisible(self) -> bool:
        """Returns whether the axis is displayed or not"""
        return self._getBackend().isYRightAxisVisible()

    def getScale(self):
        """Return the name of the scale used by this axis.

        :rtype: str
        """
        return self.__mainAxis.getScale()

    def setScale(self, scale):
        """Set the scale to be used by this axis.

        :param str scale: Name of the scale ("log", or "linear")
        """
        self.__mainAxis.setScale(scale)

    def _isLogarithmic(self):
        """Return True if Y axis scale is logarithmic, False if linear."""
        return self.__mainAxis._isLogarithmic()

    def _setLogarithmic(self, flag):
        """Set the Y axes scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        return self.__mainAxis._setLogarithmic(flag)

    def isAutoScale(self):
        """Return True if Y axes are automatically adjusting its limits."""
        return self.__mainAxis.isAutoScale()

    def setAutoScale(self, flag=True):
        """Set the Y axis limits adjusting behavior of :meth:`PlotWidget.resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        return self.__mainAxis.setAutoScale(flag)

    @docstring(Axis)
    def _getDataRange(self) -> Optional[tuple[float, float]]:
        ranges = self._getPlot().getDataRange()
        return ranges.y2
