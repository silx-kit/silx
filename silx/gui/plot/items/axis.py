# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module provides the class for axes  of the :class:`Plot`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "22/06/2017"

from ... import qt


class Axis(qt.QObject):

    sigInvertedChanged = qt.Signal(bool)
    """Signal emitted when axis orientation has changed"""

    sigLogarithmicChanged = qt.Signal(bool)
    """Signal emitted when axis scale has changed"""

    sigAutoScaleChanged = qt.Signal(bool)
    """Signal emitted when axis autoscale has changed"""

    sigLimitsChanged = qt.Signal(bool)
    """Signal emitted when axis autoscale has changed"""

    def __init__(self, plot):
        qt.QObject.__init__(self, parent=plot)
        self._isLog = False
        self._isAutoScale = True
        # Store default labels provided to setGraph[X|Y]Label
        self._defaultLabel = ''
        # Store currently displayed labels
        # Current label can differ from input one with active curve handling
        self._currentLabel = ''
        self._plot = plot

    def getLimits(self):
        raise NotImplementedError()

    def setLimits(self, limits):
        raise NotImplementedError()

    def isInverted(self):
        raise NotImplementedError()

    def setInverted(self, isInverted):
        raise NotImplementedError()

    def getLabel(self):
        """Return the current Y axis label as a str.

        :param str axis: The Y axis for which to get the label (left or right)
        """
        return self._currentLabel

    def setLabel(self, label):
        """Set the plot Y axis label.

        The provided label can be temporarily replaced by the label of the
        active curve if any.

        :param str label: The axis label
        """
        self._defaultLabel = label
        self._setCurrentLabel(label)
        self._plot._setDirtyPlot()

    def _setCurrentLabel(self, label):
        """Define the label currently displayed.

        If the label is none or empty the default label is used.

        :param str label: Currently displayed label
        :returns: str
        """
        if label is None or label == '':
            label = self._defaultLabel
        if label is None:
            label = ''
        self._currentLabel = label
        return label

    def isLogarithmic(self):
        raise NotImplementedError()

    def setLogarithmic(self, isLogarithmic):
        raise NotImplementedError()


class XAxis(Axis):

    def _setCurrentLabel(self, label):
        label = Axis._setCurrentLabel(self, label)
        self._plot._backend.setGraphXLabel(label)
        return label

    def getLimits(self):
        """Get the graph X (bottom) limits.

        :return: Minimum and maximum values of the X axis
        """
        return self._plot._backend.getGraphXLimits()

    def setLimits(self, xmin, xmax):
        """Set the graph X (bottom) limits.

        :param float xmin: minimum bottom axis value
        :param float xmax: maximum bottom axis value
        """
        xmin, xmax = self._plot._checkLimits(xmin, xmax, axis='x')

        self._plot._backend.setGraphXLimits(xmin, xmax)
        self._plot._setDirtyPlot()

        self._plot._notifyLimitsChanged()

    def isLogarithmic(self):
        """Return True if X axis scale is logarithmic, False if linear."""
        return self._isLog

    def setLogarithmic(self, flag):
        """Set the bottom X axis scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        if bool(flag) == self._isLog:
            return
        self._isLog = bool(flag)

        self._plot._backend.setXAxisLogarithmic(self._isLog)

        # TODO hackish way of forcing update of curves and images
        for item in self._plot._getItems(withhidden=True):
            item._updated()
        self._plot._invalidateDataRange()

        self._plot.resetZoom()
        self._plot.notify('setXAxisLogarithmic', state=self._isLog)

    def isAutoScale(self):
        """Return True if X axis is automatically adjusting its limits."""
        return self._isAutoScale

    def setAutoScale(self, flag=True):
        """Set the X axis limits adjusting behavior of :meth:`resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        self._isAutoScale = bool(flag)
        self._plot.notify('setXAxisAutoScale', state=self._isAutoScale)


class YAxis(Axis):

    def _setCurrentLabel(self, label):
        label = Axis._setCurrentLabel(self, label)
        self._plot._backend.setGraphYLabel(label, axis='left')
        return label

    def getLimits(self):
        """Get the graph Y limits.

        :param str axis: The axis for which to get the limits:
                         Either 'left' or 'right'
        :return: Minimum and maximum values of the X axis
        """
        return self._plot._backend.getGraphYLimits(axis='left')

    def setLimits(self, ymin, ymax):
        """Set the graph Y limits.

        :param float ymin: minimum bottom axis value
        :param float ymax: maximum bottom axis value
        :param str axis: The axis for which to get the limits:
                         Either 'left' or 'right'
        """
        ymin, ymax = self._plot._checkLimits(ymin, ymax, axis='y')
        self._plot._backend.setGraphYLimits(ymin, ymax, axis='left')
        self._plot._setDirtyPlot()

        self._plot._notifyLimitsChanged()

    def setInverted(self, flag=True):
        """Set the Y axis orientation.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        flag = bool(flag)
        self._plot._backend.setYAxisInverted(flag)
        self._plot._setDirtyPlot()
        self._plot.notify('setYAxisInverted', state=flag)

    def isInverted(self):
        """Return True if Y axis goes from top to bottom, False otherwise."""
        return self._plot._backend.isYAxisInverted()

    def isLogarithmic(self):
        """Return True if Y axis scale is logarithmic, False if linear."""
        return self._isLog

    def setLogarithmic(self, flag):
        """Set the Y axes scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        if bool(flag) == self._isLog:
            return
        self._isLog = bool(flag)

        self._plot._backend.setYAxisLogarithmic(self._isLog)

        # TODO hackish way of forcing update of curves and images
        for item in self._plot._getItems(withhidden=True):
            item._updated()
        self._plot._invalidateDataRange()

        self._plot.resetZoom()
        self._plot.notify('setYAxisLogarithmic', state=self._isLog)

    def isAutoScale(self):
        """Return True if Y axes are automatically adjusting its limits."""
        return self._isAutoScale

    def setAutoScale(self, flag=True):
        """Set the Y axis limits adjusting behavior of :meth:`resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        self._isAutoScale = bool(flag)
        self._plot.notify('setYAxisAutoScale', state=self._isAutoScale)


class YRightAxis(Axis):

    def __init__(self, plot, mainAxis):
        Axis.__init__(self, plot)
        self.__mainAxis = mainAxis

    @property
    def sigInvertedChanged(self):
        """Signal emitted when axis orientation has changed"""
        return self.__mainAxis.sigInvertedChanged

    @property
    def sigLogarithmicChanged(self):
        """Signal emitted when axis scale has changed"""
        return self.__mainAxis.sigLogarithmicChanged

    @property
    def sigAutoScaleChanged(self):
        """Signal emitted when axis autoscale has changed"""
        return self.__mainAxis.sigAutoScaleChanged

    def _setCurrentLabel(self, label):
        label = Axis._setCurrentLabel(self, label)
        self._plot._backend.setGraphYLabel(label, axis='right')
        return label

    def getLimits(self):
        """Get the graph Y limits.

        :param str axis: The axis for which to get the limits:
                         Either 'left' or 'right'
        :return: Minimum and maximum values of the X axis
        """
        return self._plot._backend.getGraphYLimits(axis='right')

    def setLimits(self, ymin, ymax):
        """Set the graph Y limits.

        :param float ymin: minimum bottom axis value
        :param float ymax: maximum bottom axis value
        :param str axis: The axis for which to get the limits:
                         Either 'left' or 'right'
        """
        ymin, ymax = self._plot._checkLimits(ymin, ymax, axis='y2')
        self._plot._backend.setGraphYLimits(ymin, ymax, axis='right')
        self._plot._setDirtyPlot()

        self._plot._notifyLimitsChanged()

    def setInverted(self, flag=True):
        """Set the Y axis orientation.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        return self.__mainAxis.setInverted(flag)

    def isInverted(self):
        """Return True if Y axis goes from top to bottom, False otherwise."""
        return self.__mainAxis.isInverted()

    def isLogarithmic(self):
        """Return True if Y axis scale is logarithmic, False if linear."""
        return self.__mainAxis.isLogarithmic()

    def setLogarithmic(self, flag):
        """Set the Y axes scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        return self.__mainAxis.setLogarithmic(flag)

    def isAutoScale(self):
        """Return True if Y axes are automatically adjusting its limits."""
        return self.__mainAxis.isAutoScale()

    def setAutoScale(self, flag=True):
        """Set the Y axis limits adjusting behavior of :meth:`resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        return self.__mainAxis.setAutoScale(flag)
