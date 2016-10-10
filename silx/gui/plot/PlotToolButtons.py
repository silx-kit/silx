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
"""This module provides a set of QToolButton to use with :class:`.PlotWidget`.

The following QToolButton are available:

- :class:`AspectToolButton`
- :class:`YAxiesOriginToolButton`
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "04/10/2016"


import logging
from .. import icons
from .. import qt


_logger = logging.getLogger(__name__)


class PlotToolButton(qt.QToolButton):
    """A QToolButton connected to a :class:`.PlotWidget`.
    """

    def __init__(self, parent=None, plot=None):
        super(PlotToolButton, self).__init__(parent)
        self._plot = None
        if plot is not None:
            self.setPlot(plot)

    def plot(self):
        """
        Returns the plot connected to the widget.

        :param plot: :class:`.PlotWidget` instance on which to operate.
        """
        return self._plot

    def setPlot(self, plot):
        """
        Set the plot connected to the widget

        :param plot: :class:`.PlotWidget` instance on which to operate.
        """
        if self._plot is plot:
            return
        if self._plot is not None:
            self._disconnectPlot(self._plot)
        self._plot = plot
        if self._plot is not None:
            self._connectPlot(self._plot)

    def _connectPlot(self, plot):
        """
        Called when the plot is connected to the widget

        :param plot: :class:`.PlotWidget` instance
        """
        pass

    def _disconnectPlot(self, plot):
        """
        Called when the plot is disconnected from the widget

        :param plot: :class:`.PlotWidget` instance
        """
        pass


class AspectToolButton(PlotToolButton):

    STATE = None
    """Lazy loaded states used to feed AspectToolButton"""

    def __init__(self, parent=None, plot=None):
        if self.STATE is None:
            self.STATE = {}
            # dont keep ration
            self.STATE[False, "icon"] = icons.getQIcon('shape-ellipse-solid')
            self.STATE[False, "state"] = "Aspect ration is not kept"
            self.STATE[False, "action"] = "Do no keep data aspect ratio"
            # keep ration
            self.STATE[True, "icon"] = icons.getQIcon('shape-circle-solid')
            self.STATE[True, "state"] = "Aspect ration is kept"
            self.STATE[True, "action"] = "Keep data aspect ratio"

        super(AspectToolButton, self).__init__(parent=parent, plot=plot)

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
        icon, toolTip = self.STATE[aspectRatio, "icon"], self.STATE[aspectRatio, "state"]
        self.setIcon(icon)
        self.setToolTip(toolTip)


class YAxisOriginToolButton(PlotToolButton):

    STATE = None
    """Lazy loaded states used to feed YAxisOriginToolButton"""

    def __init__(self, parent=None, plot=None):
        if self.STATE is None:
            self.STATE = {}
            # is down
            self.STATE[False, "icon"] = icons.getQIcon('plot-ydown')
            self.STATE[False, "state"] = "Y-axis is oriented downward"
            self.STATE[False, "action"] = "Orient Y-axis downward"
            # keep ration
            self.STATE[True, "icon"] = icons.getQIcon('plot-yup')
            self.STATE[True, "state"] = "Y-axis is oriented upward"
            self.STATE[True, "action"] = "Orient Y-axis upward"

        super(YAxisOriginToolButton, self).__init__(parent=parent, plot=plot)

        upwardAction = self._createAction(True)
        upwardAction.triggered.connect(self.setYAxisUpward)
        upwardAction.setIconVisibleInMenu(True)

        downwardAction = self._createAction(False)
        downwardAction.triggered.connect(self.setYAxisDownward)
        downwardAction.setIconVisibleInMenu(True)

        menu = qt.QMenu(self)
        menu.addAction(upwardAction)
        menu.addAction(downwardAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)

    def _createAction(self, isUpward):
        icon = self.STATE[isUpward, "icon"]
        text = self.STATE[isUpward, "action"]
        return qt.QAction(icon, text, self)

    def _connectPlot(self, plot):
        plot.sigSetYAxisInverted.connect(self._yAxisInvertedChanged)
        self._yAxisInvertedChanged(plot.isYAxisInverted())

    def _disconnectPlot(self, plot):
        plot.sigSetYAxisInverted.disconnect(self._yAxisInvertedChanged)

    def setYAxisUpward(self):
        """Configure the plot to use y-axis upward"""
        plot = self.plot()
        if plot is not None:
            # This will trigger _yAxisInvertedChanged
            plot.setYAxisInverted(False)

    def setYAxisDownward(self):
        """Configure the plot to use y-axis downward"""
        plot = self.plot()
        if plot is not None:
            # This will trigger _yAxisInvertedChanged
            plot.setYAxisInverted(True)

    def _yAxisInvertedChanged(self, inverted):
        """Handle Plot set y axis inverted signal"""
        isUpward = not inverted
        icon, toolTip = self.STATE[isUpward, "icon"], self.STATE[isUpward, "state"]
        self.setIcon(icon)
        self.setToolTip(toolTip)
