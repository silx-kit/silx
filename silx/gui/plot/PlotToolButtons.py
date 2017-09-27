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
"""This module provides a set of QToolButton to use with :class:`.PlotWidget`.

The following QToolButton are available:

- :class:`AspectToolButton`
- :class:`YAxisOriginToolButton`
- :class:`ProfileToolButton`

"""

__authors__ = ["V. Valls", "H. Payno"]
__license__ = "MIT"
__date__ = "27/06/2017"


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
            # dont keep ratio
            self.STATE[False, "icon"] = icons.getQIcon('shape-ellipse-solid')
            self.STATE[False, "state"] = "Aspect ratio is not kept"
            self.STATE[False, "action"] = "Do no keep data aspect ratio"
            # keep ratio
            self.STATE[True, "icon"] = icons.getQIcon('shape-circle-solid')
            self.STATE[True, "state"] = "Aspect ratio is kept"
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
        yAxis = plot.getYAxis()
        yAxis.sigInvertedChanged.connect(self._yAxisInvertedChanged)
        self._yAxisInvertedChanged(yAxis.isInverted())

    def _disconnectPlot(self, plot):
        plot.getYAxis().sigInvertedChanged.disconnect(self._yAxisInvertedChanged)

    def setYAxisUpward(self):
        """Configure the plot to use y-axis upward"""
        plot = self.plot()
        if plot is not None:
            # This will trigger _yAxisInvertedChanged
            plot.getYAxis().setInverted(False)

    def setYAxisDownward(self):
        """Configure the plot to use y-axis downward"""
        plot = self.plot()
        if plot is not None:
            # This will trigger _yAxisInvertedChanged
            plot.getYAxis().setInverted(True)

    def _yAxisInvertedChanged(self, inverted):
        """Handle Plot set y axis inverted signal"""
        isUpward = not inverted
        icon, toolTip = self.STATE[isUpward, "icon"], self.STATE[isUpward, "state"]
        self.setIcon(icon)
        self.setToolTip(toolTip)


class ProfileToolButton(PlotToolButton):
    """Button used in Profile3DToolbar to switch between 2D profile
    and 1D profile."""
    STATE = None
    """Lazy loaded states used to feed ProfileToolButton"""

    sigDimensionChanged = qt.Signal(int)

    def __init__(self, parent=None, plot=None):
        if self.STATE is None:
            self.STATE = {
                (1, "icon"): icons.getQIcon('profile1D'),
                (1, "state"): "1D profile is computed on visible image",
                (1, "action"): "1D profile on visible image",
                (2, "icon"): icons.getQIcon('profile2D'),
                (2, "state"): "2D profile is computed, one 1D profile for each image in the stack",
                (2, "action"): "2D profile on image stack"}
            # Compute 1D profile
            # Compute 2D profile

        super(ProfileToolButton, self).__init__(parent=parent, plot=plot)

        profile1DAction = self._createAction(1)
        profile1DAction.triggered.connect(self.computeProfileIn1D)
        profile1DAction.setIconVisibleInMenu(True)

        profile2DAction = self._createAction(2)
        profile2DAction.triggered.connect(self.computeProfileIn2D)
        profile2DAction.setIconVisibleInMenu(True)

        menu = qt.QMenu(self)
        menu.addAction(profile1DAction)
        menu.addAction(profile2DAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)
        menu.setTitle('Select profile dimension')

    def _createAction(self, profileDimension):
        icon = self.STATE[profileDimension, "icon"]
        text = self.STATE[profileDimension, "action"]
        return qt.QAction(icon, text, self)

    def _profileDimensionChanged(self, profileDimension):
        """Update icon in toolbar, emit number of dimensions for profile"""
        self.setIcon(self.STATE[profileDimension, "icon"])
        self.setToolTip(self.STATE[profileDimension, "state"])
        self.sigDimensionChanged.emit(profileDimension)

    def computeProfileIn1D(self):
        self._profileDimensionChanged(1)

    def computeProfileIn2D(self):
        self._profileDimensionChanged(2)


class MeanSumToolButton(PlotToolButton):
    """Button used in Profile3DToolbar to switch between "sum"
    and "mean" modes."""
    sigAverageMethodChanged = qt.Signal(str)

    def __init__(self, parent=None, plot=None):
        self.STATE = {
            "mean": {
                "icon": icons.getQIcon('math-mean-black'),
                "tooltip": "Compute profile by averaging samples in the" +
                           " perpendicular direction (if width > 1)"
            },
            "sum": {
                "icon": icons.getQIcon('math-sigma-black'),
                "tooltip": "Compute profile by summing samples in the" +
                           " perpendicular direction (if width > 1)"
            }
        }

        super(MeanSumToolButton, self).__init__(parent=parent, plot=plot)

        meanAction = qt.QAction(self.STATE["mean"]["icon"],
                                "Mean profile", self)
        meanAction.setToolTip(self.STATE["mean"]["tooltip"])
        meanAction.triggered.connect(self.setMeanProfile)
        meanAction.setIconVisibleInMenu(True)

        sumAction = qt.QAction(self.STATE["sum"]["icon"],
                               "Sum profile", self)
        sumAction.setToolTip(self.STATE["sum"]["tooltip"])
        sumAction.triggered.connect(self.setSumProfile)
        sumAction.setIconVisibleInMenu(True)

        menu = qt.QMenu(self)
        menu.addAction(meanAction)
        menu.addAction(sumAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)
        menu.setTitle('Select average method')

    def _averageMethodChanged(self, averageMethod):
        """Update icon in toolbar, emit averageMethod
        :param str averageMethod: "mean" or "sum"
        """
        self.setIcon(self.STATE[averageMethod]["icon"])
        self.setToolTip(self.STATE[averageMethod]["tooltip"])
        self.sigAverageMethodChanged.emit(averageMethod)

    def setMeanProfile(self):
        self._averageMethodChanged("mean")

    def setSumProfile(self):
        self._averageMethodChanged("sum")
