# /*##########################################################################
#
# Copyright (c) 2023 European Synchrotron Radiation Facility
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
This module provides :class:`PlotWidget`-related QMenu.

The following QMenu is available:

- :class:`ZoomEnabledAxesMenu`
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "12/06/2023"


import weakref
from typing import Optional

from silx.gui import qt

from ..PlotWidget import PlotWidget


class ZoomEnabledAxesMenu(qt.QMenu):
    """Menu to toggle axes for zoom interaction"""

    def __init__(self, plot: PlotWidget, parent: Optional[qt.QWidget] = None):
        super().__init__(parent)
        self.setTitle("Zoom axes")

        assert isinstance(plot, PlotWidget)
        self.__plotRef = weakref.ref(plot)

        self.addSection("Enabled axes")
        self.__xAxisAction = qt.QAction("X axis", parent=self)
        self.__yAxisAction = qt.QAction("Y left axis", parent=self)
        self.__y2AxisAction = qt.QAction("Y right axis", parent=self)

        for action in (self.__xAxisAction, self.__yAxisAction, self.__y2AxisAction):
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(self._axesActionTriggered)
            self.addAction(action)

        # Listen to interaction configuration change
        plot.interaction().sigChanged.connect(self._interactionChanged)
        # Init the state
        self._interactionChanged()

    def getPlotWidget(self) -> Optional[PlotWidget]:
        return self.__plotRef()

    def _axesActionTriggered(self, checked=False):
        plot = self.getPlotWidget()
        if plot is None:
            return

        plot.interaction().setZoomEnabledAxes(
            self.__xAxisAction.isChecked(),
            self.__yAxisAction.isChecked(),
            self.__y2AxisAction.isChecked(),
        )

    def _interactionChanged(self):
        plot = self.getPlotWidget()
        if plot is None:
            return

        enabledAxes = plot.interaction().getZoomEnabledAxes()
        self.__xAxisAction.setChecked(enabledAxes.xaxis)
        self.__yAxisAction.setChecked(enabledAxes.yaxis)
        self.__y2AxisAction.setChecked(enabledAxes.y2axis)
