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
"""This module provides an abstract PlotToolButton that can be use to create
plot tools for a toolbar.
"""

from __future__ import annotations

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "20/12/2023"


import logging
import weakref

from silx.gui import qt


_logger = logging.getLogger(__name__)


class PlotToolButton(qt.QToolButton):
    """A QToolButton connected to a :class:`~silx.gui.plot.PlotWidget`."""

    def __init__(self, parent: qt.QWidget | None = None, plot=None):
        super(PlotToolButton, self).__init__(parent)
        self._plotRef = None
        if plot is not None:
            self.setPlot(plot)

    def plot(self):
        """
        Returns the plot connected to the widget.
        """
        return None if self._plotRef is None else self._plotRef()

    def setPlot(self, plot):
        """
        Set the plot connected to the widget

        :param plot: :class:`.PlotWidget` instance on which to operate.
        """
        previousPlot = self.plot()

        if previousPlot is plot:
            return
        if previousPlot is not None:
            self._disconnectPlot(previousPlot)

        if plot is None:
            self._plotRef = None
        else:
            self._plotRef = weakref.ref(plot)
            self._connectPlot(plot)

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
