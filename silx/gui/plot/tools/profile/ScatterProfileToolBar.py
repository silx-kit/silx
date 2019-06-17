# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""This module profile tools for scatter plots.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/06/2018"


import logging
import weakref

import numpy

from ._BaseProfileToolBar import _BaseProfileToolBar
from ... import items
from ....utils.concurrent import submitToQtMainThread


_logger = logging.getLogger(__name__)


class ScatterProfileToolBar(_BaseProfileToolBar):
    """QToolBar providing scatter plot profiling tools

    :param parent: See :class:`QToolBar`.
    :param plot: :class:`~silx.gui.plot.PlotWidget` on which to operate.
    :param str title: See :class:`QToolBar`.
    """

    def __init__(self, parent=None, plot=None, title='Scatter Profile'):
        super(ScatterProfileToolBar, self).__init__(parent, plot, title)

        self.__nPoints = 1024
        self.__scatterRef = None
        self.__futureInterpolator = None

        plot = self.getPlotWidget()
        if plot is not None:
            self._setScatterItem(plot._getActiveItem(kind='scatter'))
            plot.sigActiveScatterChanged.connect(self.__activeScatterChanged)

    def __activeScatterChanged(self, previous, legend):
        """Handle change of active scatter

        :param Union[str,None] previous:
        :param Union[str,None] legend:
        """
        plot = self.getPlotWidget()
        if plot is None or legend is None:
            scatter = None
        else:
            scatter = plot.getScatter(legend)
        self._setScatterItem(scatter)

    def _getScatterItem(self):
        """Returns the scatter item currently handled by this tool.

        :rtype: ~silx.gui.plot.items.Scatter
        """
        return None if self.__scatterRef is None else self.__scatterRef()

    def _setScatterItem(self, scatter):
        """Set the scatter tracked by this tool

        :param Union[None,silx.gui.plot.items.Scatter] scatter:
        """
        self.__futureInterpolator = None  # Reset currently expected future

        previousScatter = self._getScatterItem()
        if previousScatter is not None:
            previousScatter.sigItemChanged.disconnect(
                self.__scatterItemChanged)

        if scatter is None:
            self.__scatterRef = None
        else:
            self.__scatterRef = weakref.ref(scatter)
            scatter.sigItemChanged.connect(self.__scatterItemChanged)

        # Refresh profile
        self.updateProfile()

    def __scatterItemChanged(self, event):
        """Handle update of active scatter plot item

        :param ItemChangedType event:
        """
        if event == items.ItemChangedType.DATA:
            self.updateProfile()  # Refresh profile

    def hasPendingOperations(self):
        """Returns True if waiting for an interpolator to be ready

        :rtype: bool
        """
        return (self.__futureInterpolator is not None and
                not self.__futureInterpolator.done())

    # Number of points

    def getNPoints(self):
        """Returns the number of points of the profiles

        :rtype: int
        """
        return self.__nPoints

    def setNPoints(self, npoints):
        """Set the number of points of the profiles

        :param int npoints:
        """
        npoints = int(npoints)
        if npoints < 1:
            raise ValueError("Unsupported number of points: %d" % npoints)
        elif npoints != self.__nPoints:
            self.__nPoints = npoints
            self.updateProfile()

    # Overridden methods

    def computeProfileTitle(self, x0, y0, x1, y1):
        """Compute corresponding plot title

        :param float x0: Profile start point X coord
        :param float y0: Profile start point Y coord
        :param float x1: Profile end point X coord
        :param float y1: Profile end point Y coord
        :return: Title to use
        :rtype: str
        """
        if self.hasPendingOperations():
            return 'Pre-processing data...'
        else:
            return super(ScatterProfileToolBar, self).computeProfileTitle(
                x0, y0, x1, y1)

    def __futureDone(self, future):
        """Handle completion of the interpolator creation"""
        if future is self.__futureInterpolator:
            # Only handle future callbacks for the current one
            submitToQtMainThread(self.updateProfile)

    def computeProfile(self, x0, y0, x1, y1):
        """Compute corresponding profile

        :param float x0: Profile start point X coord
        :param float y0: Profile start point Y coord
        :param float x1: Profile end point X coord
        :param float y1: Profile end point Y coord
        :return: (points, values) profile data or None
        """
        scatter = self._getScatterItem()
        if scatter is None or self.hasPendingOperations():
            return None

        # Lazy async request of the interpolator
        future = scatter._getInterpolator()
        if future is not self.__futureInterpolator:
            # First time we request this interpolator
            self.__futureInterpolator = future
            if not future.done():
                future.add_done_callback(self.__futureDone)
                return None

        if future.cancelled() or future.exception() is not None:
            return None  # Something went wrong

        interpolator = future.result()
        if interpolator is None:
            return None  # Cannot init an interpolator

        nPoints = self.getNPoints()
        points = numpy.transpose((
            numpy.linspace(x0, x1, nPoints, endpoint=True),
            numpy.linspace(y0, y1, nPoints, endpoint=True)))

        values = interpolator(points)

        if not numpy.any(numpy.isfinite(values)):
            return None  # Profile outside convex hull

        return points, values
