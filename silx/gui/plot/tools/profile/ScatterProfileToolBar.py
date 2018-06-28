# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
import threading
import time

import numpy

try:
    from scipy.interpolate import LinearNDInterpolator
except ImportError:
    LinearNDInterpolator = None

    # Fallback using local Delaunay and matplotlib interpolator
    from silx.third_party.scipy_spatial import Delaunay
    import matplotlib.tri

from ._BaseProfileToolBar import _BaseProfileToolBar
from .... import qt
from ... import items


_logger = logging.getLogger(__name__)


# TODO support log scale


class _InterpolatorInitThread(qt.QThread):
    """Thread building a scatter interpolator

    This works in greedy mode in that the signal is only emitted
    when no other request is pending
    """

    sigInterpolatorReady = qt.Signal(object)
    """Signal emitted whenever an interpolator is ready

    It provides a 3-tuple (points, values, interpolator)
    """

    _RUNNING_THREADS_TO_DELETE = []
    """Store reference of no more used threads but still running"""

    def __init__(self):
        super(_InterpolatorInitThread, self).__init__()
        self._lock = threading.RLock()
        self._pendingData = None
        self._firstFallbackRun = True

    def discard(self, obj=None):
        """Wait for pending thread to complete and delete then

        Connect this to the destroyed signal of widget using this thread
        """
        if self.isRunning():
            self.cancel()
            self._RUNNING_THREADS_TO_DELETE.append(self)  # Keep a reference
            self.finished.connect(self.__finished)

    def __finished(self):
        """Handle finished signal of threads to delete"""
        try:
            self._RUNNING_THREADS_TO_DELETE.remove(self)
        except ValueError:
            _logger.warning('Finished thread no longer in reference list')

    def request(self, points, values):
        """Request new initialisation of interpolator

        :param numpy.ndarray points: Point coordinates (N, D)
        :param numpy.ndarray values: Values the N points (1D array)
        """
        with self._lock:
            # Possibly replace already pending data
            self._pendingData = points, values

        if not self.isRunning():
            self.start()

    def cancel(self):
        """Cancel any running/pending requests"""
        with self._lock:
            self._pendingData = 'cancelled'

    def run(self):
        """Run the init of the scatter interpolator"""
        if LinearNDInterpolator is None:
            self.run_matplotlib()
        else:
            self.run_scipy()

    def run_matplotlib(self):
        """Run the init of the scatter interpolator"""
        if self._firstFallbackRun:
            self._firstFallbackRun = False
            _logger.warning(
                "scipy.spatial.LinearNDInterpolator not available: "
                "Scatter plot interpolator initialisation can freeze the GUI.")

        while True:
            with self._lock:
                data = self._pendingData
                self._pendingData = None

            if data in (None, 'cancelled'):
                return

            points, values = data

            startTime = time.time()
            try:
                delaunay = Delaunay(points)
            except:
                _logger.warning(
                    "Cannot triangulate scatter data")
            else:
                with self._lock:
                    data = self._pendingData

                if data is not None:  # Break point
                    _logger.info('Interpolator discarded after %f s',
                                 time.time() - startTime)
                else:

                    x, y = points.T
                    triangulation = matplotlib.tri.Triangulation(
                        x, y, triangles=delaunay.simplices)

                    interpolator = matplotlib.tri.LinearTriInterpolator(
                        triangulation, values)

                    with self._lock:
                        data = self._pendingData

                    if data is not None:
                        _logger.info('Interpolator discarded after %f s',
                                     time.time() - startTime)
                    else:
                        # No other processing requested: emit the signal
                        _logger.info("Interpolator initialised in %f s",
                                     time.time() - startTime)

                        # Wrap interpolator to have same API as scipy's one
                        def wrapper(points):
                            return interpolator(*points.T)

                        self.sigInterpolatorReady.emit(
                            (points, values, wrapper))

    def run_scipy(self):
        """Run the init of the scatter interpolator"""
        while True:
            with self._lock:
                data = self._pendingData
                self._pendingData = None

            if data in (None, 'cancelled'):
                return

            points, values = data

            startTime = time.time()
            try:
                interpolator = LinearNDInterpolator(points, values)
            except:
                _logger.warning(
                    "Cannot initialise scatter profile interpolator")
            else:
                with self._lock:
                    data = self._pendingData

                if data is not None:  # Break point
                    _logger.info('Interpolator discarded after %f s',
                                 time.time() - startTime)
                else:
                    # First call takes a while, do it here
                    interpolator([(0., 0.)])

                    with self._lock:
                        data = self._pendingData

                    if data is not None:
                        _logger.info('Interpolator discarded after %f s',
                                     time.time() - startTime)
                    else:
                        # No other processing requested: emit the signal
                        _logger.info("Interpolator initialised in %f s",
                                     time.time() - startTime)
                        self.sigInterpolatorReady.emit(
                            (points, values, interpolator))


class ScatterProfileToolBar(_BaseProfileToolBar):
    """QToolBar providing scatter plot profiling tools

    :param parent: See :class:`QToolBar`.
    :param plot: :class:`~silx.gui.plot.PlotWidget` on which to operate.
    :param str title: See :class:`QToolBar`.
    """

    def __init__(self, parent=None, plot=None, title='Scatter Profile'):
        super(ScatterProfileToolBar, self).__init__(parent, plot, title)

        self.__nPoints = 1024
        self.__interpolator = None
        self.__interpolatorCache = None  # points, values, interpolator

        self.__initThread = _InterpolatorInitThread()
        self.destroyed.connect(self.__initThread.discard)
        self.__initThread.sigInterpolatorReady.connect(
            self.__interpolatorReady)

        roiManager = self._getRoiManager()
        if roiManager is None:
            _logger.error(
                "Error during scatter profile toolbar initialisation")
        else:
            roiManager.sigInteractiveModeStarted.connect(
                self.__interactionStarted)
            roiManager.sigInteractiveModeFinished.connect(
                self.__interactionFinished)
            if roiManager.isStarted():
                self.__interactionStarted(roiManager.getCurrentInteractionModeRoiClass())

    def __interactionStarted(self, roiClass):
        """Handle start of ROI interaction"""
        plot = self.getPlotWidget()
        if plot is None:
            return

        plot.sigActiveScatterChanged.connect(self.__activeScatterChanged)

        scatter = plot._getActiveItem(kind='scatter')
        legend = None if scatter is None else scatter.getLegend()
        self.__activeScatterChanged(None, legend)

    def __interactionFinished(self):
        """Handle end of ROI interaction"""
        plot = self.getPlotWidget()
        if plot is None:
            return

        plot.sigActiveScatterChanged.disconnect(self.__activeScatterChanged)

        scatter = plot._getActiveItem(kind='scatter')
        legend = None if scatter is None else scatter.getLegend()
        self.__activeScatterChanged(legend, None)

    def __activeScatterChanged(self, previous, legend):
        """Handle change of active scatter

        :param Union[str,None] previous:
        :param Union[str,None] legend:
        """
        self.__initThread.cancel()

        # Reset interpolator
        self.__interpolator = None

        plot = self.getPlotWidget()
        if plot is None:
            _logger.error("Associated PlotWidget no longer exists")

        else:
            if previous is not None:  # Disconnect signal
                scatter = plot.getScatter(previous)
                if scatter is not None:
                    scatter.sigItemChanged.disconnect(
                        self.__scatterItemChanged)

            if legend is not None:
                scatter = plot.getScatter(legend)
                if scatter is None:
                    _logger.error("Cannot retrieve active scatter")

                else:
                    scatter.sigItemChanged.connect(self.__scatterItemChanged)
                    points = numpy.transpose(numpy.array((
                        scatter.getXData(copy=False),
                        scatter.getYData(copy=False))))
                    values = scatter.getValueData(copy=False)

                    self.__updateInterpolator(points, values)

        # Refresh profile
        self.updateProfile()

    def __scatterItemChanged(self, event):
        """Handle update of active scatter plot item

        :param ItemChangedType event:
        """
        if event == items.ItemChangedType.DATA:
            self.__interpolator = None
            scatter = self.sender()
            if scatter is None:
                _logger.error("Cannot retrieve updated scatter item")

            else:
                points = numpy.transpose(numpy.array((
                    scatter.getXData(copy=False),
                    scatter.getYData(copy=False))))
                values = scatter.getValueData(copy=False)

                self.__updateInterpolator(points, values)

    # Handle interpolator init thread

    def __updateInterpolator(self, points, values):
        """Update used interpolator with new data"""
        if (self.__interpolatorCache is not None and
                len(points) == len(self.__interpolatorCache[0]) and
                numpy.all(numpy.equal(self.__interpolatorCache[0], points)) and
                numpy.all(numpy.equal(self.__interpolatorCache[1], values))):
            # Reuse previous interpolator
            _logger.info(
                'Scatter changed: Reuse previous interpolator')
            self.__interpolator = self.__interpolatorCache[2]

        else:
            # Interpolator needs update: Start background processing
            _logger.info(
                'Scatter changed: Rebuild interpolator')
            self.__interpolator = None
            self.__interpolatorCache = None
            self.__initThread.request(points, values)

    def __interpolatorReady(self, data):
        """Handle end of init interpolator thread"""
        points, values, interpolator = data
        self.__interpolator = interpolator
        self.__interpolatorCache = None if interpolator is None else data
        self.updateProfile()

    def hasPendingOperations(self):
        return self.__initThread.isRunning()

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
        else:
            self.__nPoints = npoints

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

    def computeProfile(self, x0, y0, x1, y1):
        """Compute corresponding profile

        :param float x0: Profile start point X coord
        :param float y0: Profile start point Y coord
        :param float x1: Profile end point X coord
        :param float y1: Profile end point Y coord
        :return: (points, values) profile data or None
        """
        if self.__interpolator is None:
            return None

        nPoints = self.getNPoints()

        points = numpy.transpose((
            numpy.linspace(x0, x1, nPoints, endpoint=True),
            numpy.linspace(y0, y1, nPoints, endpoint=True)))

        values = self.__interpolator(points)

        if not numpy.any(numpy.isfinite(values)):
            return None  # Profile outside convex hull

        return points, values
