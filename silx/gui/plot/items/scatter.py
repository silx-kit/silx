# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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
"""This module provides the :class:`Scatter` item of the :class:`Plot`.
"""

__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "29/03/2017"


import logging
import threading
import numpy

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, CancelledError

from ....utils.weakref import WeakList
from .._utils.delaunay import delaunay
from .core import PointsBase, ColormapMixIn, ScatterVisualizationMixIn
from .axis import Axis


_logger = logging.getLogger(__name__)


class _GreedyThreadPoolExecutor(ThreadPoolExecutor):
    """:class:`ThreadPoolExecutor` with an extra :meth:`submit_greedy` method.
    """

    def __init__(self, *args, **kwargs):
        super(_GreedyThreadPoolExecutor, self).__init__(*args, **kwargs)
        self.__futures = defaultdict(WeakList)
        self.__lock = threading.RLock()

    def submit_greedy(self, queue, fn, *args, **kwargs):
        """Same as :meth:`submit` but cancel previous tasks in given queue.

        This means that when a new task is submitted for a given queue,
        all other pending tasks of that queue are cancelled.

        :param queue: Identifier of the queue. This must be hashable.
        :param callable fn: The callable to call with provided extra arguments
        :return: Future corresponding to this task
        :rtype: concurrent.futures.Future
        """
        with self.__lock:
            # Cancel previous tasks in given queue
            for future in self.__futures.pop(queue, []):
                if not future.done():
                    future.cancel()

            future = super(_GreedyThreadPoolExecutor, self).submit(
                fn, *args, **kwargs)
            self.__futures[queue].append(future)

        return future


class Scatter(PointsBase, ColormapMixIn, ScatterVisualizationMixIn):
    """Description of a scatter"""

    _DEFAULT_SELECTABLE = True
    """Default selectable state for scatter plots"""

    _SUPPORTED_SCATTER_VISUALIZATION = (
        ScatterVisualizationMixIn.Visualization.POINTS,
        ScatterVisualizationMixIn.Visualization.SOLID)
    """Overrides supported Visualizations"""

    def __init__(self):
        PointsBase.__init__(self)
        ColormapMixIn.__init__(self)
        ScatterVisualizationMixIn.__init__(self)
        self._value = ()
        self.__alpha = None
        # Cache Delaunay triangulation future object
        self.__delaunayFuture = None
        # Cache interpolator future object
        self.__interpolatorFuture = None
        self.__executor = None

        # Cache triangles: x, y, indices
        self.__cacheTriangles = None, None, None
        
    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        # Filter-out values <= 0
        xFiltered, yFiltered, valueFiltered, xerror, yerror = self.getData(
            copy=False, displayed=True)

        # Remove not finite numbers (this includes filtered out x, y <= 0)
        mask = numpy.logical_and(numpy.isfinite(xFiltered), numpy.isfinite(yFiltered))
        xFiltered = xFiltered[mask]
        yFiltered = yFiltered[mask]

        if len(xFiltered) == 0:
            return None  # No data to display, do not add renderer to backend

        # Compute colors
        cmap = self.getColormap()
        rgbacolors = cmap.applyToData(self._value)

        if self.__alpha is not None:
            rgbacolors[:, -1] = (rgbacolors[:, -1] * self.__alpha).astype(numpy.uint8)

        # Apply mask to colors
        rgbacolors = rgbacolors[mask]

        if self.getVisualization() is self.Visualization.POINTS:
            return backend.addCurve(xFiltered, yFiltered, self.getLegend(),
                                    color=rgbacolors,
                                    symbol=self.getSymbol(),
                                    linewidth=0,
                                    linestyle="",
                                    yaxis='left',
                                    xerror=xerror,
                                    yerror=yerror,
                                    z=self.getZValue(),
                                    selectable=self.isSelectable(),
                                    fill=False,
                                    alpha=self.getAlpha(),
                                    symbolsize=self.getSymbolSize())

        else:  # 'solid'
            plot = self.getPlot()
            if (plot is None or
                    plot.getXAxis().getScale() != Axis.LINEAR or
                    plot.getYAxis().getScale() != Axis.LINEAR):
                # Solid visualization is not available with log scaled axes
                return None

            triangulation = self._getDelaunay().result()
            if triangulation is None:
                return None
            else:
                triangles = triangulation.simplices.astype(numpy.int32)
                return backend.addTriangles(xFiltered,
                                            yFiltered,
                                            triangles,
                                            legend=self.getLegend(),
                                            color=rgbacolors,
                                            z=self.getZValue(),
                                            selectable=self.isSelectable(),
                                            alpha=self.getAlpha())

    def __getExecutor(self):
        """Returns async greedy executor

        :rtype: _GreedyThreadPoolExecutor
        """
        if self.__executor is None:
            self.__executor = _GreedyThreadPoolExecutor(max_workers=2)
        return self.__executor

    def _getDelaunay(self):
        """Returns a :class:`Future` which result is the Delaunay object.

        :rtype: concurrent.futures.Future
        """
        if self.__delaunayFuture is None or self.__delaunayFuture.cancelled():
            # Need to init a new delaunay
            x, y = self.getData(copy=False)[:2]
            # Remove not finite points
            mask = numpy.logical_and(numpy.isfinite(x), numpy.isfinite(y))

            self.__delaunayFuture = self.__getExecutor().submit_greedy(
                'delaunay', delaunay, x[mask], y[mask])

        return self.__delaunayFuture

    @staticmethod
    def __initInterpolator(delaunayFuture, values):
        """Returns an interpolator for the given data points

        :param concurrent.futures.Future delaunayFuture:
            Future object which result is a Delaunay object
        :param numpy.ndarray values: The data value of valid points.
        :rtype: Union[callable,None]
        """
        # Wait for Delaunay to complete
        try:
            triangulation = delaunayFuture.result()
        except CancelledError:
            triangulation = None

        if triangulation is None:
            interpolator = None  # Error case
        else:
            # Lazy-loading of interpolator
            try:
                from scipy.interpolate import LinearNDInterpolator
            except ImportError:
                LinearNDInterpolator = None

            if LinearNDInterpolator is not None:
                interpolator = LinearNDInterpolator(triangulation, values)

                # First call takes a while, do it here
                interpolator([(0., 0.)])

            else:
                # Fallback using matplotlib interpolator
                import matplotlib.tri

                x, y = triangulation.points.T
                tri = matplotlib.tri.Triangulation(
                    x, y, triangles=triangulation.simplices)
                mplInterpolator = matplotlib.tri.LinearTriInterpolator(
                    tri, values)

                # Wrap interpolator to have same API as scipy's one
                def interpolator(points):
                    return mplInterpolator(*points.T)

        return interpolator

    def _getInterpolator(self):
        """Returns a :class:`Future` which result is the interpolator.

        The interpolator is a callable taking an array Nx2 of points
        as a single argument.
        The :class:`Future` result is None in case the interpolator cannot
        be initialized.

        :rtype: concurrent.futures.Future
        """
        if (self.__interpolatorFuture is None or
                self.__interpolatorFuture.cancelled()):
            # Need to init a new interpolator
            x, y, values = self.getData(copy=False)[:3]
            # Remove not finite points
            mask = numpy.logical_and(numpy.isfinite(x), numpy.isfinite(y))
            x, y, values = x[mask], y[mask], values[mask]

            self.__interpolatorFuture = self.__getExecutor().submit_greedy(
                'interpolator',
                self.__initInterpolator, self._getDelaunay(), values)
        return self.__interpolatorFuture

    def _logFilterData(self, xPositive, yPositive):
        """Filter out values with x or y <= 0 on log axes

        :param bool xPositive: True to filter arrays according to X coords.
        :param bool yPositive: True to filter arrays according to Y coords.
        :return: The filtered arrays or unchanged object if not filtering needed
        :rtype: (x, y, value, xerror, yerror)
        """
        # overloaded from PointsBase to filter also value.
        value = self.getValueData(copy=False)

        if xPositive or yPositive:
            clipped = self._getClippingBoolArray(xPositive, yPositive)

            if numpy.any(clipped):
                # copy to keep original array and convert to float
                value = numpy.array(value, copy=True, dtype=numpy.float)
                value[clipped] = numpy.nan

        x, y, xerror, yerror = PointsBase._logFilterData(self, xPositive, yPositive)

        return x, y, value, xerror, yerror

    def getValueData(self, copy=True):
        """Returns the value assigned to the scatter data points.

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._value, copy=copy)

    def getAlphaData(self, copy=True):
        """Returns the alpha (transparency) assigned to the scatter data points.

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self.__alpha, copy=copy)

    def getData(self, copy=True, displayed=False):
        """Returns the x, y coordinates and the value of the data points

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :param bool displayed: True to only get curve points that are displayed
                               in the plot. Default: False.
                               Note: If plot has log scale, negative points
                               are not displayed.
        :returns: (x, y, value, xerror, yerror)
        :rtype: 5-tuple of numpy.ndarray
        """
        if displayed:
            data = self._getCachedData()
            if data is not None:
                assert len(data) == 5
                return data

        return (self.getXData(copy),
                self.getYData(copy),
                self.getValueData(copy),
                self.getXErrorData(copy),
                self.getYErrorData(copy))

    # reimplemented from PointsBase to handle `value`
    def setData(self, x, y, value, xerror=None, yerror=None, alpha=None, copy=True):
        """Set the data of the scatter.

        :param numpy.ndarray x: The data corresponding to the x coordinates.
        :param numpy.ndarray y: The data corresponding to the y coordinates.
        :param numpy.ndarray value: The data corresponding to the value of
                                    the data points.
        :param xerror: Values with the uncertainties on the x values
        :type xerror: A float, or a numpy.ndarray of float32.
                      If it is an array, it can either be a 1D array of
                      same length as the data or a 2D array with 2 rows
                      of same length as the data: row 0 for positive errors,
                      row 1 for negative errors.
        :param yerror: Values with the uncertainties on the y values
        :type yerror: A float, or a numpy.ndarray of float32. See xerror.
        :param alpha: Values with the transparency (between 0 and 1)
        :type alpha: A float, or a numpy.ndarray of float32 
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        """
        value = numpy.array(value, copy=copy)
        assert value.ndim == 1
        assert len(x) == len(value)

        # Reset triangulation and interpolator
        if self.__delaunayFuture is not None:
            self.__delaunayFuture.cancel()
            self.__delaunayFuture = None
        if self.__interpolatorFuture is not None:
            self.__interpolatorFuture.cancel()
            self.__interpolatorFuture = None

        self._value = value

        if alpha is not None:
            # Make sure alpha is an array of float in [0, 1]
            alpha = numpy.array(alpha, copy=copy)
            assert alpha.ndim == 1
            assert len(x) == len(alpha)
            if alpha.dtype.kind != 'f':
                alpha = alpha.astype(numpy.float32)
            if numpy.any(numpy.logical_or(alpha < 0., alpha > 1.)):
                alpha = numpy.clip(alpha, 0., 1.)
        self.__alpha = alpha

        # set x, y, xerror, yerror

        # call self._updated + plot._invalidateDataRange()
        PointsBase.setData(self, x, y, xerror, yerror, copy)
