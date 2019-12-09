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

from __future__ import division


__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "29/03/2017"


from collections import namedtuple
import logging
import threading
import numpy

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, CancelledError

from ....utils.proxy import docstring
from ....math.combo import min_max
from ....utils.weakref import WeakList
from .._utils.delaunay import delaunay
from .core import PointsBase, ColormapMixIn, ScatterVisualizationMixIn
from .axis import Axis
from ._pick import PickingResult


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

# Functions to guess grid size from coordinates

def get_Z_line_length(array):
    """Return length of line if array is a Z-like 2D regular grid.

    :param numpy.ndarray array: The 1D array of coordinates to check
    :return: 0 if no line length could be found,
        else the number of element per line.
    :rtype: int
    """
    array = numpy.array(array, copy=False).reshape(-1)
    sign = numpy.sign(numpy.diff(array))
    if len(sign) == 0 or sign[0] == 0:  # We don't handle that
        return 0
    # Check this way to account for 0 sign (i.e., diff == 0)
    beginnings = numpy.where(sign == - sign[0])[0] + 1
    if len(beginnings) == 0:
        return 0
    length = beginnings[0]
    if numpy.all(numpy.equal(numpy.diff(beginnings), length)):
        return length
    return 0


def guess_Z_grid_size(x, y):
    """Guess the size of a grid from (x, y) coordinates.

    The grid might contain more elements than x and y,
    as the last line might be partly filled.

    :param numpy.ndarray x:
    :paran numpy.ndarray y:
    :returns: (order, (width, height), directions (dir x, dir y))
        of the regular grid, or None if could not guess one.
        is_transposed is 'C' if 'X' (i.e., column) is the fast dimension, else 'F'
        direction is either 1 or -1
    :rtype: Union[List(str,int),None]
    """
    width = get_Z_line_length(x)
    if width != 0:
        height = int(numpy.ceil(len(x) / width))
        dir_x = numpy.sign(x[width - 1] - x[0])
        dir_y = numpy.sign(y[-1] - y[0])
        if dir_x == 0 or dir_y == 0:
            return None
        else:
            return 'C', (width, height), (dir_x, dir_y)
    else:
        height = get_Z_line_length(y)
        if height != 0:
            width = int(numpy.ceil(len(y) / height))
            dir_x = numpy.sign(x[-1] - x[0])
            dir_y = numpy.sign(y[height - 1] - y[0])
            if dir_x == 0 or dir_y == 0:
                return None
            else:
                return 'F', (width, height), (dir_x, dir_y)
    return None


def is_monotonic(array):
    """Returns whether array is monotonic (increasing or decreasing).

    :param numpy.ndarray array: 1D array-like container.
    :returns: 1 if array is monotonically increasing,
       -1 if array is monotonically decreasing,
       0 if array is not monotonic
    :rtype: int
    """
    diff = numpy.diff(numpy.ravel(array))
    if numpy.all(diff >= 0):
        return 1
    elif numpy.all(diff <= 0):
        return -1
    else:
        return 0


GuessedGrid = namedtuple('GuessedGrid',
                         ['origin', 'scale', 'size', 'order'])


def guess_grid(x, y):
    """Guess a regular grid from the points.

    Result convention is (x, y)

    :param numpy.ndarray x: X coordinates of the points
    :param numpy.ndarray y: Y coordinates of the points
    :returns:
        origin: (ox, oy), scale: (sx, sy), size: (width, height), order: 'C' or 'F'
    :rtype: Union[GuessedGrid,None]
    """
    x_min, x_max = min_max(x)
    y_min, y_max = min_max(y)

    guess = guess_Z_grid_size(x, y)
    if guess is not None:
        order, size, directions = guess

    else:
        # Cannot guess a regular grid
        # Let's assume it's a single line
        order = 'C'  # or 'F' doesn't matter for a single line
        y_monotonic = is_monotonic(y)
        if is_monotonic(x) or y_monotonic:  # we can guess a line
            if not y_monotonic or x_max - x_min >= y_max - y_min:
                # x only is monotonic or both are and X varies more
                # line along X
                size = len(x), 1
                directions = numpy.sign(x[-1] - x[0]), 1
            else:
                # y only is monotonic or both are and Y varies more
                # line along Y
                size = 1, len(y)
                directions = 1, numpy.sign(y[-1] - y[0])

        else:  # Cannot guess a line from the points
            return None

    if directions[0] == 0 or directions[1] == 0:
        # Can happens (e.g., with a single point)
        # Avoid further issues like scale = 0
        return None

    scale = ((x_max - x_min) / max(1, size[0] - 1),
             (y_max - y_min) / max(1, size[1] - 1))
    if scale[0] == 0 and scale[1] == 0:
        scale = 1., 1.
    elif scale[0] == 0:
        scale = scale[1], scale[1]
    elif scale[1] == 0:
        scale = scale[0], scale[0]
    scale = directions[0] * scale[0], directions[1] * scale[1]

    origin = ((x_min if directions[0] > 0 else x_max) - 0.5 * scale[0],
              (y_min if directions[1] > 0 else y_max) - 0.5 * scale[1])

    return GuessedGrid(
        origin=origin, scale=scale, size=size, order=order)


class Scatter(PointsBase, ColormapMixIn, ScatterVisualizationMixIn):
    """Description of a scatter"""

    _DEFAULT_SELECTABLE = True
    """Default selectable state for scatter plots"""

    _SUPPORTED_SCATTER_VISUALIZATION = (
        ScatterVisualizationMixIn.Visualization.POINTS,
        ScatterVisualizationMixIn.Visualization.SOLID,
        ScatterVisualizationMixIn.Visualization.REGULAR_GRID,
        )
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

        # Cache regular grid info: origin, scale, size
        self.__cacheRegularGridInfo = None
        
    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        # Reset cache
        self.__cacheRegularGridInfo = None

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

        visualization = self.getVisualization()

        if visualization is self.Visualization.POINTS:
            return backend.addCurve(xFiltered, yFiltered,
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
                                    symbolsize=self.getSymbolSize(),
                                    baseline=None)

        elif visualization is self.Visualization.SOLID:
            plot = self.getPlot()
            if (plot is None or
                    plot.getXAxis().getScale() != Axis.LINEAR or
                    plot.getYAxis().getScale() != Axis.LINEAR):
                # Solid visualization is not available with log scaled axes
                return None

            triangulation = self._getDelaunay().result()
            if triangulation is None:
                _logger.warning(
                    'Cannot get a triangulation: Cannot display as solid surface')
                return None
            else:
                triangles = triangulation.simplices.astype(numpy.int32)
                return backend.addTriangles(xFiltered,
                                            yFiltered,
                                            triangles,
                                            color=rgbacolors,
                                            z=self.getZValue(),
                                            selectable=self.isSelectable(),
                                            alpha=self.getAlpha())

        elif visualization is self.Visualization.REGULAR_GRID:
            plot = self.getPlot()
            if (plot is None or
                    plot.getXAxis().getScale() != Axis.LINEAR or
                    plot.getYAxis().getScale() != Axis.LINEAR):
                # regular grid visualization is not available with log scaled axes
                return None

            guess = guess_grid(xFiltered, yFiltered)
            if guess is None:
                _logger.warning(
                    'Cannot guess a grid: Cannot display as regular grid image')
                return None

            width, height = guess.size
            dim0, dim1 = (height, width) if guess.order == 'C' else (width, height)

            if len(rgbacolors) != dim0 * dim1:
                # The points do not fill the whole image
                image = numpy.empty((dim0 * dim1, 4), dtype=rgbacolors.dtype)
                image[:len(rgbacolors)] = rgbacolors
                image[len(rgbacolors):] = 0, 0, 0, 0  # Transparent pixels
                image.shape = dim0, dim1, -1
            else:
                image = rgbacolors.reshape(dim0, dim1, -1)

            if guess.order == 'F':
                image = numpy.transpose(image, axes=(1, 0, 2))

            return backend.addImage(
                data=image,
                origin=guess.origin,
                scale=guess.scale,
                z=self.getZValue(),
                selectable=self.isSelectable(),
                draggable=False,
                colormap=None,
                alpha=self.getAlpha())

        else:
            _logger.error("Unhandled visualization %s", visualization)
            return None

    @docstring(PointsBase)
    def pick(self, x, y):
        result = super(Scatter, self).pick(x, y)

        # Specific handling of picking for the regular grid mode
        if (self.getVisualization() is self.Visualization.REGULAR_GRID and
                result is not None):
            plot = self.getPlot()
            if plot is None:
                return None

            dataPos = plot.pixelToData(x, y)
            if dataPos is None:
                return None

            if self.__cacheRegularGridInfo is None:
                return None

            origin, scale, size = self.__cacheRegularGridInfo
            column = int((dataPos[0] - origin[0]) / scale[0])
            row = int((dataPos[1] - origin[1]) / scale[1])
            index = row * size[0] + column
            if index >= len(self.getXData(copy=False)):  # OK as long as not log scale
                return None  # Image can be larger than scatter

            result = PickingResult(self, (index,))

        return result

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
