# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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
from ....math.histogram import Histogramnd
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


# Functions to guess grid shape from coordinates

def _get_z_line_length(array):
    """Return length of line if array is a Z-like 2D regular grid.

    :param numpy.ndarray array: The 1D array of coordinates to check
    :return: 0 if no line length could be found,
        else the number of element per line.
    :rtype: int
    """
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


def _guess_z_grid_shape(x, y):
    """Guess the shape of a grid from (x, y) coordinates.

    The grid might contain more elements than x and y,
    as the last line might be partly filled.

    :param numpy.ndarray x:
    :paran numpy.ndarray y:
    :returns: (order, (height, width)) of the regular grid,
        or None if could not guess one.
        'order' is 'row' if X (i.e., column) is the fast dimension, else 'column'.
    :rtype: Union[List(str,int),None]
    """
    width = _get_z_line_length(x)
    if width != 0:
        return 'row', (int(numpy.ceil(len(x) / width)), width)
    else:
        height = _get_z_line_length(y)
        if height != 0:
            return 'column', (height, int(numpy.ceil(len(y) / height)))
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
    with numpy.errstate(invalid='ignore'):
        if numpy.all(diff >= 0):
            return 1
        elif numpy.all(diff <= 0):
            return -1
        else:
            return 0


def _guess_grid(x, y):
    """Guess a regular grid from the points.

    Result convention is (x, y)

    :param numpy.ndarray x: X coordinates of the points
    :param numpy.ndarray y: Y coordinates of the points
    :returns: (order, (height, width)
        order is 'row' or 'column'
    :rtype: Union[List[str,List[int]],None]
    """
    x, y = numpy.ravel(x), numpy.ravel(y)

    guess = _guess_z_grid_shape(x, y)
    if guess is not None:
        return guess

    else:
        # Cannot guess a regular grid
        # Let's assume it's a single line
        order = 'row'  # or 'column' doesn't matter for a single line
        y_monotonic = is_monotonic(y)
        if is_monotonic(x) or y_monotonic:  # we can guess a line
            x_min, x_max = min_max(x)
            y_min, y_max = min_max(y)

            if not y_monotonic or x_max - x_min >= y_max - y_min:
                # x only is monotonic or both are and X varies more
                # line along X
                shape = 1, len(x)
            else:
                # y only is monotonic or both are and Y varies more
                # line along Y
                shape = len(y), 1

        else:  # Cannot guess a line from the points
            return None

    return order, shape


def _quadrilateral_grid_coords(points):
    """Compute an irregular grid of quadrilaterals from a set of points

    The input points are expected to lie on a grid.

    :param numpy.ndarray points:
       3D data set of 2D input coordinates (height, width, 2)
       height and width must be at least 2.
    :return: 3D dataset of 2D coordinates of the grid (height+1, width+1, 2)
    """
    assert points.ndim == 3
    assert points.shape[0] >= 2
    assert points.shape[1] >= 2
    assert points.shape[2] == 2

    dim0, dim1 = points.shape[:2]
    grid_points = numpy.zeros((dim0 + 1, dim1 + 1, 2), dtype=numpy.float64)

    # Compute inner points as mean of 4 neighbours
    neighbour_view = numpy.lib.stride_tricks.as_strided(
        points,
        shape=(dim0 - 1, dim1 - 1, 2, 2, points.shape[2]),
        strides=points.strides[:2] + points.strides[:2] + points.strides[-1:], writeable=False)
    inner_points = numpy.mean(neighbour_view, axis=(2, 3))
    grid_points[1:-1, 1:-1] = inner_points

    # Compute 'vertical' sides
    # Alternative: grid_points[1:-1, [0, -1]] = points[:-1, [0, -1]] + points[1:, [0, -1]] - inner_points[:, [0, -1]]
    grid_points[1:-1, [0, -1], 0] = points[:-1, [0, -1], 0] + points[1:, [0, -1], 0] - inner_points[:, [0, -1], 0]
    grid_points[1:-1, [0, -1], 1] = inner_points[:, [0, -1], 1]

    # Compute 'horizontal' sides
    grid_points[[0, -1], 1:-1, 0] = inner_points[[0, -1], :, 0]
    grid_points[[0, -1], 1:-1, 1] = points[[0, -1], :-1, 1] + points[[0, -1], 1:, 1] - inner_points[[0, -1], :, 1]

    # Compute corners
    d0, d1 = [0, 0, -1, -1], [0, -1, -1, 0]
    grid_points[d0, d1] = 2 * points[d0, d1] - inner_points[d0, d1]
    return grid_points


def _quadrilateral_grid_as_triangles(points):
    """Returns the points and indices to make a grid of quadirlaterals

    :param numpy.ndarray points:
        3D array of points (height, width, 2)
    :return: triangle corners (4 * N, 2), triangle indices (2 * N, 3)
        With N = height * width, the number of input points
    """
    nbpoints = numpy.prod(points.shape[:2])

    grid = _quadrilateral_grid_coords(points)
    coords = numpy.empty((4 * nbpoints, 2), dtype=grid.dtype)
    coords[::4] = grid[:-1, :-1].reshape(-1, 2)
    coords[1::4] = grid[1:, :-1].reshape(-1, 2)
    coords[2::4] = grid[:-1, 1:].reshape(-1, 2)
    coords[3::4] = grid[1:, 1:].reshape(-1, 2)

    indices = numpy.empty((2 * nbpoints, 3), dtype=numpy.uint32)
    indices[::2, 0] = numpy.arange(0, 4 * nbpoints, 4)
    indices[::2, 1] = numpy.arange(1, 4 * nbpoints, 4)
    indices[::2, 2] = numpy.arange(2, 4 * nbpoints, 4)
    indices[1::2, 0] = indices[::2, 1]
    indices[1::2, 1] = indices[::2, 2]
    indices[1::2, 2] = numpy.arange(3, 4 * nbpoints, 4)

    return coords, indices


_RegularGridInfo = namedtuple(
    '_RegularGridInfo', ['bounds', 'origin', 'scale', 'shape', 'order'])


_HistogramInfo = namedtuple(
    '_HistogramInfo', ['mean', 'count', 'sum', 'origin', 'scale', 'shape'])


class Scatter(PointsBase, ColormapMixIn, ScatterVisualizationMixIn):
    """Description of a scatter"""

    _DEFAULT_SELECTABLE = True
    """Default selectable state for scatter plots"""

    _SUPPORTED_SCATTER_VISUALIZATION = (
        ScatterVisualizationMixIn.Visualization.POINTS,
        ScatterVisualizationMixIn.Visualization.SOLID,
        ScatterVisualizationMixIn.Visualization.REGULAR_GRID,
        ScatterVisualizationMixIn.Visualization.IRREGULAR_GRID,
        ScatterVisualizationMixIn.Visualization.BINNED_STATISTIC,
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

        # Cache regular grid and histogram info
        self.__cacheRegularGridInfo = None
        self.__cacheHistogramInfo = None

    def _updateColormappedData(self):
        """Update the colormapped data, to be called when changed"""
        if self.getVisualization() is self.Visualization.BINNED_STATISTIC:
            histoInfo = self.__getHistogramInfo()
            if histoInfo is None:
                data = None
            else:
                data = getattr(
                    histoInfo,
                    self.getVisualizationParameter(
                        self.VisualizationParameter.BINNED_STATISTIC_FUNCTION))
        else:
            data = self.getValueData(copy=False)
        self._setColormappedData(data, copy=False)

    @docstring(ScatterVisualizationMixIn)
    def setVisualization(self, mode):
        previous = self.getVisualization()
        if super().setVisualization(mode):
            if (bool(mode is self.Visualization.BINNED_STATISTIC) ^
                    bool(previous is self.Visualization.BINNED_STATISTIC)):
                self._updateColormappedData()
            return True
        else:
            return False

    @docstring(ScatterVisualizationMixIn)
    def setVisualizationParameter(self, parameter, value):
        if super(Scatter, self).setVisualizationParameter(parameter, value):
            if parameter in (self.VisualizationParameter.GRID_BOUNDS,
                             self.VisualizationParameter.GRID_MAJOR_ORDER,
                             self.VisualizationParameter.GRID_SHAPE):
                self.__cacheRegularGridInfo = None

            if parameter in (self.VisualizationParameter.BINNED_STATISTIC_SHAPE,
                             self.VisualizationParameter.BINNED_STATISTIC_FUNCTION):
                if parameter == self.VisualizationParameter.BINNED_STATISTIC_SHAPE:
                    self.__cacheHistogramInfo = None  # Clean-up cache
                if self.getVisualization() is self.Visualization.BINNED_STATISTIC:
                    self._updateColormappedData()
            return True
        else:
            return False

    @docstring(ScatterVisualizationMixIn)
    def getCurrentVisualizationParameter(self, parameter):
        value = self.getVisualizationParameter(parameter)
        if value is not None:
            return value  # Value has been set, return it

        elif parameter is self.VisualizationParameter.GRID_BOUNDS:
            grid = self.__getRegularGridInfo()
            return None if grid is None else grid.bounds
        
        elif parameter is self.VisualizationParameter.GRID_MAJOR_ORDER:
            grid = self.__getRegularGridInfo()
            return None if grid is None else grid.order

        elif parameter is self.VisualizationParameter.GRID_SHAPE:
            grid = self.__getRegularGridInfo()
            return None if grid is None else grid.shape

        elif parameter is self.VisualizationParameter.BINNED_STATISTIC_SHAPE:
            info = self.__getHistogramInfo()
            return None if info is None else info.shape

        else:
            raise NotImplementedError()

    def __getRegularGridInfo(self):
        """Get grid info"""
        if self.__cacheRegularGridInfo is None:
            shape = self.getVisualizationParameter(
                self.VisualizationParameter.GRID_SHAPE)
            order = self.getVisualizationParameter(
                self.VisualizationParameter.GRID_MAJOR_ORDER)
            if shape is None or order is None:
                guess = _guess_grid(self.getXData(copy=False),
                                    self.getYData(copy=False))
                if guess is None:
                    _logger.warning(
                        'Cannot guess a grid: Cannot display as regular grid image')
                    return None
                if shape is None:
                    shape = guess[1]
                if order is None:
                    order = guess[0]

            nbpoints = len(self.getXData(copy=False))
            if nbpoints > shape[0] * shape[1]:
                # More data points that provided grid shape: enlarge grid
                _logger.warning(
                    "More data points than provided grid shape size: extends grid")
                dim0, dim1 = shape
                if order == 'row':  # keep dim1, enlarge dim0
                    dim0 = nbpoints // dim1 + (1 if nbpoints % dim1 else 0)
                else:  # keep dim0, enlarge dim1
                    dim1 = nbpoints // dim0 + (1 if nbpoints % dim0 else 0)
                shape = dim0, dim1

            bounds = self.getVisualizationParameter(
                self.VisualizationParameter.GRID_BOUNDS)
            if bounds is None:
                x, y = self.getXData(copy=False), self.getYData(copy=False)
                min_, max_ = min_max(x)
                xRange = (min_, max_) if (x[0] - min_) < (max_ - x[0]) else (max_, min_)
                min_, max_ = min_max(y)
                yRange = (min_, max_) if (y[0] - min_) < (max_ - y[0]) else (max_, min_)
                bounds = (xRange[0], yRange[0]), (xRange[1], yRange[1])

            begin, end = bounds
            scale = ((end[0] - begin[0]) / max(1, shape[1] - 1),
                     (end[1] - begin[1]) / max(1, shape[0] - 1))
            if scale[0] == 0 and scale[1] == 0:
                scale = 1., 1.
            elif scale[0] == 0:
                scale = scale[1], scale[1]
            elif scale[1] == 0:
                scale = scale[0], scale[0]

            origin = begin[0] - 0.5 * scale[0], begin[1] - 0.5 * scale[1]

            self.__cacheRegularGridInfo = _RegularGridInfo(
                bounds=bounds, origin=origin, scale=scale, shape=shape, order=order)

        return self.__cacheRegularGridInfo

    def __getHistogramInfo(self):
        """Get histogram info"""
        if self.__cacheHistogramInfo is None:
            shape = self.getVisualizationParameter(
                self.VisualizationParameter.BINNED_STATISTIC_SHAPE)
            if shape is None:
                shape = 100, 100 # TODO compute auto shape

            x, y, values = self.getData(copy=False)[:3]
            if len(x) == 0:  # No histogram
                return None

            if not numpy.issubdtype(x.dtype, numpy.floating):
                x = x.astype(numpy.float64)
            if not numpy.issubdtype(y.dtype, numpy.floating):
                y = y.astype(numpy.float64)
            if not numpy.issubdtype(values.dtype, numpy.floating):
                values = values.astype(numpy.float64)

            ranges = (tuple(min_max(y, finite=True)),
                      tuple(min_max(x, finite=True)))
            points = numpy.transpose(numpy.array((y, x)))
            counts, sums, bin_edges = Histogramnd(
                points,
                histo_range=ranges,
                n_bins=shape,
                weights=values)
            yEdges, xEdges = bin_edges
            origin = xEdges[0], yEdges[0]
            scale = ((xEdges[-1] - xEdges[0]) / (len(xEdges) - 1),
                     (yEdges[-1] - yEdges[0]) / (len(yEdges) - 1))

            with numpy.errstate(divide='ignore', invalid='ignore'):
                histo = sums / counts

            self.__cacheHistogramInfo = _HistogramInfo(
                mean=histo, count=counts, sum=sums,
                origin=origin, scale=scale, shape=shape)

        return self.__cacheHistogramInfo

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

        visualization = self.getVisualization()

        if visualization is self.Visualization.BINNED_STATISTIC:
            plot = self.getPlot()
            if (plot is None or
                    plot.getXAxis().getScale() != Axis.LINEAR or
                    plot.getYAxis().getScale() != Axis.LINEAR):
                # Those visualizations are not available with log scaled axes
                return None

            histoInfo = self.__getHistogramInfo()
            if histoInfo is None:
                return None
            data = getattr(histoInfo, self.getVisualizationParameter(
                self.VisualizationParameter.BINNED_STATISTIC_FUNCTION))

            return backend.addImage(
                data=data,
                origin=histoInfo.origin,
                scale=histoInfo.scale,
                colormap=self.getColormap(),
                alpha=self.getAlpha())

        # Compute colors
        cmap = self.getColormap()
        rgbacolors = cmap.applyToData(self)

        if self.__alpha is not None:
            rgbacolors[:, -1] = (rgbacolors[:, -1] * self.__alpha).astype(numpy.uint8)

        visualization = self.getVisualization()

        if visualization is self.Visualization.POINTS:
            return backend.addCurve(xFiltered, yFiltered,
                                    color=rgbacolors[mask],
                                    symbol=self.getSymbol(),
                                    linewidth=0,
                                    linestyle="",
                                    yaxis='left',
                                    xerror=xerror,
                                    yerror=yerror,
                                    fill=False,
                                    alpha=self.getAlpha(),
                                    symbolsize=self.getSymbolSize(),
                                    baseline=None)

        else:
            plot = self.getPlot()
            if (plot is None or
                    plot.getXAxis().getScale() != Axis.LINEAR or
                    plot.getYAxis().getScale() != Axis.LINEAR):
                # Those visualizations are not available with log scaled axes
                return None

            if visualization is self.Visualization.SOLID:
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
                                                color=rgbacolors[mask],
                                                alpha=self.getAlpha())

            elif visualization is self.Visualization.REGULAR_GRID:
                gridInfo = self.__getRegularGridInfo()
                if gridInfo is None:
                    return None

                dim0, dim1 = gridInfo.shape
                if gridInfo.order == 'column':  # transposition needed
                    dim0, dim1 = dim1, dim0

                if len(rgbacolors) == dim0 * dim1:
                    image = rgbacolors.reshape(dim0, dim1, -1)
                else:
                    # The points do not fill the whole image
                    image = numpy.empty((dim0 * dim1, 4), dtype=rgbacolors.dtype)
                    image[:len(rgbacolors)] = rgbacolors
                    image[len(rgbacolors):] = 0, 0, 0, 0  # Transparent pixels
                    image.shape = dim0, dim1, -1

                if gridInfo.order == 'column':
                    image = numpy.transpose(image, axes=(1, 0, 2))

                return backend.addImage(
                    data=image,
                    origin=gridInfo.origin,
                    scale=gridInfo.scale,
                    colormap=None,
                    alpha=self.getAlpha())

            elif visualization is self.Visualization.IRREGULAR_GRID:
                gridInfo = self.__getRegularGridInfo()
                if gridInfo is None:
                    return None

                shape = gridInfo.shape
                if shape is None:  # No shape, no display
                    return None

                nbpoints = len(xFiltered)
                if nbpoints == 1:
                    # single point, render as a square points
                    return backend.addCurve(xFiltered, yFiltered,
                                            color=rgbacolors[mask],
                                            symbol='s',
                                            linewidth=0,
                                            linestyle="",
                                            yaxis='left',
                                            xerror=None,
                                            yerror=None,
                                            fill=False,
                                            alpha=self.getAlpha(),
                                            symbolsize=7,
                                            baseline=None)

                # Make shape include all points
                gridOrder = gridInfo.order
                if nbpoints != numpy.prod(shape):
                    if gridOrder == 'row':
                        shape = int(numpy.ceil(nbpoints / shape[1])), shape[1]
                    else:   # column-major order
                        shape = shape[0], int(numpy.ceil(nbpoints / shape[0]))

                if shape[0] < 2 or shape[1] < 2:  # Single line, at least 2 points
                    points = numpy.ones((2, nbpoints, 2), dtype=numpy.float64)
                    # Use row/column major depending on shape, not on info value
                    gridOrder = 'row' if shape[0] == 1 else 'column'

                    if gridOrder == 'row':
                        points[0, :, 0] = xFiltered
                        points[0, :, 1] = yFiltered
                    else:  # column-major order
                        points[0, :, 0] = yFiltered
                        points[0, :, 1] = xFiltered

                    # Add a second line that will be clipped in the end
                    points[1, :-1] = points[0, :-1] + numpy.cross(
                        points[0, 1:] - points[0, :-1], (0., 0., 1.))[:, :2]
                    points[1, -1] = points[0, -1] + numpy.cross(
                        points[0, -1] - points[0, -2], (0., 0., 1.))[:2]

                    points.shape = 2, nbpoints, 2  # Use same shape for both orders
                    coords, indices = _quadrilateral_grid_as_triangles(points)

                elif gridOrder == 'row':  # row-major order
                    if nbpoints != numpy.prod(shape):
                        points = numpy.empty((numpy.prod(shape), 2), dtype=numpy.float64)
                        points[:nbpoints, 0] = xFiltered
                        points[:nbpoints, 1] = yFiltered
                        # Index of last element of last fully filled row
                        index = (nbpoints // shape[1]) * shape[1]
                        points[nbpoints:, 0] = xFiltered[index - (numpy.prod(shape) - nbpoints):index]
                        points[nbpoints:, 1] = yFiltered[-1]
                    else:
                        points = numpy.transpose((xFiltered, yFiltered))
                    points.shape = shape[0], shape[1], 2

                else:   # column-major order
                    if nbpoints != numpy.prod(shape):
                        points = numpy.empty((numpy.prod(shape), 2), dtype=numpy.float64)
                        points[:nbpoints, 0] = yFiltered
                        points[:nbpoints, 1] = xFiltered
                        # Index of last element of last fully filled column
                        index = (nbpoints // shape[0]) * shape[0]
                        points[nbpoints:, 0] = yFiltered[index - (numpy.prod(shape) - nbpoints):index]
                        points[nbpoints:, 1] = xFiltered[-1]
                    else:
                        points = numpy.transpose((yFiltered, xFiltered))
                    points.shape = shape[1], shape[0], 2

                coords, indices = _quadrilateral_grid_as_triangles(points)

                # Remove unused extra triangles
                coords = coords[:4*nbpoints]
                indices = indices[:2*nbpoints]

                if gridOrder == 'row':
                    x, y = coords[:, 0], coords[:, 1]
                else:  # column-major order
                    y, x = coords[:, 0], coords[:, 1]

                rgbacolors = rgbacolors[mask]  # Filter-out not finite points
                gridcolors = numpy.empty(
                    (4 * nbpoints, rgbacolors.shape[-1]), dtype=rgbacolors.dtype)
                for first in range(4):
                    gridcolors[first::4] = rgbacolors[:nbpoints]

                return backend.addTriangles(x,
                                            y,
                                            indices,
                                            color=gridcolors,
                                            alpha=self.getAlpha())

            else:
                _logger.error("Unhandled visualization %s", visualization)
                return None

    @docstring(PointsBase)
    def pick(self, x, y):
        result = super(Scatter, self).pick(x, y)

        if result is not None:
            visualization = self.getVisualization()

            if visualization is self.Visualization.IRREGULAR_GRID:
                # Specific handling of picking for the irregular grid mode
                index = result.getIndices(copy=False)[0] // 4
                result = PickingResult(self, (index,))

            elif visualization is self.Visualization.REGULAR_GRID:
                # Specific handling of picking for the regular grid mode
                picked = result.getIndices(copy=False)
                if picked is None:
                    return None
                row, column = picked[0][0], picked[1][0]

                gridInfo = self.__getRegularGridInfo()
                if gridInfo is None:
                    return None

                if gridInfo.order == 'row':
                    index = row * gridInfo.shape[1] + column
                else:
                    index = row + column * gridInfo.shape[0]
                if index >= len(self.getXData(copy=False)):  # OK as long as not log scale
                    return None  # Image can be larger than scatter

                result = PickingResult(self, (index,))

            elif visualization is self.Visualization.BINNED_STATISTIC:
                picked = result.getIndices(copy=False)
                if picked is None or len(picked) == 0 or len(picked[0]) == 0:
                    return None
                row, col = picked[0][0], picked[1][0]
                histoInfo = self.__getHistogramInfo()
                if histoInfo is None:
                    return None
                sx, sy = histoInfo.scale
                ox, oy = histoInfo.origin
                xdata = self.getXData(copy=False)
                ydata = self.getYData(copy=False)
                indices = numpy.nonzero(numpy.logical_and(
                    numpy.logical_and(xdata >= ox + sx * col, xdata < ox + sx * (col + 1)),
                    numpy.logical_and(ydata >= oy + sy * row, ydata < oy + sy * (row + 1))))[0]
                result = None if len(indices) == 0 else PickingResult(self, indices)

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

        # Data changed, this needs update
        self.__cacheRegularGridInfo = None
        self.__cacheHistogramInfo = None

        self._value = value
        self._updateColormappedData()

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
