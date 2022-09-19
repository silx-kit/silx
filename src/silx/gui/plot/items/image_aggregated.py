# /*##########################################################################
#
# Copyright (c) 2021 European Synchrotron Radiation Facility
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
"""This module provides the :class:`ImageDataAggregated` items of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "07/07/2021"

import enum
import logging
from typing import Tuple, Union

import numpy

from ....utils.enum import Enum as _Enum
from ....utils.proxy import docstring
from .axis import Axis
from .core import ItemChangedType
from .image import ImageDataBase
from ._pick import PickingResult


_logger = logging.getLogger(__name__)


class ImageDataAggregated(ImageDataBase):
    """Item displaying an image as a density map."""

    @enum.unique
    class Aggregation(_Enum):
        NONE = "none"
        "Do not aggregate data, display as is (default)"

        MAX = "max"
        "Aggregates elements with max (ignore NaNs)"

        MEAN = "mean"
        "Aggregates elements with mean (ignore NaNs)"

        MIN = "min"
        "Aggregates elements with min (ignore NaNs)"

    def __init__(self):
        super().__init__()
        self.__cacheLODData = {}
        self.__currentLOD = 0, 0
        self.__aggregationMode = self.Aggregation.NONE

    def setAggregationMode(self, mode: Union[str,Aggregation]):
        """Set the aggregation method used to reduce the data to screen resolution.

        :param Aggregation mode: The aggregation method
        """
        aggregationMode = self.Aggregation.from_value(mode)
        if aggregationMode != self.__aggregationMode:
            self.__aggregationMode = aggregationMode
            self.__cacheLODData = {}  # Clear cache
            self._updated(ItemChangedType.VISUALIZATION_MODE)

    def getAggregationMode(self) -> Aggregation:
        """Returns the currently used aggregation method."""
        return self.__aggregationMode

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        plot = self.getPlot()
        assert plot is not None
        if not self._isPlotLinear(plot):
            # Do not render with non linear scales
            return None

        data = self.getData(copy=False)
        if data.size == 0:
            return None  # No data to display

        aggregationMode = self.getAggregationMode()
        if aggregationMode == self.Aggregation.NONE:  # Pass data as it is
            displayedData = data
            scale = self.getScale()

        else:  # Aggregate data according to level of details
            if aggregationMode == self.Aggregation.MAX:
                aggregator = numpy.nanmax
            elif aggregationMode == self.Aggregation.MEAN:
                aggregator = numpy.nanmean
            elif aggregationMode == self.Aggregation.MIN:
                aggregator = numpy.nanmin
            else:
                _logger.error("Unsupported aggregation mode")
                return None

            lodx, lody = self._getLevelOfDetails()

            if (lodx, lody) not in self.__cacheLODData:
                height, width = data.shape
                self.__cacheLODData[(lodx, lody)] = aggregator(
                    data[: (height // lody) * lody, : (width // lodx) * lodx].reshape(
                        height // lody, lody, width // lodx, lodx
                    ),
                    axis=(1, 3),
                )

            self.__currentLOD = lodx, lody
            displayedData = self.__cacheLODData[self.__currentLOD]

            sx, sy = self.getScale()
            scale = sx * lodx, sy * lody

        return backend.addImage(
            displayedData,
            origin=self.getOrigin(),
            scale=scale,
            colormap=self._getColormapForRendering(),
            alpha=self.getAlpha(),
        )

    def _getPixelSizeInData(self, axis="left"):
        """Returns the size of a pixel in plot data coordinates

        :param str axis: Y axis to use in: 'left' (default), 'right'
        :return:
            Size (width, height) of a Qt pixel in data coordinates.
            Size is None if it cannot be computed
        :rtype: Union[List[float],None]
        """
        assert axis in ("left", "right")
        plot = self.getPlot()
        if plot is None:
            return None

        xaxis = plot.getXAxis()
        yaxis = plot.getYAxis(axis)

        if (
            xaxis.getScale() != Axis.LINEAR
            or yaxis.getScale() != Axis.LINEAR
        ):
            raise RuntimeError("Only available with linear axes")

        xmin, xmax = xaxis.getLimits()
        ymin, ymax = yaxis.getLimits()
        width, height = plot.getPlotBoundsInPixels()[2:]
        if width == 0 or height == 0:
            return None
        else:
            return (xmax - xmin) / width, (ymax - ymin) / height

    def _getLevelOfDetails(self) -> Tuple[int, int]:
        """Return current level of details the image is displayed with."""
        plot = self.getPlot()
        if plot is None or not self._isPlotLinear(plot):
            return 1, 1  # Fallback to bas LOD

        sx, sy = self.getScale()
        xUnitPerPixel, yUnitPerPixel = self._getPixelSizeInData()
        lodx = max(1, int(numpy.ceil(xUnitPerPixel / sx)))
        lody = max(1, int(numpy.ceil(yUnitPerPixel / sy)))
        return lodx, lody

    @docstring(ImageDataBase)
    def setData(self, data, copy=True):
        self.__cacheLODData = {}  # Reset cache
        super().setData(data)

    @docstring(ImageDataBase)
    def _setPlot(self, plot):
        """Refresh image when plot limits change"""
        previousPlot = self.getPlot()
        if previousPlot is not None:
            for axis in (previousPlot.getXAxis(), previousPlot.getYAxis()):
                axis.sigLimitsChanged.disconnect(self.__plotLimitsChanged)

        super()._setPlot(plot)

        if plot is not None:
            for axis in (plot.getXAxis(), plot.getYAxis()):
                axis.sigLimitsChanged.connect(self.__plotLimitsChanged)

    def __plotLimitsChanged(self):
        """Trigger update if level of details has changed"""
        if (self.getAggregationMode() != self.Aggregation.NONE and
                self.__currentLOD != self._getLevelOfDetails()):
            self._updated()

    @docstring(ImageDataBase)
    def pick(self, x, y):
        result = super().pick(x, y)
        if result is None:
            return None

        # Compute indices in initial data
        plot = self.getPlot()
        if plot is None:
            return None
        dataPos = plot.pixelToData(x, y, axis="left", check=True)
        if dataPos is None:
            return None  # Outside plot area

        ox, oy = self.getOrigin()
        sx, sy = self.getScale()
        col = int((dataPos[0] - ox) / sx)
        row = int((dataPos[1] - oy) / sy)
        height, width = self.getData(copy=False).shape[:2]
        if 0 <= col < width and 0 <= row < height:
            return PickingResult(self, ((row,), (col,)))
        return None
