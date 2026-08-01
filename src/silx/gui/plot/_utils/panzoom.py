# /*##########################################################################
#
# Copyright (c) 2004-2023 European Synchrotron Radiation Facility
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
"""Functions to apply pan and zoom on a Plot"""

__authors__ = ["T. Vincent", "V. Valls"]
__license__ = "MIT"
__date__ = "08/08/2017"


import logging
from typing import NamedTuple

import numpy

from .axis_scale import (
    FLOAT32_MINPOS,
    isValid,
    apply,
    revert,
    clipToSafeRange,
)
from ..items.types import AxisScaleType

_logger = logging.getLogger(__name__)


def checkAxisLimits(
    axisScale: AxisScaleType, vmin: float, vmax: float, name: str = ""
) -> tuple[float, float]:
    """Makes sure axis range is not empty and within supported range.

    :return: (min, max) making sure min < max
    """
    vmax = clipToSafeRange(axisScale, vmax)
    vmin = clipToSafeRange(axisScale, vmin)

    if vmax < vmin:
        _logger.debug("%s axis: max < min, inverting limits.", name)
        vmin, vmax = vmax, vmin
    elif vmax == vmin:
        _logger.debug("%s axis: max == min, expanding limits.", name)
        if vmin == 0.0:
            vmin, vmax = -0.1, 0.1
        elif vmin < 0:
            vmax *= 0.9
            vmin = clipToSafeRange(axisScale, vmin * 1.1)
        else:  # vmin > 0
            vmax = clipToSafeRange(axisScale, vmin * 1.1)
            vmin *= 0.9

    return vmin, vmax


def scale1DRange(
    axisScale: AxisScaleType, min_: float, max_: float, center: float, scale: float
) -> tuple[float, float]:
    """Scale a 1D range given a scale factor and an center point.

    Keeps the values in a smaller range than float32.

    :param axisScale:
    :param min_: The current min value of the range.
    :param max_: The current max value of the range.
    :param center: The center of the zoom (i.e., invariant point).
    :param scale: The scaling factor to apply for zoom
    :return: The zoomed range (min, max)
    """
    # Min and center can be < 0 when
    # autoscale is off and switch to log scale
    # max_ < 0 should not happen
    # TODO better and weird previous behavior
    scaledMin = apply(axisScale, min_) if isValid(axisScale, min_) else FLOAT32_MINPOS
    scaledCenter = (
        apply(axisScale, center) if isValid(axisScale, center) else FLOAT32_MINPOS
    )
    scaledMax = apply(axisScale, max_) if isValid(axisScale, max_) else FLOAT32_MINPOS

    if scaledMin == scaledMax:
        return scaledMin, scaledMax

    offset = (scaledCenter - scaledMin) / (scaledMax - scaledMin)
    range_ = (scaledMax - scaledMin) / scale
    newScaledMin = scaledCenter - offset * range_
    newScaledMax = scaledCenter + (1.0 - offset) * range_

    newRange = clipToSafeRange(
        axisScale, revert(axisScale, (newScaledMin, newScaledMax))
    )
    if not all(numpy.isfinite(newRange)):
        # There should be no overflow as exponent is log10 of a float32
        # but better safe than sorry
        return scaledMin, scaledMax

    return tuple(newRange)


class EnabledAxes(NamedTuple):
    """Toggle zoom for each axis"""

    xaxis: bool = True
    yaxis: bool = True
    y2axis: bool = True

    def isDisabled(self) -> bool:
        """True only if all axes are disabled"""
        return not (self.xaxis or self.yaxis or self.y2axis)


def applyZoomToPlot(
    plot,
    scale: float,
    center: tuple[float, float] = None,
    enabled: EnabledAxes = EnabledAxes(),
):
    """Zoom in/out plot given a scale and a center point.

    :param plot: The plot on which to apply zoom.
    :param scale: Scale factor of zoom.
    :param center: (x, y) coords in pixel coordinates of the zoom center.
    :param enabled: Toggle zoom for each axis independently
    """
    xMin, xMax = plot.getXAxis().getLimits()
    yMin, yMax = plot.getYAxis().getLimits()
    y2Min, y2Max = plot.getYAxis(axis="right").getLimits()

    if center is None:
        left, top, width, height = plot.getPlotBoundsInPixels()
        cx, cy = left + width // 2, top + height // 2
    else:
        cx, cy = center

    dataCenterPos = plot.pixelToData(cx, cy)
    assert dataCenterPos is not None

    if enabled.xaxis:
        xMin, xMax = scale1DRange(
            plot.getXAxis().getScale(), xMin, xMax, dataCenterPos[0], scale
        )

    if enabled.yaxis:
        yMin, yMax = scale1DRange(
            plot.getYAxis("left").getScale(), yMin, yMax, dataCenterPos[1], scale
        )

    if enabled.y2axis:
        dataPos = plot.pixelToData(cx, cy, axis="right")
        assert dataPos is not None
        y2Center = dataPos[1]
        y2Min, y2Max = scale1DRange(
            plot.getYAxis("right").getScale(), y2Min, y2Max, y2Center, scale
        )

    plot.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)


def applyPan(
    axisScale: AxisScaleType, min_: float, max_: float, panFactor: float
) -> tuple[float, float]:
    """Returns a new range with applied panning.

    Moves the range according to panFactor after applying axis scale.

    :param axisScale:
    :param min_: Min value of the data range to pan.
    :param max_: Max value of the data range to pan.
                 Must be >= min.
    :param panFactor: Signed proportion of the range to use for pan.
    :return: New min and max value with pan applied.
    """
    scaledMin = apply(axisScale, min_)
    scaledMax = apply(axisScale, max_)
    scaledOffset = panFactor * (scaledMax - scaledMin)

    newMin = revert(axisScale, scaledMin + scaledOffset)
    newMax = revert(axisScale, scaledMax + scaledOffset)
    if isValid(axisScale, newMin) and isValid(axisScale, newMax):
        return tuple(clipToSafeRange(axisScale, (newMin, newMax)))
    else:
        return min_, max_


class _Unset:
    """To be able to have distinction between None and unset"""

    pass


class ViewConstraints:
    """
    Store constraints applied on the view box and compute the resulting view box.
    """

    def __init__(self):
        self._min = [None, None]
        self._max = [None, None]
        self._minRange = [None, None]
        self._maxRange = [None, None]

    def update(
        self,
        xMin=_Unset,
        xMax=_Unset,
        yMin=_Unset,
        yMax=_Unset,
        minXRange=_Unset,
        maxXRange=_Unset,
        minYRange=_Unset,
        maxYRange=_Unset,
    ):
        """
        Update the constraints managed by the object

        The constraints are the same as the ones provided by PyQtGraph.

        :param float xMin: Minimum allowed x-axis value.
            (default do not change the stat, None remove the constraint)
        :param float xMax: Maximum allowed x-axis value.
            (default do not change the stat, None remove the constraint)
        :param float yMin: Minimum allowed y-axis value.
            (default do not change the stat, None remove the constraint)
        :param float yMax: Maximum allowed y-axis value.
            (default do not change the stat, None remove the constraint)
        :param float minXRange: Minimum allowed left-to-right span across the
            view (default do not change the stat, None remove the constraint)
        :param float maxXRange: Maximum allowed left-to-right span across the
            view (default do not change the stat, None remove the constraint)
        :param float minYRange: Minimum allowed top-to-bottom span across the
            view (default do not change the stat, None remove the constraint)
        :param float maxYRange: Maximum allowed top-to-bottom span across the
            view (default do not change the stat, None remove the constraint)
        :return: True if the constraints was changed
        """
        updated = False

        minRange = [minXRange, minYRange]
        maxRange = [maxXRange, maxYRange]
        minPos = [xMin, yMin]
        maxPos = [xMax, yMax]

        for axis in range(2):
            value = minPos[axis]
            if value is not _Unset and value != self._min[axis]:
                self._min[axis] = value
                updated = True

            value = maxPos[axis]
            if value is not _Unset and value != self._max[axis]:
                self._max[axis] = value
                updated = True

            value = minRange[axis]
            if value is not _Unset and value != self._minRange[axis]:
                self._minRange[axis] = value
                updated = True

            value = maxRange[axis]
            if value is not _Unset and value != self._maxRange[axis]:
                self._maxRange[axis] = value
                updated = True

        # Sanity checks

        for axis in range(2):
            if (
                self._maxRange[axis] is not None
                and self._min[axis] is not None
                and self._max[axis] is not None
            ):
                # max range cannot be larger than bounds
                diff = self._max[axis] - self._min[axis]
                self._maxRange[axis] = min(self._maxRange[axis], diff)
                updated = True

        return updated

    def normalize(self, xMin, xMax, yMin, yMax, allow_scaling=True):
        """Normalize a view range defined by x and y corners using predefined
        containts.

        :param float xMin: Min position of the x-axis
        :param float xMax: Max position of the x-axis
        :param float yMin: Min position of the y-axis
        :param float yMax: Max position of the y-axis
        :param bool allow_scaling: Allow or not to apply scaling for the
            normalization. Used according to the interaction mode.
        :return: A normalized tuple of (xMin, xMax, yMin, yMax)
        """
        viewRange = [[xMin, xMax], [yMin, yMax]]

        for axis in range(2):
            # clamp xRange and yRange
            if allow_scaling:
                diff = viewRange[axis][1] - viewRange[axis][0]
                delta = None
                if self._maxRange[axis] is not None and diff > self._maxRange[axis]:
                    delta = self._maxRange[axis] - diff
                elif self._minRange[axis] is not None and diff < self._minRange[axis]:
                    delta = self._minRange[axis] - diff
                if delta is not None:
                    viewRange[axis][0] -= delta * 0.5
                    viewRange[axis][1] += delta * 0.5

            # clamp min and max positions
            outMin = (
                self._min[axis] is not None and viewRange[axis][0] < self._min[axis]
            )
            outMax = (
                self._max[axis] is not None and viewRange[axis][1] > self._max[axis]
            )

            if outMin and outMax:
                if allow_scaling:
                    # we can clamp both sides
                    viewRange[axis][0] = self._min[axis]
                    viewRange[axis][1] = self._max[axis]
                else:
                    # center the result
                    delta = viewRange[axis][1] - viewRange[axis][0]
                    mid = self._min[axis] + self._max[axis] - self._min[axis]
                    viewRange[axis][0] = mid - delta
                    viewRange[axis][1] = mid + delta
            elif outMin:
                delta = self._min[axis] - viewRange[axis][0]
                viewRange[axis][0] += delta
                viewRange[axis][1] += delta
            elif outMax:
                delta = self._max[axis] - viewRange[axis][1]
                viewRange[axis][0] += delta
                viewRange[axis][1] += delta

        return viewRange[0][0], viewRange[0][1], viewRange[1][0], viewRange[1][1]
