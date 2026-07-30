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
"""Miscellaneous utility functions for the Plot"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "21/03/2017"


from .axis_scale import FLOAT32_SAFE_MIN, FLOAT32_MINPOS, FLOAT32_SAFE_MAX  # noqa: F401
from . import axis_scale
from .panzoom import (  # noqa: F401
    applyZoomToPlot,
    applyPan,
    checkAxisLimits,
    EnabledAxes,
)
from ..items.types import AxisScaleType


def _addMargins(
    scale: AxisScaleType,
    minMargin: float,
    maxMargin: float,
    minLimit: float,
    maxLimit: float,
):
    if not axis_scale.isValid(scale, minLimit) or not axis_scale.isValid(
        scale, maxLimit
    ):
        return minLimit, maxLimit

    minScaled = axis_scale.apply(scale, minLimit)
    maxScaled = axis_scale.apply(scale, maxLimit)
    rangeLimit = maxScaled - minScaled

    min_ = axis_scale.revert(scale, minScaled - minMargin * rangeLimit)
    max_ = axis_scale.revert(scale, maxScaled + maxMargin * rangeLimit)
    if all(axis_scale.inSafeRange(scale, (min_, max_))):
        return min_, max_
    return minLimit, maxLimit


def addMarginsToLimits(
    margins,
    xScale: AxisScaleType,
    yScale: AxisScaleType,
    xMin,
    xMax,
    yMin,
    yMax,
    y2Min=None,
    y2Max=None,
):
    """Returns updated limits by extending them with margins.

    :param margins: The ratio of the margins to add or None for no margins.
    :type margins: A 4-tuple of floats as
                   (xMinMargin, xMaxMargin, yMinMargin, yMaxMargin)

    :return: The updated limits
    :rtype: tuple of 4 or 6 floats: Either (xMin, xMax, yMin, yMax) or
            (xMin, xMax, yMin, yMax, y2Min, y2Max) if y2Min and y2Max
            are provided.
    """
    if margins is not None:
        xMinMargin, xMaxMargin, yMinMargin, yMaxMargin = margins

        xMin, xMax = _addMargins(xScale, xMinMargin, xMaxMargin, xMin, xMax)
        yMin, yMax = _addMargins(yScale, yMinMargin, yMaxMargin, yMin, yMax)
        if y2Min is not None and y2Max is not None:
            y2Min, y2Max = _addMargins(yScale, yMinMargin, yMaxMargin, y2Min, y2Max)

    xMin, xMax = checkAxisLimits(xScale, xMin, xMax)
    yMin, yMax = checkAxisLimits(yScale, yMin, yMax)

    if y2Min is None or y2Max is None:
        return xMin, xMax, yMin, yMax
    else:
        y2Min, y2Max = checkAxisLimits(yScale, y2Min, y2Max)
        return xMin, xMax, yMin, yMax, y2Min, y2Max
