# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "21/03/2017"


import math

import numpy


# Float 32 info ###############################################################
# Using min/max value below limits of float32
# so operation with such value (e.g., max - min) do not overflow

FLOAT32_SAFE_MIN = -1e37
FLOAT32_MINPOS = numpy.finfo(numpy.float32).tiny
FLOAT32_SAFE_MAX = 1e37
# TODO double support


def scale1DRange(min_, max_, center, scale, isLog):
    """Scale a 1D range given a scale factor and an center point.

    Keeps the values in a smaller range than float32.

    :param float min_: The current min value of the range.
    :param float max_: The current max value of the range.
    :param float center: The center of the zoom (i.e., invariant point).
    :param float scale: The scale to use for zoom
    :param bool isLog: Whether using log scale or not.
    :return: The zoomed range.
    :rtype: tuple of 2 floats: (min, max)
    """
    if isLog:
        # Min and center can be < 0 when
        # autoscale is off and switch to log scale
        # max_ < 0 should not happen
        min_ = numpy.log10(min_) if min_ > 0. else FLOAT32_MINPOS
        center = numpy.log10(center) if center > 0. else FLOAT32_MINPOS
        max_ = numpy.log10(max_) if max_ > 0. else FLOAT32_MINPOS

    if min_ == max_:
        return min_, max_

    offset = (center - min_) / (max_ - min_)
    range_ = (max_ - min_) / scale
    newMin = center - offset * range_
    newMax = center + (1. - offset) * range_

    if isLog:
        # No overflow as exponent is log10 of a float32
        newMin = pow(10., newMin)
        newMax = pow(10., newMax)
        newMin = numpy.clip(newMin, FLOAT32_MINPOS, FLOAT32_SAFE_MAX)
        newMax = numpy.clip(newMax, FLOAT32_MINPOS, FLOAT32_SAFE_MAX)
    else:
        newMin = numpy.clip(newMin, FLOAT32_SAFE_MIN, FLOAT32_SAFE_MAX)
        newMax = numpy.clip(newMax, FLOAT32_SAFE_MIN, FLOAT32_SAFE_MAX)
    return newMin, newMax


def applyZoomToPlot(plot, scaleF, center=None):
    """Zoom in/out plot given a scale and a center point.

    :param plot: The plot on which to apply zoom.
    :param float scaleF: Scale factor of zoom.
    :param center: (x, y) coords in pixel coordinates of the zoom center.
    :type center: 2-tuple of float
    """
    xMin, xMax = plot.getGraphXLimits()
    yMin, yMax = plot.getGraphYLimits()

    if center is None:
        left, top, width, height = plot.getPlotBoundsInPixels()
        cx, cy = left + width // 2, top + height // 2
    else:
        cx, cy = center

    dataCenterPos = plot.pixelToData(cx, cy)
    assert dataCenterPos is not None

    xMin, xMax = scale1DRange(xMin, xMax, dataCenterPos[0], scaleF,
                              plot.isXAxisLogarithmic())

    yMin, yMax = scale1DRange(yMin, yMax, dataCenterPos[1], scaleF,
                              plot.isYAxisLogarithmic())

    dataPos = plot.pixelToData(cx, cy, axis="right")
    assert dataPos is not None
    y2Center = dataPos[1]
    y2Min, y2Max = plot.getGraphYLimits(axis="right")
    y2Min, y2Max = scale1DRange(y2Min, y2Max, y2Center, scaleF,
                                plot.isYAxisLogarithmic())

    plot.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)


def applyPan(min_, max_, panFactor, isLog10):
    """Returns a new range with applied panning.

    Moves the range according to panFactor.
    If isLog10 is True, converts to log10 before moving.

    :param float min_: Min value of the data range to pan.
    :param float max_: Max value of the data range to pan.
                       Must be >= min.
    :param float panFactor: Signed proportion of the range to use for pan.
    :param bool isLog10: True if log10 scale, False if linear scale.
    :return: New min and max value with pan applied.
    :rtype: 2-tuple of float.
    """
    if isLog10 and min_ > 0.:
        # Negative range and log scale can happen with matplotlib
        logMin, logMax = math.log10(min_), math.log10(max_)
        logOffset = panFactor * (logMax - logMin)
        newMin = pow(10., logMin + logOffset)
        newMax = pow(10., logMax + logOffset)

        # Takes care of out-of-range values
        if newMin > 0. and newMax < float('inf'):
            min_, max_ = newMin, newMax

    else:
        offset = panFactor * (max_ - min_)
        newMin, newMax = min_ + offset, max_ + offset

        # Takes care of out-of-range values
        if newMin > - float('inf') and newMax < float('inf'):
            min_, max_ = newMin, newMax
    return min_, max_
