# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""Miscellaneous utility functions"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/02/2016"


import math

import numpy


def addMarginsToLimits(margins, isXLog, isYLog,
                       xMin, xMax, yMin, yMax, y2Min=None, y2Max=None):
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

        if not isXLog:
            xRange = xMax - xMin
            xMin -= xMinMargin * xRange
            xMax += xMaxMargin * xRange

        elif xMin > 0. and xMax > 0.:  # Log scale
            # Do not apply margins if limits < 0
            xMinLog, xMaxLog = numpy.log10(xMin), numpy.log10(xMax)
            xRangeLog = xMaxLog - xMinLog
            xMin = pow(10., xMinLog - xMinMargin * xRangeLog)
            xMax = pow(10., xMaxLog + xMaxMargin * xRangeLog)

        if not isYLog:
            yRange = yMax - yMin
            yMin -= yMinMargin * yRange
            yMax += yMaxMargin * yRange
        elif yMin > 0. and yMax > 0.:  # Log scale
            # Do not apply margins if limits < 0
            yMinLog, yMaxLog = numpy.log10(yMin), numpy.log10(yMax)
            yRangeLog = yMaxLog - yMinLog
            yMin = pow(10., yMinLog - yMinMargin * yRangeLog)
            yMax = pow(10., yMaxLog + yMaxMargin * yRangeLog)

        if y2Min is not None and y2Max is not None:
            if not isYLog:
                yRange = y2Max - y2Min
                y2Min -= yMinMargin * yRange
                y2Max += yMaxMargin * yRange
            elif y2Min > 0. and y2Max > 0.:  # Log scale
                # Do not apply margins if limits < 0
                yMinLog, yMaxLog = numpy.log10(y2Min), numpy.log10(y2Max)
                yRangeLog = yMaxLog - yMinLog
                y2Min = pow(10., yMinLog - yMinMargin * yRangeLog)
                y2Max = pow(10., yMaxLog + yMaxMargin * yRangeLog)

    if y2Min is None or y2Max is None:
        return xMin, xMax, yMin, yMax
    else:
        return xMin, xMax, yMin, yMax, y2Min, y2Max


def applyPan(min_, max_, panFactor, isLog10):
    """Returns a new range with applied panning.

    Moves the range according to panFactor.
    If isLog10 is True, converts to log10 before moving.

    :param float min_: Min value of the data range to pan.
    :param float max_: Max value of the data range to pan.
                       Must be >= min_.
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


def clamp(value, min_=0., max_=1.):
    """Clip a value to a range [min, max].

    :param value: The value to clip
    :param min_: The min_ edge of the range
    :param max_: The max_ edge of the range
    :return: The clipped value
    """
    if value < min_:
        return min_
    elif value > max_:
        return max_
    else:
        return value
