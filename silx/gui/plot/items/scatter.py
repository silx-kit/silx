# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This module provides the :class:`Image` item of the :class:`Plot`.
"""

__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "29/03/2017"


import logging

import numpy

from .core import Points, LabelsMixIn, ColormapMixIn, SymbolMixIn


_logger = logging.getLogger(__name__)


class Scatter(Points, ColormapMixIn):
    """Description of a scatter plot"""
    _DEFAULT_SYMBOL = 'o'
    """Default symbol of the scatter plots"""

    def __init__(self):
        Points.__init__(self)
        ColormapMixIn.__init__(self)
        self._value = ()

        self._symbol = self._DEFAULT_SYMBOL

    # TODO _addBackendRenderer

    def _logFilterData(self, xPositive, yPositive):
        """Filter out values with x or y <= 0 on log axes

        :param bool xPositive: True to filter arrays according to X coords.
        :param bool yPositive: True to filter arrays according to Y coords.
        :return: The filter arrays or unchanged object if not filtering needed
        :rtype: (x, y, value, xerror, yerror)
        """
        x, y, value, xerror, yerror = self.getData(copy=False)

        if xPositive or yPositive:
            xclipped = (x <= 0) if xPositive else False
            yclipped = (y <= 0) if yPositive else False
            clipped = numpy.logical_or(xclipped, yclipped)

            if numpy.any(clipped):
                # copy to keep original array and convert to float
                x = numpy.array(x, copy=True, dtype=numpy.float)
                x[clipped] = numpy.nan
                y = numpy.array(y, copy=True, dtype=numpy.float)
                y[clipped] = numpy.nan
                value = numpy.array(value, copy=True, dtype=numpy.float)
                value[clipped] = numpy.nan

                if xPositive and xerror is not None:
                    xerror = self._logFilterError(x, xerror)

                if yPositive and yerror is not None:
                    yerror = self._logFilterError(y, yerror)

        return x, y, value, xerror, yerror

    def _getBounds(self):
        if self.getXData(copy=False).size == 0:  # Empty data
            return None

        plot = self.getPlot()
        if plot is not None:
            xPositive = plot.isXAxisLogarithmic()
            yPositive = plot.isYAxisLogarithmic()
        else:
            xPositive = False
            yPositive = False

        if (xPositive, yPositive) not in self._boundsCache:
            # TODO bounds do not take error bars into account
            x, y, value, xerror, yerror = self.getData(copy=False, displayed=True)
            self._boundsCache[(xPositive, yPositive)] = (
                numpy.nanmin(x),
                numpy.nanmax(x),
                numpy.nanmin(y),
                numpy.nanmax(y)
            )
        return self._boundsCache[(xPositive, yPositive)]

    def getValueData(self, copy=True):
        """Returns the value assigned to the scatter data points.

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._value, copy=copy)

    def getData(self, copy=True, displayed=False):
        """Returns the x, y coordinates and the value of the data points

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :param bool displayed: True to only get curve points that are displayed
                               in the plot. Default: False.
                               Note: If plot has log scale, negative points
                               are not displayed.
        :returns: (x, y, value xerror, yerror)
        :rtype: 5-tuple of numpy.ndarray
        """
        if displayed:
            plot = self.getPlot()
            if plot is not None:
                xPositive = plot.isXAxisLogarithmic()
                yPositive = plot.isYAxisLogarithmic()
                if xPositive or yPositive:
                    # One axis has log scale, filter data
                    if (xPositive, yPositive) not in self._filteredCache:
                        self._filteredCache[(xPositive, yPositive)] = \
                            self._logFilterData(xPositive, yPositive)
                    return self._filteredCache[(xPositive, yPositive)]

        return (self.getXData(copy),
                self.getYData(copy),
                self.getValueData(copy),
                self.getXErrorData(copy),
                self.getYErrorData(copy))

    # reimplemented from Points to handle `value`
    def setData(self, x, y, value, xerror=None, yerror=None, copy=True):
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
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        """
        value = numpy.array(value, copy=copy)
        assert value.ndim == 1
        assert len(x) == len(value)

        self._value = value

        # set the rest and call self._updated + plot._invalidateDataRange()
        Points.setData(self, x, y, xerror, yerror, copy)
