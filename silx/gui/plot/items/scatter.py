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

from .core import Item, LabelsMixIn, ColormapMixIn, SymbolMixIn


_logger = logging.getLogger(__name__)


class Scatter(Item, ColormapMixIn, SymbolMixIn, LabelsMixIn):
    """Description of a scatter plot"""
    _DEFAULT_SYMBOL = 'o'
    """Default symbol of the scatter plots"""

    def __init__(self):
        Item.__init__(self)
        ColormapMixIn.__init__(self)
        SymbolMixIn.__init__(self)
        LabelsMixIn.__init__(self)
        self._x = ()
        self._y = ()
        self._xerror = None
        self._yerror = None
        self._value = ()

        self._symbol = self._DEFAULT_SYMBOL

        # Store filtered data for x > 0 and/or y > 0
        self._filteredCache = {}

        # Store bounds depending on axes filtering >0:
        # key is (isXPositiveFilter, isYPositiveFilter)
        self._boundsCache = {}   # TODO

    @staticmethod
    def _logFilterError(value, error):
        """Filter/convert error values if they go <= 0.

        Replace error leading to negative values by nan

        :param numpy.ndarray value: 1D array of values
        :param numpy.ndarray error:
            Array of errors: scalar, N, Nx1 or 2xN or None.
        :return: Filtered error so error bars are never negative
        """
        if error is not None:
            # Convert Nx1 to N
            if error.ndim == 2 and error.shape[1] == 1 and len(value) != 1:
                error = numpy.ravel(error)

            # Supports error being scalar, N or 2xN array
            errorClipped = (value - numpy.atleast_2d(error)[0]) <= 0

            if numpy.any(errorClipped):  # Need filtering

                # expand errorbars to 2xN
                if error.size == 1:  # Scalar
                    error = numpy.full(
                        (2, len(value)), error, dtype=numpy.float)

                elif error.ndim == 1:  # N array
                    newError = numpy.empty((2, len(value)),
                                           dtype=numpy.float)
                    newError[0, :] = error
                    newError[1, :] = error
                    error = newError

                elif error.size == 2 * len(value):  # 2xN array
                    error = numpy.array(
                        error, copy=True, dtype=numpy.float)

                else:
                    _logger.error("Unhandled error array")
                    return error

                error[0, errorClipped] = numpy.nan

        return error

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

    def getXData(self, copy=True):
        """Returns the x coordinates of the data points

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._x, copy=copy)

    def getYData(self, copy=True):
        """Returns the y coordinates of the data points

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._y, copy=copy)

    def getValueData(self, copy=True):
        """Returns the value of the data points

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._value, copy=copy)

    def getXErrorData(self, copy=True):
        """Returns the x error of the curve

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray or None
        """
        if self._xerror is None:
            return None
        else:
            return numpy.array(self._xerror, copy=copy)

    def getYErrorData(self, copy=True):
        """Returns the y error of the curve

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray or None
        """
        if self._yerror is None:
            return None
        else:
            return numpy.array(self._yerror, copy=copy)

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
        x = numpy.array(x, copy=copy)
        y = numpy.array(y, copy=copy)
        value = numpy.array(value, copy=copy)
        assert x.ndim == y.ndim == value.ndim == 1
        assert len(x) == len(y) == len(value)
        if xerror is not None:
            xerror = numpy.array(xerror, copy=copy)
        if yerror is not None:
            yerror = numpy.array(yerror, copy=copy)
        # TODO checks on xerror, yerror

        self._x, self._y, self._value, = x, y, value
        self._xerror, self._yerror = xerror, yerror

        self._boundsCache = {}  # Reset cached bounds
        self._filteredCache = {}  # Reset cached filtered data

        self._updated()
        # TODO hackish data range implementation
        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()
