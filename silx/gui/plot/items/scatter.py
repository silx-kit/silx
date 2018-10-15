# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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

import numpy

from .core import Points, ColormapMixIn


_logger = logging.getLogger(__name__)


class Scatter(Points, ColormapMixIn):
    """Description of a scatter"""

    _DEFAULT_SELECTABLE = True
    """Default selectable state for scatter plots"""

    _DEFAULT_SYMBOL = 'o'
    """Default symbol of the scatter plots"""

    def __init__(self):
        Points.__init__(self)
        ColormapMixIn.__init__(self)
        self._value = ()
        self.__alpha = None
        
    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        # Filter-out values <= 0
        xFiltered, yFiltered, valueFiltered, xerror, yerror = self.getData(
            copy=False, displayed=True)

        if len(xFiltered) == 0:
            return None  # No data to display, do not add renderer to backend

        cmap = self.getColormap()
        rgbacolors = cmap.applyToData(self._value)

        if self.__alpha is not None:
            rgbacolors[:, -1] = (rgbacolors[:, -1] * self.__alpha).astype(numpy.uint8)

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

    def _logFilterData(self, xPositive, yPositive):
        """Filter out values with x or y <= 0 on log axes

        :param bool xPositive: True to filter arrays according to X coords.
        :param bool yPositive: True to filter arrays according to Y coords.
        :return: The filtered arrays or unchanged object if not filtering needed
        :rtype: (x, y, value, xerror, yerror)
        """
        # overloaded from Points to filter also value.
        value = self.getValueData(copy=False)

        if xPositive or yPositive:
            clipped = self._getClippingBoolArray(xPositive, yPositive)

            if numpy.any(clipped):
                # copy to keep original array and convert to float
                value = numpy.array(value, copy=True, dtype=numpy.float)
                value[clipped] = numpy.nan

        x, y, xerror, yerror = Points._logFilterData(self, xPositive, yPositive)

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

    # reimplemented from Points to handle `value`
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
        Points.setData(self, x, y, xerror, yerror, copy)
