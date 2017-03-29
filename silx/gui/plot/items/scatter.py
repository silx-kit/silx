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
        self._value = ()

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

    def getData(self, copy=True):
        """Returns the x, y coordinates and the value of the data points
        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :returns: (x, y, value)
        :rtype: 3-tuple of numpy.ndarray
        """
        x = self.getXData(copy)
        y = self.getYData(copy)
        value = self.getValueData(copy)
        return x, y, value

    def setData(self, x, y, value, copy=True):
        x = numpy.array(x, copy=copy)
        y = numpy.array(y, copy=copy)
        value = numpy.array(value, copy=copy)
        assert x.ndim == y.ndim == value.ndim == 1
        assert len(x) == len(y) == len(value)
        self._x, self._y, self._value = x, y, value
