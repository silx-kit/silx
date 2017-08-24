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
"""This module provides the :class:`Shape` item of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/05/2017"


import logging

import numpy

from .core import (Item, ColorMixIn, FillMixIn, ItemChangedType)


_logger = logging.getLogger(__name__)


# TODO probably make one class for each kind of shape
# TODO check fill:polygon/polyline + fill = duplicated
class Shape(Item, ColorMixIn, FillMixIn):
    """Description of a shape item

    :param str type_: The type of shape in:
                      'hline', 'polygon', 'rectangle', 'vline', 'polylines'
    """

    def __init__(self, type_):
        Item.__init__(self)
        ColorMixIn.__init__(self)
        FillMixIn.__init__(self)
        self._overlay = False
        assert type_ in ('hline', 'polygon', 'rectangle', 'vline', 'polylines')
        self._type = type_
        self._points = ()

        self._handle = None

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        points = self.getPoints(copy=False)
        x, y = points.T[0], points.T[1]
        return backend.addItem(x,
                               y,
                               legend=self.getLegend(),
                               shape=self.getType(),
                               color=self.getColor(),
                               fill=self.isFill(),
                               overlay=self.isOverlay(),
                               z=self.getZValue())

    def isOverlay(self):
        """Return true if shape is drawn as an overlay

        :rtype: bool
        """
        return self._overlay

    def setOverlay(self, overlay):
        """Set the overlay state of the shape

        :param bool overlay: True to make it an overlay
        """
        overlay = bool(overlay)
        if overlay != self._overlay:
            self._overlay = overlay
            self._updated(ItemChangedType.OVERLAY)

    def getType(self):
        """Returns the type of shape to draw.

        One of: 'hline', 'polygon', 'rectangle', 'vline', 'polylines'

        :rtype: str
        """
        return self._type

    def getPoints(self, copy=True):
        """Get the control points of the shape.

        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        :return: Array of point coordinates
        :rtype: numpy.ndarray with 2 dimensions
        """
        return numpy.array(self._points, copy=copy)

    def setPoints(self, points, copy=True):
        """Set the point coordinates

        :param numpy.ndarray points: Array of point coordinates
        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        :return:
        """
        self._points = numpy.array(points, copy=copy)
        self._updated(ItemChangedType.DATA)
