# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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
__date__ = "21/12/2018"


import logging

import numpy
import six

from ... import colors
from .core import Item, ColorMixIn, FillMixIn, ItemChangedType, LineMixIn, YAxisMixIn


_logger = logging.getLogger(__name__)


# TODO probably make one class for each kind of shape
# TODO check fill:polygon/polyline + fill = duplicated
class Shape(Item, ColorMixIn, FillMixIn, LineMixIn):
    """Description of a shape item

    :param str type_: The type of shape in:
                      'hline', 'polygon', 'rectangle', 'vline', 'polylines'
    """

    def __init__(self, type_):
        Item.__init__(self)
        ColorMixIn.__init__(self)
        FillMixIn.__init__(self)
        LineMixIn.__init__(self)
        self._overlay = False
        assert type_ in ('hline', 'polygon', 'rectangle', 'vline', 'polylines')
        self._type = type_
        self._points = ()
        self._lineBgColor = None

        self._handle = None

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        points = self.getPoints(copy=False)
        x, y = points.T[0], points.T[1]
        return backend.addShape(x,
                                y,
                                shape=self.getType(),
                                color=self.getColor(),
                                fill=self.isFill(),
                                overlay=self.isOverlay(),
                                linestyle=self.getLineStyle(),
                                linewidth=self.getLineWidth(),
                                linebgcolor=self.getLineBgColor())

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

    def getLineBgColor(self):
        """Returns the RGBA color of the item
        :rtype: 4-tuple of float in [0, 1] or array of colors
        """
        return self._lineBgColor

    def setLineBgColor(self, color, copy=True):
        """Set item color
        :param color: color(s) to be used
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in colors.py
        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        """
        if color is not None:
            if isinstance(color, six.string_types):
                color = colors.rgba(color)
            else:
                color = numpy.array(color, copy=copy)
                # TODO more checks + improve color array support
                if color.ndim == 1:  # Single RGBA color
                    color = colors.rgba(color)
                else:  # Array of colors
                    assert color.ndim == 2

        self._lineBgColor = color
        self._updated(ItemChangedType.LINE_BG_COLOR)


class BoundingRect(Item, YAxisMixIn):
    """An invisible shape which enforce the plot view to display the defined
    space on autoscale.

    This item do not display anything. But if the visible property is true,
    this bounding box is used by the plot, if not, the bounding box is
    ignored. That's the default behaviour for plot items.

    It can be applied on the "left" or "right" axes. Not both at the same time.
    """

    def __init__(self):
        Item.__init__(self)
        YAxisMixIn.__init__(self)
        self.__bounds = None

    def _updated(self, event=None, checkVisibility=True):
        if event in (ItemChangedType.YAXIS,
                     ItemChangedType.VISIBLE,
                     ItemChangedType.DATA):
            # TODO hackish data range implementation
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

        super(BoundingRect, self)._updated(event, checkVisibility)

    def setBounds(self, rect):
        """Set the bounding box of this item in data coordinates

        :param Union[None,List[float]] rect: (xmin, xmax, ymin, ymax) or None
        """
        if rect is not None:
            rect = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
            assert rect[0] <= rect[1]
            assert rect[2] <= rect[3]

        if rect != self.__bounds:
            self.__bounds = rect
            self._updated(ItemChangedType.DATA)

    def _getBounds(self):
        if self.__bounds is None:
            return None
        plot = self.getPlot()
        if plot is not None:
            xPositive = plot.getXAxis()._isLogarithmic()
            yPositive = plot.getYAxis()._isLogarithmic()
            if xPositive or yPositive:
                bounds = list(self.__bounds)
                if xPositive and bounds[1] <= 0:
                    return None
                if xPositive and bounds[0] <= 0:
                    bounds[0] = bounds[1]
                if yPositive and bounds[3] <= 0:
                    return None
                if yPositive and bounds[2] <= 0:
                    bounds[2] = bounds[3]
                return tuple(bounds)

        return self.__bounds


class _BaseExtent(Item):
    """Base class for :class:`XAxisExtent` and :class:`YAxisExtent`.

    :param str axis: Either 'x' or 'y'.
    """

    def __init__(self, axis='x'):
        assert axis in ('x', 'y')
        Item.__init__(self)
        self.__axis = axis
        self.__range = 1., 100.

    def _updated(self, event=None, checkVisibility=True):
        if event in (ItemChangedType.VISIBLE,
                     ItemChangedType.DATA):
            # TODO hackish data range implementation
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

        super(_BaseExtent, self)._updated(event, checkVisibility)

    def setRange(self, min_, max_):
        """Set the range of the extent of this item in data coordinates.

        :param float min_: Lower bound of the extent
        :param float max_: Upper bound of the extent
        :raises ValueError: If min > max or not finite bounds
        """
        range_ = float(min_), float(max_)
        if not numpy.all(numpy.isfinite(range_)):
            raise ValueError("min_ and max_ must be finite numbers.")
        if range_[0] > range_[1]:
            raise ValueError("min_ must be lesser or equal to max_")

        if range_ != self.__range:
            self.__range = range_
            self._updated(ItemChangedType.DATA)

    def getRange(self):
        """Returns the range (min, max) of the extent in data coordinates.

        :rtype: List[float]
        """
        return self.__range

    def _getBounds(self):
        min_, max_ = self.getRange()

        plot = self.getPlot()
        if plot is not None:
            axis = plot.getXAxis() if self.__axis == 'x' else plot.getYAxis()
            if axis._isLogarithmic():
                if max_ <= 0:
                    return None
                if min_ <= 0:
                    min_ = max_

        if self.__axis == 'x':
            return min_, max_, float('nan'), float('nan')
        else:
            return float('nan'), float('nan'), min_, max_


class XAxisExtent(_BaseExtent):
    """Invisible item with a settable horizontal data extent.

    This item do not display anything, but it behaves as a data
    item with a horizontal extent regarding plot data bounds, i.e.,
    :meth:`PlotWidget.resetZoom` will take this horizontal extent into account.
    """
    def __init__(self):
        _BaseExtent.__init__(self, axis='x')


class YAxisExtent(_BaseExtent, YAxisMixIn):
    """Invisible item with a settable vertical data extent.

    This item do not display anything, but it behaves as a data
    item with a vertical extent regarding plot data bounds, i.e.,
    :meth:`PlotWidget.resetZoom` will take this vertical extent into account.
    """

    def __init__(self):
        _BaseExtent.__init__(self, axis='y')
        YAxisMixIn.__init__(self)
