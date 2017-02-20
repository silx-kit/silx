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
"""This module provides classes that describes :class:`.Plot` content.

Instances of those classes are returned by :class:`.Plot` methods that give
access to its content such as :meth:`.Plot.getCurve`, :meth:`.Plot.getImage`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "09/02/2017"


from collections import Sequence
from copy import deepcopy
import logging
import weakref

import numpy

from . import Colors

from ...third_party.six import string_types
from ...utils.decorators import deprecated


_logger = logging.getLogger(__name__)


class Item(object):
    """Description of an item of the plot"""

    _DEFAULT_Z_LAYER = 0
    """Default layer for overlay rendering"""

    _DEFAULT_LEGEND = ''
    """Default legend of items"""

    _DEFAULT_SELECTABLE = False
    """Default selectable state of items"""

    def __init__(self, plot, legend=None):
        self._plot = weakref.ref(plot)
        self._visible = True
        legend = str(legend) if legend is not None else self._DEFAULT_LEGEND
        self._legend = legend
        self._selectable = self._DEFAULT_SELECTABLE
        self._z = self._DEFAULT_Z_LAYER
        self._info = None
        self._xlabel = None
        self._ylabel = None

    def getPlot(self):
        """Returns Plot this item belongs to.

        :rtype: Plot or None
        """
        return self._plot()

    def getBounds(self):  # TODO return a Bounds object rather than a tuple
        """Returns the bounding box of this item in data coordinates

        :returns: (xmin, xmax, ymin, ymax) or None
        :rtype: 4-tuple of float or None
        """
        return self._getBounds()

    def _getBounds(self):
        """:meth:`getBounds` implementation to override by sub-class"""
        return None

    def isVisible(self):
        """True if item is visible, False otherwise

        :rtype: bool
        """
        return self._visible

    def _setVisible(self, visible):
        """Set visibility of item.

        :param bool visible: True to display it, False otherwise
        """
        self._visible = bool(visible)

    def getLegend(self):
        """Returns the legend of this item (str)"""
        return self._legend

    def _setLegend(self, legend):
        legend = str(legend) if legend is not None else self._DEFAULT_LEGEND
        self._legend = legend

    def isSelectable(self):
        """Returns true if item is selectable (bool)"""
        return self._selectable

    def _setSelectable(self, selectable):
        self._selectable = bool(selectable)

    def getZLayer(self):
        """Returns the layer on which to draw this item (int)"""
        return self._z

    def _setZLayer(self, z):
        self._z = int(z) if z is not None else self._DEFAULT_Z_LAYER

    def getInfo(self, copy=True):
        """Returns the info associated to this item

        :param bool copy: True to get a deepcopy, False otherwise.
        """
        return deepcopy(self._info) if copy else self._info

    def _setInfo(self, info, copy=True):
        if copy:
            info = deepcopy(info)
        self._info = info


# Mix-in classes ##############################################################

class LabelsMixIn(object):
    """Mix-in class for items with x and y labels"""

    def __init__(self):
        self._xlabel = None
        self._ylabel = None

    def getXLabel(self):
        """Return the X axis label associated to this curve

        :rtype: str or None
        """
        return self._xlabel

    def _setXLabel(self, label):
        """Set the X axis label associated with this curve

        :param str label: The X axis label
        """
        self._xlabel = str(label)

    def getYLabel(self):
        """Return the Y axis label associated to this curve

        :rtype: str or None
        """
        return self._ylabel

    def _setYLabel(self, label):
        """Set the Y axis label associated with this curve

        :param str label: The Y axis label
        """
        self._ylabel = str(label)


class DraggableMixIn(object):
    """Mix-in class for draggable items"""

    def __init__(self):
        self._draggable = False

    def isDraggable(self):
        """Returns true if image is draggable

        :rtype: bool
        """
        return self._draggable

    def _setDraggable(self, draggable):
        """Set if image is draggable or not.

        :param bool draggable:
        """
        self._draggable = bool(draggable)


class ColormapMixIn(object):
    """Mix-in class for items with colormap"""

    _DEFAULT_COLORMAP = {'name': 'gray', 'normalization': 'linear',
                         'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
    """Default colormap of the item"""

    def __init__(self):
        self._colormap = self._DEFAULT_COLORMAP

    def getColormap(self):
        """Return the used colormap"""
        return self._colormap.copy()

    def _setColormap(self, colormap):
        """Set the colormap of this image

        :param dict colormap: colormap description
        """
        self._colormap = colormap.copy()


class SymbolMixIn(object):
    """Mix-in class for items with symbol type"""

    _DEFAULT_SYMBOL = ''
    """Default marker of the item"""

    def __init__(self):
        self._symbol = self._DEFAULT_SYMBOL

    def getSymbol(self):
        """Return the point marker type.

        Marker type::

            - 'o' circle
            - '.' point
            - ',' pixel
            - '+' cross
            - 'x' x-cross
            - 'd' diamond
            - 's' square

        :rtype: str
        """
        return self._symbol

    def _setSymbol(self, symbol):
        """Set the marker type

        See :meth:`getSymbol`.

        :param str symbol: Marker type
        """
        assert symbol in ('o', '.', ',', '+', 'x', 'd', 's', '', None)
        if symbol is None:
            symbol = self._DEFAULT_SYMBOL
        self._symbol = symbol


class ColorMixIn(object):
    """Mix-in class for item with color"""

    _DEFAULT_COLOR = (0., 0., 0., 1.)
    """Default color of the item"""

    def __init__(self):
        self._color = self._DEFAULT_COLOR

    def getColor(self):
        """Returns the RGBA color of the item

        :rtype: 4-tuple of float in [0, 1]
        """
        return self._color

    def _setColor(self, color, copy=True):
        """Set item color

        :param color: color(s) to be used
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in Colors.py
        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        """
        if isinstance(color, string_types):
            self._color = Colors.rgba(color)
        else:
            color = numpy.array(color, copy=copy)
            # TODO more checks
            if color.ndim == 1:  # Single RGBA color
                color = Colors.rgba(color)
            else:  # Array of colors
                assert color.ndim == 2
            self._color = color


class YAxisMixIn(object):
    """Mix-in class for item with yaxis"""

    _DEFAULT_YAXIS = 'left'
    """Default Y axis the item belongs to"""

    def __init__(self):
        self._yaxis = self._DEFAULT_YAXIS

    def getYAxis(self):
        """Returns the Y axis this curve belongs to.

        Either 'left' or 'right'.

        :rtype: str
        """
        return self._yaxis

    def _setYAxis(self, yaxis):
        """Set the Y axis this curve belongs to.

        :param str yaxis: 'left' or 'right'
        """
        yaxis = str(yaxis)
        assert yaxis in ('left', 'right')
        self._yaxis = yaxis


class FillMixIn(object):
    """Mix-in class for item with fill"""

    def __init__(self):
        self._fill = False

    def isFill(self):
        """Returns whether the item is filled or not.

        :rtype: bool
        """
        return self._fill

    def _setFill(self, fill):
        """Set whether to fill the item or not.

        :param bool fill:
        """
        self._fill = bool(fill)


# Items #######################################################################


class Curve(Item, LabelsMixIn, SymbolMixIn, ColorMixIn, YAxisMixIn, FillMixIn):
    """Description of a curve"""

    _DEFAULT_Z_LAYER = 1
    """Default overlay layer for curves"""

    _DEFAULT_SELECTABLE = True
    """Default selectable state for curves"""

    _DEFAULT_LINEWIDTH = 1.
    """Default line width of the curve"""

    _DEFAULT_LINESTYLE = '-'
    """Default line style of the curve"""

    _DEFAULT_HIGHLIGHT_COLOR = (0, 0, 0, 255)
    """Default highlight color of the item"""

    def __init__(self, plot, legend=None):
        Item.__init__(self, plot, legend)
        LabelsMixIn.__init__(self)
        SymbolMixIn.__init__(self)
        ColorMixIn.__init__(self)
        YAxisMixIn.__init__(self)
        FillMixIn.__init__(self)
        self._x = ()
        self._y = ()
        self._xerror = None
        self._yerror = None

        self._linewidth = self._DEFAULT_LINEWIDTH
        self._linestyle = self._DEFAULT_LINESTYLE
        self._histogram = None

        self._highlightColor = self._DEFAULT_HIGHLIGHT_COLOR
        self._highlighted = False

        # Store filtered data for x > 0 and/or y > 0
        self._filteredCache = {}

        # Store bounds depending on axes filtering >0:
        # key is (isXPositiveFilter, isYPositiveFilter)
        self._boundsCache = {}

    @deprecated
    def __getitem__(self, item):
        """Compatibility with PyMca and silx <= 0.4.0"""
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(5))]
        elif item == 0:
            return self.getXData(copy=False)
        elif item == 1:
            return self.getYData(copy=False)
        elif item == 2:
            return self.getLegend()
        elif item == 3:
            info = self.getInfo(copy=False)
            return {} if info is None else info
        elif item == 4:
            params = {
                'info': self.getInfo(),
                'color': self.getColor(),
                'symbol': self.getSymbol(),
                'linewidth': self.getLineWidth(),
                'linestyle': self.getLineStyle(),
                'xlabel': self.getXLabel(),
                'ylabel': self.getYLabel(),
                'yaxis': self.getYAxis(),
                'xerror': self.getXErrorData(copy=False),
                'yerror': self.getYErrorData(copy=False),
                'z': self.getZLayer(),
                'selectable': self.isSelectable(),
                'fill': self.isFill()
            }
            return params
        else:
            raise IndexError("Index out of range: %s", str(item))

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
        :rtype: (x, y, xerror, yerror)
        """
        x, y, xerror, yerror = self.getData(copy=False)

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

                if xPositive and xerror is not None:
                    xerror = self._logFilterError(x, xerror)

                if yPositive and yerror is not None:
                    yerror = self._logFilterError(y, yerror)

        return x, y, xerror, yerror

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
            # TODO bounds do not take error bars into account (as in silx Plot)
            x, y, xerror, yerror = self.getData(copy=False, displayed=True)
            self._boundsCache[(xPositive, yPositive)] = (
                numpy.nanmin(x),
                numpy.nanmax(x),
                numpy.nanmin(y),
                numpy.nanmax(y)
            )
        return self._boundsCache[(xPositive, yPositive)]

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
        """Returns the x, y values of the curve points and xerror, yerror

        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        :param bool displayed: True to only get curve points that are displayed
                               in the plot. Default: False.
                               Note: If plot has log scale, negative points
                               are not displayed.
        :returns: (x, y, xerror, yerror)
        :rtype: 4-tuple of numpy.ndarray
        """
        if displayed:  # Eventually filter data according to plot state
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
                self.getXErrorData(copy),
                self.getYErrorData(copy))

    def _setData(self, x, y, xerror=None, yerror=None, copy=True):
        x = numpy.array(x, copy=copy)
        y = numpy.array(y, copy=copy)
        assert x.ndim == y.ndim == 1
        assert len(x) == len(y)
        if xerror is not None:
            xerror = numpy.array(xerror, copy=copy)
        if yerror is not None:
            yerror = numpy.array(yerror, copy=copy)
        # TODO checks on xerror, yerror
        self._x, self._y = x, y
        self._xerror, self._yerror = xerror, yerror

        self._boundsCache = {}  # Reset cached bounds
        self._filteredCache = {}  # Reset cached filtered data

    def isHighlighted(self):
        """Returns True if curve is highlighted.

        :rtype: bool
        """
        return self._highlighted

    def _setHighlighted(self, highlighted):
        """Set the highlight state of the curve

        :param bool highlighted:
        """
        self._highlighted = bool(highlighted)

    def getHighlightedColor(self):
        """Returns the RGBA highlight color of the item

        :rtype: 4-tuple of int in [0, 255]
        """
        return self._highlightColor

    def _setHighlightedColor(self, color):
        """Set the color to use when highlighted

        :param color: color(s) to be used for highlight
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in Colors.py
        """
        self._highlightColor = Colors.rgba(color)

    def getCurrentColor(self):
        """Returns the current color of the curve.

        This color is either the color of the curve or the highlighted color,
        depending on the highlight state.

        :rtype: 4-tuple of int in [0, 255]
        """
        if self.isHighlighted():
            return self.getHighlightedColor()
        else:
            return self.getColor()

    def getLineWidth(self):
        """Return the curve line width in pixels (int)"""
        return self._linewidth

    def _setLineWidth(self, width):
        """Set the width in pixel of the curve line

        See :meth:`getLineWidth`.

        :param float width: Width in pixels
        """
        self._linewidth = float(width)

    def getLineStyle(self):
        """Return the type of the line

        Type of line::

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line

        :rtype: str
        """
        return self._linestyle

    def _setLineStyle(self, style):
        """Set the style of the curve line.

        See :meth:`getLineStyle`.

        :param str style: Line style
        """
        style = str(style)
        assert style in ('', ' ', '-', '--', '-.', ':', None)
        if style is None:
            style = self._DEFAULT_LINESTYLE
        self._linestyle = style

    # TODO make a separate class for histograms
    def getHistogramType(self):
        """Histogram curve rendering style.

        Histogram type::

            - None (default)
            - 'left'
            - 'right'
            - 'center'

        :rtype: str or None
        """
        return self._histogram

    def _setHistogramType(self, histogram):
        assert histogram in ('left', 'right', 'center', None)
        self._histogram = histogram


class Image(Item, LabelsMixIn, DraggableMixIn, ColormapMixIn):
    """Description of an image"""

    # TODO method to get image of data converted to RGBA with current colormap

    def __init__(self, plot, legend=None):
        Item.__init__(self, plot, legend)
        LabelsMixIn.__init__(self)
        DraggableMixIn.__init__(self)
        ColormapMixIn.__init__(self)
        self._data = ()
        self._pixmap = None

        # TODO use calibration instead of origin and scale?
        self._origin = (0., 0.)
        self._scale = (1., 1.)

    @deprecated
    def __getitem__(self, item):
        """Compatibility with PyMca and silx <= 0.4.0"""
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(5))]
        elif item == 0:
            return self.getData(copy=False)
        elif item == 1:
            return self.getLegend()
        elif item == 2:
            info = self.getInfo(copy=False)
            return {} if info is None else info
        elif item == 3:
            return self.getPixmap(copy=False)
        elif item == 4:
            params = {
                'info': self.getInfo(),
                'origin': self.getOrigin(),
                'scale': self.getScale(),
                'z': self.getZLayer(),
                'selectable': self.isSelectable(),
                'draggable': self.isDraggable(),
                'colormap': self.getColormap(),
                'xlabel': self.getXLabel(),
                'ylabel': self.getYLabel(),
            }
            return params
        else:
            raise IndexError("Index out of range: %s" % str(item))

    def _getBounds(self):
        if self.getData(copy=False).size == 0:  # Empty data
            return None

        height, width = self.getData(copy=False).shape[:2]
        origin = self.getOrigin()
        scale = self.getScale()
        # Taking care of scale might be < 0
        xmin, xmax = origin[0], origin[0] + width * scale[0]
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        # Taking care of scale might be < 0
        ymin, ymax = origin[1], origin[1] + height * scale[1]
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        plot = self.getPlot()
        if (plot is not None and
                (plot.isXAxisLogarithmic() and xmin <= 0.) or
                (plot.isYAxisLogarithmic() and ymin <= 0)):
            return None
        else:
            return xmin, xmax, ymin, ymax

    def getData(self, copy=True):
        """Returns the image data

        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._data, copy=copy)

    def getPixmap(self, copy=True):
        """Get the optional pixmap that is displayed instead of the data

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :return: The pixmap representing the data if any
        :rtype: numpy.ndarray or None
        """
        if self._pixmap is None:
            return None
        else:
            return numpy.array(self._pixmap, copy=copy)

    def _setData(self, data, pixmap=None, copy=True):
        """Set the image data

        :param data: Image data to set
        :param pixmap: Optional RGB(A) image representing the data
        :param bool copy: True (Default) to get a copy,
                          False to use internal representation (do not modify!)
        """
        data = numpy.array(data, copy=copy)
        assert data.ndim in (2, 3)
        if data.ndim == 3:
            assert data.shape[1] in (3, 4)
        self._data = data

        if pixmap is not None:
            pixmap = numpy.array(pixmap, copy=copy)
            assert pixmap.ndim == 3
            assert pixmap.shape[2] in (3, 4)
            assert pixmap.shape[:2] == data.shape[:2]
        self._pixmap = pixmap

    def getOrigin(self):
        """Returns the offset from origin at which to display the image.

        :rtype: 2-tuple of float
        """
        return self._origin

    def _setOrigin(self, origin):
        """Set the offset from origin at which to display the image.

        :param origin: (ox, oy) Offset from origin
        :type origin: float or 2-tuple of float
        """
        if isinstance(origin, Sequence):
            origin = float(origin[0]), float(origin[1])
        else:  # single value origin
            origin = float(origin), float(origin)
        self._origin = origin

    def getScale(self):
        """Returns the scale of the image in data coordinates.

        :rtype: 2-tuple of float
        """
        return self._scale

    def _setScale(self, scale):
        """Set the scale of the image

        :param scale: (sx, sy) Scale of the image
        :type scale: float or 2-tuple of float
        """
        if isinstance(scale, Sequence):
            scale = float(scale[0]), float(scale[1])
        else:  # single value scale
            scale = float(scale), float(scale)
        self._scale = scale


# Markers ####################################################################

class _BaseMarker(Item, DraggableMixIn, ColorMixIn):
    """Base class for markers"""

    _DEFAULT_COLOR = (0., 0., 0., 1.)
    """Default color of the markers"""

    def __init__(self, plot, legend=None):
        Item.__init__(self, plot, legend)
        DraggableMixIn.__init__(self)
        ColorMixIn.__init__(self)

        self._text = ''
        self._x = None
        self._y = None
        self._constraint = self._defaultConstraint

    def getText(self):
        """Returns marker text.

        :rtype: str
        """
        return self._text

    def _setText(self, text):
        """Set the text of the marker.

        :param str text: The text to use
        """
        self._text = str(text)

    def getXPosition(self):
        """Returns the X position of the marker line in data coordinates

        :rtype: float or None
        """
        return self._x

    def getYPosition(self):
        """Returns the Y position of the marker line in data coordinates

        :rtype: float or None
        """
        return self._y

    def getPosition(self):
        """Returns the (x, y) position of the marker in data coordinates

        :rtype: 2-tuple of float or None
        """
        return self._x, self._y

    def _setPosition(self, x, y):
        """Set marker position in data coordinates

        Constraint are applied if any.

        :param float x: X coordinates in data frame
        :param float y: Y coordinates in data frame
        """
        x, y = self.getConstraint()(x, y)
        self._x, self._y = float(x), float(y)

    def getConstraint(self):
        """Returns the dragging constraint of this item"""
        return self._constraint

    def _setConstraint(self, constraint):
        """

        :param callable constraint:
        :param constraint: A function filtering item displacement by
                           dragging operations or None for no filter.
                           This function is called each time the item is
                           moved.
                           This is only used if isDraggable returns True.
        :type constraint: None or a callable that takes the coordinates of
                          the current cursor position in the plot as input
                          and that returns the filtered coordinates.
        """
        if constraint is None:
            constraint = self._defaultConstraint
        assert callable(constraint)
        self._constraint = constraint

    @staticmethod
    def _defaultConstraint(*args):
        """Default constraint no doing anything"""
        return args


class Marker(_BaseMarker, SymbolMixIn):
    """Description of a marker"""

    _DEFAULT_SYMBOL = '+'
    """Default symbol of the marker"""

    def __init__(self, plot, legend=None):
        _BaseMarker.__init__(self, plot, legend)
        SymbolMixIn.__init__(self)

        self._x = 0.
        self._y = 0.

    def _setConstraint(self, constraint):
        """Set the constraint function of the marker drag.

        It also supports 'horizontal' and 'vertical' str as constraint.

        :param constraint: The constraint of the dragging of this marker
        :type: constraint: callable or str
        """
        if constraint == 'horizontal':
            constraint = self._horizontalConstraint
        elif constraint == 'vertical':
            constraint = self._verticalConstraint

        super(Marker, self)._setConstraint(constraint)

    def _horizontalConstraint(self, _, y):
        return self.getXPosition(), y

    def _verticalConstraint(self, x, _):
        return x, self.getYPosition()


class XMarker(_BaseMarker):
    """Description of a marker"""

    def __init__(self, plot, legend=None):
        _BaseMarker.__init__(self, plot, legend)
        self._x = 0.

    def _setPosition(self, x, y):
        """Set marker line position in data coordinates

        Constraint are applied if any.

        :param float x: X coordinates in data frame
        :param float y: Y coordinates in data frame
        """
        x, _ = self.getConstraint()(x, y)
        self._x = float(x)


class YMarker(_BaseMarker):
    """Description of a marker"""

    def __init__(self, plot, legend=None):
        _BaseMarker.__init__(self, plot, legend)
        self._y = 0.

    def _setPosition(self, x, y):
        """Set marker line position in data coordinates

        Constraint are applied if any.

        :param float x: X coordinates in data frame
        :param float y: Y coordinates in data frame
        """
        _, y = self.getConstraint()(x, y)
        self._y = float(y)


# TODO probably make one class for each kind of shape
# TODO check fill:polygon/polyline + fill = duplicated
class Shape(Item, ColorMixIn, FillMixIn):
    """Description of a shape item"""

    def __init__(self, plot, legend=None):
        Item.__init__(self, plot, legend)
        ColorMixIn.__init__(self)
        FillMixIn.__init__(self)
        self._overlay = False
        self._type = 'polygon'
        self._points = ()

    def isOverlay(self):
        """Return true if shape is drawn as an overlay

        :rtype: bool
        """
        return self._overlay

    def _setOverlay(self, overlay):
        """Set the overlay state of the shape

        :param overlay:
        """
        self._overlay = overlay

    def getType(self):
        """Returns the type of shape to draw.

        One of: 'hline', 'polygon', 'rectangle', 'vline', 'polyline'

        :rtype: str
        """
        return self._type

    def _setType(self, type_):
        assert type_ in ('hline', 'polygon', 'rectangle', 'vline', 'polyline')
        self._type = type_

    def getPoints(self, copy=True):
        """Get the control points of the shape.

        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        :return: Array of point coordinates
        :rtype: numpy.ndarray with 2 dimensions
        """
        return numpy.array(self._points, copy=copy)

    def _setPoints(self, points, copy=True):
        """Set the point coordinates

        :param numpy.ndarray points: Array of point coordinates
        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        :return:
        """
        self._points = numpy.array(points, copy=copy)
