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
"""This module provides the base class for items of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "14/06/2018"

import collections
from copy import deepcopy
import logging
import warnings
import weakref
import numpy
from silx.third_party import six, enum

from ... import qt
from ... import colors
from ...colors import Colormap


_logger = logging.getLogger(__name__)


@enum.unique
class ItemChangedType(enum.Enum):
    """Type of modification provided by :attr:`Item.sigItemChanged` signal."""
    # Private setters and setInfo are not emitting sigItemChanged signal.
    # Signals to consider:
    # COLORMAP_SET emitted when setColormap is called but not forward colormap object signal
    # CURRENT_COLOR_CHANGED emitted current color changed because highlight changed,
    # highlighted color changed or color changed depending on hightlight state.

    VISIBLE = 'visibleChanged'
    """Item's visibility changed flag."""

    ZVALUE = 'zValueChanged'
    """Item's Z value changed flag."""

    COLORMAP = 'colormapChanged'  # Emitted when set + forward events from the colormap object
    """Item's colormap changed flag.

    This is emitted both when setting a new colormap and
    when the current colormap object is updated.
    """

    SYMBOL = 'symbolChanged'
    """Item's symbol changed flag."""

    SYMBOL_SIZE = 'symbolSizeChanged'
    """Item's symbol size changed flag."""

    LINE_WIDTH = 'lineWidthChanged'
    """Item's line width changed flag."""

    LINE_STYLE = 'lineStyleChanged'
    """Item's line style changed flag."""

    COLOR = 'colorChanged'
    """Item's color changed flag."""

    YAXIS = 'yAxisChanged'
    """Item's Y axis binding changed flag."""

    FILL = 'fillChanged'
    """Item's fill changed flag."""

    ALPHA = 'alphaChanged'
    """Item's transparency alpha changed flag."""

    DATA = 'dataChanged'
    """Item's data changed flag"""

    HIGHLIGHTED = 'highlightedChanged'
    """Item's highlight state changed flag."""

    HIGHLIGHTED_COLOR = 'highlightedColorChanged'
    """Deprecated, use HIGHLIGHTED_STYLE instead."""

    HIGHLIGHTED_STYLE = 'highlightedStyleChanged'
    """Item's highlighted style changed flag."""

    SCALE = 'scaleChanged'
    """Item's scale changed flag."""

    TEXT = 'textChanged'
    """Item's text changed flag."""

    POSITION = 'positionChanged'
    """Item's position changed flag.

    This is emitted when a marker position changed and
    when an image origin changed.
    """

    OVERLAY = 'overlayChanged'
    """Item's overlay state changed flag."""

    VISUALIZATION_MODE = 'visualizationModeChanged'
    """Item's visualization mode changed flag."""


class Item(qt.QObject):
    """Description of an item of the plot"""

    _DEFAULT_Z_LAYER = 0
    """Default layer for overlay rendering"""

    _DEFAULT_LEGEND = ''
    """Default legend of items"""

    _DEFAULT_SELECTABLE = False
    """Default selectable state of items"""

    sigItemChanged = qt.Signal(object)
    """Signal emitted when the item has changed.

    It provides a flag describing which property of the item has changed.
    See :class:`ItemChangedType` for flags description.
    """

    def __init__(self):
        qt.QObject.__init__(self)
        self._dirty = True
        self._plotRef = None
        self._visible = True
        self._legend = self._DEFAULT_LEGEND
        self._selectable = self._DEFAULT_SELECTABLE
        self._z = self._DEFAULT_Z_LAYER
        self._info = None
        self._xlabel = None
        self._ylabel = None

        self._backendRenderer = None

    def getPlot(self):
        """Returns Plot this item belongs to.

        :rtype: Plot or None
        """
        return None if self._plotRef is None else self._plotRef()

    def _setPlot(self, plot):
        """Set the plot this item belongs to.

        WARNING: This should only be called from the Plot.

        :param Plot plot: The Plot instance.
        """
        if plot is not None and self._plotRef is not None:
            raise RuntimeError('Trying to add a node at two places.')
        self._plotRef = None if plot is None else weakref.ref(plot)
        self._updated()

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

    def setVisible(self, visible):
        """Set visibility of item.

        :param bool visible: True to display it, False otherwise
        """
        visible = bool(visible)
        if visible != self._visible:
            self._visible = visible
            # When visibility has changed, always mark as dirty
            self._updated(ItemChangedType.VISIBLE,
                          checkVisibility=False)

    def isOverlay(self):
        """Return true if item is drawn as an overlay.

        :rtype: bool
        """
        return False

    def getLegend(self):
        """Returns the legend of this item (str)"""
        return self._legend

    def _setLegend(self, legend):
        """Set the legend.

        This is private as it is used by the plot as an identifier

        :param str legend: Item legend
        """
        legend = str(legend) if legend is not None else self._DEFAULT_LEGEND
        self._legend = legend

    def isSelectable(self):
        """Returns true if item is selectable (bool)"""
        return self._selectable

    def _setSelectable(self, selectable):  # TODO support update
        """Set whether item is selectable or not.

        This is private for now as change is not handled.

        :param bool selectable: True to make item selectable
        """
        self._selectable = bool(selectable)

    def getZValue(self):
        """Returns the layer on which to draw this item (int)"""
        return self._z

    def setZValue(self, z):
        z = int(z) if z is not None else self._DEFAULT_Z_LAYER
        if z != self._z:
            self._z = z
            self._updated(ItemChangedType.ZVALUE)

    def getInfo(self, copy=True):
        """Returns the info associated to this item

        :param bool copy: True to get a deepcopy, False otherwise.
        """
        return deepcopy(self._info) if copy else self._info

    def setInfo(self, info, copy=True):
        if copy:
            info = deepcopy(info)
        self._info = info

    def _updated(self, event=None, checkVisibility=True):
        """Mark the item as dirty (i.e., needing update).

        This also triggers Plot.replot.

        :param event: The event to send to :attr:`sigItemChanged` signal.
        :param bool checkVisibility: True to only mark as dirty if visible,
                                     False to always mark as dirty.
        """
        if not checkVisibility or self.isVisible():
            if not self._dirty:
                self._dirty = True
                # TODO: send event instead of explicit call
                plot = self.getPlot()
                if plot is not None:
                    plot._itemRequiresUpdate(self)
        if event is not None:
            self.sigItemChanged.emit(event)

    def _update(self, backend):
        """Called by Plot to update the backend for this item.

        This is meant to be called asynchronously from _updated.
        This optimizes the number of call to _update.

        :param backend: The backend to update
        """
        if self._dirty:
            # Remove previous renderer from backend if any
            self._removeBackendRenderer(backend)

            # If not visible, do not add renderer to backend
            if self.isVisible():
                self._backendRenderer = self._addBackendRenderer(backend)

            self._dirty = False

    def _addBackendRenderer(self, backend):
        """Override in subclass to add specific backend renderer.

        :param BackendBase backend: The backend to update
        :return: The renderer handle to store or None if no renderer in backend
        """
        return None

    def _removeBackendRenderer(self, backend):
        """Override in subclass to remove specific backend renderer.

        :param BackendBase backend: The backend to update
        """
        if self._backendRenderer is not None:
            backend.remove(self._backendRenderer)
            self._backendRenderer = None


# Mix-in classes ##############################################################

class ItemMixInBase(qt.QObject):
    """Base class for Item mix-in"""

    def _updated(self, event=None, checkVisibility=True):
        """This is implemented in :class:`Item`.

        Mark the item as dirty (i.e., needing update).
        This also triggers Plot.replot.

        :param event: The event to send to :attr:`sigItemChanged` signal.
        :param bool checkVisibility: True to only mark as dirty if visible,
                                     False to always mark as dirty.
        """
        raise RuntimeError(
            "Issue with Mix-In class inheritance order")


class LabelsMixIn(ItemMixInBase):
    """Mix-in class for items with x and y labels

    Setters are private, otherwise it needs to check the plot
    current active curve and access the internal current labels.
    """

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


class DraggableMixIn(ItemMixInBase):
    """Mix-in class for draggable items"""

    def __init__(self):
        self._draggable = False

    def isDraggable(self):
        """Returns true if image is draggable

        :rtype: bool
        """
        return self._draggable

    def _setDraggable(self, draggable):  # TODO support update
        """Set if image is draggable or not.

        This is private for not as it does not support update.

        :param bool draggable:
        """
        self._draggable = bool(draggable)


class ColormapMixIn(ItemMixInBase):
    """Mix-in class for items with colormap"""

    def __init__(self):
        self._colormap = Colormap()
        self._colormap.sigChanged.connect(self._colormapChanged)

    def getColormap(self):
        """Return the used colormap"""
        return self._colormap

    def setColormap(self, colormap):
        """Set the colormap of this image

        :param silx.gui.colors.Colormap colormap: colormap description
        """
        if isinstance(colormap, dict):
            colormap = Colormap._fromDict(colormap)

        if self._colormap is not None:
            self._colormap.sigChanged.disconnect(self._colormapChanged)
        self._colormap = colormap
        if self._colormap is not None:
            self._colormap.sigChanged.connect(self._colormapChanged)
        self._colormapChanged()

    def _colormapChanged(self):
        """Handle updates of the colormap"""
        self._updated(ItemChangedType.COLORMAP)


class SymbolMixIn(ItemMixInBase):
    """Mix-in class for items with symbol type"""

    _DEFAULT_SYMBOL = ''
    """Default marker of the item"""

    _DEFAULT_SYMBOL_SIZE = 6.0
    """Default marker size of the item"""

    _SUPPORTED_SYMBOLS = collections.OrderedDict((
        ('o', 'Circle'),
        ('d', 'Diamond'),
        ('s', 'Square'),
        ('+', 'Plus'),
        ('x', 'Cross'),
        ('.', 'Point'),
        (',', 'Pixel'),
        ('', 'None')))
    """Dict of supported symbols"""

    def __init__(self):
        self._symbol = self._DEFAULT_SYMBOL
        self._symbol_size = self._DEFAULT_SYMBOL_SIZE

    @classmethod
    def getSupportedSymbols(cls):
        """Returns the list of supported symbol names.

        :rtype: tuple of str
        """
        return tuple(cls._SUPPORTED_SYMBOLS.keys())

    @classmethod
    def getSupportedSymbolNames(cls):
        """Returns the list of supported symbol human-readable names.

        :rtype: tuple of str
        """
        return tuple(cls._SUPPORTED_SYMBOLS.values())

    def getSymbolName(self, symbol=None):
        """Returns human-readable name for a symbol.

        :param str symbol: The symbol from which to get the name.
                           Default: current symbol.
        :rtype: str
        :raise KeyError: if symbol is not in :meth:`getSupportedSymbols`.
        """
        if symbol is None:
            symbol = self.getSymbol()
        return self._SUPPORTED_SYMBOLS[symbol]

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

    def setSymbol(self, symbol):
        """Set the marker type

        See :meth:`getSymbol`.

        :param str symbol: Marker type or marker name
        """
        if symbol is None:
            symbol = self._DEFAULT_SYMBOL

        elif symbol not in self.getSupportedSymbols():
            for symbolCode, name in self._SUPPORTED_SYMBOLS.items():
                if name.lower() == symbol.lower():
                    symbol = symbolCode
                    break
            else:
                raise ValueError('Unsupported symbol %s' % str(symbol))

        if symbol != self._symbol:
            self._symbol = symbol
            self._updated(ItemChangedType.SYMBOL)

    def getSymbolSize(self):
        """Return the point marker size in points.

        :rtype: float
        """
        return self._symbol_size

    def setSymbolSize(self, size):
        """Set the point marker size in points.

        See :meth:`getSymbolSize`.

        :param str symbol: Marker type
        """
        if size is None:
            size = self._DEFAULT_SYMBOL_SIZE
        if size != self._symbol_size:
            self._symbol_size = size
            self._updated(ItemChangedType.SYMBOL_SIZE)


class LineMixIn(ItemMixInBase):
    """Mix-in class for item with line"""

    _DEFAULT_LINEWIDTH = 1.
    """Default line width"""

    _DEFAULT_LINESTYLE = '-'
    """Default line style"""

    _SUPPORTED_LINESTYLE = '', ' ', '-', '--', '-.', ':', None
    """Supported line styles"""

    def __init__(self):
        self._linewidth = self._DEFAULT_LINEWIDTH
        self._linestyle = self._DEFAULT_LINESTYLE

    @classmethod
    def getSupportedLineStyles(cls):
        """Returns list of supported line styles.

        :rtype: List[str,None]
        """
        return cls._SUPPORTED_LINESTYLE

    def getLineWidth(self):
        """Return the curve line width in pixels

        :rtype: float
        """
        return self._linewidth

    def setLineWidth(self, width):
        """Set the width in pixel of the curve line

        See :meth:`getLineWidth`.

        :param float width: Width in pixels
        """
        width = float(width)
        if width != self._linewidth:
            self._linewidth = width
            self._updated(ItemChangedType.LINE_WIDTH)

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

    def setLineStyle(self, style):
        """Set the style of the curve line.

        See :meth:`getLineStyle`.

        :param str style: Line style
        """
        style = str(style)
        assert style in self.getSupportedLineStyles()
        if style is None:
            style = self._DEFAULT_LINESTYLE
        if style != self._linestyle:
            self._linestyle = style
            self._updated(ItemChangedType.LINE_STYLE)


class ColorMixIn(ItemMixInBase):
    """Mix-in class for item with color"""

    _DEFAULT_COLOR = (0., 0., 0., 1.)
    """Default color of the item"""

    def __init__(self):
        self._color = self._DEFAULT_COLOR

    def getColor(self):
        """Returns the RGBA color of the item

        :rtype: 4-tuple of float in [0, 1] or array of colors
        """
        return self._color

    def setColor(self, color, copy=True):
        """Set item color

        :param color: color(s) to be used
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in colors.py
        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        """
        if isinstance(color, six.string_types):
            color = colors.rgba(color)
        else:
            color = numpy.array(color, copy=copy)
            # TODO more checks + improve color array support
            if color.ndim == 1:  # Single RGBA color
                color = colors.rgba(color)
            else:  # Array of colors
                assert color.ndim == 2

        self._color = color
        self._updated(ItemChangedType.COLOR)


class YAxisMixIn(ItemMixInBase):
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

    def setYAxis(self, yaxis):
        """Set the Y axis this curve belongs to.

        :param str yaxis: 'left' or 'right'
        """
        yaxis = str(yaxis)
        assert yaxis in ('left', 'right')
        if yaxis != self._yaxis:
            self._yaxis = yaxis
            self._updated(ItemChangedType.YAXIS)


class FillMixIn(ItemMixInBase):
    """Mix-in class for item with fill"""

    def __init__(self):
        self._fill = False

    def isFill(self):
        """Returns whether the item is filled or not.

        :rtype: bool
        """
        return self._fill

    def setFill(self, fill):
        """Set whether to fill the item or not.

        :param bool fill:
        """
        fill = bool(fill)
        if fill != self._fill:
            self._fill = fill
            self._updated(ItemChangedType.FILL)


class AlphaMixIn(ItemMixInBase):
    """Mix-in class for item with opacity"""

    def __init__(self):
        self._alpha = 1.

    def getAlpha(self):
        """Returns the opacity of the item

        :rtype: float in [0, 1.]
        """
        return self._alpha

    def setAlpha(self, alpha):
        """Set the opacity of the item

        .. note::

            If the colormap already has some transparency, this alpha
            adds additional transparency. The alpha channel of the colormap
            is multiplied by this value.

        :param alpha: Opacity of the item, between 0 (full transparency)
            and 1. (full opacity)
        :type alpha: float
        """
        alpha = float(alpha)
        alpha = max(0., min(alpha, 1.))  # Clip alpha to [0., 1.] range
        if alpha != self._alpha:
            self._alpha = alpha
            self._updated(ItemChangedType.ALPHA)


class Points(Item, SymbolMixIn, AlphaMixIn):
    """Base class for :class:`Curve` and :class:`Scatter`"""
    # note: _logFilterData must be overloaded if you overload
    #       getData to change its signature

    _DEFAULT_Z_LAYER = 1
    """Default overlay layer for points,
    on top of images."""

    def __init__(self):
        Item.__init__(self)
        SymbolMixIn.__init__(self)
        AlphaMixIn.__init__(self)
        self._x = ()
        self._y = ()
        self._xerror = None
        self._yerror = None

        # Store filtered data for x > 0 and/or y > 0
        self._filteredCache = {}
        self._clippedCache = {}

        # Store bounds depending on axes filtering >0:
        # key is (isXPositiveFilter, isYPositiveFilter)
        self._boundsCache = {}

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
            valueMinusError = value - numpy.atleast_2d(error)[0]
            errorClipped = numpy.isnan(valueMinusError)
            mask = numpy.logical_not(errorClipped)
            errorClipped[mask] = valueMinusError[mask] <= 0

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

    def _getClippingBoolArray(self, xPositive, yPositive):
        """Compute a boolean array to filter out points with negative
        coordinates on log axes.

        :param bool xPositive: True to filter arrays according to X coords.
        :param bool yPositive: True to filter arrays according to Y coords.
        :rtype: boolean numpy.ndarray
        """
        assert xPositive or yPositive
        if (xPositive, yPositive) not in self._clippedCache:
            xclipped, yclipped = False, False

            if xPositive:
                x = self.getXData(copy=False)
                with warnings.catch_warnings():  # Ignore NaN warnings
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    xclipped = x <= 0

            if yPositive:
                y = self.getYData(copy=False)
                with warnings.catch_warnings():  # Ignore NaN warnings
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    yclipped = y <= 0

            self._clippedCache[(xPositive, yPositive)] = \
                numpy.logical_or(xclipped, yclipped)
        return self._clippedCache[(xPositive, yPositive)]

    def _logFilterData(self, xPositive, yPositive):
        """Filter out values with x or y <= 0 on log axes

        :param bool xPositive: True to filter arrays according to X coords.
        :param bool yPositive: True to filter arrays according to Y coords.
        :return: The filter arrays or unchanged object if filtering not needed
        :rtype: (x, y, xerror, yerror)
        """
        x = self.getXData(copy=False)
        y = self.getYData(copy=False)
        xerror = self.getXErrorData(copy=False)
        yerror = self.getYErrorData(copy=False)

        if xPositive or yPositive:
            clipped = self._getClippingBoolArray(xPositive, yPositive)

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
            xPositive = plot.getXAxis()._isLogarithmic()
            yPositive = plot.getYAxis()._isLogarithmic()
        else:
            xPositive = False
            yPositive = False

        # TODO bounds do not take error bars into account
        if (xPositive, yPositive) not in self._boundsCache:
            # use the getData class method because instance method can be
            # overloaded to return additional arrays
            data = Points.getData(self, copy=False,
                                 displayed=True)
            if len(data) == 5:
                # hack to avoid duplicating caching mechanism in Scatter
                # (happens when cached data is used, caching done using
                # Scatter._logFilterData)
                x, y, xerror, yerror = data[0], data[1], data[3], data[4]
            else:
                x, y, xerror, yerror = data

            self._boundsCache[(xPositive, yPositive)] = (
                numpy.nanmin(x),
                numpy.nanmax(x),
                numpy.nanmin(y),
                numpy.nanmax(y)
            )
        return self._boundsCache[(xPositive, yPositive)]

    def _getCachedData(self):
        """Return cached filtered data if applicable,
        i.e. if any axis is in log scale.
        Return None if caching is not applicable."""
        plot = self.getPlot()
        if plot is not None:
            xPositive = plot.getXAxis()._isLogarithmic()
            yPositive = plot.getYAxis()._isLogarithmic()
            if xPositive or yPositive:
                # At least one axis has log scale, filter data
                if (xPositive, yPositive) not in self._filteredCache:
                    self._filteredCache[(xPositive, yPositive)] = \
                        self._logFilterData(xPositive, yPositive)
                return self._filteredCache[(xPositive, yPositive)]
        return None

    def getData(self, copy=True, displayed=False):
        """Returns the x, y values of the curve points and xerror, yerror

        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        :param bool displayed: True to only get curve points that are displayed
                               in the plot. Default: False
                               Note: If plot has log scale, negative points
                               are not displayed.
        :returns: (x, y, xerror, yerror)
        :rtype: 4-tuple of numpy.ndarray
        """
        if displayed:  # filter data according to plot state
            cached_data = self._getCachedData()
            if cached_data is not None:
                return cached_data

        return (self.getXData(copy),
                self.getYData(copy),
                self.getXErrorData(copy),
                self.getYErrorData(copy))

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
        """Returns the x error of the points

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray, float or None
        """
        if isinstance(self._xerror, numpy.ndarray):
            return numpy.array(self._xerror, copy=copy)
        else:
            return self._xerror  # float or None

    def getYErrorData(self, copy=True):
        """Returns the y error of the points

        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray, float or None
        """
        if isinstance(self._yerror, numpy.ndarray):
            return numpy.array(self._yerror, copy=copy)
        else:
            return self._yerror  # float or None

    def setData(self, x, y, xerror=None, yerror=None, copy=True):
        """Set the data of the curve.

        :param numpy.ndarray x: The data corresponding to the x coordinates.
        :param numpy.ndarray y: The data corresponding to the y coordinates.
        :param xerror: Values with the uncertainties on the x values
        :type xerror: A float, or a numpy.ndarray of float32.
                      If it is an array, it can either be a 1D array of
                      same length as the data or a 2D array with 2 rows
                      of same length as the data: row 0 for positive errors,
                      row 1 for negative errors.
        :param yerror: Values with the uncertainties on the y values.
        :type yerror: A float, or a numpy.ndarray of float32. See xerror.
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        """
        x = numpy.array(x, copy=copy)
        y = numpy.array(y, copy=copy)
        assert len(x) == len(y)
        assert x.ndim == y.ndim == 1

        if xerror is not None:
            if isinstance(xerror, collections.Iterable):
                xerror = numpy.array(xerror, copy=copy)
            else:
                xerror = float(xerror)
        if yerror is not None:
            if isinstance(yerror, collections.Iterable):
                yerror = numpy.array(yerror, copy=copy)
            else:
                yerror = float(yerror)
        # TODO checks on xerror, yerror
        self._x, self._y = x, y
        self._xerror, self._yerror = xerror, yerror

        self._boundsCache = {}  # Reset cached bounds
        self._filteredCache = {}  # Reset cached filtered data
        self._clippedCache = {}  # Reset cached clipped bool array

        # TODO hackish data range implementation
        if self.isVisible():
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()
        self._updated(ItemChangedType.DATA)
