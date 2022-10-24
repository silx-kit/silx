# /*##########################################################################
#
# Copyright (c) 2017-2022 European Synchrotron Radiation Facility
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
__date__ = "08/12/2020"

import collections
try:
    from collections import abc
except ImportError:  # Python2 support
    import collections as abc
from copy import deepcopy
import logging
import enum
from typing import Optional, Tuple, Union
import weakref

import numpy

from ....utils.deprecation import deprecated
from ....utils.proxy import docstring
from ....utils.enum import Enum as _Enum
from ....math.combo import min_max
from ... import qt
from ... import colors
from ...colors import Colormap
from ._pick import PickingResult

from silx import config

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

    LINE_BG_COLOR = 'lineBgColorChanged'
    """Item's line background color changed flag."""

    YAXIS = 'yAxisChanged'
    """Item's Y axis binding changed flag."""

    FILL = 'fillChanged'
    """Item's fill changed flag."""

    ALPHA = 'alphaChanged'
    """Item's transparency alpha changed flag."""

    DATA = 'dataChanged'
    """Item's data changed flag"""

    MASK = 'maskChanged'
    """Item's mask changed flag"""

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

    COMPLEX_MODE = 'complexModeChanged'
    """Item's complex data visualization mode changed flag."""

    NAME = 'nameChanged'
    """Item's name changed flag."""

    EDITABLE = 'editableChanged'
    """Item's editable state changed flags."""

    SELECTABLE = 'selectableChanged'
    """Item's selectable state changed flags."""


class Item(qt.QObject):
    """Description of an item of the plot"""

    _DEFAULT_Z_LAYER = 0
    """Default layer for overlay rendering"""

    _DEFAULT_SELECTABLE = False
    """Default selectable state of items"""

    sigItemChanged = qt.Signal(object)
    """Signal emitted when the item has changed.

    It provides a flag describing which property of the item has changed.
    See :class:`ItemChangedType` for flags description.
    """

    _sigVisibleBoundsChanged = qt.Signal()
    """Signal emitted when the visible extent of the item in the plot has changed.

    This signal is emitted only if visible extent tracking is enabled
    (see :meth:`_setVisibleBoundsTracking`).
    """

    def __init__(self):
        qt.QObject.__init__(self)
        self._dirty = True
        self._plotRef = None
        self._visible = True
        self._selectable = self._DEFAULT_SELECTABLE
        self._z = self._DEFAULT_Z_LAYER
        self._info = None
        self._xlabel = None
        self._ylabel = None
        self.__name = ''

        self.__visibleBoundsTracking = False
        self.__previousVisibleBounds = None

        self._backendRenderer = None

    def getPlot(self):
        """Returns the ~silx.gui.plot.PlotWidget this item belongs to.

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        return None if self._plotRef is None else self._plotRef()

    def _setPlot(self, plot):
        """Set the plot this item belongs to.

        WARNING: This should only be called from the Plot.

        :param Union[~silx.gui.plot.PlotWidget,None] plot: The Plot instance.
        """
        if plot is not None and self._plotRef is not None:
            raise RuntimeError('Trying to add a node at two places.')
        self.__disconnectFromPlotWidget()
        self._plotRef = None if plot is None else weakref.ref(plot)
        self.__connectToPlotWidget()
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
            if visible:
                self._visibleBoundsChanged()

    def isOverlay(self):
        """Return true if item is drawn as an overlay.

        :rtype: bool
        """
        return False

    def getName(self):
        """Returns the name of the item which is used as legend.

        :rtype: str
        """
        return self.__name

    def setName(self, name):
        """Set the name of the item which is used as legend.

        :param str name: New name of the item
        :raises RuntimeError: If item belongs to a PlotWidget.
        """
        name = str(name)
        if self.__name != name:
            if self.getPlot() is not None:
                raise RuntimeError(
                    "Cannot change name while item is in a PlotWidget")

            self.__name = name
            self._updated(ItemChangedType.NAME)

    def getLegend(self):  # Replaced by getName for API consistency
        return self.getName()

    @deprecated(replacement='setName', since_version='0.13')
    def _setLegend(self, legend):
        legend = str(legend) if legend is not None else ''
        self.setName(legend)

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

    def getVisibleBounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Returns visible bounds of the item bounding box in the plot area.

        :returns:
            (xmin, xmax, ymin, ymax) in data coordinates of the visible area or
            None if item is not visible in the plot area.
        :rtype: Union[List[float],None]
        """
        plot = self.getPlot()
        bounds = self.getBounds()
        if plot is None or bounds is None or not self.isVisible():
            return None

        xmin, xmax = numpy.clip(bounds[:2], *plot.getXAxis().getLimits())
        ymin, ymax = numpy.clip(
            bounds[2:], *plot.getYAxis(self.__getYAxis()).getLimits())

        if xmin == xmax or ymin == ymax:  # Outside the plot area
            return None
        else:
            return xmin, xmax, ymin, ymax

    def _isVisibleBoundsTracking(self) -> bool:
        """Returns True if visible bounds changes are tracked.

        When enabled, :attr:`_sigVisibleBoundsChanged` is emitted upon changes.
        :rtype: bool
        """
        return self.__visibleBoundsTracking

    def _setVisibleBoundsTracking(self, enable: bool) -> None:
        """Set whether or not to track visible bounds changes.

        :param bool enable:
        """
        if enable != self.__visibleBoundsTracking:
            self.__disconnectFromPlotWidget()
            self.__previousVisibleBounds = None
            self.__visibleBoundsTracking = enable
            self.__connectToPlotWidget()

    def __getYAxis(self) -> str:
        """Returns current Y axis ('left' or 'right')"""
        return self.getYAxis() if isinstance(self, YAxisMixIn) else 'left'

    def __connectToPlotWidget(self) -> None:
        """Connect to PlotWidget signals and install event filter"""
        if not self._isVisibleBoundsTracking():
            return

        plot = self.getPlot()
        if plot is not None:
            for axis in (plot.getXAxis(), plot.getYAxis(self.__getYAxis())):
                axis.sigLimitsChanged.connect(self._visibleBoundsChanged)

            plot.installEventFilter(self)

            self._visibleBoundsChanged()

    def __disconnectFromPlotWidget(self) -> None:
        """Disconnect from PlotWidget signals and remove event filter"""
        if not self._isVisibleBoundsTracking():
            return

        plot = self.getPlot()
        if plot is not None:
            for axis in (plot.getXAxis(), plot.getYAxis(self.__getYAxis())):
                axis.sigLimitsChanged.disconnect(self._visibleBoundsChanged)

            plot.removeEventFilter(self)

    def _visibleBoundsChanged(self, *args) -> None:
        """Check if visible extent actually changed and emit signal"""
        if not self._isVisibleBoundsTracking():
            return  # No visible extent tracking

        plot = self.getPlot()
        if plot is None or not plot.isVisible():
            return  # No plot or plot not visible

        extent = self.getVisibleBounds()
        if extent != self.__previousVisibleBounds:
            self.__previousVisibleBounds = extent
            self._sigVisibleBoundsChanged.emit()

    def eventFilter(self, watched, event):
        """Event filter to handle PlotWidget show events"""
        if watched is self.getPlot() and event.type() == qt.QEvent.Show:
            self._visibleBoundsChanged()
        return super().eventFilter(watched, event)

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

    def pick(self, x, y):
        """Run picking test on this item

        :param float x: The x pixel coord where to pick.
        :param float y: The y pixel coord where to pick.
        :return: None if not picked, else the picked position information
        :rtype: Union[None,PickingResult]
        """
        if not self.isVisible() or self._backendRenderer is None:
            return None
        plot = self.getPlot()
        if plot is None:
            return None

        indices = plot._backend.pickItem(x, y, self._backendRenderer)
        if indices is None:
            return None
        else:
            return PickingResult(self, indices)


class DataItem(Item):
    """Item with a data extent in the plot"""

    def _boundsChanged(self, checkVisibility: bool=True) -> None:
        """Call this method in subclass when data bounds has changed.

        :param bool checkVisibility:
        """
        if not checkVisibility or self.isVisible():
            if self.isVisible():
                self._visibleBoundsChanged()

            # TODO hackish data range implementation
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

    @docstring(Item)
    def setVisible(self, visible: bool):
        if visible != self.isVisible():
            self._boundsChanged(checkVisibility=False)
        super().setVisible(visible)

# Mix-in classes ##############################################################


class ItemMixInBase(object):
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

    def drag(self, from_, to):
        """Perform a drag of the item.

        :param List[float] from_: (x, y) previous position in data coordinates
        :param List[float] to: (x, y) current position in data coordinates
        """
        raise NotImplementedError("Must be implemented in subclass")


class ColormapMixIn(ItemMixInBase):
    """Mix-in class for items with colormap"""

    def __init__(self):
        self._colormap = Colormap()
        self._colormap.sigChanged.connect(self._colormapChanged)
        self.__data = None
        self.__cacheColormapRange = {}  # Store {normalization: range}

    def getColormap(self):
        """Return the used colormap"""
        return self._colormap

    def setColormap(self, colormap):
        """Set the colormap of this item

        :param silx.gui.colors.Colormap colormap: colormap description
        """
        if self._colormap is colormap:
            return
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

    def _setColormappedData(self, data, copy=True,
                            min_=None, minPositive=None, max_=None):
        """Set the data used to compute the colormapped display.

        It also resets the cache of data ranges.

        This method MUST be called by inheriting classes when data is updated.

        :param Union[None,numpy.ndarray] data:
        :param Union[None,float] min_: Minimum value of the data
        :param Union[None,float] minPositive:
            Minimum of strictly positive values of the data
        :param Union[None,float] max_: Maximum value of the data
        """
        self.__data = None if data is None else numpy.array(data, copy=copy)
        self.__cacheColormapRange = {}  # Reset cache

        # Fill-up colormap range cache if values are provided
        if max_ is not None and numpy.isfinite(max_):
            if min_ is not None and numpy.isfinite(min_):
                self.__cacheColormapRange[Colormap.LINEAR, Colormap.MINMAX] = min_, max_
            if minPositive is not None and numpy.isfinite(minPositive):
                self.__cacheColormapRange[Colormap.LOGARITHM, Colormap.MINMAX] = minPositive, max_

        colormap = self.getColormap()
        if None in (colormap.getVMin(), colormap.getVMax()):
            self._colormapChanged()

    def getColormappedData(self, copy=True):
        """Returns the data used to compute the displayed colors

        :param bool copy: True to get a copy,
            False to get internal data (do not modify!).
        :rtype: Union[None,numpy.ndarray]
        """
        if self.__data is None:
            return None
        else:
            return numpy.array(self.__data, copy=copy)

    def _getColormapAutoscaleRange(self, colormap=None):
        """Returns the autoscale range for current data and colormap.

        :param Union[None,~silx.gui.colors.Colormap] colormap:
           The colormap for which to compute the autoscale range.
           If None, the default, the colormap of the item is used
        :return: (vmin, vmax) range (vmin and /or vmax might be `None`)
        """
        if colormap is None:
            colormap = self.getColormap()

        data = self.getColormappedData(copy=False)
        if colormap is None or data is None:
            return None, None

        normalization = colormap.getNormalization()
        autoscaleMode = colormap.getAutoscaleMode()
        key = normalization, autoscaleMode
        vRange = self.__cacheColormapRange.get(key, None)
        if vRange is None:
            vRange = colormap._computeAutoscaleRange(data)
            self.__cacheColormapRange[key] = vRange
        return vRange


class SymbolMixIn(ItemMixInBase):
    """Mix-in class for items with symbol type"""

    _DEFAULT_SYMBOL = None
    """Default marker of the item"""

    _DEFAULT_SYMBOL_SIZE = config.DEFAULT_PLOT_SYMBOL_SIZE
    """Default marker size of the item"""

    _SUPPORTED_SYMBOLS = collections.OrderedDict((
        ('o', 'Circle'),
        ('d', 'Diamond'),
        ('s', 'Square'),
        ('+', 'Plus'),
        ('x', 'Cross'),
        ('.', 'Point'),
        (',', 'Pixel'),
        ('|', 'Vertical line'),
        ('_', 'Horizontal line'),
        ('tickleft', 'Tick left'),
        ('tickright', 'Tick right'),
        ('tickup', 'Tick up'),
        ('tickdown', 'Tick down'),
        ('caretleft', 'Caret left'),
        ('caretright', 'Caret right'),
        ('caretup', 'Caret up'),
        ('caretdown', 'Caret down'),
        (u'\u2665', 'Heart'),
        ('', 'None')))
    """Dict of supported symbols"""

    def __init__(self):
        if self._DEFAULT_SYMBOL is None:  # Use default from config
            self._symbol = config.DEFAULT_PLOT_SYMBOL
        else:
            self._symbol = self._DEFAULT_SYMBOL

        if self._DEFAULT_SYMBOL_SIZE is None:  # Use default from config
            self._symbol_size = config.DEFAULT_PLOT_SYMBOL_SIZE
        else:
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
        if isinstance(color, str):
            color = colors.rgba(color)
        elif isinstance(color, qt.QColor):
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
            # Handle data extent changed for DataItem
            if isinstance(self, DataItem):
                self._boundsChanged()

            # Handle visible extent changed
            if self._isVisibleBoundsTracking():
                # Switch Y axis signal connection
                plot = self.getPlot()
                if plot is not None:
                    previousYAxis = 'left' if self.getXAxis() == 'right' else 'right'
                    plot.getYAxis(previousYAxis).sigLimitsChanged.disconnect(
                        self._visibleBoundsChanged)
                    plot.getYAxis(self.getYAxis()).sigLimitsChanged.connect(
                        self._visibleBoundsChanged)
                self._visibleBoundsChanged()

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


class ComplexMixIn(ItemMixInBase):
    """Mix-in class for complex data mode"""

    _SUPPORTED_COMPLEX_MODES = None
    """Override to only support a subset of all ComplexMode"""

    class ComplexMode(_Enum):
        """Identify available display mode for complex"""
        NONE = 'none'
        ABSOLUTE = 'amplitude'
        PHASE = 'phase'
        REAL = 'real'
        IMAGINARY = 'imaginary'
        AMPLITUDE_PHASE = 'amplitude_phase'
        LOG10_AMPLITUDE_PHASE = 'log10_amplitude_phase'
        SQUARE_AMPLITUDE = 'square_amplitude'

    def __init__(self):
        self.__complex_mode = self.ComplexMode.ABSOLUTE

    def getComplexMode(self):
        """Returns the current complex visualization mode.

        :rtype: ComplexMode
        """
        return self.__complex_mode

    def setComplexMode(self, mode):
        """Set the complex visualization mode.

        :param ComplexMode mode: The visualization mode in:
            'real', 'imaginary', 'phase', 'amplitude'
        :return: True if value was set, False if is was already set
        :rtype: bool
        """
        mode = self.ComplexMode.from_value(mode)
        assert mode in self.supportedComplexModes()

        if mode != self.__complex_mode:
            self.__complex_mode = mode
            self._updated(ItemChangedType.COMPLEX_MODE)
            return True
        else:
            return False

    def _convertComplexData(self, data, mode=None):
        """Convert complex data to the specific mode.

        :param Union[ComplexMode,None] mode:
            The kind of value to compute.
            If None (the default), the current complex mode is used.
        :return: The converted dataset
        :rtype: Union[numpy.ndarray[float],None]
        """
        if data is None:
            return None

        if mode is None:
            mode = self.getComplexMode()

        if mode is self.ComplexMode.REAL:
            return numpy.real(data)
        elif mode is self.ComplexMode.IMAGINARY:
            return numpy.imag(data)
        elif mode is self.ComplexMode.ABSOLUTE:
            return numpy.absolute(data)
        elif mode is self.ComplexMode.PHASE:
            return numpy.angle(data)
        elif mode is self.ComplexMode.SQUARE_AMPLITUDE:
            return numpy.absolute(data) ** 2
        else:
            raise ValueError('Unsupported conversion mode: %s', str(mode))

    @classmethod
    def supportedComplexModes(cls):
        """Returns the list of supported complex visualization modes.

        See :class:`ComplexMode` and :meth:`setComplexMode`.

        :rtype: List[ComplexMode]
        """
        if cls._SUPPORTED_COMPLEX_MODES is None:
            return cls.ComplexMode.members()
        else:
            return cls._SUPPORTED_COMPLEX_MODES


class ScatterVisualizationMixIn(ItemMixInBase):
    """Mix-in class for scatter plot visualization modes"""

    _SUPPORTED_SCATTER_VISUALIZATION = None
    """Allows to override supported Visualizations"""

    @enum.unique
    class Visualization(_Enum):
        """Different modes of scatter plot visualizations"""

        POINTS = 'points'
        """Display scatter plot as a point cloud"""

        LINES = 'lines'
        """Display scatter plot as a wireframe.

        This is based on Delaunay triangulation
        """

        SOLID = 'solid'
        """Display scatter plot as a set of filled triangles.

        This is based on Delaunay triangulation
        """

        REGULAR_GRID = 'regular_grid'
        """Display scatter plot as an image.

        It expects the points to be the intersection of a regular grid,
        and the order of points following that of an image.
        First line, then second one, and always in the same direction
        (either all lines from left to right or all from right to left).
        """

        IRREGULAR_GRID = 'irregular_grid'
        """Display scatter plot as contiguous quadrilaterals.

        It expects the points to be the intersection of an irregular grid,
        and the order of points following that of an image.
        First line, then second one, and always in the same direction
        (either all lines from left to right or all from right to left).
        """

        BINNED_STATISTIC = 'binned_statistic'
        """Display scatter plot as 2D binned statistic (i.e., generalized histogram).
        """

    @enum.unique
    class VisualizationParameter(_Enum):
        """Different parameter names for scatter plot visualizations"""

        GRID_MAJOR_ORDER = 'grid_major_order'
        """The major order of points in the regular grid.

        Either 'row' (row-major, fast X) or 'column' (column-major, fast Y).
        """

        GRID_BOUNDS = 'grid_bounds'
        """The expected range in data coordinates of the regular grid.

        A 2-tuple of 2-tuple: (begin (x, y), end (x, y)).
        This provides the data coordinates of the first point and the expected
        last on.
        As for `GRID_SHAPE`, this can be wider than the current data.
        """

        GRID_SHAPE = 'grid_shape'
        """The expected size of the regular grid (height, width).

        The given shape can be wider than the number of points,
        in which case the grid is not fully filled.
        """

        BINNED_STATISTIC_SHAPE = 'binned_statistic_shape'
        """The number of bins in each dimension (height, width).
        """

        BINNED_STATISTIC_FUNCTION = 'binned_statistic_function'
        """The reduction function to apply to each bin (str).

        Available reduction functions are: 'mean' (default), 'count', 'sum'.
        """

        DATA_BOUNDS_HINT = 'data_bounds_hint'
        """The expected bounds of the data in data coordinates.

        A 2-tuple of 2-tuple: ((ymin, ymax), (xmin, xmax)).
        This provides a hint for the data ranges in both dimensions.
        It is eventually enlarged with actually data ranges.

        WARNING: dimension 0 i.e., Y first.
        """

    _SUPPORTED_VISUALIZATION_PARAMETER_VALUES = {
        VisualizationParameter.GRID_MAJOR_ORDER: ('row', 'column'),
        VisualizationParameter.BINNED_STATISTIC_FUNCTION: ('mean', 'count', 'sum'),
    }
    """Supported visualization parameter values.

    Defined for parameters with a set of acceptable values.
    """

    def __init__(self):
        self.__visualization = self.Visualization.POINTS
        self.__parameters = dict(# Init parameters to None
            (parameter, None) for parameter in self.VisualizationParameter)
        self.__parameters[self.VisualizationParameter.BINNED_STATISTIC_FUNCTION] = 'mean'

    @classmethod
    def supportedVisualizations(cls):
        """Returns the list of supported scatter visualization modes.

        See :meth:`setVisualization`

        :rtype: List[Visualization]
        """
        if cls._SUPPORTED_SCATTER_VISUALIZATION is None:
            return cls.Visualization.members()
        else:
            return cls._SUPPORTED_SCATTER_VISUALIZATION

    @classmethod
    def supportedVisualizationParameterValues(cls, parameter):
        """Returns the list of supported scatter visualization modes.

        See :meth:`VisualizationParameters`

        :param VisualizationParameter parameter:
            This parameter for which to retrieve the supported values.
        :returns: tuple of supported of values or None if not defined.
        """
        parameter = cls.VisualizationParameter(parameter)
        return cls._SUPPORTED_VISUALIZATION_PARAMETER_VALUES.get(
            parameter, None)

    def setVisualization(self, mode):
        """Set the scatter plot visualization mode to use.

        See :class:`Visualization` for all possible values,
        and :meth:`supportedVisualizations` for supported ones.

        :param Union[str,Visualization] mode:
            The visualization mode to use.
        :return: True if value was set, False if is was already set
        :rtype: bool
        """
        mode = self.Visualization.from_value(mode)
        assert mode in self.supportedVisualizations()

        if mode != self.__visualization:
            self.__visualization = mode

            self._updated(ItemChangedType.VISUALIZATION_MODE)
            return True
        else:
            return False

    def getVisualization(self):
        """Returns the scatter plot visualization mode in use.

        :rtype: Visualization
        """
        return self.__visualization

    def setVisualizationParameter(self, parameter, value=None):
        """Set the given visualization parameter.

        :param Union[str,VisualizationParameter] parameter:
            The name of the parameter to set
        :param value: The value to use for this parameter
            Set to None to automatically set the parameter
        :raises ValueError: If parameter is not supported
        :return: True if parameter was set, False if is was already set
        :rtype: bool
        :raise ValueError: If value is not supported
        """
        parameter = self.VisualizationParameter.from_value(parameter)

        if self.__parameters[parameter] != value:
            validValues = self.supportedVisualizationParameterValues(parameter)
            if validValues is not None and value not in validValues:
                raise ValueError("Unsupported parameter value: %s" % str(value))

            self.__parameters[parameter] = value
            self._updated(ItemChangedType.VISUALIZATION_MODE)
            return True
        return False

    def getVisualizationParameter(self, parameter):
        """Returns the value of the given visualization parameter.

        This method returns the parameter as set by
        :meth:`setVisualizationParameter`.

        :param parameter: The name of the parameter to retrieve
        :returns: The value previously set or None if automatically set
        :raises ValueError: If parameter is not supported
        """
        if parameter not in self.VisualizationParameter:
            raise ValueError("parameter not supported: %s", parameter)

        return self.__parameters[parameter]

    def getCurrentVisualizationParameter(self, parameter):
        """Returns the current value of the given visualization parameter.

        If the parameter was set by :meth:`setVisualizationParameter` to
        a value that is not None, this value is returned;
        else the current value that is automatically computed is returned.

        :param parameter: The name of the parameter to retrieve
        :returns: The current value (either set or automatically computed)
        :raises ValueError: If parameter is not supported
        """
        # Override in subclass to provide automatically computed parameters
        return self.getVisualizationParameter(parameter)


class PointsBase(DataItem, SymbolMixIn, AlphaMixIn):
    """Base class for :class:`Curve` and :class:`Scatter`"""
    # note: _logFilterData must be overloaded if you overload
    #       getData to change its signature

    _DEFAULT_Z_LAYER = 1
    """Default overlay layer for points,
    on top of images."""

    def __init__(self):
        DataItem.__init__(self)
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
                        (2, len(value)), error, dtype=numpy.float64)

                elif error.ndim == 1:  # N array
                    newError = numpy.empty((2, len(value)),
                                           dtype=numpy.float64)
                    newError[0,:] = error
                    newError[1,:] = error
                    error = newError

                elif error.size == 2 * len(value):  # 2xN array
                    error = numpy.array(
                        error, copy=True, dtype=numpy.float64)

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
                with numpy.errstate(invalid='ignore'):  # Ignore NaN warnings
                    xclipped = x <= 0

            if yPositive:
                y = self.getYData(copy=False)
                with numpy.errstate(invalid='ignore'):  # Ignore NaN warnings
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
                x = numpy.array(x, copy=True, dtype=numpy.float64)
                x[clipped] = numpy.nan
                y = numpy.array(y, copy=True, dtype=numpy.float64)
                y[clipped] = numpy.nan

                if xPositive and xerror is not None:
                    xerror = self._logFilterError(x, xerror)

                if yPositive and yerror is not None:
                    yerror = self._logFilterError(y, yerror)

        return x, y, xerror, yerror

    @staticmethod
    def __minMaxDataWithError(
        data: numpy.ndarray,
        error: Optional[Union[float, numpy.ndarray]],
        positiveOnly: bool
    ) -> Tuple[float]:
        if error is None:
            min_, max_ = min_max(data, finite=True)
            return min_, max_

        # float, 1D or 2D array
        dataMinusError = data - numpy.atleast_2d(error)[0]
        dataMinusError = dataMinusError[numpy.isfinite(dataMinusError)]
        if positiveOnly:
            dataMinusError = dataMinusError[dataMinusError > 0]
        min_ = numpy.nan if dataMinusError.size == 0 else numpy.min(dataMinusError)

        dataPlusError = data + numpy.atleast_2d(error)[-1]
        dataPlusError = dataPlusError[numpy.isfinite(dataPlusError)]
        if positiveOnly:
            dataPlusError = dataPlusError[dataPlusError > 0]
        max_ = numpy.nan if dataPlusError.size == 0 else numpy.max(dataPlusError)

        return min_, max_

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

        if (xPositive, yPositive) not in self._boundsCache:
            # use the getData class method because instance method can be
            # overloaded to return additional arrays
            data = PointsBase.getData(self, copy=False, displayed=True)
            if len(data) == 5:
                # hack to avoid duplicating caching mechanism in Scatter
                # (happens when cached data is used, caching done using
                # Scatter._logFilterData)
                x, y, xerror, yerror = data[0], data[1], data[3], data[4]
            else:
                x, y, xerror, yerror = data

            xmin, xmax = self.__minMaxDataWithError(x, xerror, xPositive)
            ymin, ymax = self.__minMaxDataWithError(y, yerror, yPositive)

            self._boundsCache[(xPositive, yPositive)] = tuple([
                (bound if bound is not None else numpy.nan)
                for bound in (xmin, xmax, ymin, ymax)])
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
                      of same length as the data: row 0 for lower errors,
                      row 1 for upper errors.
        :param yerror: Values with the uncertainties on the y values.
        :type yerror: A float, or a numpy.ndarray of float32. See xerror.
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        """
        x = numpy.array(x, copy=copy)
        y = numpy.array(y, copy=copy)
        assert len(x) == len(y)
        assert x.ndim == y.ndim == 1

        # Convert complex data
        if numpy.iscomplexobj(x):
            _logger.warning(
                'Converting x data to absolute value to plot it.')
            x = numpy.absolute(x)
        if numpy.iscomplexobj(y):
            _logger.warning(
                'Converting y data to absolute value to plot it.')
            y = numpy.absolute(y)

        if xerror is not None:
            if isinstance(xerror, abc.Iterable):
                xerror = numpy.array(xerror, copy=copy)
                if numpy.iscomplexobj(xerror):
                    _logger.warning(
                        'Converting xerror data to absolute value to plot it.')
                    xerror = numpy.absolute(xerror)
            else:
                xerror = float(xerror)
        if yerror is not None:
            if isinstance(yerror, abc.Iterable):
                yerror = numpy.array(yerror, copy=copy)
                if numpy.iscomplexobj(yerror):
                    _logger.warning(
                        'Converting yerror data to absolute value to plot it.')
                    yerror = numpy.absolute(yerror)
            else:
                yerror = float(yerror)
        # TODO checks on xerror, yerror
        self._x, self._y = x, y
        self._xerror, self._yerror = xerror, yerror

        self._boundsCache = {}  # Reset cached bounds
        self._filteredCache = {}  # Reset cached filtered data
        self._clippedCache = {}  # Reset cached clipped bool array

        self._boundsChanged()
        self._updated(ItemChangedType.DATA)


class BaselineMixIn(object):
    """Base class for Baseline mix-in"""

    def __init__(self, baseline=None):
        self._baseline = baseline

    def _setBaseline(self, baseline):
        """
        Set baseline value

        :param baseline: baseline value(s)
        :type: Union[None,float,numpy.ndarray]
        """
        if (isinstance(baseline, abc.Iterable)):
            baseline = numpy.array(baseline)
        self._baseline = baseline

    def getBaseline(self, copy=True):
        """

        :param bool copy:
        :return: histogram baseline
        :rtype: Union[None,float,numpy.ndarray]
        """
        if isinstance(self._baseline, numpy.ndarray):
            return numpy.array(self._baseline, copy=True)
        else:
            return self._baseline


class _Style:
    """Object which store styles"""


class HighlightedMixIn(ItemMixInBase):

    def __init__(self):
        self._highlightStyle = self._DEFAULT_HIGHLIGHT_STYLE
        self._highlighted = False

    def isHighlighted(self):
        """Returns True if curve is highlighted.

        :rtype: bool
        """
        return self._highlighted

    def setHighlighted(self, highlighted):
        """Set the highlight state of the curve

        :param bool highlighted:
        """
        highlighted = bool(highlighted)
        if highlighted != self._highlighted:
            self._highlighted = highlighted
            # TODO inefficient: better to use backend's setCurveColor
            self._updated(ItemChangedType.HIGHLIGHTED)

    def getHighlightedStyle(self):
        """Returns the highlighted style in use

        :rtype: CurveStyle
        """
        return self._highlightStyle

    def setHighlightedStyle(self, style):
        """Set the style to use for highlighting

        :param CurveStyle style: New style to use
        """
        previous = self.getHighlightedStyle()
        if style != previous:
            assert isinstance(style, _Style)
            self._highlightStyle = style
            self._updated(ItemChangedType.HIGHLIGHTED_STYLE)

            # Backward compatibility event
            if previous.getColor() != style.getColor():
                self._updated(ItemChangedType.HIGHLIGHTED_COLOR)
