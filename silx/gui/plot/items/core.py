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
"""This module provides the base class for items of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2017"

from copy import deepcopy
import logging
import weakref

import numpy

from .. import Colors

from ....third_party.six import string_types


_logger = logging.getLogger(__name__)


class Item(object):
    """Description of an item of the plot"""

    _DEFAULT_Z_LAYER = 0
    """Default layer for overlay rendering"""

    _DEFAULT_LEGEND = ''
    """Default legend of items"""

    _DEFAULT_SELECTABLE = False
    """Default selectable state of items"""

    def __init__(self):
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
            self._updated(checkVisibility=False)

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
            self._updated()

    def getInfo(self, copy=True):
        """Returns the info associated to this item

        :param bool copy: True to get a deepcopy, False otherwise.
        """
        return deepcopy(self._info) if copy else self._info

    def setInfo(self, info, copy=True):
        if copy:
            info = deepcopy(info)
        self._info = info

    def _updated(self, checkVisibility=True):
        """Mark the item as dirty (i.e., needing update).

        This also triggers Plot.replot.

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

class LabelsMixIn(object):
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


class DraggableMixIn(object):
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

    def setColormap(self, colormap):
        """Set the colormap of this image

        :param dict colormap: colormap description
        """
        self._colormap = colormap.copy()
        # TODO colormap comparison + colormap object and events on modification
        self._updated()


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

    def setSymbol(self, symbol):
        """Set the marker type

        See :meth:`getSymbol`.

        :param str symbol: Marker type
        """
        assert symbol in ('o', '.', ',', '+', 'x', 'd', 's', '', None)
        if symbol is None:
            symbol = self._DEFAULT_SYMBOL
        if symbol != self._symbol:
            self._symbol = symbol
            self._updated()


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

    def setColor(self, color, copy=True):
        """Set item color

        :param color: color(s) to be used
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in Colors.py
        :param bool copy: True (Default) to get a copy,
                         False to use internal representation (do not modify!)
        """
        if isinstance(color, string_types):
            color = Colors.rgba(color)
        else:
            color = numpy.array(color, copy=copy)
            # TODO more checks + improve color array support
            if color.ndim == 1:  # Single RGBA color
                color = Colors.rgba(color)
            else:  # Array of colors
                assert color.ndim == 2

        self._color = color
        self._updated()


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

    def setYAxis(self, yaxis):
        """Set the Y axis this curve belongs to.

        :param str yaxis: 'left' or 'right'
        """
        yaxis = str(yaxis)
        assert yaxis in ('left', 'right')
        if yaxis != self._yaxis:
            self._yaxis = yaxis
            self._updated()


class FillMixIn(object):
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
            self._updated()
