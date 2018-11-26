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
"""This module provides the :class:`Curve` item of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import logging
import numpy

from silx.third_party import six
from ....utils.deprecation import deprecated
from ... import colors
from .core import (Points, LabelsMixIn, ColorMixIn, YAxisMixIn,
                   FillMixIn, LineMixIn, SymbolMixIn, ItemChangedType)


_logger = logging.getLogger(__name__)


class CurveStyle(object):
    """Object storing the style of a curve.

    Set a value to None to use the default

    :param color: Color
    :param Union[str,None] linestyle: Style of the line
    :param Union[float,None] linewidth: Width of the line
    :param Union[str,None] symbol: Symbol for markers
    :param Union[float,None] symbolsize: Size of the markers
    """

    def __init__(self, color=None, linestyle=None, linewidth=None,
                 symbol=None, symbolsize=None):
        if color is None:
            self._color = None
        else:
            if isinstance(color, six.string_types):
                color = colors.rgba(color)
            else:  # array-like expected
                color = numpy.array(color, copy=False)
                if color.ndim == 1:  # Array is 1D, this is a single color
                    color = colors.rgba(color)
            self._color = color

        if linestyle is not None:
            assert linestyle in LineMixIn.getSupportedLineStyles()
        self._linestyle = linestyle

        self._linewidth = None if linewidth is None else float(linewidth)

        if symbol is not None:
            assert symbol in SymbolMixIn.getSupportedSymbols()
        self._symbol = symbol

        self._symbolsize = None if symbolsize is None else float(symbolsize)

    def getColor(self, copy=True):
        """Returns the color or None if not set.

        :param bool copy: True to get a copy (default),
            False to get internal representation (do not modify!)

        :rtype: Union[List[float],None]
        """
        if isinstance(self._color, numpy.ndarray):
            return numpy.array(self._color, copy=copy)
        else:
            return self._color

    def getLineStyle(self):
        """Return the type of the line or None if not set.

        Type of line::

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line

        :rtype: Union[str,None]
        """
        return self._linestyle

    def getLineWidth(self):
        """Return the curve line width in pixels or None if not set.

        :rtype: Union[float,None]
        """
        return self._linewidth

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

        :rtype: Union[str,None]
        """
        return self._symbol

    def getSymbolSize(self):
        """Return the point marker size in points.

        :rtype: Union[float,None]
        """
        return self._symbolsize

    def __eq__(self, other):
        if isinstance(other, CurveStyle):
            return (numpy.array_equal(self.getColor(), other.getColor()) and
                    self.getLineStyle() == other.getLineStyle() and
                    self.getLineWidth() == other.getLineWidth() and
                    self.getSymbol() == other.getSymbol() and
                    self.getSymbolSize() == other.getSymbolSize())
        else:
            return False


class Curve(Points, ColorMixIn, YAxisMixIn, FillMixIn, LabelsMixIn, LineMixIn):
    """Description of a curve"""

    _DEFAULT_Z_LAYER = 1
    """Default overlay layer for curves"""

    _DEFAULT_SELECTABLE = True
    """Default selectable state for curves"""

    _DEFAULT_LINEWIDTH = 1.
    """Default line width of the curve"""

    _DEFAULT_LINESTYLE = '-'
    """Default line style of the curve"""

    _DEFAULT_HIGHLIGHT_STYLE = CurveStyle(color='black')
    """Default highlight style of the item"""

    def __init__(self):
        Points.__init__(self)
        ColorMixIn.__init__(self)
        YAxisMixIn.__init__(self)
        FillMixIn.__init__(self)
        LabelsMixIn.__init__(self)
        LineMixIn.__init__(self)

        self._highlightStyle = self._DEFAULT_HIGHLIGHT_STYLE
        self._highlighted = False

        self.sigItemChanged.connect(self.__itemChanged)

    def __itemChanged(self, event):
        if event == ItemChangedType.YAXIS:
            # TODO hackish data range implementation
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        # Filter-out values <= 0
        xFiltered, yFiltered, xerror, yerror = self.getData(
            copy=False, displayed=True)

        if len(xFiltered) == 0 or not numpy.any(numpy.isfinite(xFiltered)):
            return None  # No data to display, do not add renderer to backend

        style = self.getCurrentStyle()

        return backend.addCurve(xFiltered, yFiltered, self.getLegend(),
                                color=style.getColor(),
                                symbol=style.getSymbol(),
                                linestyle=style.getLineStyle(),
                                linewidth=style.getLineWidth(),
                                yaxis=self.getYAxis(),
                                xerror=xerror,
                                yerror=yerror,
                                z=self.getZValue(),
                                selectable=self.isSelectable(),
                                fill=self.isFill(),
                                alpha=self.getAlpha(),
                                symbolsize=style.getSymbolSize())

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
                'z': self.getZValue(),
                'selectable': self.isSelectable(),
                'fill': self.isFill()
            }
            return params
        else:
            raise IndexError("Index out of range: %s", str(item))

    def setVisible(self, visible):
        """Set visibility of item.

        :param bool visible: True to display it, False otherwise
        """
        visible = bool(visible)
        # TODO hackish data range implementation
        if self.isVisible() != visible:
            plot = self.getPlot()
            if plot is not None:
                plot._invalidateDataRange()

        super(Curve, self).setVisible(visible)

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
            assert isinstance(style, CurveStyle)
            self._highlightStyle = style
            self._updated(ItemChangedType.HIGHLIGHTED_STYLE)

            # Backward compatibility event
            if previous.getColor() != style.getColor():
                self._updated(ItemChangedType.HIGHLIGHTED_COLOR)

    @deprecated(replacement='Curve.getHighlightedStyle().getColor()',
                since_version='0.9.0')
    def getHighlightedColor(self):
        """Returns the RGBA highlight color of the item

        :rtype: 4-tuple of float in [0, 1]
        """
        return self.getHighlightedStyle().getColor()

    @deprecated(replacement='Curve.setHighlightedStyle()',
                since_version='0.9.0')
    def setHighlightedColor(self, color):
        """Set the color to use when highlighted

        :param color: color(s) to be used for highlight
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in colors.py
        """
        self.setHighlightedStyle(CurveStyle(color))

    def getCurrentStyle(self):
        """Returns the current curve style.

        Curve style depends on curve highlighting

        :rtype: CurveStyle
        """
        if self.isHighlighted():
            style = self.getHighlightedStyle()
            color = style.getColor()
            linestyle = style.getLineStyle()
            linewidth = style.getLineWidth()
            symbol = style.getSymbol()
            symbolsize = style.getSymbolSize()

            return CurveStyle(
                color=self.getColor() if color is None else color,
                linestyle=self.getLineStyle() if linestyle is None else linestyle,
                linewidth=self.getLineWidth() if linewidth is None else linewidth,
                symbol=self.getSymbol() if symbol is None else symbol,
                symbolsize=self.getSymbolSize() if symbolsize is None else symbolsize)

        else:
             return CurveStyle(color=self.getColor(),
                               linestyle=self.getLineStyle(),
                               linewidth=self.getLineWidth(),
                               symbol=self.getSymbol(),
                               symbolsize=self.getSymbolSize())

    @deprecated(replacement='Curve.getCurrentStyle()',
                since_version='0.9.0')
    def getCurrentColor(self):
        """Returns the current color of the curve.

        This color is either the color of the curve or the highlighted color,
        depending on the highlight state.

        :rtype: 4-tuple of float in [0, 1]
        """
        return self.getCurrentStyle().getColor()
