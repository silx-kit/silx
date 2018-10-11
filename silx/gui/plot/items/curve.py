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
                   FillMixIn, LineMixIn, ItemChangedType)


_logger = logging.getLogger(__name__)


class CurveStyle(object):
    """Object storing the style of a curve.

    Set a value to None to use the default

    :param color: Color
    :param Union[str,None] linestyle: Style of the line
    :param Union[float,None] linewidth: Width of the line
    """

    def __init__(self, color=None, linestyle=None, linewidth=None):
        if color is None:
            self._color = None
        else:
            if not isinstance(color, six.string_types):
                color = numpy.array(color, copy=True)
                assert color.ndim == 1 and color.size <= 4
            self._color = colors.rgba(color)

        if linestyle is not None:
            assert linestyle in LineMixIn.getSupportedLineStyles()
        self._linestyle = linestyle

        self._linewidth = None if linewidth is None else float(linewidth)

    def getColor(self):
        """Returns the color or None if not set.

        :rtype: Union[List[float],None]
        """
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

    def __eq__(self, other):
        if isinstance(other, CurveStyle):
            return (self.getColor() == other.getColor() and
                    self.getLineStyle() == other.getLineStyle() and
                    self.getLineWidth() == other.getLineWidth())
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

        return backend.addCurve(xFiltered, yFiltered, self.getLegend(),
                                color=self.getCurrentColor(),
                                symbol=self.getSymbol(),
                                linestyle=self.getCurrentLineStyle(),
                                linewidth=self.getCurrentLineWidth(),
                                yaxis=self.getYAxis(),
                                xerror=xerror,
                                yerror=yerror,
                                z=self.getZValue(),
                                selectable=self.isSelectable(),
                                fill=self.isFill(),
                                alpha=self.getAlpha(),
                                symbolsize=self.getSymbolSize())

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

    def getCurrentColor(self):
        """Returns the current color of the curve.

        This color is either the color of the curve or the highlighted color,
        depending on the highlight state.

        :rtype: 4-tuple of int in [0, 255]
        """
        if self.isHighlighted():
            highlightedColor = self.getHighlightedStyle().getColor()
            if highlightedColor is not None:
                return highlightedColor
        return self.getColor()

    def getCurrentLineStyle(self):
        """Returns the current line style of the curve.

        :rtype: str
        """
        if self.isHighlighted():
            highlightedLineStyle = self.getHighlightedStyle().getLineStyle()
            if highlightedLineStyle is not None:
                return highlightedLineStyle
        return self.getLineStyle()

    def getCurrentLineWidth(self):
        """Returns the current line width of the curve.

        :rtype: float
        """
        if self.isHighlighted():
            highlightedLineWidth = self.getHighlightedStyle().getLineWidth()
            if highlightedLineWidth is not None:
                return highlightedLineWidth
        return self.getLineWidth()
