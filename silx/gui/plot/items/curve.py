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
"""This module provides the :class:`Curve` item of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2017"


import logging
import numpy

from .. import Colors
from .core import (Points, LabelsMixIn, ColorMixIn, YAxisMixIn,
                   FillMixIn, LineMixIn, ItemChangedType)


_logger = logging.getLogger(__name__)


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

    _DEFAULT_HIGHLIGHT_COLOR = (0, 0, 0, 255)
    """Default highlight color of the item"""

    def __init__(self):
        Points.__init__(self)
        ColorMixIn.__init__(self)
        YAxisMixIn.__init__(self)
        FillMixIn.__init__(self)
        LabelsMixIn.__init__(self)
        LineMixIn.__init__(self)

        self._highlightColor = self._DEFAULT_HIGHLIGHT_COLOR
        self._highlighted = False

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
                                linestyle=self.getLineStyle(),
                                linewidth=self.getLineWidth(),
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

    def getHighlightedColor(self):
        """Returns the RGBA highlight color of the item

        :rtype: 4-tuple of int in [0, 255]
        """
        return self._highlightColor

    def setHighlightedColor(self, color):
        """Set the color to use when highlighted

        :param color: color(s) to be used for highlight
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in Colors.py
        """
        color = Colors.rgba(color)
        if color != self._highlightColor:
            self._highlightColor = color
            self._updated(ItemChangedType.HIGHLIGHTED_COLOR)

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
