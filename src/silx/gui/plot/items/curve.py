# /*##########################################################################
#
# Copyright (c) 2017-2023 European Synchrotron Radiation Facility
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
from __future__ import annotations

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import logging

import numpy

from ....utils.deprecation import deprecated_warning
from ... import colors
from .core import (
    PointsBase,
    LabelsMixIn,
    ColorMixIn,
    YAxisMixIn,
    FillMixIn,
    LineMixIn,
    LineGapColorMixIn,
    LineStyleType,
    SymbolMixIn,
    BaselineMixIn,
    HighlightedMixIn,
    _Style,
)


_logger = logging.getLogger(__name__)


class CurveStyle(_Style):
    """Object storing the style of a curve.

    Set a value to None to use the default

    :param color: Color
    :param linestyle: Style of the line
    :param linewidth: Width of the line
    :param symbol: Symbol for markers
    :param symbolsize: Size of the markers
    :param gapcolor: Color of gaps of dashed line
    """

    def __init__(
        self,
        color: colors.RGBAColorType | None = None,
        linestyle: LineStyleType | None = None,
        linewidth: float | None = None,
        symbol: str | None = None,
        symbolsize: float | None = None,
        gapcolor: colors.RGBAColorType | None = None,
    ):
        if color is None:
            self._color = None
        else:
            if isinstance(color, str):
                color = colors.rgba(color)
            else:  # array-like expected
                color = numpy.array(color, copy=False)
                if color.ndim == 1:  # Array is 1D, this is a single color
                    color = colors.rgba(color)
            self._color = color

        if not LineMixIn.isValidLineStyle(linestyle):
            raise ValueError(f"Not a valid line style: {linestyle}")
        self._linestyle = linestyle

        self._linewidth = None if linewidth is None else float(linewidth)

        if symbol is not None:
            assert symbol in SymbolMixIn.getSupportedSymbols()
        self._symbol = symbol

        self._symbolsize = None if symbolsize is None else float(symbolsize)

        self._gapcolor = None if gapcolor is None else colors.rgba(gapcolor)

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

    def getLineGapColor(self):
        """Returns the color of dashed line gaps or None if not set.

        :rtype: Union[List[float],None]
        """
        return self._gapcolor

    def getLineStyle(self) -> LineStyleType | None:
        """Return the type of the line or None if not set.

        Type of line::

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line
            - (offset, (dash pattern))
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
            return (
                numpy.array_equal(self.getColor(), other.getColor())
                and self.getLineStyle() == other.getLineStyle()
                and self.getLineWidth() == other.getLineWidth()
                and self.getSymbol() == other.getSymbol()
                and self.getSymbolSize() == other.getSymbolSize()
                and self.getLineGapColor() == other.getLineGapColor()
            )
        else:
            return False


class Curve(
    PointsBase,
    ColorMixIn,
    YAxisMixIn,
    FillMixIn,
    LabelsMixIn,
    LineMixIn,
    LineGapColorMixIn,
    BaselineMixIn,
    HighlightedMixIn,
):
    """Description of a curve"""

    _DEFAULT_Z_LAYER = 1
    """Default overlay layer for curves"""

    _DEFAULT_SELECTABLE = True
    """Default selectable state for curves"""

    _DEFAULT_LINEWIDTH = 1.0
    """Default line width of the curve"""

    _DEFAULT_LINESTYLE = "-"
    """Default line style of the curve"""

    _DEFAULT_HIGHLIGHT_STYLE = CurveStyle(color="black")
    """Default highlight style of the item"""

    _DEFAULT_BASELINE = None

    def __init__(self):
        PointsBase.__init__(self)
        ColorMixIn.__init__(self)
        YAxisMixIn.__init__(self)
        FillMixIn.__init__(self)
        LabelsMixIn.__init__(self)
        LineMixIn.__init__(self)
        LineGapColorMixIn.__init__(self)
        BaselineMixIn.__init__(self)
        HighlightedMixIn.__init__(self)

        self._setBaseline(Curve._DEFAULT_BASELINE)

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        # Filter-out values <= 0
        xFiltered, yFiltered, xerror, yerror = self.getData(copy=False, displayed=True)

        if len(xFiltered) == 0 or not numpy.any(numpy.isfinite(xFiltered)):
            return None  # No data to display, do not add renderer to backend

        style = self.getCurrentStyle()

        return backend.addCurve(
            xFiltered,
            yFiltered,
            color=style.getColor(),
            gapcolor=style.getLineGapColor(),
            symbol=style.getSymbol(),
            linestyle=style.getLineStyle(),
            linewidth=style.getLineWidth(),
            yaxis=self.getYAxis(),
            xerror=xerror,
            yerror=yerror,
            fill=self.isFill(),
            alpha=self.getAlpha(),
            symbolsize=style.getSymbolSize(),
            baseline=self.getBaseline(copy=False),
        )

    def __getitem__(self, item):
        """Compatibility with PyMca and silx <= 0.4.0"""
        deprecated_warning(
            "Attributes",
            "__getitem__",
            since_version="2.0.0",
            replacement="Use Curve methods",
        )
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(5))]
        elif item == 0:
            return self.getXData(copy=False)
        elif item == 1:
            return self.getYData(copy=False)
        elif item == 2:
            return self.getName()
        elif item == 3:
            info = self.getInfo(copy=False)
            return {} if info is None else info
        elif item == 4:
            params = {
                "info": self.getInfo(),
                "color": self.getColor(),
                "symbol": self.getSymbol(),
                "linewidth": self.getLineWidth(),
                "linestyle": self.getLineStyle(),
                "xlabel": self.getXLabel(),
                "ylabel": self.getYLabel(),
                "yaxis": self.getYAxis(),
                "xerror": self.getXErrorData(copy=False),
                "yerror": self.getYErrorData(copy=False),
                "z": self.getZValue(),
                "selectable": self.isSelectable(),
                "fill": self.isFill(),
            }
            return params
        else:
            raise IndexError("Index out of range: %s", str(item))

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
            gapcolor = style.getLineGapColor()

            return CurveStyle(
                color=self.getColor() if color is None else color,
                linestyle=self.getLineStyle() if linestyle is None else linestyle,
                linewidth=self.getLineWidth() if linewidth is None else linewidth,
                symbol=self.getSymbol() if symbol is None else symbol,
                symbolsize=self.getSymbolSize() if symbolsize is None else symbolsize,
                gapcolor=self.getLineGapColor() if gapcolor is None else gapcolor,
            )

        else:
            return CurveStyle(
                color=self.getColor(),
                linestyle=self.getLineStyle(),
                linewidth=self.getLineWidth(),
                symbol=self.getSymbol(),
                symbolsize=self.getSymbolSize(),
                gapcolor=self.getLineGapColor(),
            )

    def setData(self, x, y, xerror=None, yerror=None, baseline=None, copy=True):
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
        :param baseline: curve baseline
        :type baseline: Union[None,float,numpy.ndarray]
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        """
        PointsBase.setData(self, x=x, y=y, xerror=xerror, yerror=yerror, copy=copy)
        self._setBaseline(baseline=baseline)
