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
"""This module provides markers item of the :class:`Plot`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2017"


import logging

from .core import (Item, DraggableMixIn, ColorMixIn, LineMixIn, SymbolMixIn,
                   ItemChangedType)


_logger = logging.getLogger(__name__)


class _BaseMarker(Item, DraggableMixIn, ColorMixIn):
    """Base class for markers"""

    _DEFAULT_COLOR = (0., 0., 0., 1.)
    """Default color of the markers"""

    def __init__(self):
        Item.__init__(self)
        DraggableMixIn.__init__(self)
        ColorMixIn.__init__(self)

        self._text = ''
        self._x = None
        self._y = None
        self._constraint = self._defaultConstraint

    def _addRendererCall(self, backend,
                         symbol=None, linestyle='-', linewidth=1):
        """Perform the update of the backend renderer"""
        return backend.addMarker(
            x=self.getXPosition(),
            y=self.getYPosition(),
            legend=self.getLegend(),
            text=self.getText(),
            color=self.getColor(),
            selectable=self.isSelectable(),
            draggable=self.isDraggable(),
            symbol=symbol,
            linestyle=linestyle,
            linewidth=linewidth,
            constraint=self.getConstraint())

    def _addBackendRenderer(self, backend):
        """Update backend renderer"""
        raise NotImplementedError()

    def isOverlay(self):
        """Return true if marker is drawn as an overlay.

        A marker is an overlay if it is draggable.

        :rtype: bool
        """
        return self.isDraggable()

    def getText(self):
        """Returns marker text.

        :rtype: str
        """
        return self._text

    def setText(self, text):
        """Set the text of the marker.

        :param str text: The text to use
        """
        text = str(text)
        if text != self._text:
            self._text = text
            self._updated(ItemChangedType.TEXT)

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

    def setPosition(self, x, y):
        """Set marker position in data coordinates

        Constraint are applied if any.

        :param float x: X coordinates in data frame
        :param float y: Y coordinates in data frame
        """
        x, y = self.getConstraint()(x, y)
        x, y = float(x), float(y)
        if x != self._x or y != self._y:
            self._x, self._y = x, y
            self._updated(ItemChangedType.POSITION)

    def getConstraint(self):
        """Returns the dragging constraint of this item"""
        return self._constraint

    def _setConstraint(self, constraint):  # TODO support update
        """Set the constraint.

        This is private for now as update is not handled.

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
        """Default constraint not doing anything"""
        return args


class Marker(_BaseMarker, SymbolMixIn):
    """Description of a marker"""

    _DEFAULT_SYMBOL = '+'
    """Default symbol of the marker"""

    def __init__(self):
        _BaseMarker.__init__(self)
        SymbolMixIn.__init__(self)

        self._x = 0.
        self._y = 0.

    def _addBackendRenderer(self, backend):
        return self._addRendererCall(backend, symbol=self.getSymbol())

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


class _LineMarker(_BaseMarker, LineMixIn):
    """Base class for line markers"""

    def __init__(self):
        _BaseMarker.__init__(self)
        LineMixIn.__init__(self)

    def _addBackendRenderer(self, backend):
        return self._addRendererCall(backend,
                                     linestyle=self.getLineStyle(),
                                     linewidth=self.getLineWidth())


class XMarker(_LineMarker):
    """Description of a marker"""

    def __init__(self):
        _LineMarker.__init__(self)
        self._x = 0.

    def setPosition(self, x, y):
        """Set marker line position in data coordinates

        Constraint are applied if any.

        :param float x: X coordinates in data frame
        :param float y: Y coordinates in data frame
        """
        x, _ = self.getConstraint()(x, y)
        x = float(x)
        if x != self._x:
            self._x = x
            self._updated(ItemChangedType.POSITION)


class YMarker(_LineMarker):
    """Description of a marker"""

    def __init__(self):
        _LineMarker.__init__(self)
        self._y = 0.

    def setPosition(self, x, y):
        """Set marker line position in data coordinates

        Constraint are applied if any.

        :param float x: X coordinates in data frame
        :param float y: Y coordinates in data frame
        """
        _, y = self.getConstraint()(x, y)
        y = float(y)
        if y != self._y:
            self._y = y
            self._updated(ItemChangedType.POSITION)
