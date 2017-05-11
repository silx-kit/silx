# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""Functions to prepare events to be sent to Plot callback."""

__author__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "11/05/2017"


import enum
import numpy
from silx.gui import qt
from silx.gui.plot import items


class Type(enum.Enum):
    """Event type used for `PlotEvent` class."""

    LimitChanged = 'limitsChanged'
    """Type of the `LimitsChangedEvent`"""

    MouseMoved = 'mouseMoved'
    """Type of the `MouseEvent` when mouse moving"""

    MouseClicked = 'mouseClicked'
    """Type of the `MouseEvent` when mouse is clicked"""

    MouseDoubleClicked = 'mouseDoubleClicked'
    """Type of the `MouseEvent` when mouse is double clicked"""

    ItemClicked = 'itemClicked'
    """Type of the `ItemClickedEvent` when an item is clicked"""

    ItemHovered = 'itemHovered'
    """Type of the `ItemHoveredEvent` when an item is hovered"""

    RegionChangeStarted = "regionChangeStarted"
    """Type of the `RegionChangeStartedEvent` when the shape of an item start
    changing."""

    RegionChanged = "regionChanged"
    """Type of the `RegionChangeStartedEvent` when the shape of an item
    changed.
    """

    RegionChangeFinished = "regionChangeFinished"
    """Type of the `RegionChangeStartedEvent` when the shape of an item finish
    changing."""


class PlotEvent(object):
    """The PlotEvent provides an abstract event that is generated by the plot
    module."""

    def __init__(self, eventType):
        """Constructor

        :param Type eventType: Type of the event.
        """
        self.__type = eventType

    def getType(self):
        """Returns the type of the event.

        :rtype: Type
        """
        return self.__type


class ItemRegionChangeEvent(PlotEvent):
    """The ItemRegionChangeEvent provides an abstract event that is generated
    when an item shape change."""

    def __init__(self, eventType, item, scenePos=None, screenPos=None):
        """Constructor

        :param silx.gui.plot.items.Item item: The changed item
        :param tuple(int,int) scenePos: Scene position of the mouse
        :param tuple(int,int) screenPos: Screen position (pixels relative to
            widget) of the mouse.
        """
        super(ItemRegionChangeEvent, self).__init__(eventType)
        self.__item = item
        self.__scenePos = scenePos
        self.__screenPos = screenPos

    def getItem(self):
        """
        Returns the item which has changed.

        :rtype: silx.gui.plot.items.Item
        """
        return self.__item

    def getScenePos(self):
        """Returns the current scene position of the mouse (x, y).

        :rtype: tuple(float,float)
        """
        return self.__scenePos

    def getScreenPos(self):
        """Return the current screen position (pixels relative to widget) of
        the mouse.

        :rtype: tuple(int,int)
        """
        return self.__screenPos

    def __getitem__(self, key):
        """Returns event content using the old dictionary-key mapping.

        This is deprecated. Look at the source code to have a description of
        available key names.

        :param str key: Name of the old key.
        :rtype: object
        :raises KeyError: If the requested key is not available
        """
        if isinstance(self.__item, items.DrawItem):
            if key == 'event':
                t = self.getType()
                if t == Type.RegionChangeStarted:
                    # NOTE: It was not part of the compatibility layer
                    # But we have to return something
                    return "drawingStarted"
                elif t == Type.RegionChanged:
                    return "drawingProgress"
                elif t == Type.RegionChangeFinished:
                    return "drawingFinished"
                else:
                    raise RuntimeError("Unexpected type %s" % t.__class__)
            elif key == "type":
                if isinstance(self.getItem(), items.LineDrawItem):
                    return "line"
                elif isinstance(self.getItem(), items.HLineDrawItem):
                    return "hline"
                elif isinstance(self.getItem(), items.VLineDrawItem):
                    return "vline"
                elif isinstance(self.getItem(), items.PolylinesDrawItem):
                    return "polylines"
                elif isinstance(self.getItem(), items.PolygonDrawItem):
                    return "polygon"
                elif isinstance(self.getItem(), items.RectangleDrawItem):
                    return "rectangle"
                else:
                    raise RuntimeError("Unexpected type %s" % self.__item.__class__)
            elif key == "points":
                x = self.getItem().getXData(copy=False)
                y = self.getItem().getYData(copy=False)
                return numpy.array([x, y]).T
            elif key == "xdata":
                return self.getItem().getXData()
            elif key == "ydata":
                return self.getItem().getYData()
            elif key == "parameters":
                return {}

            if isinstance(self.getItem(), items.RectangleDrawItem):
                if key == "x":
                    return self.getItem().getXData(copy=False).min()
                elif key == "y":
                    return self.getItem().getYData(copy=False).min()
                elif key == "width":
                    data = self.getItem().getXData(copy=False)
                    return data.max() - data.min()
                elif key == "height":
                    data = self.getItem().getYData(copy=False)
                    return data.max() - data.min()

        elif isinstance(self.__item, items.marker._BaseMarker):
            if key == 'event':
                t = self.getType()
                if t == Type.RegionChangeStarted:
                    # NOTE: It was not part of the compatibility layer
                    # But we have to return something
                    return "markerStarted"
                elif t == Type.RegionChanged:
                    return "markerMoving"
                elif t == Type.RegionChangeFinished:
                    return "markerMoved"
                else:
                    raise RuntimeError("Unexpected type %s" % t.__class__)
            elif key == "type":
                return "marker"
            elif key == "label":
                return self.getItem().getLegend()
            elif key == "button":
                # NOTE: We hardcode something, that's pointless here
                return "left"
            elif key == "x":
                return self.getScenePos()[0]
            elif key == 'y':
                return self.getScenePos()[1]
            elif key == "xpixel":
                return self.getScreenPos()[0]
            elif key == "ypixel":
                return self.getScreenPos()[1]
            elif key == 'xdata':
                xData = self.__item.getXPosition()
                if xData is None:
                    xData = [0, 1]
                return xData
            elif key == 'ydata':
                yData = self.__item.getYPosition()
                if yData is None:
                    yData = [0, 1]
                return yData
            elif key == 'draggable':
                return self.__item.isDraggable()
            elif key == 'selectable':
                return self.__item.isSelectable()

        else:
            # NOTE it is not part of the compatibility layer
            # But it have to work, then we returns something else
            if key == 'event':
                return "unknownRegionChange"
            elif key == "type":
                return "unknown"

        raise KeyError("Key %s not found" % key)


class ItemRegionChangeStartedEvent(ItemRegionChangeEvent):
    """The ItemRegionChangeStartedEvent provides an event that is generated
    when an item shape starting change.

    The shape itself is not yet changed. This event will be followed by
    `ItemRegionChangedEvent` and `ItemRegionChangeFinishedEvent`.
    """

    def __init__(self, item, scenePos=None, screenPos=None):
        """Constructor

        :param silx.gui.plot.items.Item item: The changed item
        :param tuple(int,int) scenePos: Scene position of the mouse
        :param tuple(int,int) screenPos: Screen position (pixels relative to
            widget) of the mouse.
        """
        super(ItemRegionChangeStartedEvent, self).__init__(
            Type.RegionChangeStarted,
            item, scenePos, screenPos)


class ItemRegionChangedEvent(ItemRegionChangeEvent):
    """The ItemRegionChangedEvent provides an event that is generated
    when an item shape was changed."""

    def __init__(self, item, scenePos=None, screenPos=None):
        """Constructor

        :param silx.gui.plot.items.Item item: The changed item
        :param tuple(int,int) scenePos: Scene position of the mouse
        :param tuple(int,int) screenPos: Screen position (pixels relative to
            widget) of the mouse.
        """
        super(ItemRegionChangedEvent, self).__init__(
            Type.RegionChanged,
            item, scenePos, screenPos)


class ItemRegionChangeFinishedEvent(ItemRegionChangeEvent):
    """The ItemRegionChangeFinishedEvent provides an event that is generated
    when an item shape finished changing."""

    def __init__(self, item, scenePos=None, screenPos=None):
        """Constructor

        :param silx.gui.plot.items.Item item: The changed item
        :param tuple(int,int) scenePos: Scene position of the mouse
        :param tuple(int,int) screenPos: Screen position (pixels relative to
            widget) of the mouse.
        """
        super(ItemRegionChangeFinishedEvent, self).__init__(
            Type.RegionChangeFinished,
            item, scenePos, screenPos)


class MouseEvent(PlotEvent):
    """The MouseEvent provides an event that is generated when the mouse is
    activated of the plot."""

    def __init__(self, eventType, button, scenePos, screenPos):
        """Constructor

        :param qt.Qt.MouseButton button: The clicked button if exists
        :param tuple(int,int) scenePos: Scene position of the mouse
        :param tuple(int,int) screenPos: Screen position (pixels relative to
            widget) of the mouse.
        """
        super(MouseEvent, self).__init__(eventType)
        self._button = button
        self._scenePos = scenePos
        self._screenPos = screenPos

    def getButton(self):
        """
        Returns the activated button in case of a MouseClicked event

        :rtype: qt.Qt.MouseButton
        """
        return self._button

    def getScenePos(self):
        """Returns the current scene position of the mouse (x, y).

        :rtype: tuple(float,float)
        """
        return self._scenePos

    def getScreenPos(self):
        """Return the current screen position (pixels relative to widget) of
        the mouse.

        :rtype: tuple(int,int)
        """
        return self._screenPos

    def __getitem__(self, key):
        """Returns event content using the old dictionary-key mapping.

        This is deprecated. Look at the source code to have a description of
        available key names.

        :param str key: Name of the old key.
        :rtype: object
        :raises KeyError: If the requested key is not available
        """
        if key == 'event':
            events = {
                Type.MouseMoved: 'mouseMoved',
                Type.MouseClicked: 'mouseClicked',
                Type.MouseDoubleClicked: 'mouseDoubleClicked',
            }
            return events[self.getType()]
        elif key == "x":
            return self.getScenePos()[0]
        elif key == "y":
            return self.getScenePos()[1]
        elif key == 'xpixel':
            return self.getScreenPos()[0]
        elif key == 'ypixel':
            return self.getScreenPos()[1]
        elif key == 'button':
            buttons = {
                'left': 'left',
                'right': 'right',
                'middle': 'middle',
                qt.Qt.LeftButton: 'left',
                qt.Qt.RightButton: 'right',
                qt.Qt.MiddleButton: 'middle',
                qt.Qt.NoButton: None,
            }
            return buttons[self.getButton()]
        else:
            raise KeyError("Key %s not found" % key)


class MouseMovedEvent(MouseEvent):
    """The MouseMovedEvent provides an event that is generated when the mouse
    is moved on the plot."""

    def __init__(self, scenePos, screenPos):
        """Constructor

        :param tuple(int,int) scenePos: Scene position of the mouse
        :param tuple(int,int) screenPos: Screen position (pixels relative to
            widget) of the mouse.
        """
        MouseEvent.__init__(self,
                            Type.MouseMoved,
                            qt.Qt.NoButton,
                            scenePos,
                            screenPos)


class MouseClickedEvent(MouseEvent):
    """The MouseClickedEvent provides an event that is generated when the mouse
    is clicked on the plot."""

    def __init__(self, button, scenePos, screenPos):
        """Constructor

        :param qt.Qt.MouseButton button: The clicked button if exists
        :param tuple(int,int) scenePos: Scene position of the mouse
        :param tuple(int,int) screenPos: Screen position (pixels relative to
            widget) of the mouse.
        """
        MouseEvent.__init__(self,
                            Type.MouseClicked,
                            button,
                            scenePos,
                            screenPos)


class MouseDoubleClickedEvent(MouseEvent):
    """The MouseDoubleClickedEvent provides an event that is generated when the
    mouse double clicked on the plot."""

    def __init__(self, button, scenePos, screenPos):
        """Constructor

        :param qt.Qt.MouseButton button: The clicked button if exists
        :param tuple(int,int) scenePos: Scene position of the mouse
        :param tuple(int,int) screenPos: Screen position (pixels relative to
            widget) of the mouse.
        """
        MouseEvent.__init__(self,
                            Type.MouseDoubleClicked,
                            button,
                            scenePos,
                            screenPos)


class ItemHoveredEvent(MouseEvent):
    """The ItemHoveredEvent provides an event when an item is hovered by the
    mouse."""

    def __init__(self, item, scenePos, screenPos):
        MouseEvent.__init__(self, Type.ItemHovered, qt.Qt.NoButton, scenePos, screenPos)
        self.__item = item

    def getItem(self):
        """Returns the hovered item

        :rtype: silx.gui.plot.items.Item
        """
        return self.__item

    def __getitem__(self, key):
        """Returns event content using the old dictionary-key mapping.

        This is deprecated. Look at the source code to have a description of
        available key names.

        :param str key: Name of the old key.
        :rtype: object
        :raises KeyError: If the requested key is not available
        """
        if key == 'event':
            return "hover"
        elif key == 'type':
            if isinstance(self.__item, items.marker._BaseMarker):
                return "marker"
            else:
                # NOTE it is not part of the compatibility layer
                # But it have to work, then we returns something else
                return "unknown"
        elif key == "label":
            return self.__item.getLegend()
        elif key == "x":
            return self.getScenePos()[0]
        elif key == "y":
            return self.getScenePos()[1]
        elif key == 'xpixel':
            return self.getScreenPos()[0]
        elif key == 'ypixel':
            return self.getScreenPos()[1]
        elif key == 'draggable':
            if isinstance(self.__item, items.marker._BaseMarker):
                return self.__item.isDraggable()
            else:
                return False
        elif key == 'selectable':
            if isinstance(self.__item, items.marker._BaseMarker):
                return self.__item.isSelectable()
            else:
                return False
        else:
            raise KeyError("Key %s not found" % key)


class ItemClickedEvent(MouseClickedEvent):
    """The ItemClickedEvent provides an event that is generated when the mouse
    is clicked on an item of the plot. It is a `MouseClickedEvent` which also
    contains information of the clicked item and clicked item index.
    """

    def __init__(self, button, item, itemIndices, scenePos, screenPos):
        MouseEvent.__init__(self, Type.ItemClicked, button, scenePos, screenPos)
        self.__item = item
        self.__itemIndices = itemIndices

    def getItem(self):
        """Returns the clicked item

        :rtype: silx.gui.plot.items.Item
        """
        return self.__item

    def getItemIndices(self):
        """
        Returns the list of indices of item's data under the mouse clicked.

        For an image it is a list of (row, col) of the clicked pixel.
        For a curve it is a list of point indices of the line clicked.
        """
        return self.__itemIndices

    def __getitem__(self, key):
        """Returns event content using the old dictionary-key mapping.

        This is deprecated. Look at the source code to have a description of
        available key names.

        :param str key: Name of the old key.
        :rtype: object
        :raises KeyError: If the requested key is not available
        """

        if isinstance(self.__item, items.ImageBase):
            if key == 'event':
                return "imageClicked"
            elif key == "type":
                return "image"
            elif key == "col":
                return self.__itemIndices[0][1]
            elif key == 'row':
                return self.__itemIndices[0][0]

        elif isinstance(self.__item, items.Curve):
            if key == 'event':
                return "curveClicked"
            elif key == "type":
                return "curve"
            elif key == "xdata":
                index = self.__itemIndices
                return self.__item.getXData(copy=False)[index]
            elif key == 'ydata':
                index = self.__itemIndices
                return self.__item.getYData(copy=False)[index]

        elif isinstance(self.__item, items.marker._BaseMarker):
            if key == 'event':
                return "markerClicked"
            elif key == "type":
                return "marker"
            elif key == "x":
                # NOTE: x is not about the mouse (like other items)
                # but about the marker position
                xData = self.__item.getXPosition()
                if xData is None:
                    xData = [0, 1]
                return xData
            elif key == 'y':
                # NOTE: y is not about the mouse (like other items)
                # but about the marker position
                yData = self.__item.getYPosition()
                if yData is None:
                    yData = [0, 1]
                return yData
            elif key == "xdata":
                xData = self.__item.getXPosition()
                if xData is None:
                    xData = [0, 1]
                return xData
            elif key == 'ydata':
                yData = self.__item.getYPosition()
                if yData is None:
                    yData = [0, 1]
                return yData
            elif key == 'draggable':
                return self.__item.isDraggable()
            elif key == 'selectable':
                return self.__item.isSelectable()

        else:
            # NOTE it is not part of the compatibility layer
            # But it have to work, then we returns something else
            if key == 'event':
                return "unknownMoving"
            elif key == "type":
                return "unknown"

        if key == 'label':
            return self.__item.getLegend()
        elif key == "x":
            return self.getScenePos()[0]
        elif key == "y":
            return self.getScenePos()[1]
        elif key == 'xpixel':
            return self.getScreenPos()[0]
        elif key == 'ypixel':
            return self.getScreenPos()[1]
        elif key == 'button':
            buttons = {
                'left': 'left',
                'right': 'right',
                'middle': 'middle',
                qt.Qt.LeftButton: 'left',
                qt.Qt.RightButton: 'right',
                qt.Qt.MiddleButton: 'middle',
                qt.Qt.NoButton: None,
            }
            return buttons[self.getButton()]

        raise KeyError("Key %s not found" % key)


class LimitsChangedEvent(PlotEvent):
    """The LimitsChangedEvent provides an event that is generated when the
    limits of the plot are changed."""

    def __init__(self, source, xRange, yRange, y2Range):
        """
        Constructor

        :param object source: Source of the event (only used for compatibility)
        :param tuple xRange: Range min,max of the x-axis
        :param tuple yRange: Range min,max of the y-axis
        :param tuple y2Range: Range min,max of the second y-axis
        """
        super(LimitsChangedEvent, self).__init__(Type.LimitChanged)
        self._xRange = xRange
        self._yRange = yRange
        self._y2Range = y2Range
        # stored for compatibility, but not anymore provided
        self._source = source

    def getXRange(self):
        """Returns a tuple min,max of the range of the current x-axis.

        :rtype: tuple(float, float)
        """
        return self._xRange

    def getYRange(self):
        """Returns a tuple min,max of the range of the current y-axis.

        :rtype: tuple(float, float)
        """
        return self._yRange

    def getY2Range(self):
        """Returns a tuple min,max of the range of the current second y-axis.

        :rtype: tuple(float, float)
        """
        return self._y2Range

    @classmethod
    def __getDictionaryMapping(cls):
        """Returns a cached mapping used to provide compatibility with the old
        dictionary events"""
        if hasattr(cls, "__mapping"):
            return cls._mapping
        mapping = {
            'event': lambda self: "limitsChanged",
            'source': lambda self: id(self._source),
            'xdata': lambda self: self.getXRange(),
            'ydata': lambda self: self.getYRange(),
            'y2data': lambda self: self.getY2Range(),
        }
        cls.__mapping = mapping
        return cls.__mapping

    def __getitem__(self, key):
        """Returns event content using the old dictionary-key mapping.

        This is deprecated. Look at the source code to have a description of
        available key names.

        :param str key: Name of the old key.
        :rtype: object
        :raises KeyError: If the requested key is not available
        """
        mapping = self.__getDictionaryMapping()
        return mapping[key](self)


def prepareInteractiveModeChanged(source):
    return {'event': 'interactiveModeChanged', 'source': source}


def prepareContentChanged(action, kind, legend):
    return {'event': 'contentChanged', 'action': action, 'kind': kind, 'legend': legend}


def prepareSetGraphCursor(state):
    return {'event': 'setGraphCursor', 'state': state}


def prepareActiveItemChanged(kind, updated, previous, legend):
    event = 'active' + kind[0].upper() + kind[1:] + 'Changed'
    return {'event': event, 'updated': updated, 'previous': previous, 'legend': legend}


def prepareSetYAxisInverted(state):
    return {'event': 'setYAxisInverted', 'state': state}


def prepareSetXAxisLogarithmic(state):
    return {'event': 'setXAxisLogarithmic', 'state': state}


def prepareSetYAxisLogarithmic(state):
    return {'event': 'setYAxisLogarithmic', 'state': state}


def prepareSetXAxisAutoScale(state):
    return {'event': 'setXAxisAutoScale', 'state': state}


def prepareSetYAxisAutoScale(state):
    return {'event': 'setYAxisAutoScale', 'state': state}


def prepareSetKeepDataAspectRatio(state):
    return {'event': 'setKeepDataAspectRatio', 'state': state}


def prepareSetGraphGrid(which):
    return {'event': 'setGraphGrid', 'which': which}
