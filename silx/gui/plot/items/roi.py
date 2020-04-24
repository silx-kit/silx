# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
"""This module provides ROI item for the :class:`~silx.gui.plot.PlotWidget`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/06/2018"


import logging
import collections
import numpy
import weakref

from ....utils.weakref import WeakList
from ... import qt
from ... import utils
from .. import items
from ...colors import rgba
import silx.utils.deprecation


logger = logging.getLogger(__name__)


class _RegionOfInterestBase(qt.QObject):
    """Base class of 1D and 2D region of interest

    :param QObject parent: See QObject
    :param str name: The name of the ROI
    """

    sigAboutToBeRemoved = qt.Signal()
    """Signal emitted just before this ROI is removed from its manager."""

    sigItemChanged = qt.Signal(object)
    """Signal emitted when item has changed.

    It provides a flag describing which property of the item has changed.
    See :class:`ItemChangedType` for flags description.
    """

    def __init__(self, parent=None):
        qt.QObject.__init__(self, parent=parent)
        self.__name = ''

    def getName(self):
        """Returns the name of the ROI

        :return: name of the region of interest
        :rtype: str
        """
        return self.__name

    def setName(self, name):
        """Set the name of the ROI

        :param str name: name of the region of interest
        """
        name = str(name)
        if self.__name != name:
            self.__name = name
            self._updated(items.ItemChangedType.NAME)

    def _updated(self, event=None, checkVisibility=True):
        """Implement Item mix-in update method by updating the plot items

        See :class:`~silx.gui.plot.items.Item._updated`
        """
        self.sigItemChanged.emit(event)


class RegionOfInterest(_RegionOfInterestBase):
    """Object describing a region of interest in a plot.

    :param QObject parent:
        The RegionOfInterestManager that created this object
    """

    _kind = None
    """Label for this kind of ROI.

    Should be set by inherited classes to custom the ROI manager widget.
    """

    sigRegionChanged = qt.Signal()
    """Signal emitted everytime the shape or position of the ROI changes"""

    sigEditingStarted = qt.Signal()
    """Signal emitted when the user start editing the roi"""

    sigEditingFinished = qt.Signal()
    """Signal emitted when the region edition is finished. During edition
    sigEditionChanged will be emitted several times and 
    sigRegionEditionFinished only at end"""

    def __init__(self, parent=None):
        # Avoid circular dependency
        from ..tools import roi as roi_tools
        assert parent is None or isinstance(parent, roi_tools.RegionOfInterestManager)
        _RegionOfInterestBase.__init__(self, parent)
        self._color = rgba('red')
        self._editable = False
        self._selectable = False
        self._focusProxy = None
        self._visible = True
        self._child = WeakList()

    def _connectToPlot(self, plot):
        """Called after connection to a plot"""
        for item in self.iterChild():
            plot.addItem(item)

    def _disconnectFromPlot(self, plot):
        """Called before deconnection from a plot"""
        for item in self.iterChild():
            plot.removeItem(item)

    def _setItemName(self, item):
        """Helper to generate a unique id to a plot item"""
        legend = "__ROI-%d__%d" % (id(self), id(item))
        item.setName(legend)

    def setParent(self, parent):
        """Set the parent of the RegionOfInterest

        :param Union[None,RegionOfInterestManager] parent: The new parent
        """
        # Avoid circular dependency
        from ..tools import roi as roi_tools
        if (parent is not None and not isinstance(parent, roi_tools.RegionOfInterestManager)):
            raise ValueError('Unsupported parent')

        previousParent = self.parent()
        if previousParent is not None:
            previousPlot = previousParent.parent()
            if previousPlot is not None:
                self._disconnectFromPlot(previousPlot)
        super(RegionOfInterest, self).setParent(parent)
        if parent is not None:
            plot = parent.parent()
            if plot is not None:
                self._connectToPlot(plot)

    def addItem(self, item):
        """Add an item to the set of this ROI children.

        This item will be added and removed to the plot used by the ROI.

        If the ROI is already part of a plot, the item will also be added to
        the plot.

        It the item do not have a name already, a unique one is generated to
        avoid item collision in the plot.

        :param silx.gui.plot.items.Item item: A plot item
        """
        assert item is not None
        self._child.append(item)
        if item.getName() == '':
            self._setItemName(item)
        manager = self.parent()
        if manager is not None:
            plot = manager.parent()
            if plot is not None:
                plot.addItem(item)

    def removeItem(self, item):
        """Remove an item from this ROI children.

        If the item is part of a plot it will be removed too.

        :param silx.gui.plot.items.Item item: A plot item
        """
        assert item is not None
        self._child.remove(item)
        plot = item.getPlot()
        if plot is not None:
            plot.removeItem(item)

    def iterChild(self):
        """Iterate through the all ROI child"""
        for i in self._child:
            yield i

    @classmethod
    def _getKind(cls):
        """Return an human readable kind of ROI

        :rtype: str
        """
        return cls._kind

    def getColor(self):
        """Returns the color of this ROI

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._color)

    def setColor(self, color):
        """Set the color used for this ROI.

        :param color: The color to use for ROI shape as
           either a color name, a QColor, a list of uint8 or float in [0, 1].
        """
        color = rgba(color)
        if color != self._color:
            self._color = color
            self._updated(items.ItemChangedType.COLOR)

    @silx.utils.deprecation.deprecated(reason='API modification',
                                       replacement='getName()',
                                       since_version=0.12)
    def getLabel(self):
        """Returns the label displayed for this ROI.

        :rtype: str
        """
        return self.getName()

    @silx.utils.deprecation.deprecated(reason='API modification',
                                       replacement='setName(name)',
                                       since_version=0.12)
    def setLabel(self, label):
        """Set the label displayed with this ROI.

        :param str label: The text label to display
        """
        self.setName(name=label)

    def isEditable(self):
        """Returns whether the ROI is editable by the user or not.

        :rtype: bool
        """
        return self._editable

    def setEditable(self, editable):
        """Set whether the ROI can be changed interactively.

        :param bool editable: True to allow edition by the user,
           False to disable.
        """
        editable = bool(editable)
        if self._editable != editable:
            self._editable = editable
            self._updated(items.ItemChangedType.EDITABLE)

    def isSelectable(self):
        """Returns whether the ROI is selectable by the user or not.

        :rtype: bool
        """
        return self._selectable

    def setSelectable(self, selectable):
        """Set whether the ROI can be selected interactively.

        :param bool selectable: True to allow selection by the user,
           False to disable.
        """
        selectable = bool(selectable)
        if self._selectable != selectable:
            self._selectable = selectable
            self._updated(items.ItemChangedType.SELECTABLE)

    def getFocusProxy(self):
        """Returns the ROI which have to be selected when this ROI is selected,
        else None if no proxy specified.

        :rtype: RegionOfInterest
        """
        proxy = self._focusProxy
        if proxy is None:
            return None
        proxy = proxy()
        if proxy is None:
            self._focusProxy = None
        return proxy

    def setFocusProxy(self, roi):
        """Set the real ROI which will be selected when this ROI is selected,
        else None to remove the proxy already specified.

        :param RegionOfInterest roi: A ROI
        """
        if roi is not None:
            self._focusProxy = weakref.ref(roi)
        else:
            self._focusProxy = None

    def isVisible(self):
        """Returns whether the ROI is visible in the plot.

        .. note::
            This does not take into account whether or not the plot
            widget itself is visible (unlike :meth:`QWidget.isVisible` which
            checks the visibility of all its parent widgets up to the window)

        :rtype: bool
        """
        return self._visible

    def setVisible(self, visible):
        """Set whether the plot items associated with this ROI are
        visible in the plot.

        :param bool visible: True to show the ROI in the plot, False to
            hide it.
        """
        visible = bool(visible)
        if self._visible != visible:
            self._visible = visible
            self._updated(items.ItemChangedType.VISIBLE)

    @classmethod
    def showFirstInteractionShape(cls):
        """Returns True if the shape created by the first interaction and
        managed by the plot have to be visible.

        :rtype: bool
        """
        return True

    @classmethod
    def getFirstInteractionShape(cls):
        """Returns the shape kind which will be used by the very first
        interaction with the plot.

        This interactions are hardcoded inside the plot

        :rtype: str
        """
        return cls._plotShape

    def setFirstShapePoints(self, points):
        """"Initialize the ROI using the points from the first interaction.

        This interaction is constrained by the plot API and only supports few
        shapes.
        """
        raise NotImplementedError()

    def _updateItemProperty(self, event, source, destination):
        """Update the item property of a destination from an item source.

        :param items.ItemChangedType event: Property type to update
        :param silx.gui.plot.items.Item source: The reference for the data
        :param event Union[Item,List[Item]] destination: The item(s) to update
        """
        if not isinstance(destination, (list, tuple)):
            destination = [destination]
        if event == items.ItemChangedType.NAME:
            value = source.getName()
            for d in destination:
                d.setName(value)
        elif event == items.ItemChangedType.EDITABLE:
            value = source.isEditable()
            for d in destination:
                d.setEditable(value)
        elif event == items.ItemChangedType.SELECTABLE:
            value = source.isSelectable()
            for d in destination:
                d._setSelectable(value)
        elif event == items.ItemChangedType.COLOR:
            value = rgba(source.getColor())
            for d in destination:
                d.setColor(value)
        elif event == items.ItemChangedType.LINE_STYLE:
            value = self.getLineStyle()
            for d in destination:
                d.setLineStyle(value)
        elif event == items.ItemChangedType.LINE_WIDTH:
            value = self.getLineWidth()
            for d in destination:
                d.setLineWidth(value)
        elif event == items.ItemChangedType.SYMBOL:
            value = self.getSymbol()
            for d in destination:
                d.setSymbol(value)
        elif event == items.ItemChangedType.SYMBOL_SIZE:
            value = self.getSymbolSize()
            for d in destination:
                d.setSymbolSize(value)
        elif event == items.ItemChangedType.VISIBLE:
            value = self.isVisible()
            for d in destination:
                d.setVisible(value)
        else:
            assert False

    def _editingStarted(self):
        assert self._editable is True
        self.sigEditingStarted.emit()

    def _editingFinished(self):
        self.sigEditingFinished.emit()


class _HandleBasedROI(RegionOfInterest):
    """Manage a ROI based on a set of handles"""

    def __init__(self, parent=None):
        RegionOfInterest.__init__(self, parent=parent)
        self._handles = []
        self._posOrigin = None
        self._posPrevious = None

    def addUserHandle(self, item=None):
        """
        Add a new free handle to the ROI.

        This handle do nothing. It have to be managed by the ROI
        implementing this class.

        :param Union[None,silx.gui.plot.items.Marker] item: The new marker to
            add, else None to create a default marker.
        :rtype: silx.gui.plot.items.Marker
        """
        return self.addHandle(item, role="user")

    def addLabelHandle(self, item=None):
        """
        Add a new label handle to the ROI.

        This handle is not draggable nor selectable.

        It is displayed without symbol, but it is always visible anyway
        the ROI is editable, in order to display text.

        :param Union[None,silx.gui.plot.items.Marker] item: The new marker to
            add, else None to create a default marker.
        :rtype: silx.gui.plot.items.Marker
        """
        return self.addHandle(item, role="label")

    def addTranslateHandle(self, item=None):
        """
        Add a new translate handle to the ROI.

        Dragging translate handles affect the position position of the ROI
        but not the shape itself.

        :param Union[None,silx.gui.plot.items.Marker] item: The new marker to
            add, else None to create a default marker.
        :rtype: silx.gui.plot.items.Marker
        """
        return self.addHandle(item, role="translate")

    def addHandle(self, item=None, role="default"):
        """
        Add a new handle to the ROI.

        Dragging handles while affect the position or the shape of the
        ROI.

        :param Union[None,silx.gui.plot.items.Marker] item: The new marker to
            add, else None to create a default marker.
        :rtype: silx.gui.plot.items.Marker
        """
        if item is None:
            item = items.Marker()
            color = rgba(self.getColor())
            color = self._computeHandleColor(color)
            item.setColor(color)
            if role == "default":
                item.setSymbol("s")
            elif role == "user":
                pass
            elif role == "translate":
                item.setSymbol("+")
            elif role == "label":
                item.setSymbol("")

        if role == "user":
            pass
        elif role == "label":
            item._setSelectable(False)
            item._setDraggable(False)
            item.setVisible(True)
        else:
            self.__updateEditable(item, self.isEditable(), remove=False)
            item._setSelectable(False)

        self._handles.append((item, role))
        self.addItem(item)
        return item

    def removeHandle(self, handle):
        data = [d for d in self._handles if d[0] is handle][0]
        self._handles.remove(data)
        role = data[1]
        if role not in ["user", "label"]:
            if self.isEditable():
                self.__updateEditable(handle, False)
        self.removeItem(handle)

    def iterHandles(self):
        """Iterate though all the handles"""
        for data in self._handles:
            yield data[0]

    def _updated(self, event=None, checkVisibility=True):
        """Implement Item mix-in update method by updating the plot items

        See :class:`~silx.gui.plot.items.Item._updated`
        """
        if event == items.ItemChangedType.NAME:
            self._updateText(self.getName())
        elif event == items.ItemChangedType.COLOR:
            # Update color of shape items in the plot
            color = rgba(self.getColor())
            handleColor = self._computeHandleColor(color)
            for item, role in self._handles:
                if role == 'user':
                    pass
                elif role == 'label':
                    item.setColor(color)
                else:
                    item.setColor(handleColor)
        elif event == items.ItemChangedType.VISIBLE:
            for item, role in self._handles:
                visible = self.isVisible() and self.isEditable()
                item.setVisible(visible)
        elif event == items.ItemChangedType.EDITABLE:
            for item, role in self._handles:
                editable = self.isEditable()
                if role not in ["user", "label"]:
                    self.__updateEditable(item, editable)
        super(_HandleBasedROI, self)._updated(event, checkVisibility)

    def __updateEditable(self, handle, editable, remove=True):
        # NOTE: visibility change emit a position update event
        handle.setVisible(editable and self.isVisible())
        handle._setDraggable(editable)
        if editable:
            handle.sigDragStarted.connect(self.__editingStarted)
            handle.sigItemChanged.connect(self.__editingUpdated)
            handle.sigDragFinished.connect(self.__editingFinished)
        else:
            if remove:
                handle.sigDragStarted.disconnect(self.__editingStarted)
                handle.sigItemChanged.disconnect(self.__editingUpdated)
                handle.sigDragFinished.disconnect(self.__editingFinished)

    def __editingStarted(self):
        super(_HandleBasedROI, self)._editingStarted()
        handle = self.sender()
        self._posOrigin = numpy.array(handle.getPosition())
        self._posPrevious = numpy.array(self._posOrigin)
        self.handleDragStarted(handle, self._posOrigin)

    def __editingUpdated(self):
        if self._posOrigin is None:
            # Avoid to handle events when visibility change
            return
        handle = self.sender()
        current = numpy.array(handle.getPosition())
        self.handleDragUpdated(handle, self._posOrigin, self._posPrevious, current)
        self._posPrevious = current

    def __editingFinished(self):
        handle = self.sender()
        current = numpy.array(handle.getPosition())
        self.handleDragFinished(handle, self._posOrigin, current)
        self._posPrevious = None
        self._posOrigin = None
        super(_HandleBasedROI, self)._editingFinished()

    def handleDragStarted(self, handle, origin):
        """Called when an handler drag started"""
        pass

    def handleDragUpdated(self, handle, origin, previous, current):
        """Called when an handle drag position changed"""
        pass

    def handleDragFinished(self, handle, origin, current):
        """Called when an handle drag finished"""
        pass

    def _computeHandleColor(self, color):
        """Returns the anchor color from the base ROI color

        :param Union[numpy.array,Tuple,List]: color
        :rtype: Union[numpy.array,Tuple,List]
        """
        return color[:3] + (0.5,)

    def _updateText(self, text):
        """Update the text displayed by this ROI

        :param str text: A text
        """
        pass


class PointROI(RegionOfInterest, items.SymbolMixIn):
    """A ROI identifying a point in a 2D plot."""

    _kind = "Point"
    """Label for this kind of ROI"""

    _plotShape = "point"
    """Plot shape which is used for the first interaction"""

    _DEFAULT_SYMBOL = '+'
    """Default symbol of the PointROI

    It overwrite the `SymbolMixIn` class attribte.
    """

    def __init__(self, parent=None):
        items.SymbolMixIn.__init__(self)
        RegionOfInterest.__init__(self, parent=parent)
        self._marker = items.Marker()
        self._marker.setSymbol(self._DEFAULT_SYMBOL)
        self._marker.sigDragStarted.connect(self._editingStarted)
        self._marker.sigDragFinished.connect(self._editingFinished)
        self.addItem(self._marker)
        self.__filterReentrant = utils.LockReentrant()

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def setFirstShapePoints(self, points):
        pos = points[0]
        self._marker.setPosition(pos[0], pos[1])

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.NAME:
            label = self.getName()
            self._marker.setText(label)
        elif event == items.ItemChangedType.EDITABLE:
            editable = self.isEditable()
            if editable:
                self._marker.sigItemChanged.connect(self.__positionChanged)
            else:
                self._marker.sigItemChanged.disconnect(self.__positionChanged)
            self._marker._setDraggable(editable)
        elif event in [items.ItemChangedType.COLOR,
                       items.ItemChangedType.VISIBLE,
                       items.ItemChangedType.SELECTABLE]:
            self._updateItemProperty(event, self, self._marker)
        super(PointROI, self)._updated(event, checkVisibility)

    def getPosition(self):
        """Returns the position of this ROI

        :rtype: numpy.ndarray
        """
        return self._marker.getPosition()

    def setPosition(self, pos):
        """Set the position of this ROI

        :param numpy.ndarray pos: 2d-coordinate of this point
        """
        with self.__filterReentrant:
            self._marker.setPosition(pos[0], pos[1])
        self.sigRegionChanged.emit()

    def __positionChanged(self, event):
        """Handle position changed events of the marker"""
        if self.__filterReentrant.locked():
            return
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self.setPosition(marker.getPosition())

    def __str__(self):
        params = '%f %f' % self.getPosition()
        return "%s(%s)" % (self.__class__.__name__, params)


class LineROI(_HandleBasedROI, items.LineMixIn):
    """A ROI identifying a line in a 2D plot.

    This ROI provides 1 anchor for each boundary of the line, plus an center
    in the center to translate the full ROI.
    """

    _kind = "Line"
    """Label for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        _HandleBasedROI.__init__(self, parent=parent)
        self._handleStart = self.addHandle()
        self._handleEnd = self.addHandle()
        self._handleCenter = self.addTranslateHandle()
        self._handleLabel = self.addLabelHandle()

        shape = items.Shape("polylines")
        shape.setPoints([[0, 0], [0, 0]])
        shape.setColor(rgba(self.getColor()))
        shape.setFill(False)
        shape.setOverlay(True)
        shape.setLineStyle(self.getLineStyle())
        shape.setLineWidth(self.getLineWidth())
        self.__shape = shape
        self.addItem(shape)

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def _updated(self, event=None, checkVisibility=True):
        if event in [items.ItemChangedType.COLOR,
                     items.ItemChangedType.VISIBLE,
                     items.ItemChangedType.LINE_STYLE,
                     items.ItemChangedType.LINE_WIDTH]:
            self._updateItemProperty(event, self, self.__shape)
        super(LineROI, self)._updated(event, checkVisibility)

    def setFirstShapePoints(self, points):
        assert len(points) == 2
        self.setEndPoints(points[0], points[1])

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def setEndPoints(self, startPoint, endPoint):
        """Set this line location using the ending points

        :param numpy.ndarray startPoint: Staring bounding point of the line
        :param numpy.ndarray endPoint: Ending bounding point of the line
        """
        startPoint = numpy.array(startPoint)
        endPoint = numpy.array(endPoint)
        center = (startPoint + endPoint) * 0.5

        with utils.blockSignals(self._handleStart):
            self._handleStart.setPosition(startPoint[0], startPoint[1])
        with utils.blockSignals(self._handleEnd):
            self._handleEnd.setPosition(endPoint[0], endPoint[1])
        with utils.blockSignals(self._handleCenter):
            self._handleCenter.setPosition(center[0], center[1])
        with utils.blockSignals(self._handleLabel):
            self._handleLabel.setPosition(center[0], center[1])

        line = numpy.array((startPoint, endPoint))
        self.__shape.setPoints(line)
        self.sigRegionChanged.emit()

    def getEndPoints(self):
        """Returns bounding points of this ROI.

        :rtype: Tuple(numpy.ndarray,numpy.ndarray)
        """
        startPoint = numpy.array(self._handleStart.getPosition())
        endPoint = numpy.array(self._handleEnd.getPosition())
        return (startPoint, endPoint)

    def handleDragUpdated(self, handle, origin, previous, current):
        if handle is self._handleStart:
            _start, end = self.getEndPoints()
            self.setEndPoints(current, end)
        elif handle is self._handleEnd:
            start, _end = self.getEndPoints()
            self.setEndPoints(start, current)
        elif handle is self._handleCenter:
            start, end = self.getEndPoints()
            delta = current - previous
            start += delta
            end += delta
            self.setEndPoints(start, end)

    def __str__(self):
        start, end = self.getEndPoints()
        params = start[0], start[1], end[0], end[1]
        params = 'start: %f %f; end: %f %f' % params
        return "%s(%s)" % (self.__class__.__name__, params)


class HorizontalLineROI(RegionOfInterest, items.LineMixIn):
    """A ROI identifying an horizontal line in a 2D plot."""

    _kind = "HLine"
    """Label for this kind of ROI"""

    _plotShape = "hline"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        RegionOfInterest.__init__(self, parent=parent)
        self._marker = items.YMarker()
        self._marker.sigDragStarted.connect(self._editingStarted)
        self._marker.sigDragFinished.connect(self._editingFinished)
        self.addItem(self._marker)
        self.__filterReentrant = utils.LockReentrant()

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.NAME:
            label = self.getName()
            self._marker.setText(label)
        elif event == items.ItemChangedType.EDITABLE:
            editable = self.isEditable()
            if editable:
                self._marker.sigItemChanged.connect(self.__positionChanged)
            else:
                self._marker.sigItemChanged.disconnect(self.__positionChanged)
            self._marker._setDraggable(editable)
        elif event in [items.ItemChangedType.COLOR,
                       items.ItemChangedType.LINE_STYLE,
                       items.ItemChangedType.LINE_WIDTH,
                       items.ItemChangedType.VISIBLE,
                       items.ItemChangedType.SELECTABLE]:
            self._updateItemProperty(event, self, self._marker)
        super(HorizontalLineROI, self)._updated(event, checkVisibility)

    def setFirstShapePoints(self, points):
        pos = points[0, 1]
        if pos == self.getPosition():
            return
        self.setPosition(pos)

    def getPosition(self):
        """Returns the position of this line if the horizontal axis

        :rtype: float
        """
        pos = self._marker.getPosition()
        return pos[1]

    def setPosition(self, pos):
        """Set the position of this ROI

        :param float pos: Horizontal position of this line
        """
        with self.__filterReentrant:
            self._marker.setPosition(0, pos)
        self.sigRegionChanged.emit()

    def __positionChanged(self, event):
        """Handle position changed events of the marker"""
        if self.__filterReentrant.locked():
            return
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self.setPosition(marker.getYPosition())

    def __str__(self):
        params = 'y: %f' % self.getPosition()
        return "%s(%s)" % (self.__class__.__name__, params)


class VerticalLineROI(RegionOfInterest, items.LineMixIn):
    """A ROI identifying a vertical line in a 2D plot."""

    _kind = "VLine"
    """Label for this kind of ROI"""

    _plotShape = "vline"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        RegionOfInterest.__init__(self, parent=parent)
        self._marker = items.XMarker()
        self._marker.sigDragStarted.connect(self._editingStarted)
        self._marker.sigDragFinished.connect(self._editingFinished)
        self.addItem(self._marker)
        self.__filterReentrant = utils.LockReentrant()

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.NAME:
            label = self.getName()
            self._marker.setText(label)
        elif event == items.ItemChangedType.EDITABLE:
            editable = self.isEditable()
            if editable:
                self._marker.sigItemChanged.connect(self.__positionChanged)
            else:
                self._marker.sigItemChanged.disconnect(self.__positionChanged)
            self._marker._setDraggable(editable)
        elif event in [items.ItemChangedType.COLOR,
                       items.ItemChangedType.LINE_STYLE,
                       items.ItemChangedType.LINE_WIDTH,
                       items.ItemChangedType.VISIBLE,
                       items.ItemChangedType.SELECTABLE]:
            self._updateItemProperty(event, self, self._marker)
        super(VerticalLineROI, self)._updated(event, checkVisibility)

    def setFirstShapePoints(self, points):
        pos = points[0, 0]
        if pos == self.getPosition():
            return
        self.setPosition(pos)

    def getPosition(self):
        """Returns the position of this line if the horizontal axis

        :rtype: float
        """
        pos = self._marker.getPosition()
        return pos[0]

    def setPosition(self, pos):
        """Set the position of this ROI

        :param float pos: Horizontal position of this line
        """
        with self.__filterReentrant:
            self._marker.setPosition(pos, 0)
        self.sigRegionChanged.emit()

    def __positionChanged(self, event):
        """Handle position changed events of the marker"""
        if self.__filterReentrant.locked():
            return
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self.setPosition(marker.getXPosition())

    def __str__(self):
        params = 'x: %f' % self.getPosition()
        return "%s(%s)" % (self.__class__.__name__, params)


class RectangleROI(_HandleBasedROI, items.LineMixIn):
    """A ROI identifying a rectangle in a 2D plot.

    This ROI provides 1 anchor for each corner, plus an anchor in the
    center to translate the full ROI.
    """

    _kind = "Rectangle"
    """Label for this kind of ROI"""

    _plotShape = "rectangle"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        _HandleBasedROI.__init__(self, parent=parent)
        self._handleTopLeft = self.addHandle()
        self._handleTopRight = self.addHandle()
        self._handleBottomLeft = self.addHandle()
        self._handleBottomRight = self.addHandle()
        self._handleCenter = self.addTranslateHandle()
        self._handleLabel = self.addLabelHandle()

        shape = items.Shape("rectangle")
        shape.setPoints([[0, 0], [0, 0]])
        shape.setColor(rgba(self.getColor()))
        shape.setFill(False)
        shape.setOverlay(True)
        shape.setLineStyle(self.getLineStyle())
        shape.setLineWidth(self.getLineWidth())
        shape.setColor(rgba(self.getColor()))
        self.__shape = shape
        self.addItem(shape)

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def _updated(self, event=None, checkVisibility=True):
        if event in [items.ItemChangedType.COLOR,
                     items.ItemChangedType.VISIBLE,
                     items.ItemChangedType.LINE_STYLE,
                     items.ItemChangedType.LINE_WIDTH]:
            self._updateItemProperty(event, self, self.__shape)
        super(RectangleROI, self)._updated(event, checkVisibility)

    def setFirstShapePoints(self, points):
        assert len(points) == 2
        self._setBound(points)

    def _setBound(self, points):
        """Initialize the rectangle from a bunch of points"""
        top = max(points[:, 1])
        bottom = min(points[:, 1])
        left = min(points[:, 0])
        right = max(points[:, 0])
        size = right - left, top - bottom
        self.setGeometry(origin=(left, bottom), size=size)

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def getCenter(self):
        """Returns the central point of this rectangle

        :rtype: numpy.ndarray([float,float])
        """
        pos = self._handleCenter.getPosition()
        return numpy.array(pos)

    def getOrigin(self):
        """Returns the corner point with the smaller coordinates

        :rtype: numpy.ndarray([float,float])
        """
        pos = self._handleBottomLeft.getPosition()
        return numpy.array(pos)

    def getSize(self):
        """Returns the size of this rectangle

        :rtype: numpy.ndarray([float,float])
        """
        vmin = self._handleBottomLeft.getPosition()
        vmax = self._handleTopRight.getPosition()
        vmin, vmax = numpy.array(vmin), numpy.array(vmax)
        return vmax - vmin

    def setOrigin(self, position):
        """Set the origin position of this ROI

        :param numpy.ndarray position: Location of the smaller corner of the ROI
        """
        size = self.getSize()
        self.setGeometry(origin=position, size=size)

    def setSize(self, size):
        """Set the size of this ROI

        :param numpy.ndarray size: Size of the center of the ROI
        """
        origin = self.getOrigin()
        self.setGeometry(origin=origin, size=size)

    def setCenter(self, position):
        """Set the size of this ROI

        :param numpy.ndarray position: Location of the center of the ROI
        """
        size = self.getSize()
        self.setGeometry(center=position, size=size)

    def setGeometry(self, origin=None, size=None, center=None):
        """Set the geometry of the ROI
        """
        if origin is not None:
            origin = numpy.array(origin)
            size = numpy.array(size)
            points = numpy.array([origin, origin + size])
            center = origin + size * 0.5
        elif center is not None:
            center = numpy.array(center)
            size = numpy.array(size)
            points = numpy.array([center - size * 0.5, center + size * 0.5])
        else:
            raise ValueError("Origin or center expected")

        with utils.blockSignals(self._handleBottomLeft):
            self._handleBottomLeft.setPosition(points[0, 0], points[0, 1])
        with utils.blockSignals(self._handleBottomRight):
            self._handleBottomRight.setPosition(points[1, 0], points[0, 1])
        with utils.blockSignals(self._handleTopLeft):
            self._handleTopLeft.setPosition(points[0, 0], points[1, 1])
        with utils.blockSignals(self._handleTopRight):
            self._handleTopRight.setPosition(points[1, 0], points[1, 1])
        with utils.blockSignals(self._handleCenter):
            self._handleCenter.setPosition(center[0], center[1])
        with utils.blockSignals(self._handleLabel):
            self._handleLabel.setPosition(points[0, 0], points[0, 1])

        self.__shape.setPoints(points)
        self.sigRegionChanged.emit()

    def handleDragUpdated(self, handle, origin, previous, current):
        if handle is self._handleCenter:
            # It is the center anchor
            size = self.getSize()
            self.setGeometry(center=current, size=size)
        else:
            opposed = {
                self._handleBottomLeft: self._handleTopRight,
                self._handleTopRight: self._handleBottomLeft,
                self._handleBottomRight: self._handleTopLeft,
                self._handleTopLeft: self._handleBottomRight,
            }
            handle2 = opposed[handle]
            current2 = handle2.getPosition()
            points = numpy.array([current, current2])
            self._setBound(points)

    def __str__(self):
        origin = self.getOrigin()
        w, h = self.getSize()
        params = origin[0], origin[1], w, h
        params = 'origin: %f %f; width: %f; height: %f' % params
        return "%s(%s)" % (self.__class__.__name__, params)


class PolygonROI(_HandleBasedROI, items.LineMixIn):
    """A ROI identifying a closed polygon in a 2D plot.

    This ROI provides 1 anchor for each point of the polygon.
    """

    _kind = "Polygon"
    """Label for this kind of ROI"""

    _plotShape = "polygon"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        _HandleBasedROI.__init__(self, parent=parent)
        self._handleLabel = self.addLabelHandle()
        self._handleCenter = self.addTranslateHandle()
        self._handlePoints = []
        self._points = numpy.empty((0, 2))

        shape = items.Shape("polygon")
        shape.setPoints([[0, 0], [0, 0]])
        shape.setColor(rgba(self.getColor()))
        shape.setFill(False)
        shape.setOverlay(True)
        shape.setLineStyle(self.getLineStyle())
        shape.setLineWidth(self.getLineWidth())
        shape.setColor(rgba(self.getColor()))
        self.__shape = shape
        self.addItem(shape)

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def _updated(self, event=None, checkVisibility=True):
        if event in [items.ItemChangedType.COLOR,
                     items.ItemChangedType.VISIBLE,
                     items.ItemChangedType.LINE_STYLE,
                     items.ItemChangedType.LINE_WIDTH]:
            self._updateItemProperty(event, self, self.__shape)
        super(PolygonROI, self)._updated(event, checkVisibility)

    def setFirstShapePoints(self, points):
        self.setPoints(points)

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def getPoints(self):
        """Returns the list of the points of this polygon.

        :rtype: numpy.ndarray
        """
        return self._points.copy()

    def setPoints(self, points):
        """Set the position of this ROI

        :param numpy.ndarray pos: 2d-coordinate of this point
        """
        assert(len(points.shape) == 2 and points.shape[1] == 2)

        # Update the needed handles
        while len(self._handlePoints) != len(points):
            if len(self._handlePoints) < len(points):
                handle = self.addHandle()
                self._handlePoints.append(handle)
            else:
                handle = self._handlePoints.pop(-1)
                self.removeHandle(handle)

        for handle, position in zip(self._handlePoints, points):
            with utils.blockSignals(handle):
                handle.setPosition(position[0], position[1])

        if len(points) > 0:
            vmin = numpy.min(points, axis=0)
            vmax = numpy.max(points, axis=0)
            center = (vmax + vmin) * 0.5
            with utils.blockSignals(self._handleCenter):
                self._handleCenter.setPosition(center[0], center[1])

            num = numpy.argmin(points[:, 1])
            pos = points[num]
            with utils.blockSignals(self._handleLabel):
                self._handleLabel.setPosition(pos[0], pos[1])

        if len(points) == 0:
            self._points = numpy.empty((0, 2))
        else:
            self._points = points
        self.__shape.setPoints(self._points)
        self.sigRegionChanged.emit()

    def translate(self, x, y):
        points = self.getPoints()
        delta = numpy.array([x, y])
        self.setPoints(points)
        self.setPoints(points + delta)

    def handleDragUpdated(self, handle, origin, previous, current):
        if handle is self._handleCenter:
            delta = current - previous
            self.translate(delta[0], delta[1])
        else:
            points = self.getPoints()
            num = self._handlePoints.index(handle)
            points[num] = current
            self.setPoints(points)

    def __str__(self):
        points = self._points
        params = '; '.join('%f %f' % (pt[0], pt[1]) for pt in points)
        return "%s(%s)" % (self.__class__.__name__, params)


class ArcROI(_HandleBasedROI, items.LineMixIn):
    """A ROI identifying an arc of a circle with a width.

    This ROI provides
    - 3 handle to control the curvature
    - 1 handle to control the weight
    - 1 anchor to translate the shape.
    """

    _kind = "Arc"
    """Label for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    _ArcGeometry = collections.namedtuple('ArcGeometry', ['center',
                                                          'startPoint', 'endPoint',
                                                          'radius', 'weight',
                                                          'startAngle', 'endAngle'])

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        _HandleBasedROI.__init__(self, parent=parent)
        self._geometry = None
        self._points = None

        self._handleLabel = self.addLabelHandle()
        self._arcHandles = self._createHandles()

        shape = items.Shape("polygon")
        shape.setPoints([[0, 0], [0, 0]])
        shape.setColor(rgba(self.getColor()))
        shape.setFill(False)
        shape.setOverlay(True)
        shape.setLineStyle(self.getLineStyle())
        shape.setLineWidth(self.getLineWidth())
        self.__shape = shape
        self.addItem(shape)

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def _updated(self, event=None, checkVisibility=True):
        if event in [items.ItemChangedType.COLOR,
                     items.ItemChangedType.VISIBLE,
                     items.ItemChangedType.LINE_STYLE,
                     items.ItemChangedType.LINE_WIDTH]:
            self._updateItemProperty(event, self, self.__shape)
        super(ArcROI, self)._updated(event, checkVisibility)

    def setFirstShapePoints(self, points):
        """"Initialize the ROI using the points from the first interaction.

        This interaction is constrained by the plot API and only supports few
        shapes.
        """
        points = self._createControlPointsFromFirstShape(points)
        self._setControlPoints(points)

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def _getControlPoints(self):
        """Returns the current ROI control points.

        It returns an empty tuple if there is currently no ROI.

        :return: Array of (x, y) position in plot coordinates
        :rtype: numpy.ndarray
        """
        return None if self._points is None else numpy.array(self._points)

    def _getInternalGeometry(self):
        """Returns the object storing the internal geometry of this ROI.

        This geometry is derived from the control points and cached for
        efficiency. Calling :meth:`_setControlPoints` invalidate the cache.
        """
        if self._geometry is None:
            controlPoints = self._getControlPoints()
            self._geometry = self._createGeometryFromControlPoint(controlPoints)
        return self._geometry

    def handleDragUpdated(self, handle, origin, previous, current):
        controlPoints = self._getControlPoints()
        currentWeigth = numpy.linalg.norm(controlPoints[3] - controlPoints[1]) * 2

        index = self._arcHandles.index(handle)
        if index in [0, 2]:
            # Moving start or end will maintain the same curvature
            # Then we have to custom the curvature control point
            startPoint = controlPoints[0]
            endPoint = controlPoints[2]
            center = (startPoint + endPoint) * 0.5
            normal = (endPoint - startPoint)
            normal = numpy.array((normal[1], -normal[0]))
            distance = numpy.linalg.norm(normal)
            # Compute the coeficient which have to be constrained
            if distance != 0:
                normal /= distance
                midVector = controlPoints[1] - center
                constainedCoef = numpy.dot(midVector, normal) / distance
            else:
                constainedCoef = 1.0

            # Compute the location of the curvature point
            controlPoints[index] = current
            startPoint = controlPoints[0]
            endPoint = controlPoints[2]
            center = (startPoint + endPoint) * 0.5
            normal = (endPoint - startPoint)
            normal = numpy.array((normal[1], -normal[0]))
            distance = numpy.linalg.norm(normal)
            if distance != 0:
                # BTW we dont need to divide by the distance here
                # Cause we compute normal * distance after all
                normal /= distance
            midPoint = center + normal * constainedCoef * distance
            controlPoints[1] = midPoint

            # The weight have to be fixed
            self._updateWeightControlPoint(controlPoints, currentWeigth)
            self._setControlPoints(controlPoints)

        elif index == 1:
            # The weight have to be fixed
            controlPoints[index] = current
            self._updateWeightControlPoint(controlPoints, currentWeigth)
            self._setControlPoints(controlPoints)

        elif index == 3:
            controlPoints[index] = current
            self._setControlPoints(controlPoints)

    def _updateWeightControlPoint(self, controlPoints, weigth):
        startPoint = controlPoints[0]
        midPoint = controlPoints[1]
        endPoint = controlPoints[2]
        normal = (endPoint - startPoint)
        normal = numpy.array((normal[1], -normal[0]))
        distance = numpy.linalg.norm(normal)
        if distance != 0:
            normal /= distance
        controlPoints[3] = midPoint + normal * weigth * 0.5

    def _createGeometryFromControlPoint(self, controlPoints):
        """Returns the geometry of the object"""
        weigth = numpy.linalg.norm(controlPoints[3] - controlPoints[1]) * 2
        if numpy.allclose(controlPoints[0], controlPoints[2]):
            # Special arc: It's a closed circle
            center = (controlPoints[0] + controlPoints[1]) * 0.5
            radius = numpy.linalg.norm(controlPoints[0] - center)
            v = controlPoints[0] - center
            startAngle = numpy.angle(complex(v[0], v[1]))
            endAngle = startAngle + numpy.pi * 2.0
            return self._ArcGeometry(center, controlPoints[0], controlPoints[2],
                                     radius, weigth, startAngle, endAngle)

        elif numpy.linalg.norm(
            numpy.cross(controlPoints[1] - controlPoints[0],
                        controlPoints[2] - controlPoints[0])) < 1e-5:
            # Degenerated arc, it's a rectangle
            return self._ArcGeometry(None, controlPoints[0], controlPoints[2],
                                     None, weigth, None, None)
        else:
            center, radius = self._circleEquation(*controlPoints[:3])
            v = controlPoints[0] - center
            startAngle = numpy.angle(complex(v[0], v[1]))
            v = controlPoints[1] - center
            midAngle = numpy.angle(complex(v[0], v[1]))
            v = controlPoints[2] - center
            endAngle = numpy.angle(complex(v[0], v[1]))
            # Is it clockwise or anticlockwise
            if (midAngle - startAngle + 2 * numpy.pi) % (2 * numpy.pi) <= numpy.pi:
                if endAngle < startAngle:
                    endAngle += 2 * numpy.pi
            else:
                if endAngle > startAngle:
                    endAngle -= 2 * numpy.pi

            return self._ArcGeometry(center, controlPoints[0], controlPoints[2],
                                     radius, weigth, startAngle, endAngle)

    def _isCircle(self, geometry):
        """Returns True if the geometry is a closed circle"""
        delta = numpy.abs(geometry.endAngle - geometry.startAngle)
        return numpy.isclose(delta, numpy.pi * 2)

    def _getShapeFromControlPoints(self, controlPoints):
        geometry = self._createGeometryFromControlPoint(controlPoints)
        if geometry.center is None:
            # It is not an arc
            # but we can display it as an the intermediat shape
            normal = (geometry.endPoint - geometry.startPoint)
            normal = numpy.array((normal[1], -normal[0]))
            distance = numpy.linalg.norm(normal)
            if distance != 0:
                normal /= distance
            points = numpy.array([
                geometry.startPoint + normal * geometry.weight * 0.5,
                geometry.endPoint + normal * geometry.weight * 0.5,
                geometry.endPoint - normal * geometry.weight * 0.5,
                geometry.startPoint - normal * geometry.weight * 0.5])
        else:
            innerRadius = geometry.radius - geometry.weight * 0.5
            outerRadius = geometry.radius + geometry.weight * 0.5

            if numpy.isnan(geometry.startAngle):
                # Degenerated, it's a point
                # At least 2 points are expected
                return numpy.array([geometry.startPoint, geometry.startPoint])

            delta = 0.1 if geometry.endAngle >= geometry.startAngle else -0.1
            if geometry.startAngle == geometry.endAngle:
                # Degenerated, it's a line (single radius)
                angle = geometry.startAngle
                direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
                points = []
                points.append(geometry.center + direction * innerRadius)
                points.append(geometry.center + direction * outerRadius)
                return numpy.array(points)

            angles = numpy.arange(geometry.startAngle, geometry.endAngle, delta)
            if angles[-1] != geometry.endAngle:
                angles = numpy.append(angles, geometry.endAngle)

            isCircle = self._isCircle(geometry)

            if isCircle:
                if innerRadius <= 0:
                    # It's a circle
                    points = []
                    numpy.append(angles, angles[-1])
                    for angle in angles:
                        direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
                        points.append(geometry.center + direction * outerRadius)
                else:
                    # It's a donut
                    points = []
                    # NOTE: NaN value allow to create 2 separated circle shapes
                    # using a single plot item. It's a kind of cheat
                    points.append(numpy.array([float("nan"), float("nan")]))
                    for angle in angles:
                        direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
                        points.insert(0, geometry.center + direction * innerRadius)
                        points.append(geometry.center + direction * outerRadius)
                    points.append(numpy.array([float("nan"), float("nan")]))
            else:
                if innerRadius <= 0:
                    # It's a part of camembert
                    points = []
                    points.append(geometry.center)
                    points.append(geometry.startPoint)
                    delta = 0.1 if geometry.endAngle >= geometry.startAngle else -0.1
                    for angle in angles:
                        direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
                        points.append(geometry.center + direction * outerRadius)
                    points.append(geometry.endPoint)
                    points.append(geometry.center)
                else:
                    # It's a part of donut
                    points = []
                    points.append(geometry.startPoint)
                    for angle in angles:
                        direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
                        points.insert(0, geometry.center + direction * innerRadius)
                        points.append(geometry.center + direction * outerRadius)
                    points.insert(0, geometry.endPoint)
                    points.append(geometry.endPoint)
            points = numpy.array(points)

        return points

    def _setControlPoints(self, points):
        # Invalidate the geometry
        self._geometry = None
        self._points = points

        for handle, pos in zip(self._arcHandles, points):
            with utils.blockSignals(handle):
                handle.setPosition(pos[0], pos[1])

        points = self._getShapeFromControlPoints(points)
        self.__shape.setPoints(points)

        pos = numpy.min(points, axis=0)
        with utils.blockSignals(self._handleLabel):
            self._handleLabel.setPosition(pos[0], pos[1])

        self.sigRegionChanged.emit()

    def getGeometry(self):
        """Returns a tuple containing the geometry of this ROI

        It is a symmetric function of :meth:`setGeometry`.

        If `startAngle` is smaller than `endAngle` the rotation is clockwise,
        else the rotation is anticlockwise.

        :rtype: Tuple[numpy.ndarray,float,float,float,float]
        :raise ValueError: In case the ROI can't be represented as section of
            a circle
        """
        geometry = self._getInternalGeometry()
        if geometry.center is None:
            raise ValueError("This ROI can't be represented as a section of circle")
        return geometry.center, self.getInnerRadius(), self.getOuterRadius(), geometry.startAngle, geometry.endAngle

    def isClosed(self):
        """Returns true if the arc is a closed shape, like a circle or a donut.

        :rtype: bool
        """
        geometry = self._getInternalGeometry()
        return self._isCircle(geometry)

    def getCenter(self):
        """Returns the center of the circle used to draw arcs of this ROI.

        This center is usually outside the the shape itself.

        :rtype: numpy.ndarray
        """
        geometry = self._getInternalGeometry()
        return geometry.center

    def getStartAngle(self):
        """Returns the angle of the start of the section of this ROI (in radian).

        If `startAngle` is smaller than `endAngle` the rotation is clockwise,
        else the rotation is anticlockwise.

        :rtype: float
        """
        geometry = self._getInternalGeometry()
        return geometry.startAngle

    def getEndAngle(self):
        """Returns the angle of the end of the section of this ROI (in radian).

        If `startAngle` is smaller than `endAngle` the rotation is clockwise,
        else the rotation is anticlockwise.

        :rtype: float
        """
        geometry = self._getInternalGeometry()
        return geometry.endAngle

    def getInnerRadius(self):
        """Returns the radius of the smaller arc used to draw this ROI.

        :rtype: float
        """
        geometry = self._getInternalGeometry()
        radius = geometry.radius - geometry.weight * 0.5
        if radius < 0:
            radius = 0
        return radius

    def getOuterRadius(self):
        """Returns the radius of the bigger arc used to draw this ROI.

        :rtype: float
        """
        geometry = self._getInternalGeometry()
        radius = geometry.radius + geometry.weight * 0.5
        return radius

    def setGeometry(self, center, innerRadius, outerRadius, startAngle, endAngle):
        """
        Set the geometry of this arc.

        :param numpy.ndarray center: Center of the circle.
        :param float innerRadius: Radius of the smaller arc of the section.
        :param float outerRadius: Weight of the bigger arc of the section.
            It have to be bigger than `innerRadius`
        :param float startAngle: Location of the start of the section (in radian)
        :param float endAngle: Location of the end of the section (in radian).
            If `startAngle` is smaller than `endAngle` the rotation is clockwise,
            else the rotation is anticlockwise.
        """
        assert(innerRadius <= outerRadius)
        assert(numpy.abs(startAngle - endAngle) <= 2 * numpy.pi)
        center = numpy.array(center)
        radius = (innerRadius + outerRadius) * 0.5
        weight = outerRadius - innerRadius
        geometry = self._ArcGeometry(center, None, None, radius, weight, startAngle, endAngle)
        controlPoints = self._createControlPointsFromGeometry(geometry)
        self._setControlPoints(controlPoints)

    def _createControlPointsFromGeometry(self, geometry):
        if geometry.startPoint or geometry.endPoint:
            # Duplication with the angles
            raise NotImplementedError("This general case is not implemented")

        angle = geometry.startAngle
        direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
        startPoint = geometry.center + direction * geometry.radius

        angle = geometry.endAngle
        direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
        endPoint = geometry.center + direction * geometry.radius

        angle = (geometry.startAngle + geometry.endAngle) * 0.5
        direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
        curvaturePoint = geometry.center + direction * geometry.radius
        weightPoint = curvaturePoint + direction * geometry.weight * 0.5

        return numpy.array([startPoint, curvaturePoint, endPoint, weightPoint])

    def _createControlPointsFromFirstShape(self, points):
        # The first shape is a line
        point0 = points[0]
        point1 = points[1]

        # Compute a non colineate point for the curvature
        center = (point1 + point0) * 0.5
        normal = point1 - center
        normal = numpy.array((normal[1], -normal[0]))
        defaultCurvature = numpy.pi / 5.0
        defaultWeight = 0.20  # percentage
        curvaturePoint = center - normal * defaultCurvature
        weightPoint = center - normal * defaultCurvature * (1.0 + defaultWeight)

        # 3 corners
        controlPoints = numpy.array([
            point0,
            curvaturePoint,
            point1,
            weightPoint
        ])
        return controlPoints

    def _createHandles(self):
        handles = []
        symbols = ['o', 'o', 'o', 's']

        for index, symbol in enumerate(symbols):
            if index in [1, 3]:
                constraint = self._arcCurvatureMarkerConstraint
            else:
                constraint = None
            handle = self.addHandle()
            handle.setText('')
            handle.setSymbol(symbol)
            if constraint is not None:
                handle._setConstraint(constraint)
            handles.append(handle)

        return handles

    def _arcCurvatureMarkerConstraint(self, x, y):
        """Curvature marker remains on "mediatrice" """
        start = self._points[0]
        end = self._points[2]
        midPoint = (start + end) / 2.
        normal = (end - start)
        normal = numpy.array((normal[1], -normal[0]))
        distance = numpy.linalg.norm(normal)
        if distance != 0:
            normal /= distance
        v = numpy.dot(normal, (numpy.array((x, y)) - midPoint))
        x, y = midPoint + v * normal
        return x, y

    @staticmethod
    def _circleEquation(pt1, pt2, pt3):
        """Circle equation from 3 (x, y) points

        :return: Position of the center of the circle and the radius
        :rtype: Tuple[Tuple[float,float],float]
        """
        x, y, z = complex(*pt1), complex(*pt2), complex(*pt3)
        w = z - x
        w /= y - x
        c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
        return ((-c.real, -c.imag), abs(c + x))

    def __str__(self):
        try:
            center, innerRadius, outerRadius, startAngle, endAngle = self.getGeometry()
            params = center[0], center[1], innerRadius, outerRadius, startAngle, endAngle
            params = 'center: %f %f; radius: %f %f; angles: %f %f' % params
        except ValueError:
            params = "invalid"
        return "%s(%s)" % (self.__class__.__name__, params)


class HorizontalRangeROI(RegionOfInterest, items.LineMixIn):
    """A ROI identifying an horizontal range in a 1D plot."""

    _kind = "HRange"
    """Label for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        RegionOfInterest.__init__(self, parent=parent)
        self._markerMin = items.XMarker()
        self._markerMax = items.XMarker()
        self._markerCen = items.XMarker()
        self._markerCen.setLineStyle(" ")
        self._markerMin._setConstraint(self.__positionMinConstraint)
        self._markerMax._setConstraint(self.__positionMaxConstraint)
        self._markerMin.sigDragStarted.connect(self._editingStarted)
        self._markerMin.sigDragFinished.connect(self._editingFinished)
        self._markerMax.sigDragStarted.connect(self._editingStarted)
        self._markerMax.sigDragFinished.connect(self._editingFinished)
        self._markerCen.sigDragStarted.connect(self._editingStarted)
        self._markerCen.sigDragFinished.connect(self._editingFinished)
        self.addItem(self._markerMin)
        self.addItem(self._markerMax)
        self.addItem(self._markerCen)
        self.__filterReentrant = utils.LockReentrant()

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def setFirstShapePoints(self, points):
        vmin = min(points[:, 0])
        vmax = max(points[:, 0])
        self._updatePos(vmin, vmax)

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.NAME:
            self._updateText()
        elif event == items.ItemChangedType.EDITABLE:
            self._updateEditable()
            self._updateText()
        elif event == items.ItemChangedType.LINE_STYLE:
            markers = [self._markerMin, self._markerMax]
            self._updateItemProperty(event, self, markers)
        elif event in [items.ItemChangedType.COLOR,
                       items.ItemChangedType.LINE_WIDTH,
                       items.ItemChangedType.VISIBLE,
                       items.ItemChangedType.SELECTABLE]:
            markers = [self._markerMin, self._markerMax, self._markerCen]
            self._updateItemProperty(event, self, markers)
        super(HorizontalRangeROI, self)._updated(event, checkVisibility)

    def _updateText(self):
        text = self.getName()
        if self.isEditable():
            self._markerMin.setText("")
            self._markerCen.setText(text)
        else:
            self._markerMin.setText(text)
            self._markerCen.setText("")

    def _updateEditable(self):
        editable = self.isEditable()
        self._markerMin._setDraggable(editable)
        self._markerMax._setDraggable(editable)
        self._markerCen._setDraggable(editable)
        if self.isEditable():
            self._markerMin.sigItemChanged.connect(self.__positionMinChanged)
            self._markerMax.sigItemChanged.connect(self.__positionMaxChanged)
            self._markerCen.sigItemChanged.connect(self.__positionCenChanged)
            self._markerCen.setLineStyle(":")
        else:
            self._markerMin.sigItemChanged.disconnect(self.__positionMinChanged)
            self._markerMax.sigItemChanged.disconnect(self.__positionMaxChanged)
            self._markerCen.sigItemChanged.disconnect(self.__positionCenChanged)
            self._markerCen.setLineStyle(" ")

    def _updatePos(self, vmin, vmax):
        center = (vmin + vmax) * 0.5
        with self.__filterReentrant:
            self._markerMin.setPosition(vmin, 0)
            self._markerCen.setPosition(center, 0)
            self._markerMax.setPosition(vmax, 0)
        self.sigRegionChanged.emit()

    def setRange(self, vmin, vmax):
        """Set the range of this ROI.

        :param float vmin: Staring location of the range
        :param float vmax: Ending location of the range
        """
        if vmin is None or vmax is None:
            err = "Can't set vmin or vmax to None"
            raise ValueError(err)
        if vmin > vmax:
            err = "Can't set vmin and vmax because vmin >= vmax " \
                  "vmin = %s, vmax = %s" % (vmin, vmax)
            raise ValueError(err)
        self._updatePos(vmin, vmax)

    def getRange(self):
        """Returns the range of this ROI.

        :rtype: Tuple[float,float]
        """
        vmin = self.getMin()
        vmax = self.getMax()
        return vmin, vmax

    def setMin(self, vmin):
        """Set the min of this ROI.

        :param float vmin: New min
        """
        vmax = self.getMax()
        self._updatePos(vmin, vmax)

    def getMin(self):
        """Returns the min value of this ROI.

        :rtype: float
        """
        return self._markerMin.getPosition()[0]

    def setMax(self, vmax):
        """Set the max of this ROI.

        :param float vmax: New max
        """
        vmin = self.getMin()
        self._updatePos(vmin, vmax)

    def getMax(self):
        """Returns the max value of this ROI.

        :rtype: float
        """
        return self._markerMax.getPosition()[0]

    def setCenter(self, center):
        """Set the center of this ROI.

        :param float center: New center
        """
        vmin, vmax = self.getRange()
        previousCenter = (vmin + vmax) * 0.5
        delta = center - previousCenter
        self._updatePos(vmin + delta, vmax + delta)

    def getCenter(self):
        """Returns the center location of this ROI.

        :rtype: float
        """
        vmin, vmax = self.getRange()
        return (vmin + vmax) * 0.5

    def __positionMinConstraint(self, x, y):
        """Constraint of the min marker"""
        if self.__filterReentrant.locked():
            # Ignore the constraint when we set an explicit value
            return x, y
        vmax = self.getMax()
        if vmax is None:
            return x, y
        return min(x, vmax), y

    def __positionMaxConstraint(self, x, y):
        """Constraint of the max marker"""
        if self.__filterReentrant.locked():
            # Ignore the constraint when we set an explicit value
            return x, y
        vmin = self.getMin()
        if vmin is None:
            return x, y
        return max(x, vmin), y

    def __positionMinChanged(self, event):
        """Handle position changed events of the marker"""
        if self.__filterReentrant.locked():
            return
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self.setMin(marker.getXPosition())

    def __positionMaxChanged(self, event):
        """Handle position changed events of the marker"""
        if self.__filterReentrant.locked():
            return
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self.setMax(marker.getXPosition())

    def __positionCenChanged(self, event):
        """Handle position changed events of the marker"""
        if self.__filterReentrant.locked():
            return
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self.setCenter(marker.getXPosition())

    def __str__(self):
        vrange = self.getRange()
        params = 'min: %f; max: %f' % vrange
        return "%s(%s)" % (self.__class__.__name__, params)
