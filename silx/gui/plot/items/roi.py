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

.. inheritance-diagram::
   silx.gui.plot.items.roi
   :parts: 1
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/06/2018"


import logging
import numpy
import weakref
from silx.image.shapes import Polygon

from ....utils.weakref import WeakList
from ... import qt
from ... import utils
from .. import items
from ..items import core
from ...colors import rgba
import silx.utils.deprecation
from silx.image._boundingbox import _BoundingBox
from ....utils.proxy import docstring
from ..utils.intersections import segments_intersection


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

    def contains(self, position):
        """Returns True if the `position` is in this ROI.

        :param tuple[float,float] position: position to check
        :return: True if the value / point is consider to be in the region of
                 interest.
        :rtype: bool
        """
        raise NotImplementedError("Base class")


class RegionOfInterest(_RegionOfInterestBase, core.HighlightedMixIn):
    """Object describing a region of interest in a plot.

    :param QObject parent:
        The RegionOfInterestManager that created this object
    """

    _DEFAULT_LINEWIDTH = 1.
    """Default line width of the curve"""

    _DEFAULT_LINESTYLE = '-'
    """Default line style of the curve"""

    _DEFAULT_HIGHLIGHT_STYLE = items.CurveStyle(linewidth=2)
    """Default highlight style of the item"""

    ICON, NAME, SHORT_NAME = None, None, None
    """Metadata to describe the ROI in labels, tooltips and widgets

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
        core.HighlightedMixIn.__init__(self)
        self._color = rgba('red')
        self._editable = False
        self._selectable = False
        self._focusProxy = None
        self._visible = True
        self._child = WeakList()

    def _connectToPlot(self, plot):
        """Called after connection to a plot"""
        for item in self.getItems():
            # This hack is needed to avoid reentrant call from _disconnectFromPlot
            # to the ROI manager. It also speed up the item tests in _itemRemoved
            item._roiGroup = True
            plot.addItem(item)

    def _disconnectFromPlot(self, plot):
        """Called before disconnection from a plot"""
        for item in self.getItems():
            # The item could be already be removed by the plot
            if item.getPlot() is not None:
                del item._roiGroup
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
                item._roiGroup = True
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
            del item._roiGroup
            plot.removeItem(item)

    def getItems(self):
        """Returns the list of PlotWidget items of this RegionOfInterest.

        :rtype: List[~silx.gui.plot.items.Item]
        """
        return tuple(self._child)

    @classmethod
    def _getShortName(cls):
        """Return an human readable kind of ROI

        :rtype: str
        """
        if hasattr(cls, "SHORT_NAME"):
            name = cls.SHORT_NAME
        if name is None:
            name = cls.__name__
        return name

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
        return False

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

    def creationStarted(self):
        """"Called when the ROI creation interaction was started.
        """
        pass

    @docstring(_RegionOfInterestBase)
    def contains(self, position):
        raise NotImplementedError("Base class")

    def creationFinalized(self):
        """"Called when the ROI creation interaction was finalized.
        """
        pass

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

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.HIGHLIGHTED:
            style = self.getCurrentStyle()
            self._updatedStyle(event, style)
        else:
            hilighted = self.isHighlighted()
            if hilighted:
                if event == items.ItemChangedType.HIGHLIGHTED_STYLE:
                    style = self.getCurrentStyle()
                    self._updatedStyle(event, style)
            else:
                if event in [items.ItemChangedType.COLOR,
                             items.ItemChangedType.LINE_STYLE,
                             items.ItemChangedType.LINE_WIDTH,
                             items.ItemChangedType.SYMBOL,
                             items.ItemChangedType.SYMBOL_SIZE]:
                    style = self.getCurrentStyle()
                    self._updatedStyle(event, style)
        super(RegionOfInterest, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        """Called when the current displayed style of the ROI was changed.

        :param event: The event responsible of the change of the style
        :param items.CurveStyle style: The current style
        """
        pass

    def getCurrentStyle(self):
        """Returns the current curve style.

        Curve style depends on curve highlighting

        :rtype: CurveStyle
        """
        baseColor = rgba(self.getColor())
        if isinstance(self, core.LineMixIn):
            baseLinestyle = self.getLineStyle()
            baseLinewidth = self.getLineWidth()
        else:
            baseLinestyle = self._DEFAULT_LINESTYLE
            baseLinewidth = self._DEFAULT_LINEWIDTH
        if isinstance(self, core.SymbolMixIn):
            baseSymbol = self.getSymbol()
            baseSymbolsize = self.getSymbolSize()
        else:
            baseSymbol = 'o'
            baseSymbolsize = 1

        if self.isHighlighted():
            style = self.getHighlightedStyle()
            color = style.getColor()
            linestyle = style.getLineStyle()
            linewidth = style.getLineWidth()
            symbol = style.getSymbol()
            symbolsize = style.getSymbolSize()

            return items.CurveStyle(
                color=baseColor if color is None else color,
                linestyle=baseLinestyle if linestyle is None else linestyle,
                linewidth=baseLinewidth if linewidth is None else linewidth,
                symbol=baseSymbol if symbol is None else symbol,
                symbolsize=baseSymbolsize if symbolsize is None else symbolsize)
        else:
            return items.CurveStyle(color=baseColor,
                                    linestyle=baseLinestyle,
                                    linewidth=baseLinewidth,
                                    symbol=baseSymbol,
                                    symbolsize=baseSymbolsize)

    def _editingStarted(self):
        assert self._editable is True
        self.sigEditingStarted.emit()

    def _editingFinished(self):
        self.sigEditingFinished.emit()


class HandleBasedROI(RegionOfInterest):
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

    def getHandles(self):
        """Returns the list of handles of this HandleBasedROI.

        :rtype: List[~silx.gui.plot.items.Marker]
        """
        return tuple(data[0] for data in self._handles)

    def _updated(self, event=None, checkVisibility=True):
        """Implement Item mix-in update method by updating the plot items

        See :class:`~silx.gui.plot.items.Item._updated`
        """
        if event == items.ItemChangedType.NAME:
            self._updateText(self.getName())
        elif event == items.ItemChangedType.VISIBLE:
            for item, role in self._handles:
                visible = self.isVisible()
                editionVisible = visible and self.isEditable()
                if role not in ["user", "label"]:
                    item.setVisible(editionVisible)
                else:
                    item.setVisible(visible)
        elif event == items.ItemChangedType.EDITABLE:
            for item, role in self._handles:
                editable = self.isEditable()
                if role not in ["user", "label"]:
                    self.__updateEditable(item, editable)
        super(HandleBasedROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super(HandleBasedROI, self)._updatedStyle(event, style)

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

    def __updateEditable(self, handle, editable, remove=True):
        # NOTE: visibility change emit a position update event
        handle.setVisible(editable and self.isVisible())
        handle._setDraggable(editable)
        if editable:
            handle.sigDragStarted.connect(self._handleEditingStarted)
            handle.sigItemChanged.connect(self._handleEditingUpdated)
            handle.sigDragFinished.connect(self._handleEditingFinished)
        else:
            if remove:
                handle.sigDragStarted.disconnect(self._handleEditingStarted)
                handle.sigItemChanged.disconnect(self._handleEditingUpdated)
                handle.sigDragFinished.disconnect(self._handleEditingFinished)

    def _handleEditingStarted(self):
        super(HandleBasedROI, self)._editingStarted()
        handle = self.sender()
        self._posOrigin = numpy.array(handle.getPosition())
        self._posPrevious = numpy.array(self._posOrigin)
        self.handleDragStarted(handle, self._posOrigin)

    def _handleEditingUpdated(self):
        if self._posOrigin is None:
            # Avoid to handle events when visibility change
            return
        handle = self.sender()
        current = numpy.array(handle.getPosition())
        self.handleDragUpdated(handle, self._posOrigin, self._posPrevious, current)
        self._posPrevious = current

    def _handleEditingFinished(self):
        handle = self.sender()
        current = numpy.array(handle.getPosition())
        self.handleDragFinished(handle, self._posOrigin, current)
        self._posPrevious = None
        self._posOrigin = None
        super(HandleBasedROI, self)._editingFinished()

    def isHandleBeingDragged(self):
        """Returns True if one of the handles is currently being dragged.

        :rtype: bool
        """
        return self._posOrigin is not None

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

    ICON = 'add-shape-point'
    NAME = 'point markers'
    SHORT_NAME = "point"
    """Metadata for this kind of ROI"""

    _plotShape = "point"
    """Plot shape which is used for the first interaction"""

    _DEFAULT_SYMBOL = '+'
    """Default symbol of the PointROI

    It overwrite the `SymbolMixIn` class attribte.
    """

    def __init__(self, parent=None):
        RegionOfInterest.__init__(self, parent=parent)
        items.SymbolMixIn.__init__(self)
        self._marker = items.Marker()
        self._marker.sigItemChanged.connect(self._pointPositionChanged)
        self._marker.setSymbol(self._DEFAULT_SYMBOL)
        self._marker.sigDragStarted.connect(self._editingStarted)
        self._marker.sigDragFinished.connect(self._editingFinished)
        self.addItem(self._marker)

    def setFirstShapePoints(self, points):
        self.setPosition(points[0])

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.NAME:
            label = self.getName()
            self._marker.setText(label)
        elif event == items.ItemChangedType.EDITABLE:
            self._marker._setDraggable(self.isEditable())
        elif event in [items.ItemChangedType.VISIBLE,
                       items.ItemChangedType.SELECTABLE]:
            self._updateItemProperty(event, self, self._marker)
        super(PointROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        self._marker.setColor(style.getColor())

    def getPosition(self):
        """Returns the position of this ROI

        :rtype: numpy.ndarray
        """
        return self._marker.getPosition()

    def setPosition(self, pos):
        """Set the position of this ROI

        :param numpy.ndarray pos: 2d-coordinate of this point
        """
        self._marker.setPosition(*pos)

    @docstring(_RegionOfInterestBase)
    def contains(self, position):
        raise NotImplementedError('Base class')

    def _pointPositionChanged(self, event):
        """Handle position changed events of the marker"""
        if event is items.ItemChangedType.POSITION:
            self.sigRegionChanged.emit()

    def __str__(self):
        params = '%f %f' % self.getPosition()
        return "%s(%s)" % (self.__class__.__name__, params)


class CrossROI(HandleBasedROI, items.LineMixIn):
    """A ROI identifying a point in a 2D plot and displayed as a cross
    """

    ICON = 'add-shape-cross'
    NAME = 'cross marker'
    SHORT_NAME = "cross"
    """Metadata for this kind of ROI"""

    _plotShape = "point"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        HandleBasedROI.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        self._handle = self.addHandle()
        self._handle.sigItemChanged.connect(self._handlePositionChanged)
        self._handleLabel = self.addLabelHandle()
        self._vmarker = self.addUserHandle(items.YMarker())
        self._vmarker._setSelectable(False)
        self._vmarker._setDraggable(False)
        self._vmarker.setPosition(*self.getPosition())
        self._hmarker = self.addUserHandle(items.XMarker())
        self._hmarker._setSelectable(False)
        self._hmarker._setDraggable(False)
        self._hmarker.setPosition(*self.getPosition())

    def _updated(self, event=None, checkVisibility=True):
        if event in [items.ItemChangedType.VISIBLE]:
            markers = (self._vmarker, self._hmarker)
            self._updateItemProperty(event, self, markers)
        super(CrossROI, self)._updated(event, checkVisibility)

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def _updatedStyle(self, event, style):
        super(CrossROI, self)._updatedStyle(event, style)
        for marker in [self._vmarker, self._hmarker]:
            marker.setColor(style.getColor())
            marker.setLineStyle(style.getLineStyle())
            marker.setLineWidth(style.getLineWidth())

    def setFirstShapePoints(self, points):
        pos = points[0]
        self.setPosition(pos)

    def getPosition(self):
        """Returns the position of this ROI

        :rtype: numpy.ndarray
        """
        return self._handle.getPosition()

    def setPosition(self, pos):
        """Set the position of this ROI

        :param numpy.ndarray pos: 2d-coordinate of this point
        """
        self._handle.setPosition(*pos)

    def _handlePositionChanged(self, event):
        """Handle center marker position updates"""
        if event is items.ItemChangedType.POSITION:
            position = self.getPosition()
            self._handleLabel.setPosition(*position)
            self._vmarker.setPosition(*position)
            self._hmarker.setPosition(*position)
            self.sigRegionChanged.emit()

    @docstring(HandleBasedROI)
    def contains(self, position):
        roiPos = self.getPosition()
        return position[0] == roiPos[0] or position[1] == roiPos[1]


class LineROI(HandleBasedROI, items.LineMixIn):
    """A ROI identifying a line in a 2D plot.

    This ROI provides 1 anchor for each boundary of the line, plus an center
    in the center to translate the full ROI.
    """

    ICON = 'add-shape-diagonal'
    NAME = 'line ROI'
    SHORT_NAME = "line"
    """Metadata for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        HandleBasedROI.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
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

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.VISIBLE:
            self._updateItemProperty(event, self, self.__shape)
        super(LineROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super(LineROI, self)._updatedStyle(event, style)
        self.__shape.setColor(style.getColor())
        self.__shape.setLineStyle(style.getLineStyle())
        self.__shape.setLineWidth(style.getLineWidth())

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
        if not numpy.array_equal((startPoint, endPoint), self.getEndPoints()):
            self.__updateEndPoints(startPoint, endPoint)

    def __updateEndPoints(self, startPoint, endPoint):
        """Update marker and shape to match given end points

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
            self.__updateEndPoints(current, end)
        elif handle is self._handleEnd:
            start, _end = self.getEndPoints()
            self.__updateEndPoints(start, current)
        elif handle is self._handleCenter:
            start, end = self.getEndPoints()
            delta = current - previous
            start += delta
            end += delta
            self.setEndPoints(start, end)

    @docstring(_RegionOfInterestBase)
    def contains(self, position):
        bottom_left = position[0], position[1]
        bottom_right = position[0] + 1, position[1]
        top_left = position[0], position[1] + 1
        top_right = position[0] + 1, position[1] + 1

        line_pt1 = self._points[0]
        line_pt2 = self._points[1]

        bb1 = _BoundingBox.from_points(self._points)
        if bb1.contains(position) is False:
            return False

        return (
            segments_intersection(seg1_start_pt=line_pt1, seg1_end_pt=line_pt2,
                                  seg2_start_pt=bottom_left, seg2_end_pt=bottom_right) or
            segments_intersection(seg1_start_pt=line_pt1, seg1_end_pt=line_pt2,
                                  seg2_start_pt=bottom_right, seg2_end_pt=top_right) or
            segments_intersection(seg1_start_pt=line_pt1, seg1_end_pt=line_pt2,
                                  seg2_start_pt=top_right, seg2_end_pt=top_left) or
            segments_intersection(seg1_start_pt=line_pt1, seg1_end_pt=line_pt2,
                                  seg2_start_pt=top_left, seg2_end_pt=bottom_left)
        )

    def __str__(self):
        start, end = self.getEndPoints()
        params = start[0], start[1], end[0], end[1]
        params = 'start: %f %f; end: %f %f' % params
        return "%s(%s)" % (self.__class__.__name__, params)


class HorizontalLineROI(RegionOfInterest, items.LineMixIn):
    """A ROI identifying an horizontal line in a 2D plot."""

    ICON = 'add-shape-horizontal'
    NAME = 'horizontal line ROI'
    SHORT_NAME = "hline"
    """Metadata for this kind of ROI"""

    _plotShape = "hline"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        RegionOfInterest.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        self._marker = items.YMarker()
        self._marker.sigItemChanged.connect(self._linePositionChanged)
        self._marker.sigDragStarted.connect(self._editingStarted)
        self._marker.sigDragFinished.connect(self._editingFinished)
        self.addItem(self._marker)

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.NAME:
            label = self.getName()
            self._marker.setText(label)
        elif event == items.ItemChangedType.EDITABLE:
            self._marker._setDraggable(self.isEditable())
        elif event in [items.ItemChangedType.VISIBLE,
                       items.ItemChangedType.SELECTABLE]:
            self._updateItemProperty(event, self, self._marker)
        super(HorizontalLineROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        self._marker.setColor(style.getColor())
        self._marker.setLineStyle(style.getLineStyle())
        self._marker.setLineWidth(style.getLineWidth())

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
        self._marker.setPosition(0, pos)

    @docstring(_RegionOfInterestBase)
    def contains(self, position):
        return position[1] == self.getPosition()[1]

    def _linePositionChanged(self, event):
        """Handle position changed events of the marker"""
        if event is items.ItemChangedType.POSITION:
            self.sigRegionChanged.emit()

    def __str__(self):
        params = 'y: %f' % self.getPosition()
        return "%s(%s)" % (self.__class__.__name__, params)


class VerticalLineROI(RegionOfInterest, items.LineMixIn):
    """A ROI identifying a vertical line in a 2D plot."""

    ICON = 'add-shape-vertical'
    NAME = 'vertical line ROI'
    SHORT_NAME = "vline"
    """Metadata for this kind of ROI"""

    _plotShape = "vline"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        RegionOfInterest.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        self._marker = items.XMarker()
        self._marker.sigItemChanged.connect(self._linePositionChanged)
        self._marker.sigDragStarted.connect(self._editingStarted)
        self._marker.sigDragFinished.connect(self._editingFinished)
        self.addItem(self._marker)

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.NAME:
            label = self.getName()
            self._marker.setText(label)
        elif event == items.ItemChangedType.EDITABLE:
            self._marker._setDraggable(self.isEditable())
        elif event in [items.ItemChangedType.VISIBLE,
                       items.ItemChangedType.SELECTABLE]:
            self._updateItemProperty(event, self, self._marker)
        super(VerticalLineROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        self._marker.setColor(style.getColor())
        self._marker.setLineStyle(style.getLineStyle())
        self._marker.setLineWidth(style.getLineWidth())

    def setFirstShapePoints(self, points):
        pos = points[0, 0]
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
        self._marker.setPosition(pos, 0)

    @docstring(RegionOfInterest)
    def contains(self, position):
        return position[0] == self.getPosition()[0]

    def _linePositionChanged(self, event):
        """Handle position changed events of the marker"""
        if event is items.ItemChangedType.POSITION:
            self.sigRegionChanged.emit()

    def __str__(self):
        params = 'x: %f' % self.getPosition()
        return "%s(%s)" % (self.__class__.__name__, params)


class RectangleROI(HandleBasedROI, items.LineMixIn):
    """A ROI identifying a rectangle in a 2D plot.

    This ROI provides 1 anchor for each corner, plus an anchor in the
    center to translate the full ROI.
    """

    ICON = 'add-shape-rectangle'
    NAME = 'rectangle ROI'
    SHORT_NAME = "rectangle"
    """Metadata for this kind of ROI"""

    _plotShape = "rectangle"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        HandleBasedROI.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        self._handleTopLeft = self.addHandle()
        self._handleTopRight = self.addHandle()
        self._handleBottomLeft = self.addHandle()
        self._handleBottomRight = self.addHandle()
        self._handleCenter = self.addTranslateHandle()
        self._handleLabel = self.addLabelHandle()

        shape = items.Shape("rectangle")
        shape.setPoints([[0, 0], [0, 0]])
        shape.setFill(False)
        shape.setOverlay(True)
        shape.setLineStyle(self.getLineStyle())
        shape.setLineWidth(self.getLineWidth())
        shape.setColor(rgba(self.getColor()))
        self.__shape = shape
        self.addItem(shape)

    def _updated(self, event=None, checkVisibility=True):
        if event in [items.ItemChangedType.VISIBLE]:
            self._updateItemProperty(event, self, self.__shape)
        super(RectangleROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super(RectangleROI, self)._updatedStyle(event, style)
        self.__shape.setColor(style.getColor())
        self.__shape.setLineStyle(style.getLineStyle())
        self.__shape.setLineWidth(style.getLineWidth())

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
        self._updateGeometry(origin=(left, bottom), size=size)

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
        if ((origin is None or numpy.array_equal(origin, self.getOrigin())) and
                (center is None or numpy.array_equal(center, self.getCenter())) and
                numpy.array_equal(size, self.getSize())):
            return  # Nothing has changed

        self._updateGeometry(origin, size, center)

    def _updateGeometry(self, origin=None, size=None, center=None):
        """Forced update of the geometry of the ROI"""
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

    @docstring(HandleBasedROI)
    def contains(self, position):
        assert isinstance(position, (tuple, list, numpy.array))
        points = self.__shape.getPoints()
        bb1 = _BoundingBox.from_points(points)
        return bb1.contains(position)

    def handleDragUpdated(self, handle, origin, previous, current):
        if handle is self._handleCenter:
            # It is the center anchor
            size = self.getSize()
            self._updateGeometry(center=current, size=size)
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


class CircleROI(HandleBasedROI, items.LineMixIn):
    """A ROI identifying a circle in a 2D plot.

    This ROI provides 1 anchor at the center to translate the circle,
    and one anchor on the perimeter to change the radius.
    """

    ICON = 'add-shape-circle'
    NAME = 'circle ROI'
    SHORT_NAME = "circle"
    """Metadata for this kind of ROI"""

    _kind = "Circle"
    """Label for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        HandleBasedROI.__init__(self, parent=parent)
        self._handlePerimeter = self.addHandle()
        self._handleCenter = self.addTranslateHandle()
        self._handleCenter.sigItemChanged.connect(self._centerPositionChanged)
        self._handleLabel = self.addLabelHandle()

        shape = items.Shape("polygon")
        shape.setPoints([[0, 0], [0, 0]])
        shape.setColor(rgba(self.getColor()))
        shape.setFill(False)
        shape.setOverlay(True)
        shape.setLineStyle(self.getLineStyle())
        shape.setLineWidth(self.getLineWidth())
        self.__shape = shape
        self.addItem(shape)

        self.__radius = 0

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.VISIBLE:
            self._updateItemProperty(event, self, self.__shape)
        super(CircleROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super(CircleROI, self)._updatedStyle(event, style)
        self.__shape.setColor(style.getColor())
        self.__shape.setLineStyle(style.getLineStyle())
        self.__shape.setLineWidth(style.getLineWidth())

    def setFirstShapePoints(self, points):
        assert len(points) == 2
        self._setRay(points)

    def _setRay(self, points):
        """Initialize the circle from the center point and a
        perimeter point."""
        center = points[0]
        radius = numpy.linalg.norm(points[0] - points[1])
        self.setGeometry(center=center, radius=radius)

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def getCenter(self):
        """Returns the central point of this rectangle

        :rtype: numpy.ndarray([float,float])
        """
        pos = self._handleCenter.getPosition()
        return numpy.array(pos)

    def getRadius(self):
        """Returns the radius of this circle

        :rtype: float
        """
        return self.__radius

    def setCenter(self, position):
        """Set the center point of this ROI

        :param numpy.ndarray position: Location of the center of the circle
        """
        self._handleCenter.setPosition(*position)

    def setRadius(self, radius):
        """Set the size of this ROI

        :param float size: Radius of the circle
        """
        radius = float(radius)
        if radius != self.__radius:
            self.__radius = radius
            self._updateGeometry()

    def setGeometry(self, center, radius):
        """Set the geometry of the ROI
        """
        if numpy.array_equal(center, self.getCenter()):
            self.setRadius(radius)
        else:
            self.__radius = float(radius)  # Update radius directly
            self.setCenter(center)  # Calls _updateGeometry

    def _updateGeometry(self):
        """Update the handles and shape according to given parameters"""
        center = self.getCenter()
        perimeter_point = numpy.array([center[0] + self.__radius, center[1]])

        self._handlePerimeter.setPosition(perimeter_point[0], perimeter_point[1])
        self._handleLabel.setPosition(center[0], center[1])

        nbpoints = 27
        angles = numpy.arange(nbpoints) * 2.0 * numpy.pi / nbpoints
        circleShape = numpy.array((numpy.cos(angles) * self.__radius,
                                   numpy.sin(angles) * self.__radius)).T
        circleShape += center
        self.__shape.setPoints(circleShape)
        self.sigRegionChanged.emit()

    def _centerPositionChanged(self, event):
        """Handle position changed events of the center marker"""
        if event is items.ItemChangedType.POSITION:
            self._updateGeometry()

    def handleDragUpdated(self, handle, origin, previous, current):
        if handle is self._handlePerimeter:
            center = self.getCenter()
            self.setRadius(numpy.linalg.norm(center - current))

    def __str__(self):
        center = self.getCenter()
        radius = self.getRadius()
        params = center[0], center[1], radius
        params = 'center: %f %f; radius: %f;' % params
        return "%s(%s)" % (self.__class__.__name__, params)


class EllipseROI(HandleBasedROI, items.LineMixIn):
    """A ROI identifying an oriented ellipse in a 2D plot.

    This ROI provides 1 anchor at the center to translate the circle,
    and two anchors on the perimeter to modify the major-radius and
    minor-radius. These two anchors also allow to change the orientation.
    """

    ICON = 'add-shape-ellipse'
    NAME = 'ellipse ROI'
    SHORT_NAME = "ellipse"
    """Metadata for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        items.LineMixIn.__init__(self)
        HandleBasedROI.__init__(self, parent=parent)
        self._handleAxis0 = self.addHandle()
        self._handleAxis1 = self.addHandle()
        self._handleCenter = self.addTranslateHandle()
        self._handleCenter.sigItemChanged.connect(self._centerPositionChanged)
        self._handleLabel = self.addLabelHandle()

        shape = items.Shape("polygon")
        shape.setPoints([[0, 0], [0, 0]])
        shape.setColor(rgba(self.getColor()))
        shape.setFill(False)
        shape.setOverlay(True)
        shape.setLineStyle(self.getLineStyle())
        shape.setLineWidth(self.getLineWidth())
        self.__shape = shape
        self.addItem(shape)

        self._radius = 0., 0.
        self._orientation = 0.  # angle in radians between the X-axis and the _handleAxis0

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.VISIBLE:
            self._updateItemProperty(event, self, self.__shape)
        super(EllipseROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super(EllipseROI, self)._updatedStyle(event, style)
        self.__shape.setColor(style.getColor())
        self.__shape.setLineStyle(style.getLineStyle())
        self.__shape.setLineWidth(style.getLineWidth())

    def setFirstShapePoints(self, points):
        assert len(points) == 2
        self._setRay(points)

    @staticmethod
    def _calculateOrientation(p0, p1):
        """return angle in radians between the vector p0-p1
        and the X axis

        :param p0: first point coordinates (x, y)
        :param p1:  second point coordinates
        :return:
        """
        vector = (p1[0] - p0[0], p1[1] - p0[1])
        x_unit_vector = (1, 0)
        norm = numpy.linalg.norm(vector)
        if norm != 0:
            theta = numpy.arccos(numpy.dot(vector, x_unit_vector) / norm)
        else:
            theta = 0
        if vector[1] < 0:
            # arccos always returns values in range [0, pi]
            theta = 2 * numpy.pi - theta
        return theta

    def _setRay(self, points):
        """Initialize the circle from the center point and a
        perimeter point."""
        center = points[0]
        radius = numpy.linalg.norm(points[0] - points[1])
        orientation = self._calculateOrientation(points[0], points[1])
        self.setGeometry(center=center,
                         radius=(radius, radius),
                         orientation=orientation)

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def getCenter(self):
        """Returns the central point of this rectangle

        :rtype: numpy.ndarray([float,float])
        """
        pos = self._handleCenter.getPosition()
        return numpy.array(pos)

    def getMajorRadius(self):
        """Returns the half-diameter of the major axis.

        :rtype: float
        """
        return max(self._radius)

    def getMinorRadius(self):
        """Returns the half-diameter of the minor axis.

        :rtype: float
        """
        return min(self._radius)

    def getOrientation(self):
        """Return angle in radians between the horizontal (X) axis
        and the major axis of the ellipse in [0, 2*pi[

        :rtype: float:
        """
        return self._orientation

    def setCenter(self, center):
        """Set the center point of this ROI

        :param numpy.ndarray position: Coordinates (X, Y) of the center
            of the ellipse
        """
        self._handleCenter.setPosition(*center)

    def setMajorRadius(self, radius):
        """Set the half-diameter of the major axis of the ellipse.

        :param float radius:
            Major radius of the ellipsis. Must be a positive value.
        """
        if self._radius[0] > self._radius[1]:
            newRadius = radius, self._radius[1]
        else:
            newRadius = self._radius[0], radius
        self.setGeometry(radius=newRadius)

    def setMinorRadius(self, radius):
        """Set the half-diameter of the minor axis of the ellipse.

        :param float radius:
            Minor radius of the ellipsis. Must be a positive value.
        """
        if self._radius[0] > self._radius[1]:
            newRadius = self._radius[0], radius
        else:
            newRadius = radius, self._radius[1]
        self.setGeometry(radius=newRadius)

    def setOrientation(self, orientation):
        """Rotate the ellipse

        :param float orientation: Angle in radians between the horizontal and
            the major axis.
        :return:
        """
        self.setGeometry(orientation=orientation)

    def setGeometry(self, center=None, radius=None, orientation=None):
        """

        :param center: (X, Y) coordinates
        :param float majorRadius:
        :param float minorRadius:
        :param float orientation: angle in radians between the major axis and the
            horizontal
        :return:
        """
        if center is None:
            center = self.getCenter()

        if radius is None:
            radius = self._radius
        else:
            radius = float(radius[0]), float(radius[1])

        if orientation is None:
            orientation = self._orientation
        else:
            # ensure that we store the orientation in range [0, 2*pi
            orientation = numpy.mod(orientation, 2 * numpy.pi)

        if (numpy.array_equal(center, self.getCenter()) or
                radius != self._radius or
                orientation != self._orientation):

            # Update parameters directly
            self._radius = radius
            self._orientation = orientation

            if numpy.array_equal(center, self.getCenter()):
                self._updateGeometry()
            else:
                # This will call _updateGeometry
                self.setCenter(center)

    def _updateGeometry(self):
        """Update shape and markers"""
        center = self.getCenter()

        orientation = self.getOrientation()
        if self._radius[1] > self._radius[0]:
            # _handleAxis1 is the major axis
            orientation -= numpy.pi/2

        point0 = numpy.array([center[0] + self._radius[0] * numpy.cos(orientation),
                              center[1] + self._radius[0] * numpy.sin(orientation)])
        point1 = numpy.array([center[0] - self._radius[1] * numpy.sin(orientation),
                              center[1] + self._radius[1] * numpy.cos(orientation)])
        with utils.blockSignals(self._handleAxis0):
            self._handleAxis0.setPosition(*point0)
        with utils.blockSignals(self._handleAxis1):
            self._handleAxis1.setPosition(*point1)
        with utils.blockSignals(self._handleLabel):
            self._handleLabel.setPosition(*center)

        nbpoints = 27
        angles = numpy.arange(nbpoints) * 2.0 * numpy.pi / nbpoints
        X = (self._radius[0] * numpy.cos(angles) * numpy.cos(orientation)
             - self._radius[1] * numpy.sin(angles) * numpy.sin(orientation))
        Y = (self._radius[0] * numpy.cos(angles) * numpy.sin(orientation)
             + self._radius[1] * numpy.sin(angles) * numpy.cos(orientation))

        ellipseShape = numpy.array((X, Y)).T
        ellipseShape += center
        self.__shape.setPoints(ellipseShape)
        self.sigRegionChanged.emit()

    def handleDragUpdated(self, handle, origin, previous, current):
        if handle in (self._handleAxis0, self._handleAxis1):
            center = self.getCenter()
            orientation = self._calculateOrientation(center, current)
            distance = numpy.linalg.norm(center - current)

            if handle is self._handleAxis1:
                if self._radius[0] > distance:
                    # _handleAxis1 is not the major axis, rotate -90 degrees
                    orientation -= numpy.pi/2
                radius = self._radius[0], distance

            else:  # _handleAxis0
                if self._radius[1] > distance:
                    # _handleAxis0 is not the major axis, rotate +90 degrees
                    orientation += numpy.pi/2
                radius = distance, self._radius[1]

            self.setGeometry(radius=radius, orientation=orientation)

    def _centerPositionChanged(self, event):
        """Handle position changed events of the center marker"""
        if event is items.ItemChangedType.POSITION:
            self._updateGeometry()

    def __str__(self):
        center = self.getCenter()
        major = self.getMajorRadius()
        minor = self.getMinorRadius()
        orientation = self.getOrientation()
        params = center[0], center[1], major, minor, orientation
        params = 'center: %f %f; major radius: %f: minor radius: %f; orientation: %f' % params
        return "%s(%s)" % (self.__class__.__name__, params)


class PolygonROI(HandleBasedROI, items.LineMixIn):
    """A ROI identifying a closed polygon in a 2D plot.

    This ROI provides 1 anchor for each point of the polygon.
    """

    ICON = 'add-shape-polygon'
    NAME = 'polygon ROI'
    SHORT_NAME = "polygon"
    """Metadata for this kind of ROI"""

    _plotShape = "polygon"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        HandleBasedROI.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        self._handleLabel = self.addLabelHandle()
        self._handleCenter = self.addTranslateHandle()
        self._handlePoints = []
        self._points = numpy.empty((0, 2))
        self._handleClose = None

        self._polygon_shape = None
        shape = self.__createShape()
        self.__shape = shape
        self.addItem(shape)

    def _updated(self, event=None, checkVisibility=True):
        if event in [items.ItemChangedType.VISIBLE]:
            self._updateItemProperty(event, self, self.__shape)
        super(PolygonROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super(PolygonROI, self)._updatedStyle(event, style)
        self.__shape.setColor(style.getColor())
        self.__shape.setLineStyle(style.getLineStyle())
        self.__shape.setLineWidth(style.getLineWidth())
        if self._handleClose is not None:
            color = self._computeHandleColor(style.getColor())
            self._handleClose.setColor(color)

    def __createShape(self, interaction=False):
        kind = "polygon" if not interaction else "polylines"
        shape = items.Shape(kind)
        shape.setPoints([[0, 0], [0, 0]])
        shape.setFill(False)
        shape.setOverlay(True)
        style = self.getCurrentStyle()
        shape.setLineStyle(style.getLineStyle())
        shape.setLineWidth(style.getLineWidth())
        shape.setColor(rgba(style.getColor()))
        return shape

    def setFirstShapePoints(self, points):
        if self._handleClose is not None:
            self._handleClose.setPosition(*points[0])
        self.setPoints(points)

    def creationStarted(self):
        """"Called when the ROI creation interaction was started.
        """
        # Handle to see where to close the polygon
        self._handleClose = self.addUserHandle()
        self._handleClose.setSymbol("o")
        color = self._computeHandleColor(rgba(self.getColor()))
        self._handleClose.setColor(color)

        # Hide the center while creating the first shape
        self._handleCenter.setSymbol("")

        # In interaction replace the polygon by a line, to display something unclosed
        self.removeItem(self.__shape)
        self.__shape = self.__createShape(interaction=True)
        self.__shape.setPoints(self._points)
        self.addItem(self.__shape)

    def isBeingCreated(self):
        """Returns true if the ROI is in creation step"""
        return self._handleClose is not None

    def creationFinalized(self):
        """"Called when the ROI creation interaction was finalized.
        """
        self.removeHandle(self._handleClose)
        self._handleClose = None
        self.removeItem(self.__shape)
        self.__shape = self.__createShape()
        self.__shape.setPoints(self._points)
        self.addItem(self.__shape)
        # Hide the center while creating the first shape
        self._handleCenter.setSymbol("+")
        for handle in self._handlePoints:
            handle.setSymbol("s")

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

        if numpy.array_equal(points, self._points):
            return  # Nothing has changed

        self._polygon_shape = None

        # Update the needed handles
        while len(self._handlePoints) != len(points):
            if len(self._handlePoints) < len(points):
                handle = self.addHandle()
                self._handlePoints.append(handle)
                if self.isBeingCreated():
                    handle.setSymbol("")
            else:
                handle = self._handlePoints.pop(-1)
                self.removeHandle(handle)

        for handle, position in zip(self._handlePoints, points):
            with utils.blockSignals(handle):
                handle.setPosition(position[0], position[1])

        if len(points) > 0:
            if not self.isHandleBeingDragged():
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

    def handleDragFinished(self, handle, origin, current):
        points = self._points
        if len(points) > 0:
            # Only update the center at the end
            # To avoid to disturb the interaction
            vmin = numpy.min(points, axis=0)
            vmax = numpy.max(points, axis=0)
            center = (vmax + vmin) * 0.5
            with utils.blockSignals(self._handleCenter):
                self._handleCenter.setPosition(center[0], center[1])

    def __str__(self):
        points = self._points
        params = '; '.join('%f %f' % (pt[0], pt[1]) for pt in points)
        return "%s(%s)" % (self.__class__.__name__, params)

    @docstring(HandleBasedROI)
    def contains(self, position):
        bb1 = _BoundingBox.from_points(self.getPoints())
        if bb1.contains(position) is False:
            return False

        if self._polygon_shape is None:
            self._polygon_shape = Polygon(vertices=self.getPoints())

        # warning: both the polygon and the value are inverted
        return self._polygon_shape.is_inside(row=position[0], col=position[1])

    def _setControlPoints(self, points):
        RegionOfInterest._setControlPoints(self, points=points)
        self._polygon_shape = None


class ArcROI(HandleBasedROI, items.LineMixIn):
    """A ROI identifying an arc of a circle with a width.

    This ROI provides
    - 3 handle to control the curvature
    - 1 handle to control the weight
    - 1 anchor to translate the shape.
    """

    ICON = 'add-shape-arc'
    NAME = 'arc ROI'
    SHORT_NAME = "arc"
    """Metadata for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    class _Geometry:
        def __init__(self):
            self.center = None
            self.startPoint = None
            self.endPoint = None
            self.radius = None
            self.weight = None
            self.startAngle = None
            self.endAngle = None
            self._closed = None

        @classmethod
        def createEmpty(cls):
            zero = numpy.array([0, 0])
            return cls.create(zero, zero.copy(), zero.copy(), 0, 0, 0, 0)

        @classmethod
        def createRect(cls, startPoint, endPoint, weight):
            return cls.create(None, startPoint, endPoint, None, weight, None, None, False)

        @classmethod
        def createCircle(cls, center, startPoint, endPoint, radius,
                   weight, startAngle, endAngle):
            return cls.create(center, startPoint, endPoint, radius,
                              weight, startAngle, endAngle, True)

        @classmethod
        def create(cls, center, startPoint, endPoint, radius,
                   weight, startAngle, endAngle, closed=False):
            g = cls()
            g.center = center
            g.startPoint = startPoint
            g.endPoint = endPoint
            g.radius = radius
            g.weight = weight
            g.startAngle = startAngle
            g.endAngle = endAngle
            g._closed = closed
            return g

        def withWeight(self, weight):
            """Create a new geometry with another weight
            """
            return self.create(self.center, self.startPoint, self.endPoint,
                               self.radius, weight,
                               self.startAngle, self.endAngle, self._closed)

        def withRadius(self, radius):
            """Create a new geometry with another radius.

            The weight and the center is conserved.
            """
            startPoint = self.center + (self.startPoint - self.center) / self.radius * radius
            endPoint = self.center + (self.endPoint - self.center) / self.radius * radius
            return self.create(self.center, startPoint, endPoint,
                               radius, self.weight,
                               self.startAngle, self.endAngle, self._closed)

        def translated(self, x, y):
            delta = numpy.array([x, y])
            center = None if self.center is None else self.center + delta
            startPoint = None if self.startPoint is None else self.startPoint + delta
            endPoint = None if self.endPoint is None else self.endPoint + delta
            return self.create(center, startPoint, endPoint,
                               self.radius, self.weight,
                               self.startAngle, self.endAngle, self._closed)

        def getKind(self):
            """Returns the kind of shape defined"""
            if self.center is None:
                return "rect"
            elif numpy.isnan(self.startAngle):
                return "point"
            elif self.isClosed():
                if self.weight <= 0 or self.weight * 0.5 >= self.radius:
                    return "circle"
                else:
                    return "donut"
            else:
                if self.weight * 0.5 < self.radius:
                    return "arc"
                else:
                    return "camembert"

        def isClosed(self):
            """Returns True if the geometry is a circle like"""
            if self._closed is not None:
                return self._closed
            delta = numpy.abs(self.endAngle - self.startAngle)
            self._closed = numpy.isclose(delta, numpy.pi * 2)
            return self._closed

        def __str__(self):
            return str((self.center,
                        self.startPoint,
                        self.endPoint,
                        self.radius,
                        self.weight,
                        self.startAngle,
                        self.endAngle,
                        self._closed))

    def __init__(self, parent=None):
        HandleBasedROI.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        self._geometry  = self._Geometry.createEmpty()
        self._handleLabel = self.addLabelHandle()

        self._handleStart = self.addHandle()
        self._handleStart.setSymbol("o")
        self._handleMid = self.addHandle()
        self._handleMid.setSymbol("o")
        self._handleEnd = self.addHandle()
        self._handleEnd.setSymbol("o")
        self._handleWeight = self.addHandle()
        self._handleWeight._setConstraint(self._arcCurvatureMarkerConstraint)
        self._handleMove = self.addTranslateHandle()

        shape = items.Shape("polygon")
        shape.setPoints([[0, 0], [0, 0]])
        shape.setColor(rgba(self.getColor()))
        shape.setFill(False)
        shape.setOverlay(True)
        shape.setLineStyle(self.getLineStyle())
        shape.setLineWidth(self.getLineWidth())
        self.__shape = shape
        self.addItem(shape)

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.VISIBLE:
            self._updateItemProperty(event, self, self.__shape)
        super(ArcROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super(ArcROI, self)._updatedStyle(event, style)
        self.__shape.setColor(style.getColor())
        self.__shape.setLineStyle(style.getLineStyle())
        self.__shape.setLineWidth(style.getLineWidth())

    def setFirstShapePoints(self, points):
        """"Initialize the ROI using the points from the first interaction.

        This interaction is constrained by the plot API and only supports few
        shapes.
        """
        # The first shape is a line
        point0 = points[0]
        point1 = points[1]

        # Compute a non collinear point for the curvature
        center = (point1 + point0) * 0.5
        normal = point1 - center
        normal = numpy.array((normal[1], -normal[0]))
        defaultCurvature = numpy.pi / 5.0
        weightCoef = 0.20
        mid = center - normal * defaultCurvature
        distance = numpy.linalg.norm(point0 - point1)
        weight =  distance * weightCoef

        geometry = self._createGeometryFromControlPoints(point0, mid, point1, weight)
        self._geometry = geometry
        self._updateHandles()

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def _updateMidHandle(self):
        """Keep the same geometry, but update the location of the control
        points.

        So calling this function do not trigger sigRegionChanged.
        """
        geometry = self._geometry

        if geometry.isClosed():
            start = numpy.array(self._handleStart.getPosition())
            geometry.endPoint = start
            with utils.blockSignals(self._handleEnd):
                self._handleEnd.setPosition(*start)
            midPos = geometry.center + geometry.center - start
        else:
            if geometry.center is None:
                midPos = geometry.startPoint * 0.66 + geometry.endPoint * 0.34
            else:
                midAngle = geometry.startAngle * 0.66 + geometry.endAngle * 0.34
                vector = numpy.array([numpy.cos(midAngle), numpy.sin(midAngle)])
                midPos = geometry.center + geometry.radius * vector

        with utils.blockSignals(self._handleMid):
            self._handleMid.setPosition(*midPos)

    def _updateWeightHandle(self):
        geometry = self._geometry
        if geometry.center is None:
            # rectangle
            center = (geometry.startPoint + geometry.endPoint) * 0.5
            normal = geometry.endPoint - geometry.startPoint
            normal = numpy.array((normal[1], -normal[0]))
            distance = numpy.linalg.norm(normal)
            if distance != 0:
                normal = normal / distance
            weightPos = center + normal * geometry.weight * 0.5
        else:
            if geometry.isClosed():
                midAngle = geometry.startAngle + numpy.pi * 0.5
            elif geometry.center is not None:
                midAngle = (geometry.startAngle + geometry.endAngle) * 0.5
            vector = numpy.array([numpy.cos(midAngle), numpy.sin(midAngle)])
            weightPos = geometry.center + (geometry.radius + geometry.weight * 0.5) * vector

        with utils.blockSignals(self._handleWeight):
            self._handleWeight.setPosition(*weightPos)

    def _getWeightFromHandle(self, weightPos):
        geometry = self._geometry
        if geometry.center is None:
            # rectangle
            center = (geometry.startPoint + geometry.endPoint) * 0.5
            return numpy.linalg.norm(center - weightPos) * 2
        else:
            distance = numpy.linalg.norm(geometry.center - weightPos)
            return abs(distance - geometry.radius) * 2

    def _updateHandles(self):
        geometry = self._geometry
        with utils.blockSignals(self._handleStart):
            self._handleStart.setPosition(*geometry.startPoint)
        with utils.blockSignals(self._handleEnd):
            self._handleEnd.setPosition(*geometry.endPoint)

        self._updateMidHandle()
        self._updateWeightHandle()

        self._updateShape()

    def _updateCurvature(self, start, mid, end, updateCurveHandles, checkClosed=False):
        """Update the curvature using 3 control points in the curve

        :param bool updateCurveHandles: If False curve handles are already at
            the right location
        """
        if updateCurveHandles:
            with utils.blockSignals(self._handleStart):
                self._handleStart.setPosition(*start)
            with utils.blockSignals(self._handleMid):
                self._handleMid.setPosition(*mid)
            with utils.blockSignals(self._handleEnd):
                self._handleEnd.setPosition(*end)

        if checkClosed:
            closed = self._isCloseInPixel(start, end)
        else:
            closed = self._geometry.isClosed()

        weight = self._geometry.weight
        geometry = self._createGeometryFromControlPoints(start, mid, end, weight, closed=closed)
        self._geometry = geometry

        self._updateWeightHandle()
        self._updateShape()

    def handleDragUpdated(self, handle, origin, previous, current):
        if handle is self._handleStart:
            mid = numpy.array(self._handleMid.getPosition())
            end = numpy.array(self._handleEnd.getPosition())
            self._updateCurvature(current, mid, end,
                                  checkClosed=True, updateCurveHandles=False)
        elif handle is self._handleMid:
            if self._geometry.isClosed():
                radius = numpy.linalg.norm(self._geometry.center - current)
                self._geometry = self._geometry.withRadius(radius)
                self._updateHandles()
            else:
                start = numpy.array(self._handleStart.getPosition())
                end = numpy.array(self._handleEnd.getPosition())
                self._updateCurvature(start, current, end, updateCurveHandles=False)
        elif handle is self._handleEnd:
            start = numpy.array(self._handleStart.getPosition())
            mid = numpy.array(self._handleMid.getPosition())
            self._updateCurvature(start, mid, current,
                                  checkClosed=True, updateCurveHandles=False)
        elif handle is self._handleWeight:
            weight = self._getWeightFromHandle(current)
            self._geometry = self._geometry.withWeight(weight)
            self._updateShape()
        elif handle is self._handleMove:
            delta = current - previous
            self.translate(*delta)

    def _isCloseInPixel(self, point1, point2):
        manager = self.parent()
        if manager is None:
            return False
        plot = manager.parent()
        if plot is None:
            return False
        point1 = plot.dataToPixel(*point1)
        if point1 is None:
            return False
        point2 = plot.dataToPixel(*point2)
        if point2 is None:
            return False
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]) < 15

    def _normalizeGeometry(self):
        """Keep the same phisical geometry, but with normalized parameters.
        """
        geometry = self._geometry
        if geometry.weight * 0.5 >= geometry.radius:
            radius = (geometry.weight * 0.5 + geometry.radius) * 0.5
            geometry = geometry.withRadius(radius)
            geometry = geometry.withWeight(radius * 2)
            self._geometry = geometry
            return True
        return False

    def handleDragFinished(self, handle, origin, current):
        if handle in [self._handleStart, self._handleMid, self._handleEnd]:
            if self._normalizeGeometry():
                self._updateHandles()
            else:
                self._updateMidHandle()
        if self._geometry.isClosed():
            self._handleStart.setSymbol("x")
            self._handleEnd.setSymbol("x")
        else:
            self._handleStart.setSymbol("o")
            self._handleEnd.setSymbol("o")

    def _createGeometryFromControlPoints(self, start, mid, end, weight, closed=None):
        """Returns the geometry of the object"""
        if closed or (closed is None and numpy.allclose(start, end)):
            # Special arc: It's a closed circle
            center = (start + mid) * 0.5
            radius = numpy.linalg.norm(start - center)
            v = start - center
            startAngle = numpy.angle(complex(v[0], v[1]))
            endAngle = startAngle + numpy.pi * 2.0
            return self._Geometry.createCircle(center, start, end, radius,
                                               weight, startAngle, endAngle)

        elif numpy.linalg.norm(numpy.cross(mid - start, end - start)) < 1e-5:
            # Degenerated arc, it's a rectangle
            return self._Geometry.createRect(start, end, weight)
        else:
            center, radius = self._circleEquation(start, mid, end)
            v = start - center
            startAngle = numpy.angle(complex(v[0], v[1]))
            v = mid - center
            midAngle = numpy.angle(complex(v[0], v[1]))
            v = end - center
            endAngle = numpy.angle(complex(v[0], v[1]))

            # Is it clockwise or anticlockwise
            relativeMid = (endAngle - midAngle + 2 * numpy.pi) % (2 * numpy.pi)
            relativeEnd = (endAngle - startAngle + 2 * numpy.pi) % (2 * numpy.pi)
            if relativeMid < relativeEnd:
                if endAngle < startAngle:
                    endAngle += 2 * numpy.pi
            else:
                if endAngle > startAngle:
                    endAngle -= 2 * numpy.pi

            return self._Geometry.create(center, start, end,
                                         radius, weight, startAngle, endAngle)

    def _createShapeFromGeometry(self, geometry):
        kind = geometry.getKind()
        if kind == "rect":
            # It is not an arc
            # but we can display it as an intermediate shape
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
        elif kind == "point":
            # It is not an arc
            # but we can display it as an intermediate shape
            # NOTE: At least 2 points are expected
            points = numpy.array([geometry.startPoint, geometry.startPoint])
        elif kind == "circle":
            outerRadius = geometry.radius + geometry.weight * 0.5
            angles = numpy.arange(0, 2 * numpy.pi, 0.1)
            # It's a circle
            points = []
            numpy.append(angles, angles[-1])
            for angle in angles:
                direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
                points.append(geometry.center + direction * outerRadius)
            points = numpy.array(points)
        elif kind == "donut":
            innerRadius = geometry.radius - geometry.weight * 0.5
            outerRadius = geometry.radius + geometry.weight * 0.5
            angles = numpy.arange(0, 2 * numpy.pi, 0.1)
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
            points = numpy.array(points)
        else:
            innerRadius = geometry.radius - geometry.weight * 0.5
            outerRadius = geometry.radius + geometry.weight * 0.5

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

            if kind ==  "camembert":
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
            elif kind == "arc":
                # It's a part of donut
                points = []
                points.append(geometry.startPoint)
                for angle in angles:
                    direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
                    points.insert(0, geometry.center + direction * innerRadius)
                    points.append(geometry.center + direction * outerRadius)
                points.insert(0, geometry.endPoint)
                points.append(geometry.endPoint)
            else:
                assert False

            points = numpy.array(points)

        return points

    def _updateShape(self):
        geometry = self._geometry
        points = self._createShapeFromGeometry(geometry)
        self.__shape.setPoints(points)

        index = numpy.nanargmin(points[:, 1])
        pos = points[index]
        with utils.blockSignals(self._handleLabel):
            self._handleLabel.setPosition(pos[0], pos[1])

        if geometry.center is None:
            movePos = geometry.startPoint * 0.34 + geometry.endPoint * 0.66
        elif (geometry.isClosed()
              or abs(geometry.endAngle - geometry.startAngle) > numpy.pi * 0.7):
            movePos = geometry.center
        else:
            moveAngle = geometry.startAngle * 0.34 + geometry.endAngle * 0.66
            vector = numpy.array([numpy.cos(moveAngle), numpy.sin(moveAngle)])
            movePos = geometry.center + geometry.radius * vector

        with utils.blockSignals(self._handleMove):
            self._handleMove.setPosition(*movePos)

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
        geometry = self._geometry
        if geometry.center is None:
            raise ValueError("This ROI can't be represented as a section of circle")
        return geometry.center, self.getInnerRadius(), self.getOuterRadius(), geometry.startAngle, geometry.endAngle

    def isClosed(self):
        """Returns true if the arc is a closed shape, like a circle or a donut.

        :rtype: bool
        """
        return self._geometry.isClosed()

    def getCenter(self):
        """Returns the center of the circle used to draw arcs of this ROI.

        This center is usually outside the the shape itself.

        :rtype: numpy.ndarray
        """
        return self._geometry.center

    def getStartAngle(self):
        """Returns the angle of the start of the section of this ROI (in radian).

        If `startAngle` is smaller than `endAngle` the rotation is clockwise,
        else the rotation is anticlockwise.

        :rtype: float
        """
        return self._geometry.startAngle

    def getEndAngle(self):
        """Returns the angle of the end of the section of this ROI (in radian).

        If `startAngle` is smaller than `endAngle` the rotation is clockwise,
        else the rotation is anticlockwise.

        :rtype: float
        """
        return self._geometry.endAngle

    def getInnerRadius(self):
        """Returns the radius of the smaller arc used to draw this ROI.

        :rtype: float
        """
        geometry = self._geometry
        radius = geometry.radius - geometry.weight * 0.5
        if radius < 0:
            radius = 0
        return radius

    def getOuterRadius(self):
        """Returns the radius of the bigger arc used to draw this ROI.

        :rtype: float
        """
        geometry = self._geometry
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

        vector = numpy.array([numpy.cos(startAngle), numpy.sin(startAngle)])
        startPoint = center + vector * radius
        vector = numpy.array([numpy.cos(endAngle), numpy.sin(endAngle)])
        endPoint = center + vector * radius

        geometry = self._Geometry.create(center, startPoint, endPoint,
                                         radius, weight,
                                         startAngle, endAngle, closed=None)
        self._geometry = geometry
        self._updateHandles()

    @docstring(HandleBasedROI)
    def contains(self, position):
        # first check distance, fastest
        center = self.getCenter()
        distance = numpy.sqrt((position[1] - center[1]) ** 2 + ((position[0] - center[0])) ** 2)
        is_in_distance = self.getInnerRadius() <= distance <= self.getOuterRadius()
        if not is_in_distance:
            return False
        rel_pos = position[1] - center[1], position[0] - center[0]
        angle = numpy.arctan2(*rel_pos)
        start_angle = self.getStartAngle()
        end_angle = self.getEndAngle()

        if start_angle < end_angle:
            # I never succeed to find a condition where start_angle < end_angle
            # so this is untested
            is_in_angle = start_angle <= angle <= end_angle
        else:
            if end_angle < -numpy.pi and angle > 0:
                angle = angle - (numpy.pi *2.0)
            is_in_angle = end_angle <= angle <= start_angle
        return is_in_angle

    def translate(self, x, y):
        self._geometry = self._geometry.translated(x, y)
        self._updateHandles()

    def _arcCurvatureMarkerConstraint(self, x, y):
        """Curvature marker remains on perpendicular bisector"""
        geometry = self._geometry
        if geometry.center is None:
            center = (geometry.startPoint + geometry.endPoint) * 0.5
            vector = geometry.startPoint - geometry.endPoint
            vector = numpy.array((vector[1], -vector[0]))
            vdist = numpy.linalg.norm(vector)
            if vdist != 0:
                normal = numpy.array((vector[1], -vector[0])) / vdist
            else:
                normal = numpy.array((0, 0))
        else:
            if geometry.isClosed():
                midAngle = geometry.startAngle + numpy.pi * 0.5
            else:
                midAngle = (geometry.startAngle + geometry.endAngle) * 0.5
            normal = numpy.array([numpy.cos(midAngle), numpy.sin(midAngle)])
            center = geometry.center
        dist = numpy.dot(normal, (numpy.array((x, y)) - center))
        dist = numpy.clip(dist, geometry.radius, geometry.radius * 2)
        x, y = center + dist * normal
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
        return numpy.array((-c.real, -c.imag)), abs(c + x)

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

    ICON = 'add-range-horizontal'
    NAME = 'horizontal range ROI'
    SHORT_NAME = "hrange"

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        RegionOfInterest.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
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
        self.addItem(self._markerCen)
        self.addItem(self._markerMin)
        self.addItem(self._markerMax)
        self.__filterReentrant = utils.LockReentrant()

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
        elif event in [items.ItemChangedType.VISIBLE,
                       items.ItemChangedType.SELECTABLE]:
            markers = [self._markerMin, self._markerMax, self._markerCen]
            self._updateItemProperty(event, self, markers)
        super(HorizontalRangeROI, self)._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        markers = [self._markerMin, self._markerMax, self._markerCen]
        for m in markers:
            m.setColor(style.getColor())
            m.setLineWidth(style.getLineWidth())

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
            self._markerMin.sigItemChanged.connect(self._minPositionChanged)
            self._markerMax.sigItemChanged.connect(self._maxPositionChanged)
            self._markerCen.sigItemChanged.connect(self._cenPositionChanged)
            self._markerCen.setLineStyle(":")
        else:
            self._markerMin.sigItemChanged.disconnect(self._minPositionChanged)
            self._markerMax.sigItemChanged.disconnect(self._maxPositionChanged)
            self._markerCen.sigItemChanged.disconnect(self._cenPositionChanged)
            self._markerCen.setLineStyle(" ")

    def _updatePos(self, vmin, vmax, force=False):
        """Update marker position and emit signal.

        :param float vmin:
        :param float vmax:
        :param bool force:
            True to update even if already at the right position.
        """
        if not force and numpy.array_equal((vmin, vmax), self.getRange()):
            return  # Nothing has changed

        center = (vmin + vmax) * 0.5
        with self.__filterReentrant:
            with utils.blockSignals(self._markerMin):
                self._markerMin.setPosition(vmin, 0)
            with utils.blockSignals(self._markerCen):
                self._markerCen.setPosition(center, 0)
            with utils.blockSignals(self._markerMax):
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

    def _minPositionChanged(self, event):
        """Handle position changed events of the marker"""
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self._updatePos(marker.getXPosition(), self.getMax(), force=True)

    def _maxPositionChanged(self, event):
        """Handle position changed events of the marker"""
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self._updatePos(self.getMin(), marker.getXPosition(), force=True)

    def _cenPositionChanged(self, event):
        """Handle position changed events of the marker"""
        if event is items.ItemChangedType.POSITION:
            marker = self.sender()
            self.setCenter(marker.getXPosition())

    def __str__(self):
        vrange = self.getRange()
        params = 'min: %f; max: %f' % vrange
        return "%s(%s)" % (self.__class__.__name__, params)
