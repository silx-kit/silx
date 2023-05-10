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
"""This module provides base components to create ROI item for
the :class:`~silx.gui.plot.PlotWidget`.

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

from ....utils.weakref import WeakList
from ... import qt
from .. import items
from ..items import core
from ...colors import rgba
import silx.utils.deprecation
from ....utils.proxy import docstring


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
        return False  # Override in subclass to perform actual test


class RoiInteractionMode(object):
    """Description of an interaction mode.

    An interaction mode provide a specific kind of interaction for a ROI.
    A ROI can implement many interaction.
    """

    def __init__(self, label, description=None):
        self._label = label
        self._description = description

    @property
    def label(self):
        return self._label

    @property
    def description(self):
        return self._description


class InteractionModeMixIn(object):
    """Mix in feature which can be implemented by a ROI object.

    This provides user interaction to switch between different
    interaction mode to edit the ROI.

    This ROI modes have to be described using `RoiInteractionMode`,
    and taken into account during interation with handles.
    """

    sigInteractionModeChanged = qt.Signal(object)

    def __init__(self):
        self.__modeId = None

    def _initInteractionMode(self, modeId):
        """Set the mode without updating anything.

        Must be one of the returned :meth:`availableInteractionModes`.

        :param RoiInteractionMode modeId: Mode to use
        """
        self.__modeId = modeId

    def availableInteractionModes(self):
        """Returns the list of available interaction modes

        Must be implemented when inherited to provide all available modes.

        :rtype: List[RoiInteractionMode]
        """
        raise NotImplementedError()

    def setInteractionMode(self, modeId):
        """Set the interaction mode.

        :param RoiInteractionMode modeId: Mode to use
        """
        self.__modeId = modeId
        self._interactiveModeUpdated(modeId)
        self.sigInteractionModeChanged.emit(modeId)

    def _interactiveModeUpdated(self, modeId):
        """Called directly after an update of the mode.

        The signal `sigInteractionModeChanged` is triggered after this
        call.

        Must be implemented when inherited to take care of the change.
        """
        raise NotImplementedError()

    def getInteractionMode(self):
        """Returns the interaction mode.

        Must be one of the returned :meth:`availableInteractionModes`.

        :rtype: RoiInteractionMode
        """
        return self.__modeId


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
            styleEvents = [items.ItemChangedType.COLOR,
                           items.ItemChangedType.LINE_STYLE,
                           items.ItemChangedType.LINE_WIDTH,
                           items.ItemChangedType.SYMBOL,
                           items.ItemChangedType.SYMBOL_SIZE]
            if self.isHighlighted():
                styleEvents.append(items.ItemChangedType.HIGHLIGHTED_STYLE)

            if event in styleEvents:
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
