# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This module provides ROI interaction for :class:`~silx.gui.plot.PlotWidget`.

This API is not mature and will probably change in the future.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/03/2018"


import collections
import functools
import itertools
import logging
import time
import weakref

import numpy

from ....third_party import enum
from ....utils.weakref import WeakList
from ... import qt, icons
from .. import PlotWidget
from .. import items
from ...colors import rgba


logger = logging.getLogger(__name__)


class RegionOfInterest(qt.QObject):
    """Object describing a region of interest in a plot.

    :param QObject parent:
        The RegionOfInterestManager that created this object
    :param str kind: The kind of selection represented by this object
    """

    sigControlPointsChanged = qt.Signal()
    """Signal emitted when this control points has changed"""

    def __init__(self, parent, kind):
        assert isinstance(parent, RegionOfInterestManager)
        super(RegionOfInterest, self).__init__(parent)
        self._color = rgba('red')
        self._items = WeakList()
        self._editAnchors = WeakList()
        self._points = None
        self._label = ''
        self._kind = str(kind)
        self._editable = False

    def __del__(self):
        # Clean-up plot items
        self._removePlotItems()

    def _removedFromSelector(self):
        """Remove this selection from the selector that created it"""
        self._removePlotItems()
        self.setParent(None)

    def getKind(self):
        """Return kind of selection

        :rtype: str
        """
        return self._kind

    def getColor(self):
        """Returns the color of this selection

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._color)

    def setColor(self, color):
        """Set the color used for this selection.

        :param color: The color to use for selection shape as
           either a color name, a QColor, a list of uint8 or float in [0, 1].
        """
        color = rgba(color)
        if color != self._color:
            self._color = color

            # Update color of selection items in the plot
            rgbaColor = rgba(color)
            for item in list(self._items):
                if isinstance(item, items.ColorMixIn):
                    item.setColor(rgbaColor)

            # Use transparency for anchors
            rgbaColor = rgbaColor[:3] + (0.5,)

            for item in list(self._editAnchors):
                if isinstance(item, items.ColorMixIn):
                    item.setColor(rgbaColor)

    def getLabel(self):
        """Returns the label displayed for this selection.

        :rtype: str
        """
        return self._label

    def setLabel(self, label):
        """Set the label displayed with this selection.

        :param str label: The text label to display
        """
        label = str(label)
        if label != self._label:
            self._label = label
            for item in self._items:
                if isinstance(
                        item, (items.Marker, items.XMarker, items.YMarker)):
                    item.setText(self._label)

    def isEditable(self):
        """Returns whether the selection is editable by the user or not.

        :rtype: bool
        """
        return self._editable

    def setEditable(self, editable):
        """Set whether selection can be changed interactively.

        :param bool editable: True to allow edition by the user,
           False to disable.
        """
        editable = bool(editable)
        if self._editable != editable:
            self._editable = editable
            # Recreate plot items
            # This can be avoided once marker.setDraggable is public
            self._removePlotItems()
            self._createPlotItems()

    def getControlPoints(self):
        """Returns the current selection control points.

        It returns an empty tuple if there is currently no selection.

        :return: Array of (x, y) position in plot coordinates
        :rtype: numpy.ndarray
        """
        return None if self._points is None else numpy.array(self._points)

    def setControlPoints(self, points):
        """Set this selection control points.

        :param points: Iterable of (x, y) control points
        """
        points = numpy.array(points)
        assert points.ndim == 2
        assert points.shape[1] == 2

        if (self._points is None or
                not numpy.all(numpy.equal(points, self._points))):
            kind = self.getKind()

            if kind == 'polygon':
                # Makes sure first and last points are different
                if numpy.all(numpy.equal(points[0], points[-1])):
                    points = points[:-1]
                assert len(points) > 2

            self._points = points

            if self._items:  # Update plot items
                for item in self._items:
                    if isinstance(item, items.Shape):
                        item.setPoints(points)
                    elif isinstance(item, (items.Marker,
                                           items.XMarker,
                                           items.YMarker)):
                        markerPos = self._getMarkerPosition(points, kind)
                        item.setPosition(*markerPos)

                if self._editAnchors:  # Update anchors
                    if len(self._editAnchors) == len(points) + 1:
                        # Also update center anchor
                        points = numpy.append(points,
                                              [numpy.mean(points, axis=0)], axis=0)

                    for anchor, point in zip(self._editAnchors, points):
                        anchor.setPosition(*point)

            else:  # Create plot items
                self._createPlotItems()

            self.sigControlPointsChanged.emit()

    @staticmethod
    def _getMarkerPosition(points, kind):
        """Compute marker position.

        :param numpy.ndarray points: Array of (x, y) control points
        :param str kind: The kind of selection shape to use
        :return: (x, y) position of the marker
        """
        if kind in ('point', 'hline', 'vline'):
            assert len(points) == 1
            return points[0]

        elif kind == 'rectangle':
            assert len(points) == 2
            return points.min(axis=0)

        elif kind == 'line':
            assert len(points) == 2
            return points[numpy.argmin(points[:, 0])]

        elif kind == 'polygon':
            return points[numpy.argmin(points[:, 1])]

        else:
            raise RuntimeError('Unsupported selection kind: %s' % kind)

    def _createPlotItems(self):
        """Create items displaying the selection in the plot."""
        roiManager = self.parent()
        plot = roiManager.parent()

        x, y = self._points.T
        kind = self.getKind()
        legend = "__RegionOfInterest-%d__" % id(self)

        self._items = WeakList()

        if kind == 'point':
            plot.addMarker(
                x[0], y[0],
                legend=legend,
                text=self.getLabel(),
                color=rgba(self.getColor()),
                draggable=self.isEditable())
            item = plot._getItem(kind='marker', legend=legend)
            self._items.append(item)

            if self.isEditable():
                item.sigItemChanged.connect(self._markerChanged)

        elif kind == 'hline':
            plot.addYMarker(
                y[0],
                legend=legend,
                text=self.getLabel(),
                color=rgba(self.getColor()),
                draggable=self.isEditable())
            item = plot._getItem(kind='marker', legend=legend)
            self._items.append(item)

            if self.isEditable():
                item.sigItemChanged.connect(self._markerChanged)

        elif kind == 'vline':
            plot.addXMarker(
                x[0],
                legend=legend,
                text=self.getLabel(),
                color=rgba(self.getColor()),
                draggable=self.isEditable())
            item = plot._getItem(kind='marker', legend=legend)
            self._items.append(item)

            if self.isEditable():
                item.sigItemChanged.connect(self._markerChanged)

        else:  # rectangle, line, polygon
            plot.addItem(x, y,
                         legend=legend,
                         shape='polylines' if kind == 'line' else kind,
                         color=rgba(self.getColor()),
                         fill=False)
            self._items.append(plot._getItem(kind='item', legend=legend))

            # Add label marker
            markerPos = self._getMarkerPosition(self._points, kind)
            plot.addMarker(*markerPos,
                           legend=legend + '-name',
                           text=self.getLabel(),
                           color=rgba(self.getColor()),
                           symbol='',
                           draggable=False)
            self._items.append(
                plot._getItem(kind='marker', legend=legend + '-name'))

            if self.isEditable():  # Add draggable anchors
                self._editAnchors = WeakList()

                color = rgba(self.getColor())
                color = color[:3] + (0.5,)

                for index, point in enumerate(self._points):
                    anchorLegend = legend + '-anchor-%d' % index
                    plot.addMarker(*point,
                                   legend=anchorLegend,
                                   text='',
                                   color=color,
                                   symbol='s',
                                   draggable=True)
                    item = plot._getItem(kind='marker', legend=anchorLegend)
                    item.sigItemChanged.connect(functools.partial(
                        self._controlPointAnchorChanged, index))
                    self._editAnchors.append(item)

                # Add an anchor to the center of the rectangle
                if kind == 'rectangle':
                    center = numpy.mean(self._points, axis=0)
                    anchorLegend = legend + '-anchor-center'
                    plot.addMarker(*center,
                                   legend=anchorLegend,
                                   text='',
                                   color=color,
                                   symbol='o',
                                   draggable=True)
                    item = plot._getItem(kind='marker', legend=anchorLegend)
                    item.sigItemChanged.connect(self._centerAnchorChanged)
                    self._editAnchors.append(item)

    def _markerChanged(self, event):
        """Handle draggable marker changed.

        Used for 'point', 'hline', 'vline'.

        :param ItemChangeType event:
        """
        if event == items.ItemChangedType.POSITION:
            kind = self.getKind()

            marker = self.sender()
            position = marker.getPosition()

            if kind == 'point':
                points = [position]
            elif kind == 'hline':
                points = self.getControlPoints()
                points[:, 1] = position[1]
            elif kind == 'vline':
                points = self.getControlPoints()
                points[:, 0] = position[0]
            else:
                raise RuntimeError('Unhandled kind %s' % kind)

            self.setControlPoints(points)

    def _controlPointAnchorChanged(self, index, event):
        """Handle update of position of an edition anchor

        :param int index: Index of the anchor
        :param ItemChangedType event: Event type
        """
        if event == items.ItemChangedType.POSITION:
            anchor = self._editAnchors[index]
            points = self.getControlPoints()
            points[index] = anchor.getPosition()
            self.setControlPoints(points)

    def _centerAnchorChanged(self, event):
        """Handle update of position of the center anchor

        :param ItemChangedType event: Event type
        """
        if event == items.ItemChangedType.POSITION:
            anchor = self._editAnchors[-1]
            points = self.getControlPoints()
            center = numpy.mean(points, axis=0)
            offset = anchor.getPosition() - center
            points = points + offset
            self.setControlPoints(points)

    def _removePlotItems(self):
        """Remove items from their plot."""
        for item in itertools.chain(list(self._items),
                                    list(self._editAnchors)):
            plot = item.getPlot()
            if plot is not None:
                plot._remove(item)
        self._items = WeakList()
        self._editAnchors = WeakList()


class RegionOfInterestManager(qt.QObject):
    """Class handling a selection interaction on a PlotWidget.

    It supports the selection of multiple points, rectangles, polygons,
    lines, horizontal and vertical lines.

    See ``plotInteractiveImageROI.py`` sample code (:ref:`sample-code`).

    :param silx.gui.plot.PlotWidget parent:
        The plot widget the selection is done on
    """

    sigSelectionAdded = qt.Signal(RegionOfInterest)
    """Signal emitted when a new selection has been added.

    It provides the newly add :class:`RegionOfInterest` object.
    """

    sigSelectionAboutToBeRemoved = qt.Signal(RegionOfInterest)
    """Signal emitted just before a selection is removed.

    It provides the :class:`RegionOfInterest` object that is about to be removed.
    """

    sigSelectionChanged = qt.Signal(tuple)
    """Signal emitted whenever the selection has changed.

    It provides the selection.
    """

    sigSelectionStarted = qt.Signal(str)
    """Signal emitted whenever an interactive selection has started.

    It provides the shape used for the selection.
    """

    sigSelectionFinished = qt.Signal(tuple)
    """Signal emitted when an interactive selection has ended.

    It provides the selection.
    """

    _MODE_ACTIONS_PARAMS = collections.OrderedDict()
    # Interactive mode: (icon name, text)
    _MODE_ACTIONS_PARAMS['point'] = 'normal', 'Add point selection'
    _MODE_ACTIONS_PARAMS['rectangle'] = 'shape-rectangle', 'Add Rectangle ROI'
    _MODE_ACTIONS_PARAMS['polygon'] = 'shape-polygon', 'Add Polygon ROI'
    _MODE_ACTIONS_PARAMS['line'] = 'shape-diagonal', 'Add Line ROI'
    _MODE_ACTIONS_PARAMS['hline'] = 'shape-horizontal', 'Add Horizontal Line ROI'
    _MODE_ACTIONS_PARAMS['vline'] = 'shape-vertical', 'Add Vertical Line ROI'

    def __init__(self, parent):
        assert isinstance(parent, PlotWidget)
        super(RegionOfInterestManager, self).__init__(parent)
        self._selections = []
        self._maxSelection = None

        self._shapeKind = None
        self._color = rgba('red')

        self._label = "__RegionOfInterestManager__%d" % id(self)

        self._eventLoop = None

        self._modeActions = {}

        parent.sigInteractiveModeChanged.connect(
            self._plotInteractiveModeChanged)

    @classmethod
    def getSupportedSelectionKinds(cls):
        """Returns available selection kinds

        :rtype: List[str]
        """
        return tuple(cls._MODE_ACTIONS_PARAMS.keys())

    # Associated QActions

    def getDrawSelectionModeAction(self, kind):
        """Returns the QAction corresponding to a kind of selection

        The QAction allows to enable the corresponding drawing
        interactive mode.

        :param str kind: Kind of selection
        :rtype: QAction
        :raise ValueError: If kind is not supported
        """
        if kind not in self.getSupportedSelectionKinds():
            raise ValueError('Unsupported kind %s' % kind)

        action = self._modeActions.get(kind, None)
        if action is None:  # Lazy-loading
            iconName, text = self._MODE_ACTIONS_PARAMS[kind]
            action = qt.QAction(self)
            action.setIcon(icons.getQIcon(iconName))
            action.setText(text)
            action.setCheckable(True)
            action.setChecked(self.getSelectionKind() == kind)

            action.triggered.connect(
                functools.partial(self._modeActionTriggered, kind=kind))
            self._modeActions[kind] = action
        return action

    def _modeActionTriggered(self, checked, kind):
        """Handle mode actions being checked by the user

        :param bool checked:
        :param str kind: Corresponding shape kind
        """
        if checked:
            self.start(kind)

    def _updateModeActions(self):
        """Enable/Disable mode actions depending on max selections"""
        for kind, action in self._modeActions.items():
            action.setChecked(kind == self.getSelectionKind())

    # PlotWidget eventFilter and listeners

    def _plotInteractiveModeChanged(self, source):
        """Handle change of interactive mode in the plot"""
        if source is not self:
            self.stop()  # Stop any current interaction mode

        else:  # Check the corresponding action
            self._updateModeActions()

    # Handle selection interaction

    def _handleInteraction(self, event):
        """Handle mouse interaction for selection"""
        kind = self.getSelectionKind()
        if kind is None:
            return  # Should not happen

        points = None

        if kind == 'point':
            if event['event'] == 'mouseClicked' and event['button'] == 'left':
                points = numpy.array([(event['x'], event['y'])],
                                     dtype=numpy.float64)

        else:  # other shapes
            if (event['event'] == 'drawingFinished' and
                    event['parameters']['label'] == self._label):
                points = numpy.array((event['xdata'], event['ydata']),
                                     dtype=numpy.float64).T

                if kind == 'hline':
                    points = numpy.array([(float('nan'), points[0, 1])],
                                         dtype=numpy.float64)
                elif kind == 'vline':
                    points = numpy.array([(points[0, 0], float('nan'))],
                                         dtype=numpy.float64)

        if points is not None:
            if self.isMaxSelections():
                # When reaching max number of selections, redo last one
                selections = self.getSelections()
                if len(selections) > 0:
                    self.removeSelection(selections[-1])
            self.addSelection(kind=kind, points=points)

    # RegionOfInterest API

    def getSelectionPoints(self):
        """Returns the current selection control points

        :return: Tuple of arrays of (x, y) points in plot coordinates
        :rtype: tuple of Nx2 numpy.ndarray
        """
        return tuple(s.getControlPoints() for s in self.getSelections())

    def getSelections(self):
        """Returns the list of current selections.

        It returns an empty tuple if there is currently no selection.

        :return: Tuple of arrays of objects describing the selection
        """
        return tuple(self._selections)

    def clearSelections(self):
        """Reset current selections

        :return: True if selections were reset.
        :rtype: bool
        """
        if self.getSelections():  # Something to reset
            for selection in self._selections:
                selection.sigControlPointsChanged.disconnect(
                    self._selectionPointsChanged)
                selection._removedFromSelector()
            self._selections = []
            self._selectionUpdated()
            return True

        else:
            return False

    def _selectionPointsChanged(self):
        """Handle selection object points changed"""
        self.sigSelectionChanged.emit(self.getSelections())

    def getMaxSelections(self):
        """Returns the maximum number of selections or None if no limit

        :rtype: Union[int,None]
        """
        return self._maxSelection

    def setMaxSelections(self, max_):
        """Set the maximum number of selections

        :param Union[int,None] max_: The max limit or None for no limit.
        :raise ValueError: If there is more selections than max value
        """
        if max_ is not None:
            max_ = int(max_)
            if max_ <= 0:
                raise ValueError('Max limit must be strictly positive')

            if len(self.getSelections()) > max_:
                raise ValueError(
                    'Cannot set max limit: Already too many selections')

        self._maxSelection = max_

    def isMaxSelections(self):
        """Returns True if the maximum number of selections is reached.

        :rtype: bool
        """
        max_ = self.getMaxSelections()
        return max_ is not None and len(self.getSelections()) >= max_

    def addSelection(self, kind, points, label='', index=None):
        """Add a selection to current selections

        :param str kind: The kind of selection to add
        :param numpy.ndarray points: The control points of the selection shape
        :param str label: The label to display along with the selection.
        :param int index: The position where to insert the selection,
            By default it is appended to the end of the list of selections
        :return: The created Selection object
        :rtype: RegionOfInterest
        :raise RuntimeError: When selection cannot be added because the maximum
           number of selection has been reached.
        """
        if self.isMaxSelections():
            raise RuntimeError(
                'Cannot add selection: Maximum number of selections reached')

        plot = self.parent()
        if plot is None:
            raise RuntimeError(
                'Cannot add selection: PlotWidget no more available')

        # Create new selection object
        selection = RegionOfInterest(parent=self, kind=kind)
        selection.setColor(self.getColor())
        selection.setLabel(str(label))
        selection.setControlPoints(points)
        selection.sigControlPointsChanged.connect(
            self._selectionPointsChanged)

        if index is None:
            self._selections.append(selection)
        else:
            self._selections.insert(index, selection)
        self.sigSelectionAdded.emit(selection)
        self._selectionUpdated()

    def removeSelection(self, selection):
        """Remove a selection from the list of current selections

        :param RegionOfInterest selection: The selection to remove
        :raise ValueError: When selection is not a selection in this object
        """
        if not (isinstance(selection, RegionOfInterest) and
                selection.parent() is self and
                selection in self._selections):
            raise ValueError('RegionOfInterest does not belong to this instance')

        self.sigSelectionAboutToBeRemoved.emit(selection)

        self._selections.remove(selection)
        selection.sigControlPointsChanged.disconnect(
            self._selectionPointsChanged)
        selection._removedFromSelector()
        self._selectionUpdated()

    def _selectionUpdated(self):
        """Handle update of the selection"""
        self.sigSelectionChanged.emit(self.getSelections())

    # RegionOfInterest parameters

    def getColor(self):
        """Return the default color of the selections

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._color)

    def setColor(self, color):
        """Set the default color to use for selections.

        Existing selections are not affected.

        :param color: The color to use for displaying selections as
           either a color name, a QColor, a list of uint8 or float in [0, 1].
        """
        self._color = rgba(color)

    # Control selection

    def getSelectionKind(self):
        """Returns the current interactive selection mode or None.

        :rtype: Union[str,None]
        """
        return self._shapeKind

    def isStarted(self):
        """Returns True if the selection is requesting user input.

        :rtype: bool
        """
        return self._shapeKind is not None

    def start(self, kind):
        """Start an interactive selection.

        :param str kind: The kind of shape to select in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :return: True if interactive selection was started, False otherwise
        :rtype: bool
        :raise ValueError: If kind is not supported
        """
        self.stop()

        plot = self.parent()
        if plot is None:
            return False

        if kind not in self.getSupportedSelectionKinds():
            raise ValueError('Unsupported kind %s' % kind)
        self._shapeKind = kind

        if self._shapeKind == 'point':
            plot.setInteractiveMode(mode='select', source=self)
        else:
            plot.setInteractiveMode(mode='select-draw',
                                    source=self,
                                    shape=self._shapeKind,
                                    color=rgba(self.getColor()),
                                    label=self._label)

        plot.sigPlotSignal.connect(self._handleInteraction)

        self.sigSelectionStarted.emit(kind)

        return True

    def stop(self):
        """Stop interactive selection

        :return: True if a selection was actually stopped
        :rtype: bool
        """
        if not self.isStarted():
            return False

        self._shapeKind = None

        plot = self.parent()
        if plot is not None:
            plot.sigPlotSignal.disconnect(self._handleInteraction)
            plot.setInteractiveMode(mode='zoom', source=None)

        self._updateModeActions()

        self.sigSelectionFinished.emit(self.getSelectionPoints())

        return True

    def exec_(self, kind):
        """Block until :meth:`quit` is called.

        :param str kind: The kind of shape to select in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :return: The current selection
        :rtype: tuple
        """
        self.start(kind=kind)

        plot = self.parent()
        plot.show()
        plot.raise_()

        self._eventLoop = qt.QEventLoop()
        self._eventLoop.exec_()
        self._eventLoop = None

        self.stop()

        selection = self.getSelectionPoints()
        self.clearSelections()
        return selection

    def quit(self):
        """Stop a blocking :meth:`exec_` and call :meth:`stop`"""
        if self._eventLoop is not None:
            self._eventLoop.quit()
            self._eventLoop = None
        self.stop()


class InteractiveRegionOfInterestManager(RegionOfInterestManager):
    """RegionOfInterestManager with features for use from interpreter.

    It is meant to be used through the :meth:`exec_`.
    It provides some messages to display in a status bar and
    different modes to end blocking calls to :meth:`exec_`.

    :param parent: See QObject
    """

    sigMessageChanged = qt.Signal(str)
    """Signal emitted when a new message should be displayed to the user

    It provides the message as a str.
    """

    def __init__(self, parent):
        super(InteractiveRegionOfInterestManager, self).__init__(parent)
        self.__timeoutEndTime = None
        self.__message = ''
        self.__validationMode = self.ValidationMode.ENTER
        self.__execKind = None

        self.sigSelectionAdded.connect(self.__added)
        self.sigSelectionAboutToBeRemoved.connect(self.__aboutToBeRemoved)
        self.sigSelectionStarted.connect(self.__started)
        self.sigSelectionFinished.connect(self.__finished)

    # Validation mode

    @ enum.unique
    class ValidationMode(enum.Enum):
        """Mode of validation of the selection"""

        AUTO = 'auto'
        """Automatically ends the selection once the user terminates the last shape"""

        ENTER = 'enter'
        """Ends the selection when the *Enter* key is pressed"""

        AUTO_ENTER = 'auto_enter'
        """Ends selection if reaching max selection or on *Enter* key press
        """

        NONE = 'none'
        """Do not provide the user a way to end the selection.

        The end of :meth:`exec_` is done through :meth:`quit` or timeout.
        """

    def getValidationMode(self):
        """Returns the selection validation mode in use.

        :rtype: ValidationMode
        """
        return self.__validationMode

    def setValidationMode(self, mode):
        """Set the way to perform selection validation.

        See :class:`ValidationMode` enumeration for the supported
        validation modes.

        :param ValidationMode mode: The mode of selection validation to use.
        """
        assert isinstance(mode, self.ValidationMode)
        if mode != self.__validationMode:
            self.__validationMode = mode

        if self.isExec():
            if (self.isMaxSelections() and self.getValidationMode() in
                    (self.ValidationMode.AUTO,
                     self.ValidationMode.AUTO_ENTER)):
                self.quit()

            self.__updateMessage()

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.Hide:
            self.quit()

        if event.type() == qt.QEvent.KeyPress:
            key = event.key()
            if (key == qt.Qt.Key_Return and
                    self.getValidationMode() in (self.ValidationMode.ENTER,
                                                 self.ValidationMode.AUTO_ENTER)):
                # Stop on return key pressed
                self.quit()
                return True  # Stop further handling of this keys

            if (key in (qt.Qt.Key_Delete, qt.Qt.Key_Backspace) or (
                    key == qt.Qt.Key_Z and
                    event.modifiers() & qt.Qt.ControlModifier)):
                selections = self.getSelections()
                if selections:  # Something to undo
                    self.removeSelection(selections[-1])
                    # Stop further handling of keys if something was undone
                    return True

        return super(InteractiveRegionOfInterestManager, self).eventFilter(obj, event)

    # Message API

    def getMessage(self):
        """Returns the current status message.

        This message is meant to be displayed in a status bar.

        :rtype: str
        """
        if self.__timeoutEndTime is None:
            return self.__message
        else:
            remaining = self.__timeoutEndTime - time.time()
            return self.__message + (' - %d seconds remaining' %
                                     max(1, int(remaining)))

    # Listen to selection updates

    def __added(self, *args, **kwargs):
        """Handle new selection added"""
        self.__updateMessage()
        if (self.isMaxSelections() and
                self.getValidationMode() in (self.ValidationMode.AUTO,
                                             self.ValidationMode.AUTO_ENTER)):
            self.quit()

    def __aboutToBeRemoved(self, *args, **kwargs):
        """Handle removal of a selection"""
        # RegionOfInterest not removed yet
        self.__updateMessage(nbSelections=len(self.getSelections()) - 1)

    def __started(self, *args, **kwargs):
        """Handle interactive mode started"""
        self.__updateMessage()

    def __finished(self, *args, **kwargs):
        """Handle interactive mode finished"""
        self.__updateMessage()

    def __updateMessage(self, nbSelections=None):
        """Update message"""
        if not self.isExec():
            message = 'Done'

        elif not self.isStarted():
            message = 'Use %s ROI edition mode' % self.__execKind

        else:
            if nbSelections is None:
                nbSelections = len(self.getSelections())

            kind = self.__execKind
            maxNbSelection = self.getMaxSelections()

            if maxNbSelection is None:
                message = 'Select %ss (%d selected)' % (kind, nbSelections)

            elif maxNbSelection <= 1:
                message = 'Select a %s' % kind
            else:
                message = 'Select %d/%d %ss' % (nbSelections, maxNbSelection, kind)

            if (self.getValidationMode() == self.ValidationMode.ENTER and
                    self.isMaxSelections()):
                message += ' - Press Enter to confirm'

        if message != self.__message:
            self.__message = message
            # Use getMessage to add timeout message
            self.sigMessageChanged.emit(self.getMessage())

    # Handle blocking call

    def __timeoutUpdate(self):
        """Handle update of timeout"""
        if (self.__timeoutEndTime is not None and
                (self.__timeoutEndTime - time.time()) > 0):
                self.sigMessageChanged.emit(self.getMessage())
        else:  # Stop selection and message timer
            timer = self.sender()
            timer.stop()
            self.__timeoutEndTime = None
            self.quit()

    def isExec(self):
        """Returns True if :meth:`exec_` is currently running.

        :rtype: bool"""
        return self.__execKind is not None

    def exec_(self, kind, timeout=0):
        """Block until selection is done or timeout is elapsed.

        :meth:`quit` also ends this blocking call.

        :param str kind: The kind of shape to select in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :param int timeout: Maximum duration in seconds to block.
            Default: No timeout
        :return: The current selection
        :rtype: tuple
        """
        plot = self.parent()
        if plot is None:
            return

        self.__execKind = kind

        plot.installEventFilter(self)

        if timeout > 0:
            self.__timeoutEndTime = time.time() + timeout
            timer = qt.QTimer(self)
            timer.timeout.connect(self.__timeoutUpdate)
            timer.start(1000)

            selection = super(InteractiveRegionOfInterestManager, self).exec_(kind)

            timer.stop()
            self.__timeoutEndTime = None

        else:
            selection = super(InteractiveRegionOfInterestManager, self).exec_(kind)

        plot.removeEventFilter(self)

        self.__execKind = None
        self.__updateMessage()

        return selection


class _DeleteSelectionToolButton(qt.QToolButton):
    """Tool button deleting a selection object

    :param parent: See QWidget
    :param RegionOfInterest selection: The selection to delete
    """

    def __init__(self, parent, selection):
        super(_DeleteSelectionToolButton, self).__init__(parent)
        self.setIcon(icons.getQIcon('remove'))
        self.__selection = selection
        self.clicked.connect(self.__clicked)

    def __clicked(self, checked):
        """Handle button clicked"""
        if self.__selection is not None:
            selector = self.__selection.parent()
            if selector is not None:
                selector.removeSelection(self.__selection)
                self.__selection = None


class RegionOfInterestTableWidget(qt.QTableWidget):
    """Widget displaying the selection of a :class:`RegionOfInterestManager`"""

    def __init__(self, parent=None):
        super(RegionOfInterestTableWidget, self).__init__(parent)
        self._roiManagerRef = None

        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(
            ['Label', 'Edit', 'Delete', 'Kind', 'Coordinates'])

        horizontalHeader = self.horizontalHeader()
        horizontalHeader.setDefaultAlignment(qt.Qt.AlignLeft)
        if hasattr(horizontalHeader, 'setResizeMode'):  # Qt 4
            setSectionResizeMode = horizontalHeader.setResizeMode
        else:  # Qt5
            setSectionResizeMode = horizontalHeader.setSectionResizeMode

        setSectionResizeMode(0, qt.QHeaderView.Interactive)
        setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        setSectionResizeMode(4, qt.QHeaderView.Stretch)

        verticalHeader = self.verticalHeader()
        verticalHeader.setVisible(False)

        self.setSelectionMode(qt.QAbstractItemView.NoSelection)

        self.itemChanged.connect(self.__itemChanged)

    @staticmethod
    def __itemChanged(item):
        """Handle item updates"""
        column = item.column()
        selection = item.data(qt.Qt.UserRole)
        if column == 0:
            selection.setLabel(item.text())
        elif column == 1:
            selection.setEditable(
                item.checkState() == qt.Qt.Checked)
        elif column in (2, 3, 4):
            pass  # TODO
        else:
            logger.error('Unhandled column %d', column)

    def setRegionOfInterestManager(self, selector):
        """Set the :class:`RegionOfInterestManager` object to sync with

        :param RegionOfInterestManager selector:
        """
        assert selector is None or isinstance(selector, RegionOfInterestManager)

        previousSelector = self.getRegionOfInterestManager()

        if previousSelector is not None:
            previousSelector.sigSelectionChanged.disconnect(self._sync)
        self.setRowCount(0)

        self._roiManagerRef = weakref.ref(selector)

        self._sync()

        if selector is not None:
            selector.sigSelectionChanged.connect(self._sync)

    def _sync(self, *args):
        """Update widget content according to selector"""
        selector = self.getRegionOfInterestManager()

        if selector is None:
            self.setRowCount(0)
            return

        selections = selector.getSelections()

        self.setRowCount(len(selections))
        for index, selection in enumerate(selections):
            baseFlags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled

            # Label
            label = selection.getLabel()
            item = qt.QTableWidgetItem(label)
            item.setFlags(baseFlags | qt.Qt.ItemIsEditable)
            item.setData(qt.Qt.UserRole, selection)
            self.setItem(index, 0, item)

            # Editable
            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags | qt.Qt.ItemIsUserCheckable)
            item.setData(qt.Qt.UserRole, selection)
            item.setCheckState(
                qt.Qt.Checked if selection.isEditable() else qt.Qt.Unchecked)
            self.setItem(index, 1, item)

            # Delete
            delBtn = _DeleteSelectionToolButton(None, selection)

            widget = qt.QWidget()
            layout = qt.QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            widget.setLayout(layout)
            layout.addStretch(1)
            layout.addWidget(delBtn)
            layout.addStretch(1)
            self.setCellWidget(index, 2, widget)

            # Kind
            kind = selection.getKind()
            item = qt.QTableWidgetItem(kind.capitalize())
            item.setFlags(baseFlags)
            self.setItem(index, 3, item)

            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags)

            # Coordinates
            points = selection.getControlPoints()
            if kind == 'rectangle':
                origin = numpy.min(points, axis=0)
                w, h = numpy.max(points, axis=0) - origin
                item.setText('Origin: (%f; %f); Width: %f; Height: %f' %
                             (origin[0], origin[1], w, h))

            elif kind == 'point':
                item.setText('(%f; %f)' % (points[0, 0], points[0, 1]))

            elif kind == 'hline':
                item.setText('Y: %f' % points[0, 1])

            elif kind == 'vline':
                item.setText('X: %f' % points[0, 0])

            else:  # default (polygon, line)
                item.setText('; '.join('(%f; %f)' % (pt[0], pt[1]) for pt in points))
            self.setItem(index, 4, item)

    def getRegionOfInterestManager(self):
        """Returns the :class:`RegionOfInterestManager` this widget supervise.

        It returns None if not sync with an :class:`RegionOfInterestManager`.

        :rtype: RegionOfInterestManager
        """
        return None if self._roiManagerRef is None else self._roiManagerRef()
