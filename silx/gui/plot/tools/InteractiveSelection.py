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
"""This module provides selection interaction to work with :class:`PlotWidget`.

This module is not mature and API will probably change in the future.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/03/2018"


import functools
import itertools
import logging
import numpy

from ....third_party import enum
from ....utils.weakref import WeakList
from ... import qt, icons
from .. import PlotWidget
from .. import items
from ..Colors import rgba


logger = logging.getLogger(__name__)


class Selection(qt.QObject):
    """Object describing a selection in a plot.

    :param QObject parent: The selector that created this selection
    :param str kind: The kind of selection represented by this object
    """

    sigControlPointsChanged = qt.Signal()
    """Signal emitted when this control points has changed"""

    def __init__(self, parent, kind):
        assert isinstance(parent, InteractiveSelection)
        super(Selection, self).__init__(parent)
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

            # Use tranparency for anchors
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
                if isinstance(item, items.Marker):
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
                    elif isinstance(item, items.Marker):
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
        selector = self.parent()
        plot = selector.parent()

        x, y = self._points.T
        kind = self.getKind()
        legend = "__Selection-%d__" % id(self)

        self._items = WeakList()

        if kind == 'point':
            plot.addMarker(
                x[0], y[0],
                legend=legend,
                text=self.getLabel(),
                color=rgba(self.getColor()),
                draggable=self.isEditable())
            self._items.append(plot._getItem(kind='marker', legend=legend))

        elif kind == 'hline':
            plot.addYMarker(
                y[0],
                legend=legend,
                text=self.getLabel(),
                color=rgba(self.getColor()),
                draggable=self.isEditable())
            self._items.append(plot._getItem(kind='marker', legend=legend))

        elif kind == 'vline':
            plot.addXMarker(
                x[0],
                legend=legend,
                text=self.getLabel(),
                color=rgba(self.getColor()),
                draggable=self.isEditable())
            self._items.append(plot._getItem(kind='marker', legend=legend))

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


class InteractiveSelection(qt.QObject):
    """Class handling a selection interaction on a :class:`PlotWidget`

    It supports the selection of multiple points, rectangles, polygons,
    lines, horizontal and vertical lines.

    Example:

    .. code-block:: python

       from silx import sx
       from silx.gui.plot.tools import InteractiveSelection

       plot = sx.PlotWindow()  # Create a PlotWindow
       plot.show()

       # Create object controlling interactive selection
       selector = InteractiveSelection(plot)

       # Add the selection mode action to the PlotWindow toolbar
       toolbar = plot.getInteractiveModeToolBar()
       toolbar.addAction(selector.getSelectionModeAction())

       # Connect InteractiveSelection messages to PlotWindow status bar
       statusBar = plot.statusBar()
       selector.sigMessageChanged.connect(statusBar.showMessage)

       # Start a selection of 3 points
       selector.start('point')

    :param PlotWidget parent: The plot widget the selection is done on
    """

    sigSelectionStarted = qt.Signal(str)
    """Signal emitted whenever a selection has started.

    It provides the shape used for the selection.
    """

    sigSelectionAdded = qt.Signal(Selection)
    """Signal emitted when a new selection has been added.

    It provides the newly add :class:`Selection` object.
    """

    sigSelectionAboutToBeRemoved = qt.Signal(Selection)
    """Signal emitted just before a selection is removed.

    It provides the :class:`Selection` object that is about to be removed.
    """

    sigSelectionChanged = qt.Signal(tuple)
    """Signal emitted whenever the selection has changed.

    It provides the selection.
    """

    sigSelectionFinished = qt.Signal(tuple)
    """Signal emitted when selection is terminated.

    It provides the selection.
    """

    sigMessageChanged = qt.Signal(str)
    """Signal emitted when a new message should be displayed to the user

    It provides the message as a str.
    """

    def __init__(self, parent):
        assert isinstance(parent, PlotWidget)
        super(InteractiveSelection, self).__init__(parent)
        self._selections = []
        self._isStarted = False
        self._isInteractiveModeStarted = False
        self._maxSelection = None

        self._validationMode = self.ValidationMode.ENTER
        self._statusMessage = ''

        self._shapeKind = 'point'
        self._color = rgba('red')

        self._label = "selector-%d" % id(parent)

        self._eventLoop = None

        self._action = qt.QAction(self)
        self._action.setEnabled(False)
        self._action.setCheckable(True)
        self._action.setChecked(False)
        self._action.setIcon(icons.getQIcon('normal'))
        self._action.setText('Selection Mode')
        self._action.setToolTip('Selection mode')

        self._action.changed.connect(self._actionChanged)
        self._action.triggered.connect(self._actionTriggered)

        parent.sigInteractiveModeChanged.connect(
            self._plotInteractiveModeChanged)

    # Associated QAction

    def getSelectionModeAction(self):
        """Returns a QAction for the selection interaction mode

        :rtype: QAction
        """
        return self._action

    def _actionChanged(self):
        """Handle action enabled state changed and sync selector"""
        # Makes sure action is only enabled while a selection is running
        action = self.getSelectionModeAction()
        action.setEnabled(self.isStarted())

    def _actionTriggered(self, checked=False):
        """Handle action triggered by user"""
        self._startSelectionInteraction()
        self._updateStatusMessage()

    # PlotWidget eventFilter and listeners

    def _plotInteractiveModeChanged(self, source):
        """Handle change of interactive mode in the plot"""
        action = self.getSelectionModeAction()
        action.setChecked(source is self)

        if self.isStarted():
            if source is not self:
                self._stopSelectionInteraction(resetInteractiveMode=False)
                self._updateStatusMessage(extra='Use selection mode')
            else:
                self._startSelectionInteraction()
                self._updateStatusMessage()

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.Hide:
            self.stop()

        elif event.type() == qt.QEvent.KeyPress:
            if event.key() in (qt.Qt.Key_Delete, qt.Qt.Key_Backspace) or (
                    event.key() == qt.Qt.Key_Z and
                    event.modifiers() & qt.Qt.ControlModifier):
                if self.undo():
                    # Stop further handling of keys if something was undone
                    return True

            elif (event.key() == qt.Qt.Key_Return and
                    self.getValidationMode() == self.ValidationMode.ENTER):
                self.stop()
                return True  # Stop further handling of those keys

        return super(InteractiveSelection, self).eventFilter(obj, event)

    # Handle selection interaction

    def _handleInteraction(self, event):
        """Handle mouse interaction for selection"""
        if self._shapeKind == 'point':
            if event['event'] == 'mouseClicked' and event['button'] == 'left':
                points = numpy.array([(event['x'], event['y'])],
                                     dtype=numpy.float64)
                self.addSelection(kind=self._shapeKind, points=points)

        else:  # other shapes
            if (event['event'] == 'drawingFinished' and
                    event['parameters']['label'] == self._label):
                points = numpy.array((event['xdata'], event['ydata']),
                                     dtype=numpy.float64).T
                self.addSelection(kind=self._shapeKind, points=points)

    def _stopSelectionInteraction(self, resetInteractiveMode):
        """Stop selection interaction if if was running

        :param bool resetInteractiveMode:
            True to reset interactive mode, False to avoid
        """
        if self._isInteractiveModeStarted:
            self._isInteractiveModeStarted = False
            if self.isStarted():
                plot = self.parent()

                plot.sigPlotSignal.disconnect(self._handleInteraction)
                if resetInteractiveMode and self._shapeKind != 'point':
                    plot.setInteractiveMode(mode='select', source=self)

    def _startSelectionInteraction(self):
        """Start selection interaction if it was not running

        :return: True if interaction has changed, False otherwise
        :rtype: bool
        """
        if (not self._isInteractiveModeStarted and self.isStarted() and
                (self._maxSelection is None or
                 len(self.getSelections()) < self._maxSelection)):
            self._isInteractiveModeStarted = True
            plot = self.parent()

            if self._shapeKind == 'point':
                plot.setInteractiveMode(mode='select', source=self)
            else:
                plot.setInteractiveMode(mode='draw',
                                        source=self,
                                        shape=self._shapeKind,
                                        color=rgba(self.getColor()),
                                        label=self._label)

            plot.sigPlotSignal.connect(self._handleInteraction)
            return True

        else:
            return False

    # Selection API

    def getSelectionLabels(self):
        """Returns the current selection labels

        :return: Tuple of labels
        :rtype: List[str]
        """
        return tuple(s.getLabel() for s in self.getSelections())

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

        :rtype: Union[int, None]
        """
        return self._maxSelection

    def setMaxSelections(self, max):
        """Set the maximum number of selections

        :param Union[int, None] max: The max limit or None for no limit.
        :raise ValueError: If there is more selections than max value
        """
        if max is not None:
            max = int(max)
            if len(self.getSelections()) > max:
                raise ValueError(
                    'Cannot set max limit: Already too many selections')

        self._maxSelection = max

    def addSelection(self, kind, points, label='', index=None):
        """Add a selection to current selections

        :param str kind: The kind of selection to add
        :param numpy.ndarray points: The control points of the selection shape
        :param str label: The label to display along with the selection.
        :param int index: The position where to insert the selection,
            By default it is appended to the end of the list of selections
        :return: The created Selection object
        :rtype: Selection
        :raise RuntimeError: When selection cannot be added because the maximum
           number of selection has been reached.
        """
        selections = self.getSelections()
        if (self._maxSelection is not None and
                len(selections) >= self._maxSelection):
            raise RuntimeError(
                'Cannot add selection: Maximum number of selections reached')

        plot = self.parent()
        if plot is None:
            raise RuntimeError(
                'Cannot add selection: PlotWidget no more available')

        # Create new selection object
        selection = Selection(parent=self, kind=kind)
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

        :param Selection selection: The selection to remove
        """
        assert isinstance(selection, Selection)
        assert selection.parent() is self
        assert selection in self._selections

        self.sigSelectionAboutToBeRemoved.emit(selection)

        self._selections.remove(selection)
        selection.sigControlPointsChanged.disconnect(
            self._selectionPointsChanged)
        selection._removedFromSelector()
        self._selectionUpdated()

    def _selectionUpdated(self):
        """Handle update of the selection"""
        selections = self.getSelections()
        assert self._maxSelection is None or len(selections) <= self._maxSelection

        self.sigSelectionChanged.emit(selections)

        if self.isStarted():
            if self._maxSelection is None or len(selections) < self._maxSelection:
                self._startSelectionInteraction()
                self._updateStatusMessage()

            else:
                self._stopSelectionInteraction(resetInteractiveMode=True)
                validationMode = self.getValidationMode()
                if validationMode == self.ValidationMode.AUTO:
                    self.stop()
                elif validationMode == self.ValidationMode.ENTER:
                    self._updateStatusMessage(extra='Press Enter to confirm')
                else:
                    self._updateStatusMessage()

    def undo(self):
        """Remove last selection from the selection list.

        :return: True if a selection was undone.
        :rtype: bool
        """
        selections = self.getSelections()
        if selections:  # Something to undo
            self.removeSelection(selections[-1])
            return True
        else:
            return False

    # Selection parameters

    @enum.unique
    class ValidationMode(enum.Enum):
        """Mode of validation of the selection"""

        AUTO = 'auto'
        """Automatically ends the selection once the user terminates the last shape"""

        ENTER = 'enter'
        """Ends the selection when the *Enter* key is pressed"""

        NONE = 'none'
        """Do not provide the user a way to end the selection.

        The end of a selection must be done through :meth:`stop`.
        This is useful if the application is willing to provide its own way of
        ending the selection (e.g., a button).
        """

    def getValidationMode(self):
        """Returns the selection validation mode in use.

        :rtype: ValidationMode
        """
        return self._validationMode

    def setValidationMode(self, mode):
        """Set the way to perform selection validation.

        See :class:`ValidationMode` enumeration for the supported
        validation modes.

        :param ValidationMode mode: The mode of selection validation to use.
        """
        assert isinstance(mode, self.ValidationMode)
        if mode != self._validationMode:
            self._validationMode = mode
            if self.isStarted():
                self._updateStatusMessage()

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

    # Status message

    def getStatusMessage(self):
        """Returns the current status message.

        This message is meant to be displayed in a status bar.

        :rtype: str
        """
        return self._statusMessage

    def _updateStatusMessage(self, message=None, extra=None):
        """Set the status message.

        :param str message: The main message or None to use default message
        :param str extra: Additional info to display after main message
        """
        if message is None:
            selections = self.getSelections()
            if self._maxSelection is None:
                message = 'Select %ss (%d selected)' % (
                    self._shapeKind, len(selections))

            elif self._maxSelection <= 1:
                message = 'Select a %s' % self._shapeKind
            else:
                message = 'Select %d/%d %ss' % (
                    len(selections), self._maxSelection, self._shapeKind)

            if self.getValidationMode() == self.ValidationMode.ENTER:
                message += ' - Press Enter to validate'

        else:
            message = str(message)

        if extra is not None:
            message += ' (' + str(extra) + ')'

        if self._statusMessage != message:
            self._statusMessage = message
            self.sigMessageChanged.emit(message)

    # Control selection

    def isStarted(self):
        """Returns True if the selection is requesting user input.

        :rtype: bool"""
        return self._isStarted

    def start(self, kind):
        """Start an interactive selection.

        :param str kind: The kind of shape to select in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        """
        self.stop()

        plot = self.parent()
        if plot is None:
            raise RuntimeError('No plot to perform selection')

        assert kind in ('point', 'rectangle', 'line',
                        'polygon', 'hline', 'vline')
        self._shapeKind = kind

        self._isStarted = True
        self._startSelectionInteraction()

        self.getSelectionModeAction().setEnabled(True)

        plot.installEventFilter(self)

        self._updateStatusMessage()

        self.sigSelectionStarted.emit(kind)

    def stop(self):
        """Stop interactive selection

        :return: True if a selection was actually stopped
        :rtype: bool
        """
        if not self.isStarted():
            return False

        plot = self.parent()
        if plot is not None:
            plot.removeEventFilter(self)
            plot.setInteractiveMode('zoom')

        self._stopSelectionInteraction(resetInteractiveMode=False)

        self._isStarted = False

        self.getSelectionModeAction().setEnabled(False)

        if self._eventLoop is not None:
            self._eventLoop.quit()

        self.sigSelectionFinished.emit(self.getSelectionPoints())
        self._updateStatusMessage('Selection done')

        return True

    def exec_(self, kind, timeout=0):
        """Block until selection is done or timeout is elapsed.

        :param str kind: The kind of shape to select in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :param int timeout: Maximum duration in seconds to block.
            Default: No timeout
        :return: The current selection
        :rtype: tuple
        """
        self.start(kind=kind)

        plot = self.parent()
        plot.show()
        plot.raise_()

        self._eventLoop = qt.QEventLoop()
        if timeout != 0:
            timer = qt.QTimer()
            timer.timeout.connect(self._eventLoop.quit)
            timer.start(int(timeout) * 1000)
            self._eventLoop.exec_()
            timer.stop()

        else:
            self._eventLoop.exec_()
        self._eventLoop = None

        self.stop()

        selection = self.getSelectionPoints()
        self.clearSelections()
        return selection


class InteractiveSelectionTableWidget(qt.QTableWidget):
    """Widget displaying the selection of an :class:`InteractiveSelection`"""

    def __init__(self, parent=None):
        super(InteractiveSelectionTableWidget, self).__init__(parent)
        self._selection = None

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

    def __itemChanged(self, item):
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

    def setInteractiveSelection(self, selector):
        """Set the :class:`InteractiveSelection` object to sync with

        :param InteractiveSelection selector:
        """
        assert selector is None or isinstance(selector, InteractiveSelection)

        if self._selection is not None:
            self.setRowCount(0)
            self._selection.sigSelectionChanged.disconnect(self._sync)

        self._selection = selector  # TODO weakref

        self._sync()

        if self._selection is not None:
            self._selection.sigSelectionChanged.connect(self._sync)

    def _sync(self, *args):
        """Update widget content according to selector"""
        if self._selection is None:
            self.setRowCount(0)
            return

        selections = self._selection.getSelections()

        self.setRowCount(len(selections))
        for index, selection in enumerate(selections):
            baseFlags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled

            label = selection.getLabel()
            item = qt.QTableWidgetItem(label)
            item.setFlags(baseFlags | qt.Qt.ItemIsEditable)
            item.setData(qt.Qt.UserRole, selection)
            self.setItem(index, 0, item)

            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags | qt.Qt.ItemIsUserCheckable)
            item.setData(qt.Qt.UserRole, selection)
            item.setCheckState(
                qt.Qt.Checked if selection.isEditable() else qt.Qt.Unchecked)
            self.setItem(index, 1, item)

            delBtn = qt.QToolButton()
            delBtn.setIcon(icons.getQIcon('remove'))
            self.setCellWidget(index, 2, delBtn)

            kind = selection.getKind()
            item = qt.QTableWidgetItem(kind.capitalize())
            item.setFlags(baseFlags)
            self.setItem(index, 3, item)

            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags)

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

    def getInteractiveSelection(self):
        """Returns the :class:`InteractiveSelection` this widget supervise.

        It returns None if not sync with an :class:`InteractiveSelection`.

        :rtype: InteractiveSelection
        """
        return self._selection
