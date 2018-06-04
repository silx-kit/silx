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
from ....utils.weakref import WeakList, WeakMethodProxy
from ... import qt, icons
from .. import PlotWidget
from .. import items
from ...colors import rgba


logger = logging.getLogger(__name__)


class RegionOfInterest(qt.QObject):
    """Object describing a region of interest in a plot.

    :param QObject parent:
        The RegionOfInterestManager that created this object
    :param str kind: The kind of ROI represented by this object
    """

    sigControlPointsChanged = qt.Signal()
    """Signal emitted when this control points has changed"""

    def __init__(self, parent, kind):
        assert parent is None or isinstance(parent, RegionOfInterestManager)
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

    def setParent(self, parent):
        """Set the parent of the RegionOfInterest

        :param Union[None,RegionOfInterestManager] parent:
        """
        if (parent is not None and
                not isinstance(parent, RegionOfInterestManager)):
            raise ValueError('Unsupported parent')

        self._removePlotItems()
        super(RegionOfInterest, self).setParent(parent)
        self._createPlotItems()

    def getKind(self):
        """Return kind of ROI

        :rtype: str
        """
        return self._kind

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

            # Update color of shape items in the plot
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
        """Returns the label displayed for this ROI.

        :rtype: str
        """
        return self._label

    def setLabel(self, label):
        """Set the label displayed with this ROI.

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
            # Recreate plot items
            # This can be avoided once marker.setDraggable is public
            self._removePlotItems()
            self._createPlotItems()

    def getControlPoints(self):
        """Returns the current ROI control points.

        It returns an empty tuple if there is currently no ROI.

        :return: Array of (x, y) position in plot coordinates
        :rtype: numpy.ndarray
        """
        return None if self._points is None else numpy.array(self._points)

    def setControlPoints(self, points):
        """Set this ROI control points.

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
        :param str kind: The kind of ROI shape to use
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
            raise RuntimeError('Unsupported ROI kind: %s' % kind)

    def _createPlotItems(self):
        """Create items displaying the ROI in the plot."""
        roiManager = self.parent()
        if roiManager is None:
            return
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
    """Class handling ROI interaction on a PlotWidget.

    It supports the multiple ROIs: points, rectangles, polygons,
    lines, horizontal and vertical lines.

    See ``plotInteractiveImageROI.py`` sample code (:ref:`sample-code`).

    :param silx.gui.plot.PlotWidget parent:
        The plot widget in which to control the ROIs.
    """

    sigRegionOfInterestAdded = qt.Signal(RegionOfInterest)
    """Signal emitted when a new ROI has been added.

    It provides the newly add :class:`RegionOfInterest` object.
    """

    sigRegionOfInterestAboutToBeRemoved = qt.Signal(RegionOfInterest)
    """Signal emitted just before a ROI is removed.

    It provides the :class:`RegionOfInterest` object that is about to be removed.
    """

    sigRegionOfInterestChanged = qt.Signal(tuple)
    """Signal emitted whenever the ROIs have changed.

    It provides the list of ROIs.
    """

    sigInteractionModeStarted = qt.Signal(str)
    """Signal emitted when switching to ROI drawing interactive mode.

    It provides the kind of shape of the active interactive mode.
    """

    sigInteractionModeFinished = qt.Signal(tuple)
    """Signal emitted when leaving and interactive ROI drawing.

    It provides the list of ROIs.
    """

    _MODE_ACTIONS_PARAMS = collections.OrderedDict()
    # Interactive mode: (icon name, text)
    _MODE_ACTIONS_PARAMS['point'] = 'normal', 'Add point markers'
    _MODE_ACTIONS_PARAMS['rectangle'] = 'shape-rectangle', 'Add Rectangle ROI'
    _MODE_ACTIONS_PARAMS['polygon'] = 'shape-polygon', 'Add Polygon ROI'
    _MODE_ACTIONS_PARAMS['line'] = 'shape-diagonal', 'Add Line ROI'
    _MODE_ACTIONS_PARAMS['hline'] = 'shape-horizontal', 'Add Horizontal Line ROI'
    _MODE_ACTIONS_PARAMS['vline'] = 'shape-vertical', 'Add Vertical Line ROI'

    def __init__(self, parent):
        assert isinstance(parent, PlotWidget)
        super(RegionOfInterestManager, self).__init__(parent)
        self._rois = []
        self._maxROI = None

        self._shapeKind = None
        self._color = rgba('red')

        self._label = "__RegionOfInterestManager__%d" % id(self)

        self._eventLoop = None

        self._modeActions = {}

        parent.sigInteractiveModeChanged.connect(
            self._plotInteractiveModeChanged)

    @classmethod
    def getSupportedRegionOfInterestKinds(cls):
        """Returns available ROI kinds

        :rtype: List[str]
        """
        return tuple(cls._MODE_ACTIONS_PARAMS.keys())

    # Associated QActions

    def getInteractionModeAction(self, kind):
        """Returns the QAction corresponding to a kind of ROI

        The QAction allows to enable the corresponding drawing
        interactive mode.

        :param str kind: Kind of ROI
        :rtype: QAction
        :raise ValueError: If kind is not supported
        """
        if kind not in self.getSupportedRegionOfInterestKinds():
            raise ValueError('Unsupported kind %s' % kind)

        action = self._modeActions.get(kind, None)
        if action is None:  # Lazy-loading
            iconName, text = self._MODE_ACTIONS_PARAMS[kind]
            action = qt.QAction(self)
            action.setIcon(icons.getQIcon(iconName))
            action.setText(text)
            action.setCheckable(True)
            action.setChecked(self.getRegionOfInterestKind() == kind)

            action.triggered[bool].connect(functools.partial(
                WeakMethodProxy(self._modeActionTriggered), kind=kind))
            self._modeActions[kind] = action
        return action

    def _modeActionTriggered(self, checked, kind):
        """Handle mode actions being checked by the user

        :param bool checked:
        :param str kind: Corresponding shape kind
        """
        if checked:
            self.start(kind)
        else:  # Keep action checked
            action = self.sender()
            action.setChecked(True)

    def _updateModeActions(self):
        """Check/Uncheck action corresponding to current mode"""
        for kind, action in self._modeActions.items():
            action.setChecked(kind == self.getRegionOfInterestKind())

    # PlotWidget eventFilter and listeners

    def _plotInteractiveModeChanged(self, source):
        """Handle change of interactive mode in the plot"""
        if source is not self:
            self.__roiInteractiveModeEnded()

        else:  # Check the corresponding action
            self._updateModeActions()

    # Handle ROI interaction

    def _handleInteraction(self, event):
        """Handle mouse interaction for ROI addition"""
        kind = self.getRegionOfInterestKind()
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
            if self.isMaxRegionOfInterests():
                # When reaching max number of ROIs, redo last one
                rois = self.getRegionOfInterests()
                if len(rois) > 0:
                    self.removeRegionOfInterest(rois[-1])
            self.createRegionOfInterest(kind=kind, points=points)

    # RegionOfInterest API

    def getRegionOfInterestPoints(self):
        """Returns the current ROIs control points

        :return: Tuple of arrays of (x, y) points in plot coordinates
        :rtype: tuple of Nx2 numpy.ndarray
        """
        return tuple(s.getControlPoints() for s in self.getRegionOfInterests())

    def getRegionOfInterests(self):
        """Returns the list of ROIs.

        It returns an empty tuple if there is currently no ROI.

        :return: Tuple of arrays of objects describing the ROIs
        :rtype: List[RegionOfInterest]
        """
        return tuple(self._rois)

    def clearRegionOfInterests(self):
        """Reset current ROIs

        :return: True if ROIs were reset.
        :rtype: bool
        """
        if self.getRegionOfInterests():  # Something to reset
            for roi in self._rois:
                roi.sigControlPointsChanged.disconnect(
                    self._regionOfInterestPointsChanged)
                roi.setParent(None)
            self._rois = []
            self._roisUpdated()
            return True

        else:
            return False

    def _regionOfInterestPointsChanged(self):
        """Handle ROI object points changed"""
        self.sigRegionOfInterestChanged.emit(self.getRegionOfInterests())

    def getMaxRegionOfInterests(self):
        """Returns the maximum number of ROIs or None if no limit.

        :rtype: Union[int,None]
        """
        return self._maxROI

    def setMaxRegionOfInterests(self, max_):
        """Set the maximum number of ROIs.

        :param Union[int,None] max_: The max limit or None for no limit.
        :raise ValueError: If there is more ROIs than max value
        """
        if max_ is not None:
            max_ = int(max_)
            if max_ <= 0:
                raise ValueError('Max limit must be strictly positive')

            if len(self.getRegionOfInterests()) > max_:
                raise ValueError(
                    'Cannot set max limit: Already too many ROIs')

        self._maxROI = max_

    def isMaxRegionOfInterests(self):
        """Returns True if the maximum number of ROIs is reached.

        :rtype: bool
        """
        max_ = self.getMaxRegionOfInterests()
        return max_ is not None and len(self.getRegionOfInterests()) >= max_

    def createRegionOfInterest(self, kind, points, label='', index=None):
        """Create a new ROI and add it to list of ROIs.

        :param str kind: The kind of ROI to add
        :param numpy.ndarray points: The control points of the ROI shape
        :param str label: The label to display along with the ROI.
        :param int index: The position where to insert the ROI.
            By default it is appended to the end of the list.
        :return: The created ROI object
        :rtype: RegionOfInterest
        :raise RuntimeError: When ROI cannot be added because the maximum
           number of ROIs has been reached.
        """
        roi = RegionOfInterest(parent=None, kind=kind)
        roi.setColor(self.getColor())
        roi.setLabel(str(label))
        roi.setControlPoints(points)

        self.addRegionOfInterest(roi, index)

    def addRegionOfInterest(self, roi, index=None):
        """Add the ROI to the list of ROIs.

        :param RegionOfInterest roi: The ROI to add
        :param int index: The position where to insert the ROI,
            By default it is appended to the end of the list of ROIs
        :raise RuntimeError: When ROI cannot be added because the maximum
           number of ROIs has been reached.
        """
        if self.isMaxRegionOfInterests():
            raise RuntimeError(
                'Cannot add ROI: Maximum number of ROIs reached')

        plot = self.parent()
        if plot is None:
            raise RuntimeError(
                'Cannot add ROI: PlotWidget no more available')

        roi.setParent(self)
        roi.sigControlPointsChanged.connect(
            self._regionOfInterestPointsChanged)

        if index is None:
            self._rois.append(roi)
        else:
            self._rois.insert(index, roi)
        self.sigRegionOfInterestAdded.emit(roi)
        self._roisUpdated()

    def removeRegionOfInterest(self, roi):
        """Remove a ROI from the list of ROIs.

        :param RegionOfInterest roi: The ROI to remove
        :raise ValueError: When ROI does not belong to this object
        """
        if not (isinstance(roi, RegionOfInterest) and
                roi.parent() is self and
                roi in self._rois):
            raise ValueError(
                'RegionOfInterest does not belong to this instance')

        self.sigRegionOfInterestAboutToBeRemoved.emit(roi)

        self._rois.remove(roi)
        roi.sigControlPointsChanged.disconnect(
            self._regionOfInterestPointsChanged)
        roi.setParent(None)
        self._roisUpdated()

    def _roisUpdated(self):
        """Handle update of the ROI list"""
        self.sigRegionOfInterestChanged.emit(self.getRegionOfInterests())

    # RegionOfInterest parameters

    def getColor(self):
        """Return the default color of created ROIs

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._color)

    def setColor(self, color):
        """Set the default color to use when creating ROIs.

        Existing ROIs are not affected.

        :param color: The color to use for displaying ROIs as
           either a color name, a QColor, a list of uint8 or float in [0, 1].
        """
        self._color = rgba(color)

    # Control ROI

    def getRegionOfInterestKind(self):
        """Returns the current interactive ROI drawing mode or None.

        :rtype: Union[str,None]
        """
        return self._shapeKind

    def isStarted(self):
        """Returns True  if an interactive ROI drawing mode is active.

        :rtype: bool
        """
        return self._shapeKind is not None

    def start(self, kind):
        """Start an interactive ROI drawing mode.

        :param str kind: The kind of ROI shape in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :return: True if interactive ROI drawing was started, False otherwise
        :rtype: bool
        :raise ValueError: If kind is not supported
        """
        self.stop()

        plot = self.parent()
        if plot is None:
            return False

        if kind not in self.getSupportedRegionOfInterestKinds():
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

        self.sigInteractionModeStarted.emit(kind)

        return True

    def __roiInteractiveModeEnded(self):
        """Handle end of ROI draw interactive mode"""
        if self.isStarted():
            self._shapeKind = None

            plot = self.parent()
            if plot is not None:
                plot.sigPlotSignal.disconnect(self._handleInteraction)

            self._updateModeActions()

            self.sigInteractionModeFinished.emit(self.getRegionOfInterestPoints())

    def stop(self):
        """Stop interactive ROI drawing mode.

        :return: True if an interactive ROI drawing mode was actually stopped
        :rtype: bool
        """
        if not self.isStarted():
            return False

        plot = self.parent()
        if plot is not None:
            # This leads to call __roiInteractiveModeEnded through
            # interactive mode changed signal
            plot.setInteractiveMode(mode='zoom', source=None)
        else:  # Fallback
            self.__roiInteractiveModeEnded()

        return True

    def exec_(self, kind):
        """Block until :meth:`quit` is called.

        :param str kind: The kind of ROI shape in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :return: The list of ROIs
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

        rois = self.getRegionOfInterestPoints()
        self.clearRegionOfInterests()
        return rois

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

        self.sigRegionOfInterestAdded.connect(self.__added)
        self.sigRegionOfInterestAboutToBeRemoved.connect(self.__aboutToBeRemoved)
        self.sigInteractionModeStarted.connect(self.__started)
        self.sigInteractionModeFinished.connect(self.__finished)

    # Validation mode

    @ enum.unique
    class ValidationMode(enum.Enum):
        """Mode of validation to leave blocking :meth:`exec_`"""

        AUTO = 'auto'
        """Automatically ends the interactive mode once
        the user terminates the last ROI shape."""

        ENTER = 'enter'
        """Ends the interactive mode when the *Enter* key is pressed."""

        AUTO_ENTER = 'auto_enter'
        """Ends the interactive mode when reaching max ROIs or
        when the *Enter* key is pressed.
        """

        NONE = 'none'
        """Do not provide the user a way to end the interactive mode.

        The end of :meth:`exec_` is done through :meth:`quit` or timeout.
        """

    def getValidationMode(self):
        """Returns the interactive mode validation in use.

        :rtype: ValidationMode
        """
        return self.__validationMode

    def setValidationMode(self, mode):
        """Set the way to perform interactive mode validation.

        See :class:`ValidationMode` enumeration for the supported
        validation modes.

        :param ValidationMode mode: The interactive mode validation to use.
        """
        assert isinstance(mode, self.ValidationMode)
        if mode != self.__validationMode:
            self.__validationMode = mode

        if self.isExec():
            if (self.isMaxRegionOfInterests() and self.getValidationMode() in
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
                rois = self.getRegionOfInterests()
                if rois:  # Something to undo
                    self.removeRegionOfInterest(rois[-1])
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

    # Listen to ROI updates

    def __added(self, *args, **kwargs):
        """Handle new ROI added"""
        self.__updateMessage()
        if (self.isMaxRegionOfInterests() and
                self.getValidationMode() in (self.ValidationMode.AUTO,
                                             self.ValidationMode.AUTO_ENTER)):
            self.quit()

    def __aboutToBeRemoved(self, *args, **kwargs):
        """Handle removal of a ROI"""
        # RegionOfInterest not removed yet
        self.__updateMessage(nbrois=len(self.getRegionOfInterests()) - 1)

    def __started(self, *args, **kwargs):
        """Handle interactive mode started"""
        self.__updateMessage()

    def __finished(self, *args, **kwargs):
        """Handle interactive mode finished"""
        self.__updateMessage()

    def __updateMessage(self, nbrois=None):
        """Update message"""
        if not self.isExec():
            message = 'Done'

        elif not self.isStarted():
            message = 'Use %s ROI edition mode' % self.__execKind

        else:
            if nbrois is None:
                nbrois = len(self.getRegionOfInterests())

            kind = self.__execKind
            max_ = self.getMaxRegionOfInterests()

            if max_ is None:
                message = 'Select %ss (%d selected)' % (kind, nbrois)

            elif max_ <= 1:
                message = 'Select a %s' % kind
            else:
                message = 'Select %d/%d %ss' % (nbrois, max_, kind)

            if (self.getValidationMode() == self.ValidationMode.ENTER and
                    self.isMaxRegionOfInterests()):
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
        else:  # Stop interactive mode and message timer
            timer = self.sender()
            if timer is not None:
                timer.stop()
            self.__timeoutEndTime = None
            self.quit()

    def isExec(self):
        """Returns True if :meth:`exec_` is currently running.

        :rtype: bool"""
        return self.__execKind is not None

    def exec_(self, kind, timeout=0):
        """Block until ROI selection is done or timeout is elapsed.

        :meth:`quit` also ends this blocking call.

        :param str kind: The kind of ROI shape in:
           'point', 'rectangle', 'line', 'polygon', 'hline', 'vline'
        :param int timeout: Maximum duration in seconds to block.
            Default: No timeout
        :return: The list of ROIs
        :rtype: List[RegionOfInterest]
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

            rois = super(InteractiveRegionOfInterestManager, self).exec_(kind)

            timer.stop()
            self.__timeoutEndTime = None

        else:
            rois = super(InteractiveRegionOfInterestManager, self).exec_(kind)

        plot.removeEventFilter(self)

        self.__execKind = None
        self.__updateMessage()

        return rois


class _DeleteRegionOfInterestToolButton(qt.QToolButton):
    """Tool button deleting a ROI object

    :param parent: See QWidget
    :param RegionOfInterest roi: The ROI to delete
    """

    def __init__(self, parent, roi):
        super(_DeleteRegionOfInterestToolButton, self).__init__(parent)
        self.setIcon(icons.getQIcon('remove'))
        self.__roiRef = roi if roi is None else weakref.ref(roi)
        self.clicked.connect(self.__clicked)

    def __clicked(self, checked):
        """Handle button clicked"""
        roi = None if self.__roiRef is None else self.__roiRef()
        if roi is not None:
            manager = roi.parent()
            if manager is not None:
                manager.removeRegionOfInterest(roi)
                self.__roiRef = None


class RegionOfInterestTableWidget(qt.QTableWidget):
    """Widget displaying the ROIs of a :class:`RegionOfInterestManager`"""

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
        roi = item.data(qt.Qt.UserRole)
        if column == 0:
            roi.setLabel(item.text())
        elif column == 1:
            roi.setEditable(
                item.checkState() == qt.Qt.Checked)
        elif column in (2, 3, 4):
            pass  # TODO
        else:
            logger.error('Unhandled column %d', column)

    def setRegionOfInterestManager(self, manager):
        """Set the :class:`RegionOfInterestManager` object to sync with

        :param RegionOfInterestManager manager:
        """
        assert manager is None or isinstance(manager, RegionOfInterestManager)

        previousManager = self.getRegionOfInterestManager()

        if previousManager is not None:
            previousManager.sigRegionOfInterestChanged.disconnect(self._sync)
        self.setRowCount(0)

        self._roiManagerRef = weakref.ref(manager)

        self._sync()

        if manager is not None:
            manager.sigRegionOfInterestChanged.connect(self._sync)

    def _sync(self, *args):
        """Update widget content according to ROI manger"""
        manager = self.getRegionOfInterestManager()

        if manager is None:
            self.setRowCount(0)
            return

        rois = manager.getRegionOfInterests()

        self.setRowCount(len(rois))
        for index, roi in enumerate(rois):
            baseFlags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled

            # Label
            label = roi.getLabel()
            item = qt.QTableWidgetItem(label)
            item.setFlags(baseFlags | qt.Qt.ItemIsEditable)
            item.setData(qt.Qt.UserRole, roi)
            self.setItem(index, 0, item)

            # Editable
            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags | qt.Qt.ItemIsUserCheckable)
            item.setData(qt.Qt.UserRole, roi)
            item.setCheckState(
                qt.Qt.Checked if roi.isEditable() else qt.Qt.Unchecked)
            self.setItem(index, 1, item)

            # Delete
            delBtn = _DeleteRegionOfInterestToolButton(None, roi)

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
            kind = roi.getKind()
            item = qt.QTableWidgetItem(kind.capitalize())
            item.setFlags(baseFlags)
            self.setItem(index, 3, item)

            item = qt.QTableWidgetItem()
            item.setFlags(baseFlags)

            # Coordinates
            points = roi.getControlPoints()
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
