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
"""This module provides ROI item for the :class:`~silx.gui.plot.PlotWidget`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/06/2018"


import functools
import itertools
import logging
import collections
import numpy

from ....utils.weakref import WeakList
from ... import qt
from .. import items
from ...colors import rgba


logger = logging.getLogger(__name__)


class RegionOfInterest(qt.QObject):
    """Object describing a region of interest in a plot.

    :param QObject parent:
        The RegionOfInterestManager that created this object
    """

    _kind = None
    """Label for this kind of ROI.

    Should be setted by inherited classes to custom the ROI manager widget.
    """

    sigRegionChanged = qt.Signal()
    """Signal emitted everytime the shape or position of the ROI changes"""

    def __init__(self, parent=None):
        # Avoid circular dependancy
        from ..tools import roi as roi_tools
        assert parent is None or isinstance(parent, roi_tools.RegionOfInterestManager)
        super(RegionOfInterest, self).__init__(parent)
        self._color = rgba('red')
        self._items = WeakList()
        self._editAnchors = WeakList()
        self._points = None
        self._label = ''
        self._labelItem = None
        self._editable = False

    def __del__(self):
        # Clean-up plot items
        self._removePlotItems()

    def setParent(self, parent):
        """Set the parent of the RegionOfInterest

        :param Union[None,RegionOfInterestManager] parent:
        """
        # Avoid circular dependancy
        from ..tools import roi as roi_tools
        if (parent is not None and not isinstance(parent, roi_tools.RegionOfInterestManager)):
            raise ValueError('Unsupported parent')

        self._removePlotItems()
        super(RegionOfInterest, self).setParent(parent)
        self._createPlotItems()

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

    def _getAnchorColor(self, color):
        """Returns the anchor color from the base ROI  color

        :param Union[numpy.array,Tuple,List]: color
        :rtype: Union[numpy.array,Tuple,List]
        """
        return color[:3] + (0.5,)

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
            item = self._getLabelItem()
            if isinstance(item, items.ColorMixIn):
                item.setColor(rgbaColor)

            rgbaColor = self._getAnchorColor(rgbaColor)
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
            self._updateLabelItem(label)

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
            self._createPlotItems()

    def _getControlPoints(self):
        """Returns the current ROI control points.

        It returns an empty tuple if there is currently no ROI.

        :return: Array of (x, y) position in plot coordinates
        :rtype: numpy.ndarray
        """
        return None if self._points is None else numpy.array(self._points)

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

        This interaction is constains by the plot API and only supports few
        shapes.
        """
        points = self._createControlPointsFromFirstShape(points)
        self._setControlPoints(points)

    def _createControlPointsFromFirstShape(self, points):
        """Returns the list of control points from the very first shape
        provided.

        This shape is provided by the plot interaction and constained by the
        class of the ROI itself.
        """
        return points

    def _setControlPoints(self, points):
        """Set this ROI control points.

        :param points: Iterable of (x, y) control points
        """
        points = numpy.array(points)

        nbPointsChanged = (self._points is None or
                           points.shape != self._points.shape)

        if nbPointsChanged or not numpy.all(numpy.equal(points, self._points)):
            self._points = points

            self._updateShape()
            if self._items and not nbPointsChanged:  # Update plot items
                item = self._getLabelItem()
                if item is not None:
                    markerPos = self._getLabelPosition()
                    item.setPosition(*markerPos)

                if self._editAnchors:  # Update anchors
                    for anchor, point in zip(self._editAnchors, points):
                        old = anchor.blockSignals(True)
                        anchor.setPosition(*point)
                        anchor.blockSignals(old)

            else:  # No items or new point added
                # re-create plot items
                self._createPlotItems()

            self.sigRegionChanged.emit()

    def _updateShape(self):
        """Called when shape must be updated.

        Must be reimplemented if a shape item have to be updated.
        """
        return

    def _getLabelPosition(self):
        """Compute position of the label

        :return: (x, y) position of the marker
        """
        return None

    def _createPlotItems(self):
        """Create items displaying the ROI in the plot.

        It first removes any existing plot items.
        """
        roiManager = self.parent()
        if roiManager is None:
            return
        plot = roiManager.parent()

        self._removePlotItems()

        legendPrefix = "__RegionOfInterest-%d__" % id(self)
        itemIndex = 0

        controlPoints = self._getControlPoints()

        if self._labelItem is None:
            self._labelItem = self._createLabelItem()
            if self._labelItem is not None:
                self._labelItem._setLegend(legendPrefix + "label")
                plot._add(self._labelItem)

        self._items = WeakList()
        plotItems = self._createShapeItems(controlPoints)
        for item in plotItems:
            item._setLegend(legendPrefix + str(itemIndex))
            plot._add(item)
            self._items.append(item)
            itemIndex += 1

        self._editAnchors = WeakList()
        if self.isEditable():
            plotItems = self._createAnchorItems(controlPoints)
            color = rgba(self.getColor())
            color = self._getAnchorColor(color)
            for index, item in enumerate(plotItems):
                item._setLegend(legendPrefix + str(itemIndex))
                item.setColor(color)
                plot._add(item)
                item.sigItemChanged.connect(functools.partial(
                    self._controlPointAnchorChanged, index))
                self._editAnchors.append(item)
                itemIndex += 1

    def _updateLabelItem(self, label):
        """Update the marker displaying the label.

        Inherite this method to custom the way the ROI display the label.

        :param str label: The new label to use
        """
        item = self._getLabelItem()
        if item is not None:
            item.setText(label)

    def _createLabelItem(self):
        """Returns a created marker which will be used to dipslay the label of
        this ROI.

        Inherite this method to return nothing if no new items have to be
        created, or your own marker.

        :rtype: Union[None,Marker]
        """
        # Add label marker
        markerPos = self._getLabelPosition()
        marker = items.Marker()
        marker.setPosition(*markerPos)
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker.setSymbol('')
        marker._setDraggable(False)
        return marker

    def _getLabelItem(self):
        """Returns the marker displaying the label of this ROI.

        Inherite this method to choose your own item. In case this item is also
        a control point.
        """
        return self._labelItem

    def _createShapeItems(self, points):
        """Create shape items from the current control points.

        :rtype: List[PlotItem]
        """
        return []

    def _createAnchorItems(self, points):
        """Create anchor items from the current control points.

        :rtype: List[Marker]
        """
        return []

    def _controlPointAnchorChanged(self, index, event):
        """Handle update of position of an edition anchor

        :param int index: Index of the anchor
        :param ItemChangedType event: Event type
        """
        if event == items.ItemChangedType.POSITION:
            anchor = self._editAnchors[index]
            previous = self._points[index].copy()
            current = anchor.getPosition()
            self._controlPointAnchorPositionChanged(index, current, previous)

    def _controlPointAnchorPositionChanged(self, index, current, previous):
        """Called when an anchor is manually edited.

        This function have to be inherited to change the behaviours of the
        control points. This function have to call :meth:`_getControlPoints` to
        reach the previous state of the control points. Updated the positions
        of the changed control points. Then call :meth:`_setControlPoints` to
        update the anchors and send signals.
        """
        points = self._getControlPoints()
        points[index] = current
        self._setControlPoints(points)

    def _removePlotItems(self):
        """Remove items from their plot."""
        for item in itertools.chain(list(self._items),
                                    list(self._editAnchors)):

            plot = item.getPlot()
            if plot is not None:
                plot._remove(item)
        self._items = WeakList()
        self._editAnchors = WeakList()

        if self._labelItem is not None:
            item = self._labelItem
            plot = item.getPlot()
            if plot is not None:
                plot._remove(item)
        self._labelItem = None

    def __str__(self):
        """Returns parameters of the ROI as a string."""
        points = self._getControlPoints()
        params = '; '.join('(%f; %f)' % (pt[0], pt[1]) for pt in points)
        return "%s(%s)" % (self.__class__.__name__, params)


class PointROI(RegionOfInterest):
    """A ROI identifying a point in a 2D plot."""

    _kind = "Point"
    """Label for this kind of ROI"""

    _plotShape = "point"
    """Plot shape which is used for the first interaction"""

    def getPosition(self):
        """Returns the position of this ROI

        :rtype: numpy.ndarray
        """
        return self._points[0].copy()

    def setPosition(self, pos):
        """Set the position of this ROI

        :param numpy.ndarray pos: 2d-coordinate of this point
        """
        controlPoints = numpy.array([pos])
        self._setControlPoints(controlPoints)

    def _createLabelItem(self):
        return None

    def _updateLabelItem(self, label):
        if self.isEditable():
            item = self._editAnchors[0]
        else:
            item = self._items[0]
        item.setText(label)

    def _createShapeItems(self, points):
        if self.isEditable():
            return []
        marker = items.Marker()
        marker.setPosition(points[0][0], points[0][1])
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker._setDraggable(False)
        return [marker]

    def _createAnchorItems(self, points):
        marker = items.Marker()
        marker.setPosition(points[0][0], points[0][1])
        marker.setText(self.getLabel())
        marker._setDraggable(self.isEditable())
        return [marker]

    def __str__(self):
        points = self._getControlPoints()
        params = '%f %f' % (points[0, 0], points[0, 1])
        return "%s(%s)" % (self.__class__.__name__, params)


class LineROI(RegionOfInterest):
    """A ROI identifying a line in a 2D plot.

    This ROI provides 1 anchor for each boundary of the line, plus an center
    in the center to translate the full ROI.
    """

    _kind = "Line"
    """Label for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    def _createControlPointsFromFirstShape(self, points):
        center = numpy.mean(points, axis=0)
        controlPoints = numpy.array([points[0], points[1], center])
        return controlPoints

    def setEndPoints(self, startPoint, endPoint):
        """Set this line location using the endding points

        :param numpy.ndarray startPoint: Staring bounding point of the line
        :param numpy.ndarray endPoint: Endding bounding point of the line
        """
        assert(startPoint.shape == (2,) and endPoint.shape == (2,))
        shapePoints = numpy.array([startPoint, endPoint])
        controlPoints = self._createControlPointsFromFirstShape(shapePoints)
        self._setControlPoints(controlPoints)

    def getEndPoints(self):
        """Returns bounding points of this ROI.

        :rtype: Tuple(numpy.ndarray,numpy.ndarray)
        """
        startPoint = self._points[0].copy()
        endPoint = self._points[1].copy()
        return (startPoint, endPoint)

    def _getLabelPosition(self):
        points = self._getControlPoints()
        return points[-1]

    def _updateShape(self):
        if len(self._items) == 0:
            return
        shape = self._items[0]
        points = self._getControlPoints()
        points = self._getShapeFromControlPoints(points)
        shape.setPoints(points)

    def _getShapeFromControlPoints(self, points):
        # Remove the center from the control points
        return points[0:2]

    def _createShapeItems(self, points):
        shapePoints = self._getShapeFromControlPoints(points)
        item = items.Shape("polylines")
        item.setPoints(shapePoints)
        item.setColor(rgba(self.getColor()))
        item.setFill(False)
        item.setOverlay(True)
        return [item]

    def _createAnchorItems(self, points):
        anchors = []
        for point in points[0:-1]:
            anchor = items.Marker()
            anchor.setPosition(*point)
            anchor.setText('')
            anchor.setSymbol('s')
            anchor._setDraggable(True)
            anchors.append(anchor)

        # Add an anchor to the center of the rectangle
        center = numpy.mean(points, axis=0)
        anchor = items.Marker()
        anchor.setPosition(*center)
        anchor.setText('')
        anchor.setSymbol('+')
        anchor._setDraggable(True)
        anchors.append(anchor)

        return anchors

    def _controlPointAnchorPositionChanged(self, index, current, previous):
        if index == len(self._editAnchors) - 1:
            # It is the center anchor
            points = self._getControlPoints()
            center = numpy.mean(points[0:-1], axis=0)
            offset = current - previous
            points[-1] = current
            points[0:-1] = points[0:-1] + offset
            self._setControlPoints(points)
        else:
            # Update the center
            points = self._getControlPoints()
            points[index] = current
            center = numpy.mean(points[0:-1], axis=0)
            points[-1] = center
            self._setControlPoints(points)

    def __str__(self):
        points = self._getControlPoints()
        params = points[0][0], points[0][1], points[1][0], points[1][1]
        params = 'start: %f %f; end: %f %f' % params
        return "%s(%s)" % (self.__class__.__name__, params)


class HorizontalLineROI(RegionOfInterest):
    """A ROI identifying an horizontal line in a 2D plot."""

    _kind = "HLine"
    """Label for this kind of ROI"""

    _plotShape = "hline"
    """Plot shape which is used for the first interaction"""

    def _createControlPointsFromFirstShape(self, points):
        points = numpy.array([(float('nan'), points[0, 1])],
                             dtype=numpy.float64)
        return points

    def getPosition(self):
        """Returns the position of this line if the horizontal axis

        :rtype: float
        """
        return self._points[0, 1]

    def setPosition(self, pos):
        """Set the position of this ROI

        :param float pos: Horizontal position of this line
        """
        controlPoints = numpy.array([[float('nan'), pos]])
        self._setControlPoints(controlPoints)

    def _createLabelItem(self):
        return None

    def _updateLabelItem(self, label):
        if self.isEditable():
            item = self._editAnchors[0]
        else:
            item = self._items[0]
        item.setText(label)

    def _updateShape(self):
        if not self.isEditable():
            if len(self._items) > 0:
                controlPoints = self._getControlPoints()
                item = self._items[0]
                item.setPosition(*controlPoints[0])

    def _createShapeItems(self, points):
        if self.isEditable():
            return []
        marker = items.YMarker()
        marker.setPosition(points[0][0], points[0][1])
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker._setDraggable(False)
        return [marker]

    def _createAnchorItems(self, points):
        marker = items.YMarker()
        marker.setPosition(points[0][0], points[0][1])
        marker.setText(self.getLabel())
        marker._setDraggable(self.isEditable())
        return [marker]

    def __str__(self):
        points = self._getControlPoints()
        params = 'y: %f' % points[0, 1]
        return "%s(%s)" % (self.__class__.__name__, params)


class VerticalLineROI(RegionOfInterest):
    """A ROI identifying a vertical line in a 2D plot."""

    _kind = "VLine"
    """Label for this kind of ROI"""

    _plotShape = "vline"
    """Plot shape which is used for the first interaction"""

    def _createControlPointsFromFirstShape(self, points):
        points = numpy.array([(points[0, 0], float('nan'))],
                             dtype=numpy.float64)
        return points

    def getPosition(self):
        """Returns the position of this line if the horizontal axis

        :rtype: float
        """
        return self._points[0, 0]

    def setPosition(self, pos):
        """Set the position of this ROI

        :param float pos: Horizontal position of this line
        """
        controlPoints = numpy.array([[pos, float('nan')]])
        self._setControlPoints(controlPoints)

    def _createLabelItem(self):
        return None

    def _updateLabelItem(self, label):
        if self.isEditable():
            item = self._editAnchors[0]
        else:
            item = self._items[0]
        item.setText(label)

    def _updateShape(self):
        if not self.isEditable():
            if len(self._items) > 0:
                controlPoints = self._getControlPoints()
                item = self._items[0]
                item.setPosition(*controlPoints[0])

    def _createShapeItems(self, points):
        if self.isEditable():
            return []
        marker = items.XMarker()
        marker.setPosition(points[0][0], points[0][1])
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker._setDraggable(False)
        return [marker]

    def _createAnchorItems(self, points):
        marker = items.XMarker()
        marker.setPosition(points[0][0], points[0][1])
        marker.setText(self.getLabel())
        marker._setDraggable(self.isEditable())
        return [marker]

    def __str__(self):
        points = self._getControlPoints()
        params = 'x: %f' % points[0, 0]
        return "%s(%s)" % (self.__class__.__name__, params)


class RectangleROI(RegionOfInterest):
    """A ROI identifying a rectangle in a 2D plot.

    This ROI provides 1 anchor for each corner, plus an anchor in the
    center to translate the full ROI.
    """

    _kind = "Rectangle"
    """Label for this kind of ROI"""

    _plotShape = "rectangle"
    """Plot shape which is used for the first interaction"""

    def _createControlPointsFromFirstShape(self, points):
        point0 = points[0]
        point1 = points[1]

        # 4 corners
        controlPoints = numpy.array([
            point0[0], point0[1],
            point0[0], point1[1],
            point1[0], point1[1],
            point1[0], point0[1],
        ])
        # Central
        center = numpy.mean(points, axis=0)
        controlPoints = numpy.append(controlPoints, center)
        controlPoints.shape = -1, 2
        return controlPoints

    def getCenter(self):
        """Returns the central point of this rectangle

        :rtype: numpy.ndarray([float,float])
        """
        return numpy.mean(self._points, axis=0)

    def getOrigin(self):
        """Returns the corner point with the smaller coordinates

        :rtype: numpy.ndarray([float,float])
        """
        return numpy.min(self._points, axis=0)

    def getSize(self):
        """Returns the size of this rectangle

        :rtype: numpy.ndarray([float,float])
        """
        minPoint = numpy.min(self._points, axis=0)
        maxPoint = numpy.max(self._points, axis=0)
        return maxPoint - minPoint

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
            controlPoints = self._createControlPointsFromFirstShape(points)
        elif center is not None:
            center = numpy.array(center)
            size = numpy.array(size)
            points = numpy.array([center - size * 0.5, center + size * 0.5])
            controlPoints = self._createControlPointsFromFirstShape(points)
        else:
            raise ValueError("Origin or cengter expected")
        self._setControlPoints(controlPoints)

    def _getLabelPosition(self):
        points = self._getControlPoints()
        return points.min(axis=0)

    def _updateShape(self):
        if len(self._items) == 0:
            return
        shape = self._items[0]
        points = self._getControlPoints()
        points = self._getShapeFromControlPoints(points)
        shape.setPoints(points)

    def _getShapeFromControlPoints(self, points):
        minPoint = points.min(axis=0)
        maxPoint = points.max(axis=0)
        return numpy.array([minPoint, maxPoint])

    def _createShapeItems(self, points):
        shapePoints = self._getShapeFromControlPoints(points)
        item = items.Shape("rectangle")
        item.setPoints(shapePoints)
        item.setColor(rgba(self.getColor()))
        item.setFill(False)
        item.setOverlay(True)
        return [item]

    def _createAnchorItems(self, points):
        # Remove the center control point
        points = points[0:-1]

        anchors = []
        for point in points:
            anchor = items.Marker()
            anchor.setPosition(*point)
            anchor.setText('')
            anchor.setSymbol('s')
            anchor._setDraggable(True)
            anchors.append(anchor)

        # Add an anchor to the center of the rectangle
        center = numpy.mean(points, axis=0)
        anchor = items.Marker()
        anchor.setPosition(*center)
        anchor.setText('')
        anchor.setSymbol('+')
        anchor._setDraggable(True)
        anchors.append(anchor)

        return anchors

    def _controlPointAnchorPositionChanged(self, index, current, previous):
        if index == len(self._editAnchors) - 1:
            # It is the center anchor
            points = self._getControlPoints()
            center = numpy.mean(points[0:-1], axis=0)
            offset = current - previous
            points[-1] = current
            points[0:-1] = points[0:-1] + offset
            self._setControlPoints(points)
        else:
            # Fix other corners
            constrains = [(1, 3), (0, 2), (3, 1), (2, 0)]
            constrains = constrains[index]
            points = self._getControlPoints()
            points[index] = current
            points[constrains[0]][0] = current[0]
            points[constrains[1]][1] = current[1]
            # Update the center
            center = numpy.mean(points[0:-1], axis=0)
            points[-1] = center
            self._setControlPoints(points)

    def __str__(self):
        origin = self.getOrigin()
        w, h = self.getSize()
        params = origin[0], origin[1], w, h
        params = 'origin: %f %f; width: %f; height: %f' % params
        return "%s(%s)" % (self.__class__.__name__, params)


class PolygonROI(RegionOfInterest):
    """A ROI identifying a closed polygon in a 2D plot.

    This ROI provides 1 anchor for each point of the polygon.
    """

    _kind = "Polygon"
    """Label for this kind of ROI"""

    _plotShape = "polygon"
    """Plot shape which is used for the first interaction"""

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
        if len(points) > 0:
            controlPoints = numpy.array(points)
        else:
            controlPoints = numpy.empty((0, 2))
        self._setControlPoints(controlPoints)

    def _getLabelPosition(self):
        points = self._getControlPoints()
        if len(points) == 0:
            # FIXME: we should return none, this polygon have no location
            return numpy.array([0, 0])
        return points[numpy.argmin(points[:, 1])]

    def _updateShape(self):
        if len(self._items) == 0:
            return
        shape = self._items[0]
        points = self._getControlPoints()
        shape.setPoints(points)

    def _createShapeItems(self, points):
        if len(points) == 0:
            return []
        else:
            item = items.Shape("polygon")
            item.setPoints(points)
            item.setColor(rgba(self.getColor()))
            item.setFill(False)
            item.setOverlay(True)
            return [item]

    def _createAnchorItems(self, points):
        anchors = []
        for point in points:
            anchor = items.Marker()
            anchor.setPosition(*point)
            anchor.setText('')
            anchor.setSymbol('s')
            anchor._setDraggable(True)
            anchors.append(anchor)
        return anchors

    def __str__(self):
        points = self._getControlPoints()
        params = '; '.join('%f %f' % (pt[0], pt[1]) for pt in points)
        return "%s(%s)" % (self.__class__.__name__, params)


class ArcROI(RegionOfInterest):
    """A ROI identifying an arc of a circle with a width.

    This ROI provides 3 anchors to control the curvature, 1 anchor to control
    the weigth, and 1 anchor to translate the shape.
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
        RegionOfInterest.__init__(self, parent=parent)
        self._geometry = None

    def _getInternalGeometry(self):
        """Returns the object storing the internal geometry of this ROI.

        This geometry is derived from the control points and cached for
        efficiency. Calling :meth:`_setControlPoints` invalidate the cache.
        """
        if self._geometry is None:
            controlPoints = self._getControlPoints()
            self._geometry = self._createGeometryFromControlPoint(controlPoints)
        return self._geometry

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def _getLabelPosition(self):
        points = self._getControlPoints()
        return points.min(axis=0)

    def _updateShape(self):
        if len(self._items) == 0:
            return
        shape = self._items[0]
        points = self._getControlPoints()
        points = self._getShapeFromControlPoints(points)
        shape.setPoints(points)

    def _controlPointAnchorPositionChanged(self, index, current, previous):
        controlPoints = self._getControlPoints()
        currentWeigth = numpy.linalg.norm(controlPoints[3] - controlPoints[1]) * 2

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
        else:
            super(ArcROI, self)._controlPointAnchorPositionChanged(index, current, previous)

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
        RegionOfInterest._setControlPoints(self, points)

    def getGeometry(self):
        """Returns a tuple containing the geometry of this ROI

        It is a symetric fonction of :meth:`setGeometry`.

        If `startAngle` is smaller than `endAngle` the rotation is clockwise,
        else the rotation is anticlockwise.

        :rtype: Tuple[numpy.ndarray,float,float,float,float]
        :raise ValueError: In case the ROI can't be representaed as section of
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

    def _createShapeItems(self, points):
        shapePoints = self._getShapeFromControlPoints(points)
        item = items.Shape("polygon")
        item.setPoints(shapePoints)
        item.setColor(rgba(self.getColor()))
        item.setFill(False)
        item.setOverlay(True)
        return [item]

    def _createAnchorItems(self, points):
        anchors = []
        symbols = ['o', 'o', 'o', 's']

        for index, point in enumerate(points):
            if index in [1, 3]:
                constraint = self._arcCurvatureMarkerConstraint
            else:
                constraint = None
            anchor = items.Marker()
            anchor.setPosition(*point)
            anchor.setText('')
            anchor.setSymbol(symbols[index])
            anchor._setDraggable(True)
            if constraint is not None:
                anchor._setConstraint(constraint)
            anchors.append(anchor)

        return anchors

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
