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

This API is not mature and will probably change in the future.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/06/2018"


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
    :param str kind: The kind of ROI represented by this object
    """

    sigControlPointsChanged = qt.Signal()
    """Signal emitted when this control points has changed"""

    def __init__(self, parent=None):
        # FIXME: Not very elegant: It checks class name to avoid recursive loop
        assert parent is None or "RegionOfInterestManager" in parent.__class__.__name__
        super(RegionOfInterest, self).__init__(parent)
        self._color = rgba('red')
        self._items = WeakList()
        self._editAnchors = WeakList()
        self._points = None
        self._label = ''
        self._editable = False

    def __del__(self):
        # Clean-up plot items
        self._removePlotItems()

    def setParent(self, parent):
        """Set the parent of the RegionOfInterest

        :param Union[None,RegionOfInterestManager] parent:
        """
        # FIXME: Not very elegant: It checks class name to avoid recursive loop
        if (parent is not None and "RegionOfInterestManager" not in parent.__class__.__name__):
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
            self._createPlotItems()

    def getControlPoints(self):
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
        self.setControlPoints(points)

    def _createControlPointsFromFirstShape(self, points):
        """Returns the list of control points from the very first shape
        provided.

        This shape is provided by the plot interaction and constained by the
        class of the ROI itself.
        """
        return points

    def setControlPoints(self, points):
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
                for item in self._items:
                    if isinstance(item, (items.Marker,
                                         items.XMarker,
                                         items.YMarker)):
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

            self.sigControlPointsChanged.emit()

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

        self._items = WeakList()
        controlPoints = self.getControlPoints()

        plotItems = self._createShapeItems(controlPoints)
        for item in plotItems:
            item._setLegend(legendPrefix + str(itemIndex))
            plot._add(item)
            self._items.append(item)
            itemIndex += 1

        self._editAnchors = WeakList()
        if self.isEditable():
            plotItems = self._createAnchorItems(controlPoints)
            for index, item in enumerate(plotItems):
                item._setLegend(legendPrefix + str(itemIndex))
                item.setColor(rgba(self.getColor()))
                plot._add(item)
                item.sigItemChanged.connect(functools.partial(
                    self._controlPointAnchorChanged, index))
                self._editAnchors.append(item)
                itemIndex += 1

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
        control points. This function have to call :meth:`getControlPoints` to
        reach the previous state of the control points. Updated the positions
        of the changed control points. Then call :meth:`setControlPoints` to
        update the anchors and send signals.
        """
        points = self.getControlPoints()
        points[index] = current
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

    def paramsToString(self):
        """Returns parameters of the ROI as a string."""
        points = self.getControlPoints()
        return '; '.join('(%f; %f)' % (pt[0], pt[1]) for pt in points)


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
        self.setControlPoints(controlPoints)

    def _getLabelPosition(self):
        points = self.getControlPoints()
        return points[0]

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

    def paramsToString(self):
        points = self.getControlPoints()
        return '(%f; %f)' % (points[0, 0], points[0, 1])


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
        shapePoints = numpy.array([startPoint, endPoint])
        controlPoints = self._createControlPointsFromFirstShape(shapePoints)
        self.setControlPoints(controlPoints)

    def getEndPoints(self):
        """Returns bounding points of this ROI.

        :rtype: Tuple(numpy.ndarray,numpy.ndarray)
        """
        startPoint = self._points[0].copy()
        endPoint = self._points[1].copy()
        return (startPoint, endPoint)

    def _getLabelPosition(self):
        points = self.getControlPoints()
        return points[-1]

    def _updateShape(self):
        if len(self._items) == 0:
            return
        shape = self._items[0]
        points = self.getControlPoints()
        points = self._getShapeFromControlPoints(points)
        shape.setPoints(points)

    def _getShapeFromControlPoints(self, points):
        # Remove the center from the control points
        return points[0:2]

    def _createShapeItems(self, points):
        # Add label marker
        markerPos = self._getLabelPosition()
        marker = items.Marker()
        marker.setPosition(*markerPos)
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker.setSymbol('')
        marker._setDraggable(False)

        shapePoints = self._getShapeFromControlPoints(points)
        item = items.Shape("polylines")
        item.setPoints(shapePoints)
        item.setColor(rgba(self.getColor()))
        item.setFill(False)
        item.setOverlay(True)
        return [item, marker]

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
            points = self.getControlPoints()
            center = numpy.mean(points[0:-1], axis=0)
            offset = current - previous
            points[-1] = current
            points[0:-1] = points[0:-1] + offset
            self.setControlPoints(points)
        else:
            # Update the center
            points = self.getControlPoints()
            points[index] = current
            center = numpy.mean(points[0:-1], axis=0)
            points[-1] = center
            self.setControlPoints(points)

    def paramsToString(self):
        points = self.getControlPoints()
        return '; '.join('(%f; %f)' % (pt[0], pt[1]) for pt in points[0:2])


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
        print self._points
        return self._points[0, 1]

    def setPosition(self, pos):
        """Set the position of this ROI

        :param float pos: Horizontal position of this line
        """
        controlPoints = numpy.array([[float('nan'), pos]])
        self.setControlPoints(controlPoints)

    def _getLabelPosition(self):
        points = self.getControlPoints()
        return points[0]

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

    def paramsToString(self):
        points = self.getControlPoints()
        return 'Y: %f' % points[0, 1]


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
        self.setControlPoints(controlPoints)

    def _getLabelPosition(self):
        points = self.getControlPoints()
        return points[0]

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

    def paramsToString(self):
        points = self.getControlPoints()
        return 'X: %f' % points[0, 0]


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
        self.setControlPoints(controlPoints)

    def _getLabelPosition(self):
        points = self.getControlPoints()
        return points.min(axis=0)

    def _updateShape(self):
        if len(self._items) == 0:
            return
        shape = self._items[0]
        points = self.getControlPoints()
        points = self._getShapeFromControlPoints(points)
        shape.setPoints(points)

    def _getShapeFromControlPoints(self, points):
        minPoint = points.min(axis=0)
        maxPoint = points.max(axis=0)
        return numpy.array([minPoint, maxPoint])

    def _createShapeItems(self, points):
        # Add label marker
        markerPos = self._getLabelPosition()
        marker = items.Marker()
        marker.setPosition(*markerPos)
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker.setSymbol('')
        marker._setDraggable(False)

        shapePoints = self._getShapeFromControlPoints(points)
        item = items.Shape("rectangle")
        item.setPoints(shapePoints)
        item.setColor(rgba(self.getColor()))
        item.setFill(False)
        item.setOverlay(True)
        return [item, marker]

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
            points = self.getControlPoints()
            center = numpy.mean(points[0:-1], axis=0)
            offset = current - previous
            points[-1] = current
            points[0:-1] = points[0:-1] + offset
            self.setControlPoints(points)
        else:
            # Fix other corners
            constrains = [(1, 3), (0, 2), (3, 1), (2, 0)]
            constrains = constrains[index]
            points = self.getControlPoints()
            points[index] = current
            points[constrains[0]][0] = current[0]
            points[constrains[1]][1] = current[1]
            # Update the center
            center = numpy.mean(points[0:-1], axis=0)
            points[-1] = center
            self.setControlPoints(points)

    def paramsToString(self):
        origin = self.getOrigin()
        w, h = self.getSize()
        return ('Origin: (%f; %f); Width: %f; Height: %f' %
                (origin[0], origin[1], w, h))


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
        if len(points) > 0:
            controlPoints = numpy.array(points)
        else:
            controlPoints = numpy.empty((0, 2))
        self.setControlPoints(controlPoints)

    def _getLabelPosition(self):
        points = self.getControlPoints()
        return points[numpy.argmin(points[:, 1])]

    def _updateShape(self):
        if len(self._items) == 0:
            return
        shape = self._items[0]
        points = self.getControlPoints()
        shape.setPoints(points)

    def _createShapeItems(self, points):
        # Add label marker
        markerPos = self._getLabelPosition()
        marker = items.Marker()
        marker.setPosition(*markerPos)
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker.setSymbol('')
        marker._setDraggable(False)

        item = items.Shape("polygon")
        item.setPoints(points)
        item.setColor(rgba(self.getColor()))
        item.setFill(False)
        item.setOverlay(True)
        return [item, marker]

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

    def paramsToString(self):
        points = self.getControlPoints()
        return '; '.join('(%f; %f)' % (pt[0], pt[1]) for pt in points)


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

    @classmethod
    def showFirstInteractionShape(cls):
        return False

    def _getLabelPosition(self):
        points = self.getControlPoints()
        return points.min(axis=0)

    def _updateShape(self):
        if len(self._items) == 0:
            return
        shape = self._items[0]
        points = self.getControlPoints()
        points = self._getShapeFromControlPoints(points)
        shape.setPoints(points)

    def _controlPointAnchorPositionChanged(self, index, current, previous):
        controlPoints = self.getControlPoints()
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
            # FIXME: take care of division by 0
            normal /= distance
            midVector = controlPoints[1] - center
            # Coeficient which have to be constrained
            constainedCoef = numpy.dot(midVector, normal) / distance

            # Compute the location of the curvature point
            controlPoints[index] = current
            startPoint = controlPoints[0]
            endPoint = controlPoints[2]
            center = (startPoint + endPoint) * 0.5
            normal = (endPoint - startPoint)
            normal = numpy.array((normal[1], -normal[0]))
            distance = numpy.linalg.norm(normal)
            # FIXME: take care of division by 0
            normal /= distance
            midPoint = center + normal * constainedCoef * distance
            controlPoints[1] = midPoint

            # The weight have to be fixed
            self._updateWeightControlPoint(controlPoints, currentWeigth)
            self.setControlPoints(controlPoints)

        elif index == 1:
            # The weight have to be fixed
            controlPoints[index] = current
            self._updateWeightControlPoint(controlPoints, currentWeigth)
            self.setControlPoints(controlPoints)
        else:
            super(ArcROI, self)._controlPointAnchorPositionChanged(index, current, previous)

    def _updateWeightControlPoint(self, controlPoints, weigth):
        startPoint = controlPoints[0]
        midPoint = controlPoints[1]
        endPoint = controlPoints[2]
        normal = (endPoint - startPoint)
        normal = numpy.array((normal[1], -normal[0]))
        distance = numpy.linalg.norm(normal)
        # FIXME: take care of division by 0
        normal /= distance
        controlPoints[3] = midPoint + normal * weigth * 0.5

    def _getGeometryFromControlPoint(self, controlPoints):
        """Returns the geometry of the object"""
        weigth = numpy.linalg.norm(controlPoints[3] - controlPoints[1]) * 2
        if numpy.linalg.norm(
            # Colinear
            numpy.cross(controlPoints[1] - controlPoints[0],
                        controlPoints[2] - controlPoints[0])) < 1e-5:
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

    def _getShapeFromControlPoints(self, controlPoints):
        geometry = self._getGeometryFromControlPoint(controlPoints)
        if geometry.center is None:
            # It is not an arc
            # but we can display it as an the intermediat shape
            normal = (geometry.endPoint - geometry.startPoint)
            normal = numpy.array((normal[1], -normal[0]))
            normal /= numpy.linalg.norm(normal)
            points = numpy.array([
                geometry.startPoint + normal * geometry.weight * 0.5,
                geometry.endPoint + normal * geometry.weight * 0.5,
                geometry.endPoint - normal * geometry.weight * 0.5,
                geometry.startPoint - normal * geometry.weight * 0.5])
        else:
            innerRadius = geometry.radius - geometry.weight * 0.5
            outerRadius = geometry.radius + geometry.weight * 0.5

            delta = 0.1 if geometry.endAngle >= geometry.startAngle else -0.1
            angles = numpy.arange(geometry.startAngle, geometry.endAngle, delta)
            if angles[-1] != geometry.endAngle:
                angles = numpy.append(angles, geometry.endAngle)

            if innerRadius <= 0:
                # Remove the inner radius
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
        curvaturePoint = center + normal * defaultCurvature
        weightPoint = center + normal * defaultCurvature * (1.0 + defaultWeight)

        # 3 corners
        controlPoints = numpy.array([
            point0,
            curvaturePoint,
            point1,
            weightPoint
        ])
        return controlPoints

    def _createShapeItems(self, points):
        # Add label marker
        markerPos = self._getLabelPosition()
        marker = items.Marker()
        marker.setPosition(*markerPos)
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker.setSymbol('')
        marker._setDraggable(False)

        shapePoints = self._getShapeFromControlPoints(points)
        item = items.Shape("polygon")
        item.setPoints(shapePoints)
        item.setColor(rgba(self.getColor()))
        item.setFill(False)
        item.setOverlay(True)
        return [item, marker]

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
        normal /= numpy.linalg.norm(normal)
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

    def paramsToString(self):
        points = self.getControlPoints()
        return '; '.join('(%f; %f)' % (pt[0], pt[1]) for pt in points)
