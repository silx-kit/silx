# /*##########################################################################
#
# Copyright (c) 2018-2022 European Synchrotron Radiation Facility
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

from ... import utils
from .. import items
from ...colors import rgba
from silx.image.shapes import Polygon
from silx.image._boundingbox import _BoundingBox
from ....utils.proxy import docstring
from ..utils.intersections import segments_intersection
from ._roi_base import _RegionOfInterestBase

# He following imports have to be exposed by this module
from ._roi_base import RegionOfInterest
from ._roi_base import HandleBasedROI
from ._arc_roi import ArcROI  # noqa
from ._band_roi import BandROI  # noqa
from ._roi_base import InteractionModeMixIn  # noqa
from ._roi_base import RoiInteractionMode  # noqa


logger = logging.getLogger(__name__)


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
        roiPos = self.getPosition()
        return position[0] == roiPos[0] and position[1] == roiPos[1]

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

        points = self.__shape.getPoints()
        line_pt1 = points[0]
        line_pt2 = points[1]

        bb1 = _BoundingBox.from_points(points)
        if not bb1.contains(position):
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
        ) is not None

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
        return position[1] == self.getPosition()

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
        return position[0] == self.getPosition()

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

            # Switch handles if they were crossed by interaction
            if self._handleBottomLeft.getXPosition() > self._handleBottomRight.getXPosition():
                self._handleBottomLeft, self._handleBottomRight = self._handleBottomRight, self._handleBottomLeft

            if self._handleTopLeft.getXPosition() > self._handleTopRight.getXPosition():
                self._handleTopLeft, self._handleTopRight = self._handleTopRight, self._handleTopLeft

            if self._handleBottomLeft.getYPosition() > self._handleTopLeft.getYPosition():
                self._handleBottomLeft, self._handleTopLeft = self._handleTopLeft, self._handleBottomLeft

            if self._handleBottomRight.getYPosition() > self._handleTopRight.getYPosition():
                self._handleBottomRight, self._handleTopRight = self._handleTopRight, self._handleBottomRight

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

    @docstring(HandleBasedROI)
    def contains(self, position):
        return numpy.linalg.norm(self.getCenter() - position) <= self.getRadius()

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
            orientation -= numpy.pi / 2

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
                    orientation -= numpy.pi / 2
                radius = self._radius[0], distance

            else:  # _handleAxis0
                if self._radius[1] > distance:
                    # _handleAxis0 is not the major axis, rotate +90 degrees
                    orientation += numpy.pi / 2
                radius = distance, self._radius[1]

            self.setGeometry(radius=radius, orientation=orientation)

    def _centerPositionChanged(self, event):
        """Handle position changed events of the center marker"""
        if event is items.ItemChangedType.POSITION:
            self._updateGeometry()

    @docstring(HandleBasedROI)
    def contains(self, position):
        major, minor = self.getMajorRadius(), self.getMinorRadius()
        delta = self.getOrientation()
        x, y = position - self.getCenter()
        return ((x*numpy.cos(delta) + y*numpy.sin(delta))**2/major**2 +
                (x*numpy.sin(delta) - y*numpy.cos(delta))**2/minor**2) <= 1

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

    @docstring(HandleBasedROI)
    def contains(self, position):
        return self.getMin() <= position[0] <= self.getMax()

    def __str__(self):
        vrange = self.getRange()
        params = 'min: %f; max: %f' % vrange
        return "%s(%s)" % (self.__class__.__name__, params)
