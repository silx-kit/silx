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
"""This module provides Arc ROI item for the :class:`~silx.gui.plot.PlotWidget`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "28/06/2018"

import logging
import numpy

from ... import utils
from .. import items
from ...colors import rgba
from ....utils.proxy import docstring
from ._roi_base import HandleBasedROI
from ._roi_base import InteractionModeMixIn
from ._roi_base import RoiInteractionMode


logger = logging.getLogger(__name__)


class _ArcGeometry:
    """
    Non-mutable object to store the geometry of the arc ROI.

    The aim is is to switch between consistent state without dealing with
    intermediate values.
    """
    def __init__(self, center, startPoint, endPoint, radius,
                 weight, startAngle, endAngle, closed=False):
        """Constructor for a consistent arc geometry.

        There is also specific class method to create different kind of arc
        geometry.
        """
        self.center = center
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.radius = radius
        self.weight = weight
        self.startAngle = startAngle
        self.endAngle = endAngle
        self._closed = closed

    @classmethod
    def createEmpty(cls):
        """Create an arc geometry from an empty shape
        """
        zero = numpy.array([0, 0])
        return cls(zero, zero.copy(), zero.copy(), 0, 0, 0, 0)

    @classmethod
    def createRect(cls, startPoint, endPoint, weight):
        """Create an arc geometry from a definition of a rectangle
        """
        return cls(None, startPoint, endPoint, None, weight, None, None, False)

    @classmethod
    def createCircle(cls, center, startPoint, endPoint, radius,
               weight, startAngle, endAngle):
        """Create an arc geometry from a definition of a circle
        """
        return cls(center, startPoint, endPoint, radius,
                   weight, startAngle, endAngle, True)

    def withWeight(self, weight):
        """Return a new geometry based on this object, with a specific weight
        """
        return _ArcGeometry(self.center, self.startPoint, self.endPoint,
                            self.radius, weight,
                            self.startAngle, self.endAngle, self._closed)

    def withRadius(self, radius):
        """Return a new geometry based on this object, with a specific radius.

        The weight and the center is conserved.
        """
        startPoint = self.center + (self.startPoint - self.center) / self.radius * radius
        endPoint = self.center + (self.endPoint - self.center) / self.radius * radius
        return _ArcGeometry(self.center, startPoint, endPoint,
                            radius, self.weight,
                            self.startAngle, self.endAngle, self._closed)

    def withStartAngle(self, startAngle):
        """Return a new geometry based on this object, with a specific start angle
        """
        vector = numpy.array([numpy.cos(startAngle), numpy.sin(startAngle)])
        startPoint = self.center + vector * self.radius

        # Never add more than 180 to maintain coherency
        deltaAngle = startAngle - self.startAngle
        if deltaAngle > numpy.pi:
            deltaAngle -= numpy.pi * 2
        elif deltaAngle < -numpy.pi:
            deltaAngle += numpy.pi * 2

        startAngle = self.startAngle + deltaAngle
        return _ArcGeometry(
            self.center,
            startPoint,
            self.endPoint,
            self.radius,
            self.weight,
            startAngle,
            self.endAngle,
            self._closed,
        )

    def withEndAngle(self, endAngle):
        """Return a new geometry based on this object, with a specific end angle
        """
        vector = numpy.array([numpy.cos(endAngle), numpy.sin(endAngle)])
        endPoint = self.center + vector * self.radius

        # Never add more than 180 to maintain coherency
        deltaAngle = endAngle - self.endAngle
        if deltaAngle > numpy.pi:
            deltaAngle -= numpy.pi * 2
        elif deltaAngle < -numpy.pi:
            deltaAngle += numpy.pi * 2

        endAngle = self.endAngle + deltaAngle
        return _ArcGeometry(
            self.center,
            self.startPoint,
            endPoint,
            self.radius,
            self.weight,
            self.startAngle,
            endAngle,
            self._closed,
        )

    def translated(self, dx, dy):
        """Return the translated geometry by dx, dy"""
        delta = numpy.array([dx, dy])
        center = None if self.center is None else self.center + delta
        startPoint = None if self.startPoint is None else self.startPoint + delta
        endPoint = None if self.endPoint is None else self.endPoint + delta
        return _ArcGeometry(center, startPoint, endPoint,
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


class ArcROI(HandleBasedROI, items.LineMixIn, InteractionModeMixIn):
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

    ThreePointMode = RoiInteractionMode("3 points", "Provides 3 points to define the main radius circle")
    PolarMode = RoiInteractionMode("Polar", "Provides anchors to edit the ROI in polar coords")
    # FIXME: MoveMode was designed cause there is too much anchors
    # FIXME: It would be good replace it by a dnd on the shape
    MoveMode = RoiInteractionMode("Translation", "Provides anchors to only move the ROI")

    def __init__(self, parent=None):
        HandleBasedROI.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        InteractionModeMixIn.__init__(self)

        self._geometry = _ArcGeometry.createEmpty()
        self._handleLabel = self.addLabelHandle()

        self._handleStart = self.addHandle()
        self._handleMid = self.addHandle()
        self._handleEnd = self.addHandle()
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

        self._initInteractionMode(self.ThreePointMode)
        self._interactiveModeUpdated(self.ThreePointMode)

    def availableInteractionModes(self):
        """Returns the list of available interaction modes

        :rtype: List[RoiInteractionMode]
        """
        return [self.ThreePointMode, self.PolarMode, self.MoveMode]

    def _interactiveModeUpdated(self, modeId):
        """Set the interaction mode.

        :param RoiInteractionMode modeId:
        """
        if modeId is self.ThreePointMode:
            self._handleStart.setSymbol("s")
            self._handleMid.setSymbol("s")
            self._handleEnd.setSymbol("s")
            self._handleWeight.setSymbol("d")
            self._handleMove.setSymbol("+")
        elif modeId is self.PolarMode:
            self._handleStart.setSymbol("o")
            self._handleMid.setSymbol("o")
            self._handleEnd.setSymbol("o")
            self._handleWeight.setSymbol("d")
            self._handleMove.setSymbol("+")
        elif modeId is self.MoveMode:
            self._handleStart.setSymbol("")
            self._handleMid.setSymbol("+")
            self._handleEnd.setSymbol("")
            self._handleWeight.setSymbol("")
            self._handleMove.setSymbol("+")
        else:
            assert False
        if self._geometry.isClosed():
            if modeId != self.MoveMode:
                self._handleStart.setSymbol("x")
                self._handleEnd.setSymbol("x")
        self._updateHandles()

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
        weight = distance * weightCoef

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
            midPos = geometry.center + geometry.center - start
        else:
            if geometry.center is None:
                midPos = geometry.startPoint * 0.5 + geometry.endPoint * 0.5
            else:
                midAngle = geometry.startAngle * 0.5 + geometry.endAngle * 0.5
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

    def _updateCurvature(self, start, mid, end, updateCurveHandles, checkClosed=False, updateStart=False):
        """Update the curvature using 3 control points in the curve

        :param bool updateCurveHandles: If False curve handles are already at
            the right location
        """
        if checkClosed:
            closed = self._isCloseInPixel(start, end)
        else:
            closed = self._geometry.isClosed()
        if closed:
            if updateStart:
                start = end
            else:
                end = start

        if updateCurveHandles:
            with utils.blockSignals(self._handleStart):
                self._handleStart.setPosition(*start)
            with utils.blockSignals(self._handleMid):
                self._handleMid.setPosition(*mid)
            with utils.blockSignals(self._handleEnd):
                self._handleEnd.setPosition(*end)

        weight = self._geometry.weight
        geometry = self._createGeometryFromControlPoints(start, mid, end, weight, closed=closed)
        self._geometry = geometry

        self._updateWeightHandle()
        self._updateShape()

    def _updateCloseInAngle(self, geometry, updateStart):
        azim = numpy.abs(geometry.endAngle - geometry.startAngle)
        if numpy.pi < azim < 3 * numpy.pi:
            closed = self._isCloseInPixel(geometry.startPoint, geometry.endPoint)
            geometry._closed = closed
            if closed:
                sign = 1 if geometry.startAngle < geometry.endAngle else -1
                if updateStart:
                    geometry.startPoint = geometry.endPoint
                    geometry.startAngle = geometry.endAngle - sign * 2*numpy.pi
                else:
                    geometry.endPoint = geometry.startPoint
                    geometry.endAngle = geometry.startAngle + sign * 2*numpy.pi

    def handleDragUpdated(self, handle, origin, previous, current):
        modeId = self.getInteractionMode()
        if handle is self._handleStart:
            if modeId is self.ThreePointMode:
                mid = numpy.array(self._handleMid.getPosition())
                end = numpy.array(self._handleEnd.getPosition())
                self._updateCurvature(
                    current, mid, end, checkClosed=True, updateStart=True,
                    updateCurveHandles=False
                )
            elif modeId is self.PolarMode:
                v = current - self._geometry.center
                startAngle = numpy.angle(complex(v[0], v[1]))
                geometry = self._geometry.withStartAngle(startAngle)
                self._updateCloseInAngle(geometry, updateStart=True)
                self._geometry = geometry
                self._updateHandles()
        elif handle is self._handleMid:
            if modeId is self.ThreePointMode:
                if self._geometry.isClosed():
                    radius = numpy.linalg.norm(self._geometry.center - current)
                    self._geometry = self._geometry.withRadius(radius)
                    self._updateHandles()
                else:
                    start = numpy.array(self._handleStart.getPosition())
                    end = numpy.array(self._handleEnd.getPosition())
                    self._updateCurvature(start, current, end, updateCurveHandles=False)
            elif modeId is self.PolarMode:
                radius = numpy.linalg.norm(self._geometry.center - current)
                self._geometry = self._geometry.withRadius(radius)
                self._updateHandles()
            elif modeId is self.MoveMode:
                delta = current - previous
                self.translate(*delta)
        elif handle is self._handleEnd:
            if modeId is self.ThreePointMode:
                start = numpy.array(self._handleStart.getPosition())
                mid = numpy.array(self._handleMid.getPosition())
                self._updateCurvature(
                    start, mid, current, checkClosed=True, updateStart=False,
                    updateCurveHandles=False
                )
            elif modeId is self.PolarMode:
                v = current - self._geometry.center
                endAngle = numpy.angle(complex(v[0], v[1]))
                geometry = self._geometry.withEndAngle(endAngle)
                self._updateCloseInAngle(geometry, updateStart=False)
                self._geometry = geometry
                self._updateHandles()
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
        modeId = self.getInteractionMode()
        if handle in [self._handleStart, self._handleMid, self._handleEnd]:
            if modeId is self.ThreePointMode:
                self._normalizeGeometry()
                self._updateHandles()

        if self._geometry.isClosed():
            if modeId is self.MoveMode:
                self._handleStart.setSymbol("")
                self._handleEnd.setSymbol("")
            else:
                self._handleStart.setSymbol("x")
                self._handleEnd.setSymbol("x")
        else:
            if modeId is self.ThreePointMode:
                self._handleStart.setSymbol("s")
                self._handleEnd.setSymbol("s")
            elif modeId is self.PolarMode:
                self._handleStart.setSymbol("o")
                self._handleEnd.setSymbol("o")
            if modeId is self.MoveMode:
                self._handleStart.setSymbol("")
                self._handleEnd.setSymbol("")

    def _createGeometryFromControlPoints(self, start, mid, end, weight, closed=None):
        """Returns the geometry of the object"""
        if closed or (closed is None and numpy.allclose(start, end)):
            # Special arc: It's a closed circle
            center = (start + mid) * 0.5
            radius = numpy.linalg.norm(start - center)
            v = start - center
            startAngle = numpy.angle(complex(v[0], v[1]))
            endAngle = startAngle + numpy.pi * 2.0
            return _ArcGeometry.createCircle(
                center, start, end, radius, weight, startAngle, endAngle
            )

        elif numpy.linalg.norm(numpy.cross(mid - start, end - start)) < 1e-5:
            # Degenerated arc, it's a rectangle
            return _ArcGeometry.createRect(start, end, weight)
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

            return _ArcGeometry(center, start, end,
                                radius, weight, startAngle, endAngle)

    def _createShapeFromGeometry(self, geometry):
        kind = geometry.getKind()
        if kind == "rect":
            # It is not an arc
            # but we can display it as an intermediate shape
            normal = geometry.endPoint - geometry.startPoint
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
            angles = numpy.linspace(0, 2 * numpy.pi, num=50)
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
            angles = numpy.linspace(0, 2 * numpy.pi, num=50)
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

            sign = numpy.sign(geometry.endAngle - geometry.startAngle)
            delta = min(0.1, abs(geometry.startAngle - geometry.endAngle) / 100) * sign

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

            if kind == "camembert":
                # It's a part of camembert
                points = []
                points.append(geometry.center)
                points.append(geometry.startPoint)
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
        else:
            movePos = geometry.center

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
        if innerRadius > outerRadius:
            logger.error("inner radius larger than outer radius")
            innerRadius, outerRadius = outerRadius, innerRadius
        center = numpy.array(center)
        radius = (innerRadius + outerRadius) * 0.5
        weight = outerRadius - innerRadius

        vector = numpy.array([numpy.cos(startAngle), numpy.sin(startAngle)])
        startPoint = center + vector * radius
        vector = numpy.array([numpy.cos(endAngle), numpy.sin(endAngle)])
        endPoint = center + vector * radius

        geometry = _ArcGeometry(center, startPoint, endPoint,
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
        # angle is inside [-pi, pi]

        # Normalize the start angle between [-pi, pi]
        # with a positive angle range
        start_angle = self.getStartAngle()
        end_angle = self.getEndAngle()
        azim_range = end_angle - start_angle
        if azim_range < 0:
            start_angle = end_angle
            azim_range = -azim_range
        start_angle = numpy.mod(start_angle + numpy.pi, 2 * numpy.pi) - numpy.pi

        if angle < start_angle:
            angle += 2 * numpy.pi
        return start_angle <= angle <= start_angle + azim_range

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
