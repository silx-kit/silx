# /*##########################################################################
#
# Copyright (c) 2022 European Synchrotron Radiation Facility
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
"""Rectangular ROI that can be rotated"""

import functools
import logging
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple
import numpy

from ... import qt, utils
from .. import items
from ...colors import rgba
from silx.image.shapes import Polygon
from ....utils.proxy import docstring
from ._roi_base import _RegionOfInterestBase
from ._roi_base import HandleBasedROI
from ._roi_base import InteractionModeMixIn
from ._roi_base import RoiInteractionMode


logger = logging.getLogger(__name__)


class Point(NamedTuple):
    x: float
    y: float


class BandGeometry(NamedTuple):
    begin: Point
    end: Point
    width: float

    @staticmethod
    def create(
        begin: Sequence[float] = (0.0, 0.0),
        end: Sequence[float] = (0.0, 0.0),
        width: Optional[float] = None,
    ):
        begin = Point(float(begin[0]), float(begin[1]))
        end = Point(float(end[0]), float(end[1]))
        if width is None:
            width = 0.1 * numpy.linalg.norm(numpy.array(end) - begin)
        return BandGeometry(begin, end, max(0.0, float(width)))

    @property
    @functools.lru_cache()
    def normal(self) -> Point:
        vector = numpy.array(self.end) - self.begin
        length = numpy.linalg.norm(vector)
        if length == 0:
            return Point(0.0, 0.0)
        return Point(-vector[1] / length, vector[0] / length)

    @property
    @functools.lru_cache()
    def center(self) -> Point:
        return Point(*(0.5 * (numpy.array(self.begin) + self.end)))

    @property
    @functools.lru_cache()
    def corners(self) -> Tuple[Point, Point, Point, Point]:
        """Returns a 4-uple of (x,y) position in float"""
        offset = 0.5 * self.width * numpy.array(self.normal)
        return tuple(
            map(
                lambda p: Point(*p),
                (
                    self.begin - offset,
                    self.begin + offset,
                    self.end + offset,
                    self.end - offset,
                ),
            )
        )

    @property
    @functools.lru_cache()
    def slope(self) -> float:
        """Slope of the line (begin, end), infinity for a vertical line"""
        if self.begin.x == self.end.x:
            return float('inf')
        return (self.end.y - self.begin.y) / (self.end.x - self.begin.x)

    @property
    @functools.lru_cache()
    def intercept(self) -> float:
        """Intercept of the line (begin, end) or value of x for vertical line"""
        if self.begin.x == self.end.x:
            return self.begin.x
        return self.begin.y - self.slope * self.begin.x

    @property
    @functools.lru_cache()
    def edgesIntercept(self) -> Tuple[float, float]:
        """Intercepts of lines describing band edges"""
        offset = 0.5 * self.width * numpy.array(self.normal)
        if self.begin.x == self.end.x:
            return self.begin.x - offset[0], self.begin.x + offset[0]
        return (
            self.begin.y - offset[1] - self.slope * (self.begin.x - offset[0]),
            self.begin.y + offset[1] - self.slope * (self.begin.x + offset[0]),
        )

    def contains(self, position: Sequence[float]) -> bool:
        return Polygon(self.corners).is_inside(*position)


class BandROI(HandleBasedROI, items.LineMixIn, InteractionModeMixIn):
    """A ROI identifying a line in a 2D plot.

    This ROI provides 1 anchor for each boundary of the line, plus an center
    in the center to translate the full ROI.
    """

    ICON = "add-shape-rotated-rectangle"
    NAME = "band ROI"
    SHORT_NAME = "band"
    """Metadata for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    BoundedMode = RoiInteractionMode("Bounded", "Band is bounded on both sides")
    """Interaction mode for a rectangular band ROI"""

    UnboundedMode = RoiInteractionMode("Unbounded", "Band is unbounded on both sides")
    """Interaction mode for unlimited band ROI """

    def __init__(self, parent=None):
        HandleBasedROI.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        self.__availableInteractionModes = set((self.BoundedMode, self.UnboundedMode))
        InteractionModeMixIn.__init__(self)

        self.__handleBegin = self.addHandle()
        self.__handleEnd = self.addHandle()
        self.__handleCenter = self.addTranslateHandle()
        self.__handleLabel = self.addLabelHandle()
        self.__handleWidthUp = self.addHandle()
        self.__handleWidthUp._setConstraint(self.__handleWidthUpConstraint)
        self.__handleWidthUp.setSymbol("d")
        self.__handleWidthDown = self.addHandle()
        self.__handleWidthDown._setConstraint(self.__handleWidthDownConstraint)
        self.__handleWidthDown.setSymbol("d")

        self.__geometry = BandGeometry.create()

        self.__lineUp = items.Line()
        self.__lineUp.setVisible(False)
        self.__lineMiddle = items.Line()
        self.__lineMiddle.setLineWidth(1)
        self.__lineMiddle.setVisible(False)
        self.__lineDown = items.Line()
        self.__lineDown.setVisible(False)

        self.__shape = items.Shape("polygon")
        self.__shape.setPoints(self.__geometry.corners)
        self.__shape.setFill(False)

        for item in (self.__lineUp, self.__lineMiddle, self.__lineDown, self.__shape):
            item.setColor(rgba(self.getColor()))
            item.setOverlay(True)
            item.setLineStyle(self.getLineStyle())
            if item != self.__lineMiddle:
                item.setLineWidth(self.getLineWidth())
            self.addItem(item)

        self._initInteractionMode(self.BoundedMode)
        self._interactiveModeUpdated(self.BoundedMode)

    def availableInteractionModes(self) -> List[RoiInteractionMode]:
        """Returns the list of available interaction modes"""
        return list(self.__availableInteractionModes)

    def setAvailableInteractionModes(self, modes: Iterable[RoiInteractionMode]) -> None:
        """Allows to restrict interaction modes of the ROI.

        :param modes: Subset of BandROI interaction modes:
            :attr:`BoundedMode` and :attr:`UnboundedMode`.
        """
        modes = set(modes)
        if not modes <= set((self.BoundedMode, self.UnboundedMode)):
            raise ValueError("Unsupported interaction modes")
        self.__availableInteractionModes = set(modes)
        if self.getInteractionMode() not in self.__availableInteractionModes:
            self.setInteractionMode(self.availableInteractionModes()[0])

    def _interactiveModeUpdated(self, modeId: RoiInteractionMode):
        """Set the interaction mode."""
        if modeId is self.BoundedMode:
            self.__lineDown.setVisible(False)
            self.__lineMiddle.setVisible(False)
            self.__lineUp.setVisible(False)
            self.__shape.setVisible(True)
        elif modeId is self.UnboundedMode:
            self.__lineDown.setVisible(True)
            self.__lineMiddle.setVisible(True)
            self.__lineUp.setVisible(True)
            self.__shape.setVisible(False)
        else:
            raise RuntimeError("Unsupported interactive mode")

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.VISIBLE:
            if self.isVisible():
                self._interactiveModeUpdated(self.getInteractionMode())
            else:
                self.__lineDown.setVisible(False)
                self.__lineMiddle.setVisible(False)
                self.__lineUp.setVisible(False)
                self.__shape.setVisible(False)
        super()._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super()._updatedStyle(event, style)
        for item in (self.__lineUp, self.__lineMiddle, self.__lineDown, self.__shape):
            item.setColor(style.getColor())
            item.setLineStyle(style.getLineStyle())
            if item != self.__lineMiddle:
                item.setLineWidth(style.getLineWidth())

    def setFirstShapePoints(self, points):
        assert len(points) == 2
        self.setGeometry(*points)

    def _updateText(self, text):
        self.__handleLabel.setText(text)

    def getGeometry(self) -> BandGeometry:
        """Returns the geometric description of the ROI"""
        return self.__geometry

    def setGeometry(
        self,
        begin: Sequence[float],
        end: Sequence[float],
        width: Optional[float] = None,
    ):
        """Set the geometry of the ROI

        :param begin: Starting point as (x, y)
        :paran end: Closing point as (x, y)
        :param width: Width of the ROI
        """
        geometry = BandGeometry.create(begin, end, width)
        if self.__geometry == geometry:
            return

        self.__geometry = geometry

        with utils.blockSignals(self.__handleBegin):
            self.__handleBegin.setPosition(*geometry.begin)
        with utils.blockSignals(self.__handleEnd):
            self.__handleEnd.setPosition(*geometry.end)
        with utils.blockSignals(self.__handleCenter):
            self.__handleCenter.setPosition(*geometry.center)
        with utils.blockSignals(self.__handleLabel):
            lowerCorner = geometry.corners[numpy.array(geometry.corners)[:, 1].argmin()]
            self.__handleLabel.setPosition(*lowerCorner)

        delta = 0.5 * geometry.width * numpy.array(geometry.normal)
        with utils.blockSignals(self.__handleWidthUp):
            self.__handleWidthUp.setPosition(*(geometry.center + delta))
        with utils.blockSignals(self.__handleWidthDown):
            self.__handleWidthDown.setPosition(*(geometry.center - delta))

        self.__lineDown.setSlope(geometry.slope)
        self.__lineDown.setIntercept(geometry.edgesIntercept[0])
        self.__lineMiddle.setSlope(geometry.slope)
        self.__lineMiddle.setIntercept(geometry.intercept)
        self.__lineUp.setSlope(geometry.slope)
        self.__lineUp.setIntercept(geometry.edgesIntercept[1])
        self.__shape.setPoints(geometry.corners)
        self.sigRegionChanged.emit()

    def __updateGeometry(
        self,
        begin: Optional[Sequence[float]] = None,
        end: Optional[Sequence[float]] = None,
        width: Optional[float] = None,
    ):
        geometry = self.getGeometry()
        self.setGeometry(
            geometry.begin if begin is None else begin,
            geometry.end if end is None else end,
            geometry.width if width is None else width,
        )

    @staticmethod
    def __snap(point: Tuple[float, float], fixed: Tuple[float, float]) -> Tuple[float, float]:
        """Snap point so that vector [point, fixed] snap to direction 0, 45 or 90 degrees

        :return: the snapped point position.
        """
        vector = point[0] - fixed[0], point[1] - fixed[1]
        angle = numpy.arctan2(vector[1], vector[0])
        snapAngle = numpy.pi/4 * numpy.round(angle / (numpy.pi/4))
        length = numpy.linalg.norm(vector)
        return (
            fixed[0] + length * numpy.cos(snapAngle),
            fixed[1] + length * numpy.sin(snapAngle)
        )

    def handleDragUpdated(self, handle, origin, previous, current):
        geometry = self.getGeometry()
        if handle is self.__handleBegin:
            if qt.QApplication.keyboardModifiers() & qt.Qt.ShiftModifier:
                self.__updateGeometry(begin=self.__snap(current, geometry.end))
                return
            self.__updateGeometry(begin=current)
            return
        if handle is self.__handleEnd:
            if qt.QApplication.keyboardModifiers() & qt.Qt.ShiftModifier:
                self.__updateGeometry(end=self.__snap(current, geometry.begin))
                return
            self.__updateGeometry(end=current)
            return
        if handle is self.__handleCenter:
            delta = current - previous
            self.__updateGeometry(geometry.begin + delta, geometry.end + delta)
            return
        if handle in (self.__handleWidthUp, self.__handleWidthDown):
            offset = numpy.dot(geometry.normal, current - previous)
            if handle is self.__handleWidthDown:
                offset *= -1
            self.__updateGeometry(
                geometry.begin,
                geometry.end,
                geometry.width + 2 * offset,
            )

    def __handleWidthUpConstraint(self, x: float, y: float) -> Tuple[float, float]:
        geometry = self.getGeometry()
        offset = max(0, numpy.dot(geometry.normal, numpy.array((x, y)) - geometry.center))
        return tuple(geometry.center + offset * numpy.array(geometry.normal))

    def __handleWidthDownConstraint(self, x: float, y: float) -> Tuple[float, float]:
        geometry = self.getGeometry()
        offset = max(0, -numpy.dot(geometry.normal, numpy.array((x, y)) - geometry.center))
        return tuple(geometry.center - offset * numpy.array(geometry.normal))

    @docstring(_RegionOfInterestBase)
    def contains(self, position):
        return self.getGeometry().contains(position)

    def __str__(self):
        begin, end, width = self.getGeometry()
        return f"{self.__class__.__name__}(begin=({begin[0]:g}, {begin[1]:g}), end=({end[0]:g}, {end[1]:g}), width={width:g})"
