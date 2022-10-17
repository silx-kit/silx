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
from typing import NamedTuple, Optional, Sequence, Tuple
import numpy

from ... import utils
from .. import items
from ...colors import rgba
from silx.image.shapes import Polygon
from ....utils.proxy import docstring
from ._roi_base import _RegionOfInterestBase

# He following imports have to be exposed by this module
from ._roi_base import HandleBasedROI


logger = logging.getLogger(__name__)


class Point(NamedTuple):
    x: float
    y: float

    def asarray(self) -> numpy.ndarray:
        return numpy.array((self.x, self.y))


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
            width = 0.1 * numpy.linalg.norm(end.asarray() - begin)
        return BandGeometry(begin, end, max(0.0, float(width)))

    @property
    @functools.lru_cache()
    def normal(self) -> Point:
        vector = self.end.asarray() - self.begin
        length = numpy.linalg.norm(vector)
        if length == 0:
            return Point(0.0, 0.0)
        return Point(-vector[1] / length, vector[0] / length)

    @property
    @functools.lru_cache()
    def center(self) -> Point:
        return Point(*(0.5 * (self.begin.asarray() + self.end)))

    @property
    @functools.lru_cache()
    def corners(self) -> Tuple[Point, Point, Point, Point]:
        """Returns a 4-uple of (x,y) position in float"""
        offset = 0.5 * self.width * self.normal.asarray()
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

    def contains(self, position: Sequence[float]) -> bool:
        return Polygon(self.corners).is_inside(*position)


class BandROI(HandleBasedROI, items.LineMixIn):
    """A ROI identifying a line in a 2D plot.

    This ROI provides 1 anchor for each boundary of the line, plus an center
    in the center to translate the full ROI.
    """

    ICON = "add-shape-diagonal"
    NAME = "band ROI"
    SHORT_NAME = "band"
    """Metadata for this kind of ROI"""

    _plotShape = "line"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        HandleBasedROI.__init__(self, parent=parent)
        items.LineMixIn.__init__(self)
        self._handleBegin = self.addHandle()
        self._handleEnd = self.addHandle()
        self._handleCenter = self.addTranslateHandle()
        self._handleLabel = self.addLabelHandle()
        self._handleWidthUp = self.addHandle()
        self._handleWidthUp._setConstraint(self.__handleWidthUpConstraint)
        self._handleWidthUp.setSymbol("d")
        self._handleWidthDown = self.addHandle()
        self._handleWidthDown._setConstraint(self.__handleWidthDownConstraint)
        self._handleWidthDown.setSymbol("d")

        self.__geometry = BandGeometry.create()

        self.__shape = items.Shape("polygon")
        self.__shape.setPoints(self.__geometry.corners)
        self.__shape.setColor(rgba(self.getColor()))
        self.__shape.setFill(False)
        self.__shape.setOverlay(True)
        self.__shape.setLineStyle(self.getLineStyle())
        self.__shape.setLineWidth(self.getLineWidth())
        self.addItem(self.__shape)

    def _updated(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.VISIBLE:
            self._updateItemProperty(event, self, self.__shape)
        super()._updated(event, checkVisibility)

    def _updatedStyle(self, event, style):
        super()._updatedStyle(event, style)
        self.__shape.setColor(style.getColor())
        self.__shape.setLineStyle(style.getLineStyle())
        self.__shape.setLineWidth(style.getLineWidth())

    def setFirstShapePoints(self, points):
        assert len(points) == 2
        self.setGeometry(*points)

    def _updateText(self, text):
        self._handleLabel.setText(text)

    def getGeometry(self):
        return self.__geometry

    def setGeometry(
        self,
        begin: Sequence[float],
        end: Sequence[float],
        width: Optional[float] = None,
    ):
        geometry = BandGeometry.create(begin, end, width)
        if self.__geometry == geometry:
            return

        self.__geometry = geometry

        with utils.blockSignals(self._handleBegin):
            self._handleBegin.setPosition(*geometry.begin)
        with utils.blockSignals(self._handleEnd):
            self._handleEnd.setPosition(*geometry.end)
        with utils.blockSignals(self._handleCenter):
            self._handleCenter.setPosition(*geometry.center)
        with utils.blockSignals(self._handleLabel):
            lowerCorner = geometry.corners[numpy.array(geometry.corners)[:, 1].argmin()]
            self._handleLabel.setPosition(*lowerCorner)

        delta = 0.5 * geometry.width * geometry.normal.asarray()
        with utils.blockSignals(self._handleWidthUp):
            self._handleWidthUp.setPosition(*(geometry.center + delta))
        with utils.blockSignals(self._handleWidthDown):
            self._handleWidthDown.setPosition(*(geometry.center - delta))

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

    def handleDragUpdated(self, handle, origin, previous, current):
        geometry = self.getGeometry()
        delta = current - previous
        if handle is self.__handleBegin:
            self.__updateGeometry(current, geometry.end - delta)
            return
        if handle is self.__handleEnd:
            self.__updateGeometry(geometry.begin - delta, current)
            return
        if handle is self.__handleCenter:
            self.__updateGeometry(geometry.begin + delta, geometry.end + delta)
            return
        if handle in (self.__handleWidthUp, self.__handleWidthDown):
            offset = numpy.dot(geometry.normal, delta)
            if handle is self.__handleWidthDown:
                offset *= -1
            self.__updateGeometry(
                geometry.begin,
                geometry.end,
                geometry.width + 2 * offset,
            )

    def __handleWidthUpConstraint(self, x: float, y: float) -> Tuple[float, float]:
        geometry = self.getGeometry()
        offset = max(0, numpy.dot(geometry.normal, (x, y) - geometry.center.asarray()))
        return tuple(geometry.center + offset * geometry.normal.asarray())

    def __handleWidthDownConstraint(self, x: float, y: float) -> Tuple[float, float]:
        geometry = self.getGeometry()
        offset = max(0, -numpy.dot(geometry.normal, (x, y) - geometry.center.asarray()))
        return tuple(geometry.center - offset * geometry.normal.asarray())

    @docstring(_RegionOfInterestBase)
    def contains(self, position):
        return self.getGeometry().contains(position)

    def __str__(self):
        begin, end, width = self.getGeometry()
        return f"{self.__class__.__name__}(begin=({begin[0]:g}, {begin[1]:g}), end=({end[0]:g}, {end[1]:g}), width={width:g})"
