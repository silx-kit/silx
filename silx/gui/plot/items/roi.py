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
__date__ = "20/06/2018"


import functools
import itertools
import logging

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

    def __init__(self, parent, kind):
        # FIXME: Not very elegant: It checks class name to avoid recursive loop
        assert parent is None or "RegionOfInterestManager" in parent.__class__.__name__
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

    @staticmethod
    def getFirstInteractionShape(roiKind):
        """Returns the shape kind which will be used by the very first
        interaction with the plot.

        This interactions are hardcoded inside the plot

        :param str roiKind:
        :rtype: str
        """
        return roiKind

    def setFirstShapePoints(self, points):
        """"Initialize the ROI using the points from the first interaction.

        This interaction is constains by the plot API and only supports few
        shapes.
        """
        points = self._createControlPointsFromFirstShape(points)
        self.setControlPoints(points)

    def _createControlPointsFromFirstShape(self, points):
        """"""
        kind = self._kind
        if kind == "rectangle":
            if len(points) == 2:
                # Add an extra for the central control point
                center = numpy.mean(points, axis=0)
                points = numpy.append(points, center)
                points.shape = -1, 2
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
        if len(self._items) == 0:
            return
        kind = self._kind

        if kind in ['line', 'rectangle', 'polygon']:
            shape = self._items[0]
            points = self._getShapePoints()
            shape.setPoints(points)

    def _getLabelPosition(self):
        """Compute position of the label

        :return: (x, y) position of the marker
        """
        kind = self._kind
        points = self.getControlPoints()

        if kind in ('point', 'hline', 'vline'):
            assert len(points) == 1
            return points[0]

        elif kind == 'rectangle':
            assert len(points) in [2, 3]
            return points.min(axis=0)

        elif kind == 'line':
            assert len(points) == 2
            return points[numpy.argmin(points[:, 0])]

        elif kind == 'polygon':
            return points[numpy.argmin(points[:, 1])]

        else:
            raise RuntimeError('Unsupported ROI kind: %s' % kind)

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

    def _getShapePoints(self):
        points = self.getControlPoints()
        kind = self._kind
        if kind == 'rectangle':
            points = points[0:-1]
        return points

    def _createShapeItems(self, points):
        """Create shape items from the current control points.

        :rtype: List[PlotItem]
        """
        kind = self._kind

        if kind == 'point':
            if self.isEditable():
                return []
            marker = items.Marker()
            marker.setPosition(points[0][0], points[0][1])
            marker.setText(self.getLabel())
            marker.setColor(rgba(self.getColor()))
            marker._setDraggable(False)
            return [marker]

        elif kind == 'hline':
            if self.isEditable():
                return []
            marker = items.YMarker()
            marker.setPosition(points[0][0], points[0][1])
            marker.setText(self.getLabel())
            marker.setColor(rgba(self.getColor()))
            marker._setDraggable(False)
            return [marker]

        elif kind == 'vline':
            if self.isEditable():
                return []
            marker = items.XMarker()
            marker.setPosition(points[0][0], points[0][1])
            marker.setText(self.getLabel())
            marker.setColor(rgba(self.getColor()))
            marker._setDraggable(False)
            return [marker]

        # Add label marker
        markerPos = self._getLabelPosition()
        marker = items.Marker()
        marker.setPosition(*markerPos)
        marker.setText(self.getLabel())
        marker.setColor(rgba(self.getColor()))
        marker.setSymbol('')
        marker._setDraggable(False)

        if kind == 'line':
            item = items.Shape("polylines")
            item.setPoints(points)
            item.setColor(rgba(self.getColor()))
            item.setFill(False)
            item.setOverlay(True)
            return [item, marker]

        elif kind == 'rectangle':
            item = items.Shape("rectangle")
            item.setPoints(points[0:2])
            item.setColor(rgba(self.getColor()))
            item.setFill(False)
            item.setOverlay(True)
            return [item, marker]

        elif kind == 'polygon':
            item = items.Shape("polygon")
            item.setPoints(points)
            item.setColor(rgba(self.getColor()))
            item.setFill(False)
            item.setOverlay(True)
            return [item, marker]
        else:
            return []

    def _createAnchorItems(self, points):
        """Create anchor items from the current control points.

        :rtype: List[Marker]
        """
        kind = self._kind

        if kind == 'point':
            marker = items.Marker()
            marker.setPosition(points[0][0], points[0][1])
            marker.setText(self.getLabel())
            marker._setDraggable(self.isEditable())
            return [marker]

        elif kind == 'hline':
            marker = items.YMarker()
            marker.setPosition(points[0][0], points[0][1])
            marker.setText(self.getLabel())
            marker._setDraggable(self.isEditable())
            return [marker]

        elif kind == 'vline':
            marker = items.XMarker()
            marker.setPosition(points[0][0], points[0][1])
            marker.setText(self.getLabel())
            marker._setDraggable(self.isEditable())
            return [marker]

        else:  # rectangle, line, polygon
            color = rgba(self.getColor())
            color = color[:3] + (0.5,)

            if kind == 'rectangle':
                # Remove the center control point
                points = points[0:2]

            anchors = []
            for point in points:
                anchor = items.Marker()
                anchor.setPosition(*point)
                anchor.setText('')
                anchor.setSymbol('s')
                anchor._setDraggable(True)
                anchors.append(anchor)

            # Add an anchor to the center of the rectangle
            if kind == 'rectangle':
                center = numpy.mean(points, axis=0)
                anchor = items.Marker()
                anchor.setPosition(*center)
                anchor.setText('')
                anchor.setSymbol('o')
                anchor._setDraggable(True)
                anchors.append(anchor)

            return anchors

    def _controlPointAnchorChanged(self, index, event):
        """Handle update of position of an edition anchor

        :param int index: Index of the anchor
        :param ItemChangedType event: Event type
        """
        if event == items.ItemChangedType.POSITION:
            anchor = self._editAnchors[index]
            points = self.getControlPoints()
            previous = points[index]
            current = anchor.getPosition()
            points[index] = current
            self.setControlPoints(points)
            # Custom special behaviours
            self._controlPointAnchorPositionChanged(index, current, previous)
            # Reach again the points in case some was edited
            points = self.getControlPoints()
            self.setControlPoints(points)

    def _controlPointAnchorPositionChanged(self, index, current, previous):
        kind = self._kind
        if kind == 'point':
            points = [current]
            self.setControlPoints(points)
        elif kind == 'hline':
            points = self.getControlPoints()
            points[:, 1] = current[1]
            self.setControlPoints(points)
        elif kind == 'vline':
            points = self.getControlPoints()
            points[:, 0] = current[0]
            self.setControlPoints(points)
        elif kind == "rectangle":
            if index == len(self._editAnchors) - 1:
                # It is the center anchor
                points = self.getControlPoints()
                center = numpy.mean(points, axis=0)
                offset = current - center
                points = points + offset
                self.setControlPoints(points)
            else:
                # Update the center
                points = self.getControlPoints()
                center = numpy.mean(points[0:2], axis=0)
                points[2] = center
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
        kind = self._kind
        points = self.getControlPoints()

        if kind == 'rectangle':
            origin = numpy.min(points, axis=0)
            w, h = numpy.max(points, axis=0) - origin
            return ('Origin: (%f; %f); Width: %f; Height: %f' %
                    (origin[0], origin[1], w, h))

        elif kind == 'point':
            return '(%f; %f)' % (points[0, 0], points[0, 1])

        elif kind == 'hline':
            return 'Y: %f' % points[0, 1]

        elif kind == 'vline':
            return 'X: %f' % points[0, 0]

        else:  # default (polygon, line)
            return '; '.join('(%f; %f)' % (pt[0], pt[1]) for pt in points)
