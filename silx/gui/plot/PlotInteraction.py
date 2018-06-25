#  coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2018 European Synchrotron Radiation Facility
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
"""Implementation of the interaction for the :class:`Plot`."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import math
import numpy
import time
import weakref

from .. import colors
from .. import qt
from . import items
from .Interaction import (ClickOrDrag, LEFT_BTN, RIGHT_BTN,
                          State, StateMachine)
from .PlotEvents import (prepareCurveSignal, prepareDrawingSignal,
                         prepareHoverSignal, prepareImageSignal,
                         prepareMarkerSignal, prepareMouseSignal)

from .backends.BackendBase import (CURSOR_POINTING, CURSOR_SIZE_HOR,
                                   CURSOR_SIZE_VER, CURSOR_SIZE_ALL)

from ._utils import (FLOAT32_SAFE_MIN, FLOAT32_MINPOS, FLOAT32_SAFE_MAX,
                     applyZoomToPlot)


# Base class ##################################################################

class _PlotInteraction(object):
    """Base class for interaction handler.

    It provides a weakref to the plot and methods to set/reset overlay.
    """
    def __init__(self, plot):
        """Init.

        :param plot: The plot to apply modifications to.
        """
        self._needReplot = False
        self._selectionAreas = set()
        self._plot = weakref.ref(plot)  # Avoid cyclic-ref

    @property
    def plot(self):
        plot = self._plot()
        assert plot is not None
        return plot

    def setSelectionArea(self, points, fill, color, name='', shape='polygon'):
        """Set a polygon selection area overlaid on the plot.
        Multiple simultaneous areas are supported through the name parameter.

        :param points: The 2D coordinates of the points of the polygon
        :type points: An iterable of (x, y) coordinates
        :param str fill: The fill mode: 'hatch', 'solid' or 'none'
        :param color: RGBA color to use or None to disable display
        :type color: list or tuple of 4 float in the range [0, 1]
        :param name: The key associated with this selection area
        :param str shape: Shape of the area in 'polygon', 'polylines'
        """
        assert shape in ('polygon', 'polylines')

        if color is None:
            return

        points = numpy.asarray(points)

        # TODO Not very nice, but as is for now
        legend = '__SELECTION_AREA__' + name

        fill = fill != 'none'  # TODO not very nice either

        self.plot.addItem(points[:, 0], points[:, 1], legend=legend,
                          replace=False,
                          shape=shape, color=color, fill=fill,
                          overlay=True)
        self._selectionAreas.add(legend)

    def resetSelectionArea(self):
        """Remove all selection areas set by setSelectionArea."""
        for legend in self._selectionAreas:
            self.plot.remove(legend, kind='item')
        self._selectionAreas = set()


# Zoom/Pan ####################################################################

class _ZoomOnWheel(ClickOrDrag, _PlotInteraction):
    """:class:`ClickOrDrag` state machine with zooming on mouse wheel.

    Base class for :class:`Pan` and :class:`Zoom`
    """

    _DOUBLE_CLICK_TIMEOUT = 0.4

    class ZoomIdle(ClickOrDrag.Idle):
        def onWheel(self, x, y, angle):
            scaleF = 1.1 if angle > 0 else 1. / 1.1
            applyZoomToPlot(self.machine.plot, scaleF, (x, y))

    def click(self, x, y, btn):
        """Handle clicks by sending events

        :param int x: Mouse X position in pixels
        :param int y: Mouse Y position in pixels
        :param btn: Clicked mouse button
        """
        if btn == LEFT_BTN:
            lastClickTime, lastClickPos = self._lastClick

            # Signal mouse double clicked event first
            if (time.time() - lastClickTime) <= self._DOUBLE_CLICK_TIMEOUT:
                # Use position of first click
                eventDict = prepareMouseSignal('mouseDoubleClicked', 'left',
                                               *lastClickPos)
                self.plot.notify(**eventDict)

                self._lastClick = 0., None
            else:
                # Signal mouse clicked event
                dataPos = self.plot.pixelToData(x, y)
                assert dataPos is not None
                eventDict = prepareMouseSignal('mouseClicked', 'left',
                                               dataPos[0], dataPos[1],
                                               x, y)
                self.plot.notify(**eventDict)

                self._lastClick = time.time(), (dataPos[0], dataPos[1], x, y)

        elif btn == RIGHT_BTN:
            # Signal mouse clicked event
            dataPos = self.plot.pixelToData(x, y)
            assert dataPos is not None
            eventDict = prepareMouseSignal('mouseClicked', 'right',
                                           dataPos[0], dataPos[1],
                                           x, y)
            self.plot.notify(**eventDict)

    def __init__(self, plot):
        """Init.

        :param plot: The plot to apply modifications to.
        """
        _PlotInteraction.__init__(self, plot)

        states = {
            'idle': _ZoomOnWheel.ZoomIdle,
            'rightClick': ClickOrDrag.RightClick,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        StateMachine.__init__(self, states, 'idle')

        self._lastClick = 0., None


# Pan #########################################################################

class Pan(_ZoomOnWheel):
    """Pan plot content and zoom on wheel state machine."""

    def _pixelToData(self, x, y):
        xData, yData = self.plot.pixelToData(x, y)
        _, y2Data = self.plot.pixelToData(x, y, axis='right')
        return xData, yData, y2Data

    def beginDrag(self, x, y):
        self._previousDataPos = self._pixelToData(x, y)

    def drag(self, x, y):
        xData, yData, y2Data = self._pixelToData(x, y)
        lastX, lastY, lastY2 = self._previousDataPos

        xMin, xMax = self.plot.getXAxis().getLimits()
        yMin, yMax = self.plot.getYAxis().getLimits()
        y2Min, y2Max = self.plot.getYAxis(axis='right').getLimits()

        if self.plot.getXAxis()._isLogarithmic():
            try:
                dx = math.log10(xData) - math.log10(lastX)
                newXMin = pow(10., (math.log10(xMin) - dx))
                newXMax = pow(10., (math.log10(xMax) - dx))
            except (ValueError, OverflowError):
                newXMin, newXMax = xMin, xMax

            # Makes sure both values stays in positive float32 range
            if newXMin < FLOAT32_MINPOS or newXMax > FLOAT32_SAFE_MAX:
                newXMin, newXMax = xMin, xMax
        else:
            dx = xData - lastX
            newXMin, newXMax = xMin - dx, xMax - dx

            # Makes sure both values stays in float32 range
            if newXMin < FLOAT32_SAFE_MIN or newXMax > FLOAT32_SAFE_MAX:
                newXMin, newXMax = xMin, xMax

        if self.plot.getYAxis()._isLogarithmic():
            try:
                dy = math.log10(yData) - math.log10(lastY)
                newYMin = pow(10., math.log10(yMin) - dy)
                newYMax = pow(10., math.log10(yMax) - dy)

                dy2 = math.log10(y2Data) - math.log10(lastY2)
                newY2Min = pow(10., math.log10(y2Min) - dy2)
                newY2Max = pow(10., math.log10(y2Max) - dy2)
            except (ValueError, OverflowError):
                newYMin, newYMax = yMin, yMax
                newY2Min, newY2Max = y2Min, y2Max

            # Makes sure y and y2 stays in positive float32 range
            if (newYMin < FLOAT32_MINPOS or newYMax > FLOAT32_SAFE_MAX or
                    newY2Min < FLOAT32_MINPOS or newY2Max > FLOAT32_SAFE_MAX):
                newYMin, newYMax = yMin, yMax
                newY2Min, newY2Max = y2Min, y2Max
        else:
            dy = yData - lastY
            dy2 = y2Data - lastY2
            newYMin, newYMax = yMin - dy, yMax - dy
            newY2Min, newY2Max = y2Min - dy2, y2Max - dy2

            # Makes sure y and y2 stays in float32 range
            if (newYMin < FLOAT32_SAFE_MIN or
                    newYMax > FLOAT32_SAFE_MAX or
                    newY2Min < FLOAT32_SAFE_MIN or
                    newY2Max > FLOAT32_SAFE_MAX):
                newYMin, newYMax = yMin, yMax
                newY2Min, newY2Max = y2Min, y2Max

        self.plot.setLimits(newXMin, newXMax,
                            newYMin, newYMax,
                            newY2Min, newY2Max)

        self._previousDataPos = self._pixelToData(x, y)

    def endDrag(self, startPos, endPos):
        del self._previousDataPos

    def cancel(self):
        pass


# Zoom ########################################################################

class Zoom(_ZoomOnWheel):
    """Zoom-in/out state machine.

    Zoom-in on selected area, zoom-out on right click,
    and zoom on mouse wheel.
    """

    def __init__(self, plot, color):
        self.color = color

        super(Zoom, self).__init__(plot)
        self.plot.getLimitsHistory().clear()

    def _areaWithAspectRatio(self, x0, y0, x1, y1):
        _plotLeft, _plotTop, plotW, plotH = self.plot.getPlotBoundsInPixels()

        areaX0, areaY0, areaX1, areaY1 = x0, y0, x1, y1

        if plotH != 0.:
            plotRatio = plotW / float(plotH)
            width, height = math.fabs(x1 - x0), math.fabs(y1 - y0)

            if height != 0. and width != 0.:
                if width / height > plotRatio:
                    areaHeight = width / plotRatio
                    areaX0, areaX1 = x0, x1
                    center = 0.5 * (y0 + y1)
                    areaY0 = center - numpy.sign(y1 - y0) * 0.5 * areaHeight
                    areaY1 = center + numpy.sign(y1 - y0) * 0.5 * areaHeight
                else:
                    areaWidth = height * plotRatio
                    areaY0, areaY1 = y0, y1
                    center = 0.5 * (x0 + x1)
                    areaX0 = center - numpy.sign(x1 - x0) * 0.5 * areaWidth
                    areaX1 = center + numpy.sign(x1 - x0) * 0.5 * areaWidth

        return areaX0, areaY0, areaX1, areaY1

    def beginDrag(self, x, y):
        dataPos = self.plot.pixelToData(x, y)
        assert dataPos is not None
        self.x0, self.y0 = x, y

    def drag(self, x1, y1):
        if self.color is None:
            return  # Do not draw zoom area

        dataPos = self.plot.pixelToData(x1, y1)
        assert dataPos is not None

        if self.plot.isKeepDataAspectRatio():
            area = self._areaWithAspectRatio(self.x0, self.y0, x1, y1)
            areaX0, areaY0, areaX1, areaY1 = area
            areaPoints = ((areaX0, areaY0),
                          (areaX1, areaY0),
                          (areaX1, areaY1),
                          (areaX0, areaY1))
            areaPoints = numpy.array([self.plot.pixelToData(
                x, y, check=False) for (x, y) in areaPoints])

            if self.color != 'video inverted':
                areaColor = list(self.color)
                areaColor[3] *= 0.25
            else:
                areaColor = [1., 1., 1., 1.]

            self.setSelectionArea(areaPoints,
                                  fill='none',
                                  color=areaColor,
                                  name="zoomedArea")

        corners = ((self.x0, self.y0),
                   (self.x0, y1),
                   (x1, y1),
                   (x1, self.y0))
        corners = numpy.array([self.plot.pixelToData(x, y, check=False)
                               for (x, y) in corners])

        self.setSelectionArea(corners, fill='none', color=self.color)

    def endDrag(self, startPos, endPos):
        x0, y0 = startPos
        x1, y1 = endPos

        if x0 != x1 or y0 != y1:  # Avoid empty zoom area
            # Store current zoom state in stack
            self.plot.getLimitsHistory().push()

            if self.plot.isKeepDataAspectRatio():
                x0, y0, x1, y1 = self._areaWithAspectRatio(x0, y0, x1, y1)

            # Convert to data space and set limits
            x0, y0 = self.plot.pixelToData(x0, y0, check=False)

            dataPos = self.plot.pixelToData(
                startPos[0], startPos[1], axis="right", check=False)
            y2_0 = dataPos[1]

            x1, y1 = self.plot.pixelToData(x1, y1, check=False)

            dataPos = self.plot.pixelToData(
                endPos[0], endPos[1], axis="right", check=False)
            y2_1 = dataPos[1]

            xMin, xMax = min(x0, x1), max(x0, x1)
            yMin, yMax = min(y0, y1), max(y0, y1)
            y2Min, y2Max = min(y2_0, y2_1), max(y2_0, y2_1)

            self.plot.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

        self.resetSelectionArea()

    def cancel(self):
        if isinstance(self.state, self.states['drag']):
            self.resetSelectionArea()


# Select ######################################################################

class Select(StateMachine, _PlotInteraction):
    """Base class for drawing selection areas."""

    def __init__(self, plot, parameters, states, state):
        """Init a state machine.

        :param plot: The plot to apply changes to.
        :param dict parameters: A dict of parameters such as color.
        :param dict states: The states of the state machine.
        :param str state: The name of the initial state.
        """
        _PlotInteraction.__init__(self, plot)
        self.parameters = parameters
        StateMachine.__init__(self, states, state)

    def onWheel(self, x, y, angle):
        scaleF = 1.1 if angle > 0 else 1. / 1.1
        applyZoomToPlot(self.plot, scaleF, (x, y))

    @property
    def color(self):
        return self.parameters.get('color', None)


class SelectPolygon(Select):
    """Drawing selection polygon area state machine."""

    DRAG_THRESHOLD_DIST = 4

    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enterState(self, x, y):
            dataPos = self.machine.plot.pixelToData(x, y)
            assert dataPos is not None
            self._firstPos = dataPos
            self.points = [dataPos, dataPos]

            self.updateFirstPoint()

        def updateFirstPoint(self):
            """Update drawing first point, using self._firstPos"""
            x, y = self.machine.plot.dataToPixel(*self._firstPos, check=False)

            offset = self.machine.getDragThreshold()
            points = [(x - offset, y - offset),
                      (x - offset, y + offset),
                      (x + offset, y + offset),
                      (x + offset, y - offset)]
            points = [self.machine.plot.pixelToData(xpix, ypix, check=False)
                      for xpix, ypix in points]
            self.machine.setSelectionArea(points, fill=None,
                                          color=self.machine.color,
                                          name='first_point')

        def updateSelectionArea(self):
            """Update drawing selection area using self.points"""
            self.machine.setSelectionArea(self.points,
                                          fill='hatch',
                                          color=self.machine.color)
            eventDict = prepareDrawingSignal('drawingProgress',
                                             'polygon',
                                             self.points,
                                             self.machine.parameters)
            self.machine.plot.notify(**eventDict)

        def onWheel(self, x, y, angle):
            self.machine.onWheel(x, y, angle)
            self.updateFirstPoint()

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                # checking if the position is close to the first point
                # if yes : closing the "loop"
                firstPos = self.machine.plot.dataToPixel(*self._firstPos,
                                                         check=False)
                dx, dy = abs(firstPos[0] - x), abs(firstPos[1] - y)

                threshold = self.machine.getDragThreshold()

                # Only allow to close polygon after first point
                if len(self.points) > 2 and dx <= threshold and dy <= threshold:
                    self.machine.resetSelectionArea()

                    self.points[-1] = self.points[0]

                    eventDict = prepareDrawingSignal('drawingFinished',
                                                     'polygon',
                                                     self.points,
                                                     self.machine.parameters)
                    self.machine.plot.notify(**eventDict)
                    self.goto('idle')
                    return False

                # Update polygon last point not too close to previous one
                dataPos = self.machine.plot.pixelToData(x, y)
                assert dataPos is not None
                self.updateSelectionArea()

                # checking that the new points isnt the same (within range)
                # of the previous one
                # This has to be done because sometimes the mouse release event
                # is caught right after entering the Select state (i.e : press
                # in Idle state, but with a slightly different position that
                # the mouse press. So we had the two first vertices that were
                # almost identical.
                previousPos = self.machine.plot.dataToPixel(*self.points[-2],
                                                            check=False)
                dx, dy = abs(previousPos[0] - x), abs(previousPos[1] - y)
                if dx >= threshold or dy >= threshold:
                    self.points.append(dataPos)
                else:
                    self.points[-1] = dataPos

                return True
            return False

        def onMove(self, x, y):
            firstPos = self.machine.plot.dataToPixel(*self._firstPos,
                                                     check=False)
            dx, dy = abs(firstPos[0] - x), abs(firstPos[1] - y)
            threshold = self.machine.getDragThreshold()

            if dx <= threshold and dy <= threshold:
                x, y = firstPos  # Snap to first point

            dataPos = self.machine.plot.pixelToData(x, y)
            assert dataPos is not None
            self.points[-1] = dataPos
            self.updateSelectionArea()

    def __init__(self, plot, parameters):
        states = {
            'idle': SelectPolygon.Idle,
            'select': SelectPolygon.Select
        }
        super(SelectPolygon, self).__init__(plot, parameters,
                                            states, 'idle')

    def cancel(self):
        if isinstance(self.state, self.states['select']):
            self.resetSelectionArea()

    def getDragThreshold(self):
        """Return dragging ratio with device to pixel ratio applied.

        :rtype: float
        """
        ratio = 1.
        if qt.BINDING in ('PyQt5', 'PySide2'):
            ratio = self.plot.window().windowHandle().devicePixelRatio()
        return self.DRAG_THRESHOLD_DIST * ratio



class Select2Points(Select):
    """Base class for drawing selection based on 2 input points."""
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('start', x, y)
                return True

    class Start(State):
        def enterState(self, x, y):
            self.machine.beginSelect(x, y)

        def onMove(self, x, y):
            self.goto('select', x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enterState(self, x, y):
            self.onMove(x, y)

        def onMove(self, x, y):
            self.machine.select(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endSelect(x, y)
                self.goto('idle')

    def __init__(self, plot, parameters):
        states = {
            'idle': Select2Points.Idle,
            'start': Select2Points.Start,
            'select': Select2Points.Select
        }
        super(Select2Points, self).__init__(plot, parameters,
                                            states, 'idle')

    def beginSelect(self, x, y):
        pass

    def select(self, x, y):
        pass

    def endSelect(self, x, y):
        pass

    def cancelSelect(self):
        pass

    def cancel(self):
        if isinstance(self.state, self.states['select']):
            self.cancelSelect()


class SelectRectangle(Select2Points):
    """Drawing rectangle selection area state machine."""
    def beginSelect(self, x, y):
        self.startPt = self.plot.pixelToData(x, y)
        assert self.startPt is not None

    def select(self, x, y):
        dataPos = self.plot.pixelToData(x, y)
        assert dataPos is not None

        self.setSelectionArea((self.startPt,
                              (self.startPt[0], dataPos[1]),
                              dataPos,
                              (dataPos[0], self.startPt[1])),
                              fill='hatch',
                              color=self.color)

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'rectangle',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.plot.notify(**eventDict)

    def endSelect(self, x, y):
        self.resetSelectionArea()

        dataPos = self.plot.pixelToData(x, y)
        assert dataPos is not None

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'rectangle',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.plot.notify(**eventDict)

    def cancelSelect(self):
        self.resetSelectionArea()


class SelectLine(Select2Points):
    """Drawing line selection area state machine."""
    def beginSelect(self, x, y):
        self.startPt = self.plot.pixelToData(x, y)
        assert self.startPt is not None

    def select(self, x, y):
        dataPos = self.plot.pixelToData(x, y)
        assert dataPos is not None

        self.setSelectionArea((self.startPt, dataPos),
                              fill='hatch',
                              color=self.color)

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'line',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.plot.notify(**eventDict)

    def endSelect(self, x, y):
        self.resetSelectionArea()

        dataPos = self.plot.pixelToData(x, y)
        assert dataPos is not None

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'line',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.plot.notify(**eventDict)

    def cancelSelect(self):
        self.resetSelectionArea()


class Select1Point(Select):
    """Base class for drawing selection area based on one input point."""
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enterState(self, x, y):
            self.onMove(x, y)

        def onMove(self, x, y):
            self.machine.select(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endSelect(x, y)
                self.goto('idle')

        def onWheel(self, x, y, angle):
            self.machine.onWheel(x, y, angle)  # Call select default wheel
            self.machine.select(x, y)

    def __init__(self, plot, parameters):
        states = {
            'idle': Select1Point.Idle,
            'select': Select1Point.Select
        }
        super(Select1Point, self).__init__(plot, parameters, states, 'idle')

    def select(self, x, y):
        pass

    def endSelect(self, x, y):
        pass

    def cancelSelect(self):
        pass

    def cancel(self):
        if isinstance(self.state, self.states['select']):
            self.cancelSelect()


class SelectHLine(Select1Point):
    """Drawing a horizontal line selection area state machine."""
    def _hLine(self, y):
        """Return points in data coords of the segment visible in the plot.

        Supports non-orthogonal axes.
        """
        left, _top, width, _height = self.plot.getPlotBoundsInPixels()

        dataPos1 = self.plot.pixelToData(left, y, check=False)
        dataPos2 = self.plot.pixelToData(left + width, y, check=False)
        return dataPos1, dataPos2

    def select(self, x, y):
        points = self._hLine(y)
        self.setSelectionArea(points, fill='hatch', color=self.color)

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'hline',
                                         points,
                                         self.parameters)
        self.plot.notify(**eventDict)

    def endSelect(self, x, y):
        self.resetSelectionArea()

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'hline',
                                         self._hLine(y),
                                         self.parameters)
        self.plot.notify(**eventDict)

    def cancelSelect(self):
        self.resetSelectionArea()


class SelectVLine(Select1Point):
    """Drawing a vertical line selection area state machine."""
    def _vLine(self, x):
        """Return points in data coords of the segment visible in the plot.

        Supports non-orthogonal axes.
        """
        _left, top, _width, height = self.plot.getPlotBoundsInPixels()

        dataPos1 = self.plot.pixelToData(x, top, check=False)
        dataPos2 = self.plot.pixelToData(x, top + height, check=False)
        return dataPos1, dataPos2

    def select(self, x, y):
        points = self._vLine(x)
        self.setSelectionArea(points, fill='hatch', color=self.color)

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'vline',
                                         points,
                                         self.parameters)
        self.plot.notify(**eventDict)

    def endSelect(self, x, y):
        self.resetSelectionArea()

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'vline',
                                         self._vLine(x),
                                         self.parameters)
        self.plot.notify(**eventDict)

    def cancelSelect(self):
        self.resetSelectionArea()


class DrawFreeHand(Select):
    """Interaction for drawing pencil. It display the preview of the pencil
    before pressing the mouse.
    """

    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

        def onMove(self, x, y):
            self.machine.updatePencilShape(x, y)

        def onLeave(self):
            self.machine.cancel()

    class Select(State):
        def enterState(self, x, y):
            self.__isOut = False
            self.machine.setFirstPoint(x, y)

        def onMove(self, x, y):
            self.machine.updatePencilShape(x, y)
            self.machine.select(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                if self.__isOut:
                    self.machine.resetSelectionArea()
                self.machine.endSelect(x, y)
                self.goto('idle')

        def onEnter(self):
            self.__isOut = False

        def onLeave(self):
            self.__isOut = True

    def __init__(self, plot, parameters):
        # Circle used for pencil preview
        angle = numpy.arange(13.) * numpy.pi * 2.0 / 13.
        size = parameters.get('width', 1.) * 0.5
        self._circle = size * numpy.array((numpy.cos(angle),
                                           numpy.sin(angle))).T

        states = {
            'idle': DrawFreeHand.Idle,
            'select': DrawFreeHand.Select
        }
        super(DrawFreeHand, self).__init__(plot, parameters, states, 'idle')

    @property
    def width(self):
        return self.parameters.get('width', None)

    def setFirstPoint(self, x, y):
        self._points = []
        self.select(x, y)

    def updatePencilShape(self, x, y):
        center = self.plot.pixelToData(x, y, check=False)
        assert center is not None

        polygon = center + self._circle

        self.setSelectionArea(polygon, fill='none', color=self.color)

    def select(self, x, y):
        pos = self.plot.pixelToData(x, y, check=False)
        if len(self._points) > 0:
            if self._points[-1] == pos:
                # Skip same points
                return
        self._points.append(pos)
        eventDict = prepareDrawingSignal('drawingProgress',
                                         'polylines',
                                         self._points,
                                         self.parameters)
        self.plot.notify(**eventDict)

    def endSelect(self, x, y):
        pos = self.plot.pixelToData(x, y, check=False)
        if len(self._points) > 0:
            if self._points[-1] != pos:
                # Append if different
                self._points.append(pos)

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'polylines',
                                         self._points,
                                         self.parameters)
        self.plot.notify(**eventDict)
        self._points = None

    def cancelSelect(self):
        self.resetSelectionArea()

    def cancel(self):
        self.resetSelectionArea()


class SelectFreeLine(ClickOrDrag, _PlotInteraction):
    """Base class for drawing free lines with tools such as pencil."""

    def __init__(self, plot, parameters):
        """Init a state machine.

        :param plot: The plot to apply changes to.
        :param dict parameters: A dict of parameters such as color.
        """
        # self.DRAG_THRESHOLD_SQUARE_DIST = 1  # Disable first move threshold
        self._points = []
        ClickOrDrag.__init__(self)
        _PlotInteraction.__init__(self, plot)
        self.parameters = parameters

    def onWheel(self, x, y, angle):
        scaleF = 1.1 if angle > 0 else 1. / 1.1
        applyZoomToPlot(self.plot, scaleF, (x, y))

    @property
    def color(self):
        return self.parameters.get('color', None)

    def click(self, x, y, btn):
        if btn == LEFT_BTN:
            self._processEvent(x, y, isLast=True)

    def beginDrag(self, x, y):
        self._processEvent(x, y, isLast=False)

    def drag(self, x, y):
        self._processEvent(x, y, isLast=False)

    def endDrag(self, startPos, endPos):
        x, y = endPos
        self._processEvent(x, y, isLast=True)

    def cancel(self):
        self.resetSelectionArea()
        self._points = []

    def _processEvent(self, x, y, isLast):
        dataPos = self.plot.pixelToData(x, y, check=False)
        isNewPoint = not self._points or dataPos != self._points[-1]

        if isNewPoint:
            self._points.append(dataPos)

        if isNewPoint or isLast:
            eventDict = prepareDrawingSignal(
                'drawingFinished' if isLast else 'drawingProgress',
                'polylines',
                self._points,
                self.parameters)
            self.plot.notify(**eventDict)

        if not isLast:
            self.setSelectionArea(self._points, fill='none', color=self.color,
                                  shape='polylines')
        else:
            self.cancel()


# ItemInteraction #############################################################

class ItemsInteraction(ClickOrDrag, _PlotInteraction):
    """Interaction with items (markers, curves and images).

    This class provides selection and dragging of plot primitives
    that support those interaction.
    It is also meant to be combined with the zoom interaction.
    """

    class Idle(ClickOrDrag.Idle):
        def __init__(self, *args, **kw):
            super(ItemsInteraction.Idle, self).__init__(*args, **kw)
            self._hoverMarker = None

        def onWheel(self, x, y, angle):
            scaleF = 1.1 if angle > 0 else 1. / 1.1
            applyZoomToPlot(self.machine.plot, scaleF, (x, y))

        def onMove(self, x, y):
            marker = self.machine.plot._pickMarker(x, y)
            if marker is not None:
                dataPos = self.machine.plot.pixelToData(x, y)
                assert dataPos is not None
                eventDict = prepareHoverSignal(
                    marker.getLegend(), 'marker',
                    dataPos, (x, y),
                    marker.isDraggable(),
                    marker.isSelectable())
                self.machine.plot.notify(**eventDict)

            if marker != self._hoverMarker:
                self._hoverMarker = marker

                if marker is None:
                    self.machine.plot.setGraphCursorShape()

                elif marker.isDraggable():
                    if isinstance(marker, items.YMarker):
                        self.machine.plot.setGraphCursorShape(CURSOR_SIZE_VER)
                    elif isinstance(marker, items.XMarker):
                        self.machine.plot.setGraphCursorShape(CURSOR_SIZE_HOR)
                    else:
                        self.machine.plot.setGraphCursorShape(CURSOR_SIZE_ALL)

                elif marker.isSelectable():
                    self.machine.plot.setGraphCursorShape(CURSOR_POINTING)

            return True

    def __init__(self, plot):
        _PlotInteraction.__init__(self, plot)

        states = {
            'idle': ItemsInteraction.Idle,
            'rightClick': ClickOrDrag.RightClick,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        StateMachine.__init__(self, states, 'idle')

    def click(self, x, y, btn):
        """Handle mouse click

        :param x: X position of the mouse in pixels
        :param y: Y position of the mouse in pixels
        :param btn: Pressed button id
        :return: True if click is catched by an item, False otherwise
        """
        # Signal mouse clicked event
        dataPos = self.plot.pixelToData(x, y)
        assert dataPos is not None
        eventDict = prepareMouseSignal('mouseClicked', btn,
                                       dataPos[0], dataPos[1],
                                       x, y)
        self.plot.notify(**eventDict)

        eventDict = self._handleClick(x, y, btn)
        if eventDict is not None:
            self.plot.notify(**eventDict)

    def _handleClick(self, x, y, btn):
        """Perform picking and prepare event if click is handled here

        :param x: X position of the mouse in pixels
        :param y: Y position of the mouse in pixels
        :param btn: Pressed button id
        :return: event description to send of None if not handling event.
        :rtype: dict or None
        """

        if btn == LEFT_BTN:
            marker = self.plot._pickMarker(
                x, y, lambda m: m.isSelectable())
            if marker is not None:
                xData, yData = marker.getPosition()
                if xData is None:
                    xData = [0, 1]
                if yData is None:
                    yData = [0, 1]

                eventDict = prepareMarkerSignal('markerClicked',
                                                'left',
                                                marker.getLegend(),
                                                'marker',
                                                marker.isDraggable(),
                                                marker.isSelectable(),
                                                (xData, yData),
                                                (x, y), None)
                return eventDict

            else:
                picked = self.plot._pickImageOrCurve(
                    x, y, lambda item: item.isSelectable())

                if picked is None:
                    pass

                elif picked[0] == 'curve':
                    curve = picked[1]
                    indices = picked[2]

                    dataPos = self.plot.pixelToData(x, y)
                    assert dataPos is not None

                    xData = curve.getXData(copy=False)
                    yData = curve.getYData(copy=False)

                    eventDict = prepareCurveSignal('left',
                                                   curve.getLegend(),
                                                   'curve',
                                                   xData[indices],
                                                   yData[indices],
                                                   dataPos[0], dataPos[1],
                                                   x, y)
                    return eventDict

                elif picked[0] == 'image':
                    image = picked[1]

                    dataPos = self.plot.pixelToData(x, y)
                    assert dataPos is not None

                    # Get corresponding coordinate in image
                    origin = image.getOrigin()
                    scale = image.getScale()
                    column = int((dataPos[0] - origin[0]) / float(scale[0]))
                    row = int((dataPos[1] - origin[1]) / float(scale[1]))
                    eventDict = prepareImageSignal('left',
                                                   image.getLegend(),
                                                   'image',
                                                   column, row,
                                                   dataPos[0], dataPos[1],
                                                   x, y)
                    return eventDict

        return None

    def _signalMarkerMovingEvent(self, eventType, marker, x, y):
        assert marker is not None

        xData, yData = marker.getPosition()
        if xData is None:
            xData = [0, 1]
        if yData is None:
            yData = [0, 1]

        posDataCursor = self.plot.pixelToData(x, y)
        assert posDataCursor is not None

        eventDict = prepareMarkerSignal(eventType,
                                        'left',
                                        marker.getLegend(),
                                        'marker',
                                        marker.isDraggable(),
                                        marker.isSelectable(),
                                        (xData, yData),
                                        (x, y),
                                        posDataCursor)
        self.plot.notify(**eventDict)

    def beginDrag(self, x, y):
        """Handle begining of drag interaction

        :param x: X position of the mouse in pixels
        :param y: Y position of the mouse in pixels
        :return: True if drag is catched by an item, False otherwise
        """
        self._lastPos = self.plot.pixelToData(x, y)
        assert self._lastPos is not None

        self.imageLegend = None
        self.markerLegend = None
        marker = self.plot._pickMarker(
            x, y, lambda m: m.isDraggable())

        if marker is not None:
            self.markerLegend = marker.getLegend()
            self._signalMarkerMovingEvent('markerMoving', marker, x, y)
        else:
            picked = self.plot._pickImageOrCurve(
                x,
                y,
                lambda item:
                    hasattr(item, 'isDraggable') and item.isDraggable())
            if picked is None:
                self.imageLegend = None
                self.plot.setGraphCursorShape()
                return False
            else:
                assert picked[0] == 'image'  # For now only drag images
                self.imageLegend = picked[1].getLegend()
        return True

    def drag(self, x, y):
        dataPos = self.plot.pixelToData(x, y)
        assert dataPos is not None
        xData, yData = dataPos

        if self.markerLegend is not None:
            marker = self.plot._getMarker(self.markerLegend)
            if marker is not None:
                marker.setPosition(xData, yData)

                self._signalMarkerMovingEvent(
                    'markerMoving', marker, x, y)

        if self.imageLegend is not None:
            image = self.plot.getImage(self.imageLegend)
            origin = image.getOrigin()
            xImage = origin[0] + xData - self._lastPos[0]
            yImage = origin[1] + yData - self._lastPos[1]
            image.setOrigin((xImage, yImage))

        self._lastPos = xData, yData

    def endDrag(self, startPos, endPos):
        if self.markerLegend is not None:
            marker = self.plot._getMarker(self.markerLegend)
            posData = list(marker.getPosition())
            if posData[0] is None:
                posData[0] = [0, 1]
            if posData[1] is None:
                posData[1] = [0, 1]

            eventDict = prepareMarkerSignal(
                'markerMoved',
                'left',
                marker.getLegend(),
                'marker',
                marker.isDraggable(),
                marker.isSelectable(),
                posData)
            self.plot.notify(**eventDict)

        self.plot.setGraphCursorShape()

        del self.markerLegend
        del self.imageLegend
        del self._lastPos

    def cancel(self):
        self.plot.setGraphCursorShape()


class ItemsInteractionForCombo(ItemsInteraction):
    """Interaction with items to combine through :class:`FocusManager`.
    """

    class Idle(ItemsInteraction.Idle):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                def test(item):
                    return (item.isSelectable() or
                            (isinstance(item, items.DraggableMixIn) and
                             item.isDraggable()))

                picked = self.machine.plot._pickMarker(x, y, test)
                if picked is not None:
                    itemInteraction = True

                else:
                    picked = self.machine.plot._pickImageOrCurve(x, y, test)
                    itemInteraction = picked is not None

                if itemInteraction:  # Request focus and handle interaction
                    self.goto('clickOrDrag', x, y)
                    return True
                else:  # Do not request focus
                    return False

            elif btn == RIGHT_BTN:
                self.goto('rightClick', x, y)
                return True

    def __init__(self, plot):
        _PlotInteraction.__init__(self, plot)

        states = {
            'idle': ItemsInteractionForCombo.Idle,
            'rightClick': ClickOrDrag.RightClick,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        StateMachine.__init__(self, states, 'idle')


# FocusManager ################################################################

class FocusManager(StateMachine):
    """Manages focus across multiple event handlers

    On press an event handler can acquire focus.
    By default it looses focus when all buttons are released.
    """
    class Idle(State):
        def onPress(self, x, y, btn):
            for eventHandler in self.machine.eventHandlers:
                requestFocus = eventHandler.handleEvent('press', x, y, btn)
                if requestFocus:
                    self.goto('focus', eventHandler, btn)
                    break

        def _processEvent(self, *args):
            for eventHandler in self.machine.eventHandlers:
                consumeEvent = eventHandler.handleEvent(*args)
                if consumeEvent:
                    break

        def onMove(self, x, y):
            self._processEvent('move', x, y)

        def onRelease(self, x, y, btn):
            self._processEvent('release', x, y, btn)

        def onWheel(self, x, y, angle):
            self._processEvent('wheel', x, y, angle)

    class Focus(State):
        def enterState(self, eventHandler, btn):
            self.eventHandler = eventHandler
            self.focusBtns = {btn}

        def onPress(self, x, y, btn):
            self.focusBtns.add(btn)
            self.eventHandler.handleEvent('press', x, y, btn)

        def onMove(self, x, y):
            self.eventHandler.handleEvent('move', x, y)

        def onRelease(self, x, y, btn):
            self.focusBtns.discard(btn)
            requestFocus = self.eventHandler.handleEvent('release', x, y, btn)
            if len(self.focusBtns) == 0 and not requestFocus:
                self.goto('idle')

        def onWheel(self, x, y, angleInDegrees):
            self.eventHandler.handleEvent('wheel', x, y, angleInDegrees)

    def __init__(self, eventHandlers=()):
        self.eventHandlers = list(eventHandlers)

        states = {
            'idle': FocusManager.Idle,
            'focus': FocusManager.Focus
        }
        super(FocusManager, self).__init__(states, 'idle')

    def cancel(self):
        for handler in self.eventHandlers:
            handler.cancel()


class ZoomAndSelect(ItemsInteraction):
    """Combine Zoom and ItemInteraction state machine.

    :param plot: The Plot to which this interaction is attached
    :param color: The color to use for the zoom area bounding box
    """

    def __init__(self, plot, color):
        super(ZoomAndSelect, self).__init__(plot)
        self._zoom = Zoom(plot, color)
        self._doZoom = False

    @property
    def color(self):
        """Color of the zoom area"""
        return self._zoom.color

    def click(self, x, y, btn):
        """Handle mouse click

        :param x: X position of the mouse in pixels
        :param y: Y position of the mouse in pixels
        :param btn: Pressed button id
        :return: True if click is catched by an item, False otherwise
        """
        eventDict = self._handleClick(x, y, btn)

        if eventDict is not None:
            # Signal mouse clicked event
            dataPos = self.plot.pixelToData(x, y)
            assert dataPos is not None
            clickedEventDict = prepareMouseSignal('mouseClicked', btn,
                                                  dataPos[0], dataPos[1],
                                                  x, y)
            self.plot.notify(**clickedEventDict)

            self.plot.notify(**eventDict)

        else:
            self._zoom.click(x, y, btn)

    def beginDrag(self, x, y):
        """Handle start drag and switching between zoom and item drag.

        :param x: X position in pixels
        :param y: Y position in pixels
        """
        self._doZoom = not super(ZoomAndSelect, self).beginDrag(x, y)
        if self._doZoom:
            self._zoom.beginDrag(x, y)

    def drag(self, x, y):
        """Handle drag, eventually forwarding to zoom.

        :param x: X position in pixels
        :param y: Y position in pixels
        """
        if self._doZoom:
            return self._zoom.drag(x, y)
        else:
            return super(ZoomAndSelect, self).drag(x, y)

    def endDrag(self, startPos, endPos):
        """Handle end of drag, eventually forwarding to zoom.

        :param startPos: (x, y) position at the beginning of the drag
        :param endPos: (x, y) position at the end of the drag
        """
        if self._doZoom:
            return self._zoom.endDrag(startPos, endPos)
        else:
            return super(ZoomAndSelect, self).endDrag(startPos, endPos)


class PanAndSelect(ItemsInteraction):
    """Combine Pan and ItemInteraction state machine.

    :param plot: The Plot to which this interaction is attached
    """

    def __init__(self, plot):
        super(PanAndSelect, self).__init__(plot)
        self._pan = Pan(plot)
        self._doPan = False

    def click(self, x, y, btn):
        """Handle mouse click

        :param x: X position of the mouse in pixels
        :param y: Y position of the mouse in pixels
        :param btn: Pressed button id
        :return: True if click is catched by an item, False otherwise
        """
        eventDict = self._handleClick(x, y, btn)

        if eventDict is not None:
            # Signal mouse clicked event
            dataPos = self.plot.pixelToData(x, y)
            assert dataPos is not None
            clickedEventDict = prepareMouseSignal('mouseClicked', btn,
                                                  dataPos[0], dataPos[1],
                                                  x, y)
            self.plot.notify(**clickedEventDict)

            self.plot.notify(**eventDict)

        else:
            self._pan.click(x, y, btn)

    def beginDrag(self, x, y):
        """Handle start drag and switching between zoom and item drag.

        :param x: X position in pixels
        :param y: Y position in pixels
        """
        self._doPan = not super(PanAndSelect, self).beginDrag(x, y)
        if self._doPan:
            self._pan.beginDrag(x, y)

    def drag(self, x, y):
        """Handle drag, eventually forwarding to zoom.

        :param x: X position in pixels
        :param y: Y position in pixels
        """
        if self._doPan:
            return self._pan.drag(x, y)
        else:
            return super(PanAndSelect, self).drag(x, y)

    def endDrag(self, startPos, endPos):
        """Handle end of drag, eventually forwarding to zoom.

        :param startPos: (x, y) position at the beginning of the drag
        :param endPos: (x, y) position at the end of the drag
        """
        if self._doPan:
            return self._pan.endDrag(startPos, endPos)
        else:
            return super(PanAndSelect, self).endDrag(startPos, endPos)


# Interaction mode control ####################################################

class PlotInteraction(object):
    """Proxy to currently use state machine for interaction.

    This allows to switch interactive mode.

    :param plot: The :class:`Plot` to apply interaction to
    """

    _DRAW_MODES = {
        'polygon': SelectPolygon,
        'rectangle': SelectRectangle,
        'line': SelectLine,
        'vline': SelectVLine,
        'hline': SelectHLine,
        'polylines': SelectFreeLine,
        'pencil': DrawFreeHand,
    }

    def __init__(self, plot):
        self._plot = weakref.ref(plot)  # Avoid cyclic-ref

        self.zoomOnWheel = True
        """True to enable zoom on wheel, False otherwise."""

        # Default event handler
        self._eventHandler = ItemsInteraction(plot)

    def getInteractiveMode(self):
        """Returns the current interactive mode as a dict.

        The returned dict contains at least the key 'mode'.
        Mode can be: 'draw', 'pan', 'select', 'zoom'.
        It can also contains extra keys (e.g., 'color') specific to a mode
        as provided to :meth:`setInteractiveMode`.
        """
        if isinstance(self._eventHandler, ZoomAndSelect):
            return {'mode': 'zoom', 'color': self._eventHandler.color}

        elif isinstance(self._eventHandler, FocusManager):
            drawHandler = self._eventHandler.eventHandlers[1]
            if not isinstance(drawHandler, Select):
                raise RuntimeError('Unknown interactive mode')

            result = drawHandler.parameters.copy()
            result['mode'] = 'draw'
            return result

        elif isinstance(self._eventHandler, Select):
            result = self._eventHandler.parameters.copy()
            result['mode'] = 'draw'
            return result

        elif isinstance(self._eventHandler, PanAndSelect):
            return {'mode': 'pan'}

        else:
            return {'mode': 'select'}

    def setInteractiveMode(self, mode, color='black',
                           shape='polygon', label=None, width=None):
        """Switch the interactive mode.

        :param str mode: The name of the interactive mode.
                         In 'draw', 'pan', 'select', 'select-draw', 'zoom'.
        :param color: Only for 'draw' and 'zoom' modes.
                      Color to use for drawing selection area. Default black.
                      If None, selection area is not drawn.
        :type color: Color description: The name as a str or
                     a tuple of 4 floats or None.
        :param str shape: Only for 'draw' mode. The kind of shape to draw.
                          In 'polygon', 'rectangle', 'line', 'vline', 'hline',
                          'polylines'.
                          Default is 'polygon'.
        :param str label: Only for 'draw' mode.
        :param float width: Width of the pencil. Only for draw pencil mode.
        """
        assert mode in ('draw', 'pan', 'select', 'select-draw', 'zoom')

        plot = self._plot()
        assert plot is not None

        if color not in (None, 'video inverted'):
            color = colors.rgba(color)

        if mode in ('draw', 'select-draw'):
            assert shape in self._DRAW_MODES
            eventHandlerClass = self._DRAW_MODES[shape]
            parameters = {
                'shape': shape,
                'label': label,
                'color': color,
                'width': width,
            }
            eventHandler = eventHandlerClass(plot, parameters)

            self._eventHandler.cancel()

            if mode == 'draw':
                self._eventHandler = eventHandler

            else:  # mode == 'select-draw'
                self._eventHandler = FocusManager(
                    (ItemsInteractionForCombo(plot), eventHandler))

        elif mode == 'pan':
            # Ignores color, shape and label
            self._eventHandler.cancel()
            self._eventHandler = PanAndSelect(plot)

        elif mode == 'zoom':
            # Ignores shape and label
            self._eventHandler.cancel()
            self._eventHandler = ZoomAndSelect(plot, color)

        else:  # Default mode: interaction with plot objects
            # Ignores color, shape and label
            self._eventHandler.cancel()
            self._eventHandler = ItemsInteraction(plot)

    def handleEvent(self, event, *args, **kwargs):
        """Forward event to current interactive mode state machine."""
        if not self.zoomOnWheel and event == 'wheel':
            return  # Discard wheel events
        self._eventHandler.handleEvent(event, *args, **kwargs)
