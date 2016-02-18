# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
Implementation of the interaction for the OpenGL plot backend.
"""


# import ######################################################################

import math
import numpy as np
import time
import weakref

try:
    from ..PlotBackend import PlotBackend
except ImportError:
    from PyMca5.PyMcaGraph.PlotBackend import PlotBackend

from .GLSupport import rgba, FLOAT32_MINPOS, FLOAT32_SAFE_MIN, FLOAT32_SAFE_MAX
from .Interaction import ClickOrDrag, LEFT_BTN, RIGHT_BTN,\
    State, StateMachine
from .PlotEvents import prepareCurveSignal, prepareDrawingSignal,\
    prepareHoverSignal, prepareImageSignal,\
    prepareMarkerSignal, prepareMouseSignal


# Zoom/Pan ####################################################################

def _scale1DRange(min_, max_, center, scale, isLog):
    """Scale a 1D range given a scale factor and an center point.

    Keeps the values in a smaller range than float32.

    :param float min_: The current min value of the range.
    :param float max_: The current max value of the range.
    :param float center: The center of the zoom (i.e., invariant point).
    :param float scale: The scale to use for zoom
    :param bool isLog: Whether using log scale or not.
    :return: The zoomed range.
    :rtype: tuple of 2 floats: (min, max)
    """
    if isLog:
        # Min and center can be < 0 when
        # autoscale is off and switch to log scale
        # max_ < 0 should not happen
        min_ = np.log10(min_) if min_ > 0. else FLOAT32_MINPOS
        center = np.log10(center) if center > 0. else FLOAT32_MINPOS
        max_ = np.log10(max_) if max_ > 0. else FLOAT32_MINPOS

    if min_ == max_:
        return min_, max_

    offset = (center - min_) / (max_ - min_)
    range_ = (max_ - min_) / scale
    newMin = center - offset * range_
    newMax = center + (1. - offset) * range_

    if isLog:
        # No overflow as exponent is log10 of a float32
        newMin = pow(10., newMin)
        newMax = pow(10., newMax)
        newMin = np.clip(newMin, FLOAT32_MINPOS, FLOAT32_SAFE_MAX)
        newMax = np.clip(newMax, FLOAT32_MINPOS, FLOAT32_SAFE_MAX)
    else:
        newMin = np.clip(newMin, FLOAT32_SAFE_MIN, FLOAT32_SAFE_MAX)
        newMax = np.clip(newMax, FLOAT32_SAFE_MIN, FLOAT32_SAFE_MAX)
    return newMin, newMax


def _applyZoomToBackend(backend, cx, cy, scaleF):
    """Zoom in/out backend given a scale and a center point.

    :param PlotBackend backend: The backend on which to apply zoom.
    :param float cx: X coord in data coordinates of the zoom center.
    :param float cy: Y coord in data coordinates of the zoom center.
    :param float scaleF: Scale factor of zoom.
    """
    dataCenterPos = backend.pixelToData(cx, cy)
    assert dataCenterPos is not None

    xMin, xMax = backend.getGraphXLimits()
    xMin, xMax = _scale1DRange(xMin, xMax, dataCenterPos[0], scaleF,
                               backend.isXAxisLogarithmic())

    yMin, yMax = backend.getGraphYLimits()
    yMin, yMax = _scale1DRange(yMin, yMax, dataCenterPos[1], scaleF,
                               backend.isYAxisLogarithmic())

    dataPos = backend.pixelToData(y=cy, axis="right")
    assert dataPos is not None
    y2Center = dataPos[1]
    y2Min, y2Max = backend.getGraphYLimits(axis="right")
    y2Min, y2Max = _scale1DRange(y2Min, y2Max, y2Center, scaleF,
                                 backend.isYAxisLogarithmic())

    backend.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)
    backend.replot()


class _ZoomOnWheel(ClickOrDrag):
    """:class:`ClickOrDrag` state machine with zooming on mouse wheel.

    Base class for :class:`Pan` and :class:`Zoom`
    """
    class ZoomIdle(ClickOrDrag.Idle):
        def onWheel(self, x, y, angle):
            scaleF = 1.1 if angle > 0 else 1./1.1
            _applyZoomToBackend(self.machine.backend, x, y, scaleF)

    def __init__(self, backend):
        """

        :param backend: The backend to apply modifications to.
        """
        self._backend = weakref.ref(backend)  # Avoid cyclic-ref

        states = {
            'idle': _ZoomOnWheel.ZoomIdle,
            'rightClick': ClickOrDrag.RightClick,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        StateMachine.__init__(self, states, 'idle')

    @property
    def backend(self):
        backend = self._backend()
        assert backend is not None
        return backend


# Pan #########################################################################

class Pan(_ZoomOnWheel):
    """Pan plot content and zoom on wheel state machine."""

    def _pixelToData(self, x, y):
        xData, yData = self.backend.pixelToData(x, y)
        _, y2Data = self.backend.pixelToData(x, y, axis='right')
        return xData, yData, y2Data

    def beginDrag(self, x, y):
        self._previousDataPos = self._pixelToData(x, y)

    def drag(self, x, y):
        xData, yData, y2Data = self._pixelToData(x, y)
        lastX, lastY, lastY2 = self._previousDataPos

        xMin, xMax = self.backend.getGraphXLimits()
        yMin, yMax = self.backend.getGraphYLimits(axis='left')
        y2Min, y2Max = self.backend.getGraphYLimits(axis='right')

        if self.backend.isXAxisLogarithmic():
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

        if self.backend.isYAxisLogarithmic():
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

        self.backend.setLimits(newXMin, newXMax,
                               newYMin, newYMax,
                               newY2Min, newY2Max)
        self.backend.replot()

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
    _DOUBLE_CLICK_TIMEOUT = 0.4

    def __init__(self, backend, color):
        self.color = color
        self.zoomStack = []
        self._lastClick = 0., None

        super(Zoom, self).__init__(backend)

    def _areaWithAspectRatio(self, x0, y0, x1, y1):
        plotW, plotH = self.backend.plotSizeInPixels()

        areaX0, areaY0, areaX1, areaY1 = x0, y0, x1, y1

        if plotH != 0.:
            plotRatio = plotW / float(plotH)
            width, height = math.fabs(x1 - x0), math.fabs(y1 - y0)

            if height != 0. and width != 0.:
                if width / height > plotRatio:
                    areaHeight = width / plotRatio
                    areaX0, areaX1 = x0, x1
                    center = 0.5 * (y0 + y1)
                    areaY0 = center - np.sign(y1 - y0) * 0.5 * areaHeight
                    areaY1 = center + np.sign(y1 - y0) * 0.5 * areaHeight
                else:
                    areaWidth = height * plotRatio
                    areaY0, areaY1 = y0, y1
                    center = 0.5 * (x0 + x1)
                    areaX0 = center - np.sign(x1 - x0) * 0.5 * areaWidth
                    areaX1 = center + np.sign(x1 - x0) * 0.5 * areaWidth

        return areaX0, areaY0, areaX1, areaY1

    def click(self, x, y, btn):
        if btn == LEFT_BTN:
            lastClickTime, lastClickPos = self._lastClick

            # Signal mouse double clicked event first
            if (time.time() - lastClickTime) <= self._DOUBLE_CLICK_TIMEOUT:
                # Use position of first click
                eventDict = prepareMouseSignal('mouseDoubleClicked', 'left',
                                               *lastClickPos)
                self.backend.sendEvent(eventDict)

                self._lastClick = 0., None
            else:
                # Signal mouse clicked event
                dataPos = self.backend.pixelToData(x, y)
                assert dataPos is not None
                eventDict = prepareMouseSignal('mouseClicked', 'left',
                                               dataPos[0], dataPos[1],
                                               x, y)
                self.backend.sendEvent(eventDict)

                self._lastClick = time.time(), (dataPos[0], dataPos[1], x, y)

            # Zoom-in centered on mouse cursor
            # xMin, xMax = self.backend.getGraphXLimits()
            # yMin, yMax = self.backend.getGraphYLimits()
            # y2Min, y2Max = self.backend.getGraphYLimits(axis="right")
            # self.zoomStack.append((xMin, xMax, yMin, yMax, y2Min, y2Max))
            # self._zoom(x, y, 2)
        elif btn == RIGHT_BTN:
            try:
                xMin, xMax, yMin, yMax, y2Min, y2Max = self.zoomStack.pop()
            except IndexError:
                # Signal mouse clicked event
                dataPos = self.backend.pixelToData(x, y)
                assert dataPos is not None
                eventDict = prepareMouseSignal('mouseClicked', 'right',
                                               dataPos[0], dataPos[1],
                                               x, y)
                self.backend.sendEvent(eventDict)
            else:
                self.backend.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)
            self.backend.replot()

    def beginDrag(self, x, y):
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None
        self.x0, self.y0 = x, y

    def drag(self, x1, y1):
        dataPos = self.backend.pixelToData(x1, y1)
        assert dataPos is not None

        if self.backend.isKeepDataAspectRatio():
            area = self._areaWithAspectRatio(self.x0, self.y0, x1, y1)
            areaX0, areaY0, areaX1, areaY1 = area
            areaPoints = ((areaX0, areaY0),
                          (areaX1, areaY0),
                          (areaX1, areaY1),
                          (areaX0, areaY1))
            areaPoints = np.array([self.backend.pixelToData(x, y, check=False)
                for (x, y) in areaPoints])

            if self.color != 'video inverted':
                areaColor = list(self.color)
                areaColor[3] *= 0.25
            else:
                areaColor = [1., 1., 1., 1.]

            self.backend.setSelectionArea(areaPoints,
                                          fill=None,
                                          color=areaColor,
                                          name="zoomedArea")

        corners = ((self.x0, self.y0),
                  (self.x0, y1),
                  (x1, y1),
                  (x1, self.y0))
        corners = np.array([self.backend.pixelToData(x, y, check=False)
            for (x, y) in corners])

        self.backend.setSelectionArea(corners,
                                      fill=None,
                                      color=self.color)
        self.backend.replot()

    def endDrag(self, startPos, endPos):
        x0, y0 = startPos
        x1, y1 = endPos

        if x0 != x1 or y0 != y1:  # Avoid empty zoom area
            # Store current zoom state in stack
            xMin, xMax = self.backend.getGraphXLimits()
            yMin, yMax = self.backend.getGraphYLimits()
            y2Min, y2Max = self.backend.getGraphYLimits(axis="right")
            self.zoomStack.append((xMin, xMax, yMin, yMax, y2Min, y2Max))

            if self.backend.isKeepDataAspectRatio():
                x0, y0, x1, y1 = self._areaWithAspectRatio(x0, y0, x1, y1)

            if self.backend.isDefaultBaseVectors():
                # Convert to data space and set limits
                x0, y0 = self.backend.pixelToData(x0, y0, check=False)

                dataPos = self.backend.pixelToData(
                    y=startPos[1], axis="right", check=False)
                y2_0 = dataPos[1]

                x1, y1 = self.backend.pixelToData(x1, y1, check=False)

                dataPos = self.backend.pixelToData(
                    y=endPos[1], axis="right", check=False)
                y2_1 = dataPos[1]

                xMin, xMax = min(x0, x1), max(x0, x1)
                yMin, yMax = min(y0, y1), max(y0, y1)
                y2Min, y2Max = min(y2_0, y2_1), max(y2_0, y2_1)

                self.backend.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

            else:  # Non-orthogonal axes
                # Get zoom bbox in pixels
                xPixelMin, xPixelMax = min(x0, x1), max(x0, x1)
                yPixelMin, yPixelMax = min(y0, y1), max(y0, y1)

                # Take the data bounds that fit entirely in the bbox
                corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
                corners = np.array([self.backend.pixelToData(x, y, check=False)
                    for (x, y) in corners], dtype=np.float32)

                # Drop the 2 extermes and use the other 2 as the range
                xMin, xMax = np.sort(corners[:, 0])[1:3]
                yMin, yMax = np.sort(corners[:, 1])[1:3]
                # TODO y2 axis
                self.backend.setLimits(xMin, xMax, yMin, yMax)

        self.backend.resetSelectionArea()
        self.backend.replot()

    def cancel(self):
        if isinstance(self.state, self.states['drag']):
            self.backend.resetSelectionArea()
            self.backend.replot()


# Select ######################################################################

class Select(StateMachine):
    """Base class for drawing selection areas."""

    def __init__(self, backend, parameters, states, state):
        """Init a state machine.

        :param backend: The backend to apply changes to.
        :param dict parameters: A dict of parameters such as color.
        :param dict states: The states of the state machine.
        :param str state: The name of the initial state.
        """
        self._backend = weakref.ref(backend)  # Avoid cyclic-ref
        self.parameters = parameters
        super(Select, self).__init__(states, state)

    def onWheel(self, x, y, angle):
        scaleF = 1.1 if angle > 0 else 1./1.1
        _applyZoomToBackend(self.backend, x, y, scaleF)

    @property
    def backend(self):
        backend = self._backend()
        assert backend is not None
        return backend

    @property
    def color(self):
        return self.parameters.get('color', None)


class SelectPolygon(Select):
    """Drawing selection polygon area state machine."""
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enter(self, x, y):
            dataPos = self.machine.backend.pixelToData(x, y)
            assert dataPos is not None
            self.points = [dataPos, dataPos]

        def updateSelectionArea(self):
            self.machine.backend.setSelectionArea(self.points,
                                                  fill='hatch',
                                                  color=self.machine.color)
            self.machine.backend.replot()
            eventDict = prepareDrawingSignal('drawingProgress',
                                             'polygon',
                                             self.points,
                                             self.machine.parameters)
            self.machine.backend.sendEvent(eventDict)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                dataPos = self.machine.backend.pixelToData(x, y)
                assert dataPos is not None
                self.points[-1] = dataPos
                self.updateSelectionArea()
                if self.points[-2] != self.points[-1]:
                    self.points.append(dataPos)
                return True

        def onMove(self, x, y):
            dataPos = self.machine.backend.pixelToData(x, y)
            assert dataPos is not None
            self.points[-1] = dataPos
            self.updateSelectionArea()

        def onPress(self, x, y, btn):
            if btn == RIGHT_BTN:
                self.machine.backend.resetSelectionArea()
                self.machine.backend.replot()

                dataPos = self.machine.backend.pixelToData(x, y)
                assert dataPos is not None
                self.points[-1] = dataPos
                if self.points[-2] == self.points[-1]:
                    self.points.pop()
                self.points.append(self.points[0])

                eventDict = prepareDrawingSignal('drawingFinished',
                                                 'polygon',
                                                 self.points,
                                                 self.machine.parameters)
                self.machine.backend.sendEvent(eventDict)
                self.goto('idle')

    def __init__(self, backend, parameters):
        states = {
            'idle': SelectPolygon.Idle,
            'select': SelectPolygon.Select
        }
        super(SelectPolygon, self).__init__(backend, parameters,
                                            states, 'idle')

    def cancel(self):
        if isinstance(self.state, self.states['select']):
            self.backend.resetSelectionArea()
            self.backend.replot()


class Select2Points(Select):
    """Base class for drawing selection based on 2 input points."""
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('start', x, y)
                return True

    class Start(State):
        def enter(self, x, y):
            self.machine.beginSelect(x, y)

        def onMove(self, x, y):
            self.goto('select', x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enter(self, x, y):
            self.onMove(x, y)

        def onMove(self, x, y):
            self.machine.select(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endSelect(x, y)
                self.goto('idle')

    def __init__(self, backend, parameters):
        states = {
            'idle': Select2Points.Idle,
            'start': Select2Points.Start,
            'select': Select2Points.Select
        }
        super(Select2Points, self).__init__(backend, parameters,
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
        self.startPt = self.backend.pixelToData(x, y)
        assert self.startPt is not None

    def select(self, x, y):
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None

        self.backend.setSelectionArea((self.startPt,
                                      (self.startPt[0], dataPos[1]),
                                      dataPos,
                                      (dataPos[0], self.startPt[1])),
                                      fill='hatch',
                                      color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'rectangle',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.backend.sendEvent(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'rectangle',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.backend.sendEvent(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


class SelectLine(Select2Points):
    """Drawing line selection area state machine."""
    def beginSelect(self, x, y):
        self.startPt = self.backend.pixelToData(x, y)
        assert self.startPt is not None

    def select(self, x, y):
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None

        self.backend.setSelectionArea((self.startPt, dataPos),
                                      fill='hatch',
                                      color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'line',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.backend.sendEvent(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'line',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.backend.sendEvent(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


class Select1Point(Select):
    """Base class for drawing selection area based on one input point."""
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enter(self, x, y):
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

    def __init__(self, backend, parameters):
        states = {
            'idle': Select1Point.Idle,
            'select': Select1Point.Select
        }
        super(Select1Point, self).__init__(backend, parameters,
                                           states, 'idle')

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
        plotWidth, plotHeight = self.backend.plotSizeInPixels()
        plotLeft, plotTop = self.backend.plotOriginInPixels()

        dataPos1 = self.backend.pixelToData(plotLeft, y, check=False)
        dataPos2 = self.backend.pixelToData(
            plotLeft + plotWidth, y, check=False)
        return dataPos1, dataPos2

    def select(self, x, y):
        points = self._hLine(y)
        self.backend.setSelectionArea(points, fill='hatch', color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'hline',
                                         points,
                                         self.parameters)
        self.backend.sendEvent(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'hline',
                                         self._hLine(y),
                                         self.parameters)
        self.backend.sendEvent(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


class SelectVLine(Select1Point):
    """Drawing a vertical line selection area state machine."""
    def _vLine(self, x):
        """Return points in data coords of the segment visible in the plot.

        Supports non-orthogonal axes.
        """
        plotWidth, plotHeight = self.backend.plotSizeInPixels()
        plotLeft, plotTop = self.backend.plotOriginInPixels()

        dataPos1 = self.backend.pixelToData(x, plotTop, check=False)
        dataPos2 = self.backend.pixelToData(
            x, plotTop + plotHeight, check=False)
        return dataPos1, dataPos2

    def select(self, x, y):
        points = self._vLine(x)
        self.backend.setSelectionArea(points, fill='hatch', color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'vline',
                                         points,
                                         self.parameters)
        self.backend.sendEvent(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'vline',
                                         self._vLine(x),
                                         self.parameters)
        self.backend.sendEvent(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


# ItemInteraction #############################################################

class ItemsInteraction(ClickOrDrag):
    class Idle(ClickOrDrag.Idle):
        def __init__(self, *args, **kw):
            super(ItemsInteraction.Idle, self).__init__(*args, **kw)
            self._hoverMarker = None

        def onWheel(self, x, y, angle):
            scaleF = 1.1 if angle > 0 else 1./1.1
            _applyZoomToBackend(self.machine.backend, x, y, scaleF)

        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                testBehaviors = set(('selectable', 'draggable'))

                marker = self.machine.backend.pickMarker(
                    x, y,
                    lambda marker: marker['behaviors'] & testBehaviors)
                if marker is not None:
                    self.goto('clickOrDrag', x, y)
                    return True

                else:
                    picked = self.machine.backend.pickImageOrCurve(
                        x,
                        y,
                        lambda item: item.info['behaviors'] & testBehaviors)
                    if picked is not None:
                        self.goto('clickOrDrag', x, y)
                        return True

            return False

        def onMove(self, x, y):
            marker = self.machine.backend.pickMarker(x, y)
            if marker is not None:
                dataPos = self.machine.backend.pixelToData(x, y)
                assert dataPos is not None
                eventDict = prepareHoverSignal(
                    marker['legend'], 'marker',
                    dataPos, (x, y),
                    'draggable' in marker['behaviors'],
                    'selectable' in marker['behaviors'])
                self.machine.backend.sendEvent(eventDict)

            if marker != self._hoverMarker:
                self._hoverMarker = marker

                if marker is None:
                    self.machine.backend.setCursor()

                elif 'draggable' in marker['behaviors']:
                    if marker['x'] is None:
                        self.machine.backend.setCursor('size vertical')
                    elif marker['y'] is None:
                        self.machine.backend.setCursor('size horizontal')
                    else:
                        self.machine.backend.setCursor('size all')

                elif 'selectable' in marker['behaviors']:
                    self.machine.backend.setCursor('pointing')

            return True

    def __init__(self, backend):
        self._backend = weakref.ref(backend)  # Avoid cyclic-ref

        states = {
            'idle': ItemsInteraction.Idle,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        StateMachine.__init__(self, states, 'idle')

    @property
    def backend(self):
        backend = self._backend()
        assert backend is not None
        return backend

    def click(self, x, y, btn):
        # Signal mouse clicked event
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None
        eventDict = prepareMouseSignal('mouseClicked', btn,
                                       dataPos[0], dataPos[1],
                                       x, y)
        self.backend.sendEvent(eventDict)

        if btn == LEFT_BTN:
            marker = self.backend.pickMarker(
                x, y, lambda marker: 'selectable' in marker['behaviors'])
            if marker is not None:
                # Mimic MatplotlibBackend signal
                xData, yData = marker['x'], marker['y']
                if xData is None:
                    xData = [0, 1]
                if yData is None:
                    yData = [0, 1]

                draggable = 'draggable' in marker['behaviors']
                selectable = 'selectable' in marker['behaviors']
                eventDict = prepareMarkerSignal('markerClicked',
                                                'left',
                                                marker['legend'],
                                                'marker',
                                                draggable,
                                                selectable,
                                                (xData, yData),
                                                (x, y), None)
                self.backend.sendEvent(eventDict)

                self.backend.replot()
            else:
                picked = self.backend.pickImageOrCurve(
                    x,
                    y,
                    lambda item: 'selectable' in item.info['behaviors'])

                if picked is None:
                    pass
                elif picked[0] == 'curve':
                    _, curve, indices = picked
                    dataPos = self.backend.pixelToData(x, y)
                    assert dataPos is not None
                    eventDict = prepareCurveSignal('left',
                                                   curve.info['legend'],
                                                   'curve',
                                                   curve.xData[indices],
                                                   curve.yData[indices],
                                                   dataPos[0], dataPos[1],
                                                   x, y)
                    self.backend.sendEvent(eventDict)

                elif picked[0] == 'image':
                    _, image, posImg = picked

                    dataPos = self.backend.pixelToData(x, y)
                    assert dataPos is not None
                    eventDict = prepareImageSignal('left',
                                                   image.info['legend'],
                                                   'image',
                                                   posImg[0], posImg[1],
                                                   dataPos[0], dataPos[1],
                                                   x, y)
                    self.backend.sendEvent(eventDict)

    def _signalMarkerMovingEvent(self, eventType, marker, x, y):
        assert marker is not None

        # Mimic MatplotlibBackend signal
        xData, yData = marker['x'], marker['y']
        if xData is None:
            xData = [0, 1]
        if yData is None:
            yData = [0, 1]

        posDataCursor = self.backend.pixelToData(x, y)
        assert posDataCursor is not None

        eventDict = prepareMarkerSignal(eventType,
                                        'left',
                                        marker['legend'],
                                        'marker',
                                        'draggable' in marker['behaviors'],
                                        'selectable' in marker['behaviors'],
                                        (xData, yData),
                                        (x, y),
                                        posDataCursor)
        self.backend.sendEvent(eventDict)

    def beginDrag(self, x, y):
        self._lastPos = self.backend.pixelToData(x, y)
        assert self._lastPos is not None

        self.image = None
        self.marker = self.backend.pickMarker(
            x, y, lambda marker: 'draggable' in marker['behaviors'])
        if self.marker is not None:
            self._signalMarkerMovingEvent('markerMoving', self.marker, x, y)
        else:
            picked = self.backend.pickImageOrCurve(
                x,
                y,
                lambda item: 'draggable' in item.info['behaviors'])
            if picked is None:
                self.image = None
                self.backend.setCursor()
            else:
                assert picked[0] == 'image'  # For now, only drag images
                self.image = picked[1]

    def drag(self, x, y):
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None
        xData, yData = dataPos

        if self.marker is not None:
            if self.marker['constraint'] is not None:
                xData, yData = self.marker['constraint'](xData, yData)

            if self.marker['x'] is not None:
                self.marker['x'] = xData
            if self.marker['y'] is not None:
                self.marker['y'] = yData

            self._signalMarkerMovingEvent('markerMoving', self.marker, x, y)

            self.backend.replot()

        if self.image is not None:
            dx, dy = xData - self._lastPos[0], yData - self._lastPos[1]
            self.image.xMin += dx
            self.image.yMin += dy

            self.backend._plotDirtyFlag = True
            self.backend.replot()

        self._lastPos = xData, yData

    def endDrag(self, startPos, endPos):
        if self.marker is not None:
            posData = [self.marker['x'], self.marker['y']]
            # Mimic MatplotlibBackend signal
            if posData[0] is None:
                posData[0] = [0, 1]
            if posData[1] is None:
                posData[1] = [0, 1]

            eventDict = prepareMarkerSignal(
                'markerMoved',
                'left',
                self.marker['legend'],
                'marker',
                'draggable' in self.marker['behaviors'],
                'selectable' in self.marker['behaviors'],
                posData)
            self.backend.sendEvent(eventDict)

        self.backend.setCursor()

        del self.marker
        del self.image
        del self._lastPos

    def cancel(self):
        self.backend.setCursor()


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
        def enter(self, eventHandler, btn):
            self.eventHandler = eventHandler
            self.focusBtns = set((btn,))

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


class ZoomAndSelect(FocusManager):
    """Combine Zoom and ItemInteraction state machine."""
    def __init__(self, backend, color):
        eventHandlers = ItemsInteraction(backend), Zoom(backend, color)
        super(ZoomAndSelect, self).__init__(eventHandlers)

    @property
    def color(self):
        return self.eventHandlers[1].color


# Interaction mode control ####################################################

class PlotInteraction(object):
    """Proxy to currently use state machine for interaction.

    This allows to switch interactive mode.
    """

    _DRAW_MODES = {
        'polygon': SelectPolygon,
        'rectangle': SelectRectangle,
        'line': SelectLine,
        'vline': SelectVLine,
        'hline': SelectHLine,
    }

    def __init__(self, backend):
        self._backend = weakref.ref(backend)  # Avoid cyclic-ref

        # Default event handler
        self._eventHandler = ItemsInteraction(backend)

    def getInteractiveMode(self):
        """Returns the current interactive mode as a dict.

        The returned dict contains at least the key 'mode'.
        Mode can be: 'draw', 'pan', 'select', 'zoom'.
        It can also contains extra keys (e.g., 'color') specific to a mode
        as provided to :meth:`setInteractiveMode`.
        """
        if isinstance(self._eventHandler, ZoomAndSelect):
            return {'mode': 'zoom', 'color': self._eventHandler.color}

        elif isinstance(self._eventHandler, Select):
            result = self._eventHandler.parameters.copy()
            result['mode'] = 'draw'
            return result

        elif isinstance(self._eventHandler, Pan):
            return {'mode': 'pan'}

        else:
            return {'mode': 'select'}

    def setInteractiveMode(self, mode, color=None,
                           shape='polygon', label=None):
        """Switch the interactive mode.

        :param str mode: The name of the interactive mode.
                         In 'draw', 'pan', 'select', 'zoom'.
        :param color: Only for 'draw' and 'zoom' modes.
                      Color to use for drawing selection area. Default black.
        :type color: Color description: The name as a str or
                     a tuple of 4 floats.
        :param str shape: Only for 'draw' mode. The kind of shape to draw.
                          In 'polygon', 'rectangle', 'line', 'vline', 'hline'.
                          Default is 'polygon'.
        :param str label: Only for 'draw' mode.
        """
        assert mode in ('draw', 'pan', 'select', 'zoom')

        if color is None:
            color = 'black'

        backend = self._backend()
        assert backend is not None

        if mode == 'draw':
            assert shape in self._DRAW_MODES
            eventHandlerClass = self._DRAW_MODES[shape]
            parameters = {
                'shape': shape,
                'label': label,
                'color': rgba(color, PlotBackend.COLORDICT)
            }

            self._eventHandler.cancel()
            self._eventHandler = eventHandlerClass(backend, parameters)

        elif mode == 'pan':
            # Ignores color, shape and label
            self._eventHandler.cancel()
            self._eventHandler = Pan(backend)

        elif mode == 'zoom':
            # Ignores shape and label
            if color != 'video inverted':
                color = rgba(color, PlotBackend.COLORDICT)
            self._eventHandler.cancel()
            self._eventHandler = ZoomAndSelect(backend, color)

        else:  # Default mode: interaction with plot objects
            # Ignores color, shape and label
            self._eventHandler.cancel()
            self._eventHandler = ItemsInteraction(backend)

    def handleEvent(self, *args, **kwargs):
        """Forward event to current interactive mode state machine."""
        self._eventHandler.handleEvent(*args, **kwargs)
