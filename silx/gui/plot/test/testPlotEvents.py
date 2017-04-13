# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""Basic tests for Plot"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "13/04/2017"


import unittest
import numpy
from .. import PlotEvents


class TestEvents(unittest.TestCase):
    """Test object events"""

    def testLimitsChangedEvent(self):
        sourceObj = "a"
        xRange = "b"
        yRange = "c"
        y2Range = "d"
        event = PlotEvents.LimitsChangedEvent(
            sourceObj, xRange, yRange, y2Range)
        self.assertEquals(event.getType(), PlotEvents.Type.LimitChanged)
        self.assertEquals(event.getXRange(), xRange)
        self.assertEquals(event.getYRange(), yRange)
        self.assertEquals(event.getY2Range(), y2Range)


class TestDictionaryLikeGetter(unittest.TestCase):
    """Test old getter in plot events. Events have to support a set of
    dictionnary-like getter"""

    def testDrawingProgressEvent(self):
        eventName = "drawingProgress"
        eventType = "rectangle"
        points = [[-1, 0], [1, 1], [2, 2]]
        params = {"foobar"}
        event = PlotEvents.prepareDrawingSignal(eventName, eventType, points, params)

        self.assertEquals(event['event'], eventName)
        self.assertEquals(event['type'], eventType)
        numpy.testing.assert_almost_equal(event['points'], numpy.array(points))
        numpy.testing.assert_almost_equal(event['xdata'], numpy.array(points)[:, 0])
        numpy.testing.assert_almost_equal(event['ydata'], numpy.array(points)[:, 1])
        self.assertEquals(event['x'], -1)
        self.assertEquals(event['y'], 0)
        self.assertEquals(event['width'], 3)
        self.assertEquals(event['height'], 2)
        self.assertEquals(event['parameters'], params)

    def testDrawingFinishedEvent(self):
        eventName = "drawingFinished"
        eventType = "rectangle"
        points = [[-1, 0], [1, 1], [2, 2]]
        params = {"foobar"}
        event = PlotEvents.prepareDrawingSignal(eventName, eventType, points, params)

        self.assertEquals(event['event'], eventName)
        self.assertEquals(event['type'], eventType)
        numpy.testing.assert_almost_equal(event['points'], numpy.array(points))
        numpy.testing.assert_almost_equal(event['xdata'], numpy.array(points)[:, 0])
        numpy.testing.assert_almost_equal(event['ydata'], numpy.array(points)[:, 1])
        self.assertEquals(event['x'], -1)
        self.assertEquals(event['y'], 0)
        self.assertEquals(event['width'], 3)
        self.assertEquals(event['height'], 2)
        self.assertEquals(event['parameters'], params)

    def testMouseEvent(self):
        eventType = "mouseMoved"
        button = "left"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.prepareMouseSignal(eventType, button, xData, yData, xPixel, yPixel)
        self.assertEquals(event['event'], eventType)
        self.assertEquals(event['button'], button)
        self.assertEquals(event['x'], xData)
        self.assertEquals(event['y'], yData)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testHoverEvent(self):
        label = "a"
        eventType = "b"
        posData = ["c0", "c1"]
        posPixel = ["d0", "d1"]
        draggable = "e"
        selectable = "f"
        event = PlotEvents.prepareHoverSignal(label, eventType, posData, posPixel, draggable, selectable)
        self.assertEquals(event['event'], "hover")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['x'], posData[0])
        self.assertEquals(event['y'], posData[1])
        self.assertEquals(event['xpixel'], posPixel[0])
        self.assertEquals(event['ypixel'], posPixel[1])
        self.assertEquals(event['draggable'], draggable)
        self.assertEquals(event['selectable'], selectable)

    def testImageEvent(self):
        button = "a"
        label = "b"
        eventType = "c"
        col = "d"
        row = "e"
        x = "f"
        y = "g"
        xPixel = "h"
        yPixel = "i"
        event = PlotEvents.prepareImageSignal(
            button, label, eventType, col, row, x, y, xPixel, yPixel)
        self.assertEquals(event['event'], "imageClicked")
        self.assertEquals(event['button'], button)
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], eventType)
        self.assertEquals(event['col'], col)
        self.assertEquals(event['row'], row)
        self.assertEquals(event['x'], x)
        self.assertEquals(event['y'], y)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testCurveEvent(self):
        button = "a"
        label = "b"
        eventType = "c"
        xData = "d"
        yData = "e"
        x = "f"
        y = "g"
        xPixel = "h"
        yPixel = "i"
        event = PlotEvents.prepareCurveSignal(
            button, label, eventType, xData, yData, x, y, xPixel, yPixel)
        self.assertEquals(event['event'], "curveClicked")
        self.assertEquals(event['button'], button)
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], eventType)
        self.assertEquals(event['xdata'], xData)
        self.assertEquals(event['ydata'], yData)
        self.assertEquals(event['x'], x)
        self.assertEquals(event['y'], y)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testLimitsChangedEvent(self):
        sourceObj = "a"
        xRange = "b"
        yRange = "c"
        y2Range = "d"
        event = PlotEvents.LimitsChangedEvent(
            sourceObj, xRange, yRange, y2Range)
        self.assertEquals(event['event'], "limitsChanged")
        self.assertEquals(event['source'], id(sourceObj))
        self.assertEquals(event['xdata'], xRange)
        self.assertEquals(event['ydata'], yRange)
        self.assertEquals(event['y2data'], y2Range)

    def testMarkerEvent(self):
        eventType = "markerMoving"
        button = "b"
        label = "l"
        type_ = "t"
        draggable = "d"
        selectable = "s"
        posDataMarker = ["dm0", "dm1"]
        posPixelCursor = ["pc0", "pc1"]
        posDataCursor = ["dc0", "dc1"]
        event = PlotEvents.prepareMarkerSignal(
            eventType, button, label, type_, draggable, selectable,
            posDataMarker, posPixelCursor, posDataCursor)
        self.assertEquals(event['event'], eventType)
        self.assertEquals(event['button'], button)
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], type_)
        self.assertEquals(event['x'], posDataCursor[0])
        self.assertEquals(event['y'], posDataCursor[1])
        self.assertEquals(event['xdata'], posDataMarker[0])
        self.assertEquals(event['ydata'], posDataMarker[1])
        self.assertEquals(event['draggable'], draggable)
        self.assertEquals(event['selectable'], selectable)
        self.assertEquals(event['xpixel'], posPixelCursor[0])
        self.assertEquals(event['ypixel'], posPixelCursor[1])


def suite():
    test_suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loader(TestEvents))
    test_suite.addTest(loader(TestDictionaryLikeGetter))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
