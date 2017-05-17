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
__date__ = "17/05/2017"


import unittest
import numpy
from silx.gui import qt
from .. import items
from .. import PlotEvents


class TestEvents(unittest.TestCase):
    """Test object events"""

    def testLimitsChangedEvent(self):
        source = "a"
        xRange = "b"
        yRange = "c"
        y2Range = "d"
        event = PlotEvents.LimitsChangedEvent(
            source, xRange, yRange, y2Range)
        self.assertEquals(event.getType(), PlotEvents.Type.LimitChanged)
        self.assertEquals(event.getSource(), source)
        self.assertEquals(event.getXRange(), xRange)
        self.assertEquals(event.getYRange(), yRange)
        self.assertEquals(event.getYRange('left'), yRange)
        self.assertEquals(event.getYRange('right'), y2Range)

    def testMouseEvent(self):
        event = "e"
        button = "b"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseEvent(event, button, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), "e")
        self.assertEquals(event.getButton(), "b")
        self.assertEquals(event.getScenePos()[0], xData)
        self.assertEquals(event.getScenePos()[1], yData)
        self.assertEquals(event.getScreenPos()[0], xPixel)
        self.assertEquals(event.getScreenPos()[1], yPixel)

    def testMouseClickedEvent(self):
        button = "b"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseClickedEvent(button, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), PlotEvents.Type.MouseClicked)

    def testMouseDoubleClickedEvent(self):
        button = "b"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseDoubleClickedEvent(button, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), PlotEvents.Type.MouseDoubleClicked)

    def testMouseMovedEvent(self):
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseMovedEvent([xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), PlotEvents.Type.MouseMoved)
        self.assertEquals(event.getButton(), qt.Qt.NoButton)

    def testItemClickedEvent(self):
        button = "b"
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        item = "foo"
        itemIndices = [10]
        event = PlotEvents.ItemClickedEvent(button, item, itemIndices, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event.getType(), PlotEvents.Type.ItemClicked)
        self.assertEquals(event.getButton(), "b")
        self.assertEquals(event.getScenePos()[0], xData)
        self.assertEquals(event.getScenePos()[1], yData)
        self.assertEquals(event.getScreenPos()[0], xPixel)
        self.assertEquals(event.getScreenPos()[1], yPixel)
        self.assertEquals(event.getItem(), item)
        self.assertEquals(event.getItemIndices(), itemIndices)

    def testItemRegionChangeEvent(self):
        item = "foo"
        eventType = "e"
        event = PlotEvents.ItemRegionChangeEvent(eventType, item)
        self.assertEquals(event.getType(), eventType)
        self.assertEquals(event.getItem(), item)

    def testItemRegionChangeFinishedEvent(self):
        event = PlotEvents.ItemRegionChangeFinishedEvent("item")
        self.assertEquals(event.getType(), PlotEvents.Type.RegionChangeFinished)

    def testItemRegionChangeStrartedEvent(self):
        event = PlotEvents.ItemRegionChangeStartedEvent("item")
        self.assertEquals(event.getType(), PlotEvents.Type.RegionChangeStarted)

    def testItemRegionChangedEvent(self):
        event = PlotEvents.ItemRegionChangedEvent("item")
        self.assertEquals(event.getType(), PlotEvents.Type.RegionChanged)

    def testItemHoverdEvent(self):
        posData = [10, 11]
        posPixel = [12, 13]
        item = "a"
        event = PlotEvents.ItemHoveredEvent(item, posData, posPixel)
        self.assertEquals(event.getType(), PlotEvents.Type.ItemHovered)
        self.assertEquals(event.getItem(), item)
        self.assertEquals(event.getScenePos(), posData)
        self.assertEquals(event.getScreenPos(), posPixel)

    def testInteractiveModeChangedEvent(self):
        source = "a"
        event = PlotEvents.InteractiveModeChangedEvent(source)
        self.assertEquals(event.getType(), PlotEvents.Type.InteractiveModeChanged)
        self.assertEquals(event.getSource(), source)

    def testChildAddedEvent(self):
        legend = "l"
        item = items.ImageData()
        item._setLegend(legend)
        event = PlotEvents.ChildAddedEvent(item)
        self.assertEquals(event.getType(), PlotEvents.Type.ChildAdded)
        self.assertIs(event.getChild(), item)

    def testChildRemovedEvent(self):
        legend = "l"
        item = items.ImageData()
        item._setLegend(legend)
        event = PlotEvents.ChildRemovedEvent(item)
        self.assertEquals(event.getType(), PlotEvents.Type.ChildRemoved)
        self.assertIs(event.getChild(), item)

    def testCursorChangedEvent(self):
        state = "s"
        event = PlotEvents.CursorChangedEvent(state)
        self.assertEquals(event.getType(), PlotEvents.Type.CursorChanged)
        self.assertEquals(event.getState(), state)

    def testActiveItemChanged(self):
        legend1 = "l1"
        item1 = items.ImageData()
        item1._setLegend(legend1)
        legend2 = "l2"
        item2 = items.ImageData()
        item2._setLegend(legend2)
        updated = True
        event = PlotEvents.ActiveItemChangedEvent(item1, item2, updated)
        self.assertEquals(event.getType(), PlotEvents.Type.ActiveItemChanged)
        self.assertIs(event.getActiveItem(), item1)
        self.assertIs(event.getPreviousActiveItem(), item2)
        self.assertIs(event.isUpdated(), updated)

    def testGridChangedEvent(self):
        which = "w"
        event = PlotEvents.GridChangedEvent(which)
        self.assertEquals(event.getType(), PlotEvents.Type.GridChanged)
        self.assertEquals(event.getWhich(), which)


class TestDictionaryLikeGetter(unittest.TestCase):
    """Test old getter in plot events. Events have to support a set of
    dictionnary-like getter"""

    def testDrawingProgressEvent(self):
        points = numpy.array([[-1, 0], [1, 1], [2, 2]])
        item = items.Shape('rectangle')
        item.setPoints(points)
        event = PlotEvents.ItemRegionChangedEvent(item)

        self.assertEquals(event['event'], "drawingProgress")
        self.assertEquals(event['type'], "rectangle")
        numpy.testing.assert_almost_equal(event['points'], points)
        numpy.testing.assert_almost_equal(event['xdata'], points[:, 0])
        numpy.testing.assert_almost_equal(event['ydata'], points[:, 1])
        self.assertEquals(event['x'], -1)
        self.assertEquals(event['y'], 0)
        self.assertEquals(event['width'], 3)
        self.assertEquals(event['height'], 2)
        self.assertEquals(event['parameters'], {})

    def testDrawingFinishedEvent(self):
        points = numpy.array([[0, 1], [1, 1]])
        item = items.Shape('vline')
        item.setPoints(points)
        event = PlotEvents.ItemRegionChangeFinishedEvent(item)

        self.assertEquals(event['event'], "drawingFinished")
        self.assertEquals(event['type'], "vline")
        numpy.testing.assert_almost_equal(event['points'], points)
        numpy.testing.assert_almost_equal(event['xdata'], points[:, 0])
        numpy.testing.assert_almost_equal(event['ydata'], points[:, 1])
        self.assertEquals(event['parameters'], {})

    def testMouseEvent(self):
        button = qt.Qt.LeftButton
        xData = "xd"
        yData = "yd"
        xPixel = "xp"
        yPixel = "yp"
        event = PlotEvents.MouseClickedEvent(button, [xData, yData], [xPixel, yPixel])
        self.assertEquals(event['event'], "mouseClicked")
        self.assertEquals(event['button'], "left")
        self.assertEquals(event['x'], xData)
        self.assertEquals(event['y'], yData)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testHoverEvent(self):
        label = "a"
        posData = [10, 11]
        posPixel = [12, 13]
        item = items.YMarker()
        item._setLegend(label)
        item._setSelectable(True)
        item._setDraggable(False)
        event = PlotEvents.ItemHoveredEvent(item, posData, posPixel)
        self.assertEquals(event['event'], "hover")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['x'], posData[0])
        self.assertEquals(event['y'], posData[1])
        self.assertEquals(event['xpixel'], posPixel[0])
        self.assertEquals(event['ypixel'], posPixel[1])
        self.assertEquals(event['draggable'], False)
        self.assertEquals(event['selectable'], True)

    def testImageEvent(self):
        button = qt.Qt.LeftButton
        label = "b"
        eventType = "image"
        col = "d"
        row = "e"
        x = "f"
        y = "g"
        xPixel = "h"
        yPixel = "i"
        image = items.ImageBase()
        image._setLegend(label)
        event = PlotEvents.ItemClickedEvent(
            button,
            image,
            [(row, col)],
            (x, y),
            (xPixel, yPixel))
        self.assertEquals(event['event'], "imageClicked")
        self.assertEquals(event['button'], "left")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], eventType)
        self.assertEquals(event['col'], col)
        self.assertEquals(event['row'], row)
        self.assertEquals(event['x'], x)
        self.assertEquals(event['y'], y)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testCurveEvent(self):
        button = qt.Qt.RightButton
        label = "b"
        eventType = "curve"
        xData2 = 11
        yData2 = 12
        x = "f"
        y = "g"
        xPixel = "h"
        yPixel = "i"
        index = 2
        xData = [0, 0, xData2, 0, 0]
        yData = [0, 0, yData2, 0, 0]
        curve = items.Curve()
        curve._setLegend(label)
        curve.setData(xData, yData)
        event = PlotEvents.ItemClickedEvent(
            button,
            curve,
            index,
            (x, y),
            (xPixel, yPixel))
        self.assertEquals(event['event'], "curveClicked")
        self.assertEquals(event['button'], "right")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], eventType)
        self.assertEquals(event['xdata'], xData2)
        self.assertEquals(event['ydata'], yData2)
        self.assertEquals(event['x'], x)
        self.assertEquals(event['y'], y)
        self.assertEquals(event['xpixel'], xPixel)
        self.assertEquals(event['ypixel'], yPixel)

    def testLimitsChangedEvent(self):
        source = "a"

        class Source(object):
            def getWidgetHandle(self):
                return source

        xRange = "b"
        yRange = "c"
        y2Range = "d"
        event = PlotEvents.LimitsChangedEvent(
            Source(), xRange, yRange, y2Range)
        self.assertEquals(event['event'], "limitsChanged")
        self.assertEquals(event['source'], id(source))
        self.assertEquals(event['xdata'], xRange)
        self.assertEquals(event['ydata'], yRange)
        self.assertEquals(event['y2data'], y2Range)

    def testMarkerMovingEvent(self):
        eventType = "markerMoving"
        label = "l"
        draggable = True
        selectable = False
        posDataMarker = [10, 11]
        posPixelCursor = [12, 13]
        posDataCursor = [14, 15]

        item = items.marker.Marker()
        item._setLegend(label)
        item.setPosition(posDataMarker[0], posDataMarker[1])
        item._setDraggable(draggable)
        item._setSelectable(selectable)

        event = PlotEvents.ItemRegionChangedEvent(item)
        event._setMousePosition(posDataCursor, posPixelCursor)
        self.assertEquals(event['event'], eventType)
        self.assertEquals(event['button'], "left")
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], "marker")
        self.assertEquals(event['x'], posDataCursor[0])
        self.assertEquals(event['y'], posDataCursor[1])
        self.assertEquals(event['xdata'], posDataMarker[0])
        self.assertEquals(event['ydata'], posDataMarker[1])
        self.assertEquals(event['draggable'], draggable)
        self.assertEquals(event['selectable'], selectable)
        self.assertEquals(event['xpixel'], posPixelCursor[0])
        self.assertEquals(event['ypixel'], posPixelCursor[1])

    def testMarkerClicked(self):
        button = "left"
        label = "l"
        draggable = True
        selectable = False
        posDataMarker = (10, 11)
        posPixelCursor = (12, 13)
        scenePos = (14, 15)
        index = "foo"
        item = items.marker.Marker()
        item._setLegend(label)
        item.setPosition(posDataMarker[0], posDataMarker[1])
        item._setDraggable(draggable)
        item._setSelectable(selectable)

        event = PlotEvents.ItemClickedEvent(button, item, index, scenePos, posPixelCursor)
        self.assertEquals(event['event'], "markerClicked")
        self.assertEquals(event['button'], button)
        self.assertEquals(event['label'], label)
        self.assertEquals(event['type'], "marker")
        self.assertEquals(event['x'], posDataMarker[0])
        self.assertEquals(event['y'], posDataMarker[1])
        self.assertEquals(event['xdata'], posDataMarker[0])
        self.assertEquals(event['ydata'], posDataMarker[1])
        self.assertEquals(event['draggable'], draggable)
        self.assertEquals(event['selectable'], selectable)
        self.assertEquals(event['xpixel'], posPixelCursor[0])
        self.assertEquals(event['ypixel'], posPixelCursor[1])

    def testInteractiveModeChanged(self):
        source = "a"
        event = PlotEvents.InteractiveModeChangedEvent(source)
        self.assertEquals(event['event'], "interactiveModeChanged")
        self.assertEquals(event['source'], source)

    def testContentAdded(self):
        legend = "l"
        item = items.ImageData()
        item._setLegend(legend)
        event = PlotEvents.ChildAddedEvent(item)
        self.assertEquals(event['event'], "contentChanged")
        self.assertEquals(event['action'], "add")
        self.assertEquals(event['kind'], "image")
        self.assertEquals(event['legend'], legend)

    def testContentRemoved(self):
        legend = "l"
        item = items.ImageData()
        item._setLegend(legend)
        event = PlotEvents.ChildRemovedEvent(item)
        self.assertEquals(event['event'], "contentChanged")
        self.assertEquals(event['action'], "remove")
        self.assertEquals(event['kind'], "image")
        self.assertEquals(event['legend'], legend)

    def testSetGraphCursor(self):
        state = "s"
        event = PlotEvents.CursorChangedEvent(state)
        self.assertEquals(event['event'], "setGraphCursor")
        self.assertEquals(event['state'], state)

    def testActiveItemChanged(self):
        legend1 = "l1"
        item1 = items.ImageData()
        item1._setLegend(legend1)
        legend2 = "l2"
        item2 = items.ImageData()
        item2._setLegend(legend2)
        updated = True
        event = PlotEvents.ActiveItemChangedEvent(item1, item2, updated)
        self.assertEquals(event['event'], "activeImageChanged")
        self.assertEquals(event['updated'], updated)
        self.assertEquals(event['previous'], legend2)
        self.assertEquals(event['legend'], legend1)

    def testActiveItemChangedWithNone(self):
        legend1 = "l1"
        item1 = items.ImageData()
        item1._setLegend(legend1)
        updated = True
        event = PlotEvents.ActiveItemChangedEvent(None, item1, updated)
        self.assertEquals(event['event'], "activeImageChanged")
        self.assertEquals(event['legend'], None)

    def testActiveItemChangedWithNone2(self):
        legend1 = "l1"
        item1 = items.ImageData()
        item1._setLegend(legend1)
        updated = True
        event = PlotEvents.ActiveItemChangedEvent(item1, None, updated)
        self.assertEquals(event['event'], "activeImageChanged")
        self.assertEquals(event['previous'], None)

    def testSetGraphGrid(self):
        which = "w"
        event = PlotEvents.GridChangedEvent(which)
        self.assertEquals(event['event'], "setGraphGrid")
        self.assertEquals(event['which'], which)


def suite():
    test_suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loader(TestEvents))
    test_suite.addTest(loader(TestDictionaryLikeGetter))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
