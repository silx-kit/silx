# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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
"""Tests for PlotWidget items."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/09/2017"


import unittest

import numpy

from silx.gui.utils.testutils import SignalListener
from silx.gui.plot.items import ItemChangedType
from silx.gui.plot import items
from .utils import PlotWidgetTestCase


class TestSigItemChangedSignal(PlotWidgetTestCase):
    """Test item's sigItemChanged signal"""

    def testCurveChanged(self):
        """Test sigItemChanged for curve"""
        self.plot.addCurve(numpy.arange(10), numpy.arange(10), legend='test')
        curve = self.plot.getCurve('test')

        listener = SignalListener()
        curve.sigItemChanged.connect(listener)

        # Test for signal in Item class
        curve.setVisible(False)
        curve.setVisible(True)
        curve.setZValue(100)

        # Test for signals in PointsBase class
        curve.setData(numpy.arange(100), numpy.arange(100))

        # SymbolMixIn
        curve.setSymbol('Circle')
        curve.setSymbol('d')
        curve.setSymbolSize(20)

        # AlphaMixIn
        curve.setAlpha(0.5)

        # Test for signals in Curve class
        # ColorMixIn
        curve.setColor('yellow')
        # YAxisMixIn
        curve.setYAxis('right')
        # FillMixIn
        curve.setFill(True)
        # LineMixIn
        curve.setLineStyle(':')
        curve.setLineStyle(':')  # Not sending event
        curve.setLineWidth(2)

        self.assertEqual(listener.arguments(argumentIndex=0),
                         [ItemChangedType.VISIBLE,
                          ItemChangedType.VISIBLE,
                          ItemChangedType.ZVALUE,
                          ItemChangedType.DATA,
                          ItemChangedType.SYMBOL,
                          ItemChangedType.SYMBOL,
                          ItemChangedType.SYMBOL_SIZE,
                          ItemChangedType.ALPHA,
                          ItemChangedType.COLOR,
                          ItemChangedType.YAXIS,
                          ItemChangedType.FILL,
                          ItemChangedType.LINE_STYLE,
                          ItemChangedType.LINE_WIDTH])

    def testHistogramChanged(self):
        """Test sigItemChanged for Histogram"""
        self.plot.addHistogram(
            numpy.arange(10), edges=numpy.arange(11), legend='test')
        histogram = self.plot.getHistogram('test')
        listener = SignalListener()
        histogram.sigItemChanged.connect(listener)

        # Test signals in Histogram class
        histogram.setData(numpy.zeros(10), numpy.arange(11))

        self.assertEqual(listener.arguments(argumentIndex=0),
                         [ItemChangedType.DATA])

    def testImageDataChanged(self):
        """Test sigItemChanged for ImageData"""
        self.plot.addImage(numpy.arange(100).reshape(10, 10), legend='test')
        image = self.plot.getImage('test')

        listener = SignalListener()
        image.sigItemChanged.connect(listener)

        # ColormapMixIn
        colormap = self.plot.getDefaultColormap().copy()
        image.setColormap(colormap)
        image.getColormap().setName('viridis')

        # Test of signals in ImageBase class
        image.setOrigin(10)
        image.setScale(2)

        # Test of signals in ImageData class
        image.setData(numpy.ones((10, 10)))

        self.assertEqual(listener.arguments(argumentIndex=0),
                         [ItemChangedType.COLORMAP,
                          ItemChangedType.COLORMAP,
                          ItemChangedType.POSITION,
                          ItemChangedType.SCALE,
                          ItemChangedType.COLORMAP,
                          ItemChangedType.DATA])

    def testImageRgbaChanged(self):
        """Test sigItemChanged for ImageRgba"""
        self.plot.addImage(numpy.ones((10, 10, 3)), legend='rgb')
        image = self.plot.getImage('rgb')

        listener = SignalListener()
        image.sigItemChanged.connect(listener)

        # Test of signals in ImageRgba class
        image.setData(numpy.zeros((10, 10, 3)))

        self.assertEqual(listener.arguments(argumentIndex=0),
                         [ItemChangedType.DATA])

    def testMarkerChanged(self):
        """Test sigItemChanged for markers"""
        self.plot.addMarker(10, 20, legend='test')
        marker = self.plot._getMarker('test')

        listener = SignalListener()
        marker.sigItemChanged.connect(listener)

        # Test signals in _BaseMarker
        marker.setPosition(10, 10)
        marker.setPosition(10, 10)  # Not sending event
        marker.setText('toto')
        self.assertEqual(listener.arguments(argumentIndex=0),
                         [ItemChangedType.POSITION,
                          ItemChangedType.TEXT])

        # XMarker
        self.plot.addXMarker(10, legend='x')
        marker = self.plot._getMarker('x')

        listener = SignalListener()
        marker.sigItemChanged.connect(listener)
        marker.setPosition(20, 20)
        self.assertEqual(listener.arguments(argumentIndex=0),
                         [ItemChangedType.POSITION])

        # YMarker
        self.plot.addYMarker(10, legend='x')
        marker = self.plot._getMarker('x')

        listener = SignalListener()
        marker.sigItemChanged.connect(listener)
        marker.setPosition(20, 20)
        self.assertEqual(listener.arguments(argumentIndex=0),
                         [ItemChangedType.POSITION])

    def testScatterChanged(self):
        """Test sigItemChanged for scatter"""
        data = numpy.arange(10)
        self.plot.addScatter(data, data, data, legend='test')
        scatter = self.plot.getScatter('test')

        listener = SignalListener()
        scatter.sigItemChanged.connect(listener)

        # ColormapMixIn
        scatter.getColormap().setName('viridis')

        # Test of signals in Scatter class
        scatter.setData((0, 1, 2), (1, 0, 2), (0, 1, 2))

        # Visualization mode changed
        scatter.setVisualization(scatter.Visualization.SOLID)

        self.assertEqual(listener.arguments(),
                         [(ItemChangedType.COLORMAP,),
                          (ItemChangedType.DATA,),
                          (ItemChangedType.COLORMAP,),
                          (ItemChangedType.VISUALIZATION_MODE,)])

    def testShapeChanged(self):
        """Test sigItemChanged for shape"""
        data = numpy.array((1., 10.))
        self.plot.addShape(data, data, legend='test', shape='rectangle')
        shape = self.plot._getItem(kind='item', legend='test')

        listener = SignalListener()
        shape.sigItemChanged.connect(listener)

        shape.setOverlay(True)
        shape.setPoints(((2., 2.), (3., 3.)))

        self.assertEqual(listener.arguments(),
                         [(ItemChangedType.OVERLAY,),
                          (ItemChangedType.DATA,)])


class TestSymbol(PlotWidgetTestCase):
    """Test item's symbol """

    def test(self):
        """Test sigItemChanged for curve"""
        self.plot.addCurve(numpy.arange(10), numpy.arange(10), legend='test')
        curve = self.plot.getCurve('test')

        # SymbolMixIn
        curve.setSymbol('o')
        name = curve.getSymbolName()
        self.assertEqual('Circle', name)

        name = curve.getSymbolName('d')
        self.assertEqual('Diamond', name)


class TestVisibleExtent(PlotWidgetTestCase):
    """Test item's visible extent feature"""

    def testGetVisibleBounds(self):
        """Test Item.getVisibleBounds"""

        # Create test items (with a bounding box of x: [1,3], y: [0,2])
        curve = items.Curve()
        curve.setData((1, 2, 3), (0, 1, 2))

        histogram = items.Histogram()
        histogram.setData((0, 1, 2), (1, 5/3, 7/3, 3))

        image = items.ImageData()
        image.setOrigin((1, 0))
        image.setData(numpy.arange(4).reshape(2, 2))

        scatter = items.Scatter()
        scatter.setData((1, 2, 3), (0, 1, 2), (1, 2, 3))

        bbox = items.BoundingRect()
        bbox.setBounds((1, 3, 0, 2))

        xaxis, yaxis = self.plot.getXAxis(), self.plot.getYAxis()
        for item in (curve, histogram, image, scatter, bbox):
            with self.subTest(item=item):
                xaxis.setLimits(0, 100)
                yaxis.setLimits(0, 100)
                self.plot.addItem(item)
                self.assertEqual(item.getVisibleBounds(), (1., 3., 0., 2.))

                xaxis.setLimits(0.5, 2.5)
                self.assertEqual(item.getVisibleBounds(), (1, 2.5, 0., 2.))

                yaxis.setLimits(0.5, 1.5)
                self.assertEqual(item.getVisibleBounds(), (1, 2.5, 0.5, 1.5))

                item.setVisible(False)
                self.assertIsNone(item.getVisibleBounds())

                self.plot.clear()

    def testVisibleExtentTracking(self):
        """Test Item's visible extent tracking"""
        image = items.ImageData()
        image.setData(numpy.arange(6).reshape(2, 3))

        listener = SignalListener()
        image._sigVisibleBoundsChanged.connect(listener)
        image._setVisibleBoundsTracking(True)
        self.assertTrue(image._isVisibleBoundsTracking())

        self.plot.addItem(image)
        self.assertEqual(listener.callCount(), 1)

        self.plot.getXAxis().setLimits(0, 1)
        self.assertEqual(listener.callCount(), 2)

        self.plot.hide()
        self.qapp.processEvents()
        # No event here
        self.assertEqual(listener.callCount(), 2)

        self.plot.getXAxis().setLimits(1, 2)
        # No event since PlotWidget is hidden, delayed to PlotWidget show
        self.assertEqual(listener.callCount(), 2)

        self.plot.show()
        self.qapp.processEvents()
        # Receives delayed event now
        self.assertEqual(listener.callCount(), 3)

        image.setOrigin((-1, -1))
        self.assertEqual(listener.callCount(), 4)

        image.setVisible(False)
        image.setOrigin((0, 0))
        # No event since item is not visible
        self.assertEqual(listener.callCount(), 4)

        image.setVisible(True)
        # Receives delayed event now
        self.assertEqual(listener.callCount(), 5)


class TestImageDataAggregated(PlotWidgetTestCase):
    """Test ImageDataAggregated item"""

    def test(self):
        data = numpy.random.random(1024**2).reshape(1024, 1024)

        item = items.ImageDataAggregated()
        item.setData(data)
        self.assertEqual(item.getAggregationMode(), item.Aggregation.NONE)
        self.plot.addItem(item)

        for mode in item.Aggregation.members():
            with self.subTest(mode=mode):
                self.plot.resetZoom()
                self.qapp.processEvents()

                item.setAggregationMode(mode)
                self.qapp.processEvents()

                # Zoom-out
                for i in range(4):
                    xmin, xmax = self.plot.getXAxis().getLimits()
                    ymin, ymax =  self.plot.getYAxis().getLimits()
                    self.plot.setLimits(
                        xmin - (xmax - xmin)/2,
                        xmax + (xmax - xmin)/2,
                        ymin - (ymax - ymin)/2,
                        ymax + (ymax - ymin)/2,
                    )
                    self.qapp.processEvents()
