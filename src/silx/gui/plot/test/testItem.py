# /*##########################################################################
#
# Copyright (c) 2017-2023 European Synchrotron Radiation Facility
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


import numpy
import pytest

from silx.gui.utils.testutils import SignalListener
from silx.gui.plot.items.roi import RegionOfInterest
from silx.gui.plot.items import ItemChangedType
from silx.gui.plot import items
from .utils import PlotWidgetTestCase


class TestSigItemChangedSignal(PlotWidgetTestCase):
    """Test item's sigItemChanged signal"""

    def testCurveChanged(self):
        """Test sigItemChanged for curve"""
        self.plot.addCurve(numpy.arange(10), numpy.arange(10), legend="test")
        curve = self.plot.getCurve("test")

        listener = SignalListener()
        curve.sigItemChanged.connect(listener)

        # Test for signal in Item class
        curve.setVisible(False)
        curve.setVisible(True)
        curve.setZValue(100)

        # Test for signals in PointsBase class
        curve.setData(numpy.arange(100), numpy.arange(100))

        # SymbolMixIn
        curve.setSymbol("Circle")
        curve.setSymbol("d")
        curve.setSymbolSize(20)

        # AlphaMixIn
        curve.setAlpha(0.5)

        # Test for signals in Curve class
        # ColorMixIn
        curve.setColor("yellow")
        # YAxisMixIn
        curve.setYAxis("right")
        # FillMixIn
        curve.setFill(True)
        # LineMixIn
        curve.setLineStyle(":")
        curve.setLineStyle(":")  # Not sending event
        curve.setLineWidth(2)

        self.assertEqual(
            listener.arguments(argumentIndex=0),
            [
                ItemChangedType.VISIBLE,
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
                ItemChangedType.LINE_WIDTH,
            ],
        )

    def testHistogramChanged(self):
        """Test sigItemChanged for Histogram"""
        self.plot.addHistogram(numpy.arange(10), edges=numpy.arange(11), legend="test")
        histogram = self.plot.getHistogram("test")
        listener = SignalListener()
        histogram.sigItemChanged.connect(listener)

        # Test signals in Histogram class
        histogram.setData(numpy.zeros(10), numpy.arange(11))

        self.assertEqual(listener.arguments(argumentIndex=0), [ItemChangedType.DATA])

    def testImageDataChanged(self):
        """Test sigItemChanged for ImageData"""
        self.plot.addImage(numpy.arange(100).reshape(10, 10), legend="test")
        image = self.plot.getImage("test")

        listener = SignalListener()
        image.sigItemChanged.connect(listener)

        # ColormapMixIn
        colormap = self.plot.getDefaultColormap().copy()
        image.setColormap(colormap)
        image.getColormap().setName("viridis")

        # Test of signals in ImageBase class
        image.setOrigin(10)
        image.setScale(2)

        # Test of signals in ImageData class
        image.setData(numpy.ones((10, 10)))

        self.assertEqual(
            listener.arguments(argumentIndex=0),
            [
                ItemChangedType.COLORMAP,
                ItemChangedType.COLORMAP,
                ItemChangedType.POSITION,
                ItemChangedType.SCALE,
                ItemChangedType.COLORMAP,
                ItemChangedType.DATA,
            ],
        )

    def testImageRgbaChanged(self):
        """Test sigItemChanged for ImageRgba"""
        self.plot.addImage(numpy.ones((10, 10, 3)), legend="rgb")
        image = self.plot.getImage("rgb")

        listener = SignalListener()
        image.sigItemChanged.connect(listener)

        # Test of signals in ImageRgba class
        image.setData(numpy.zeros((10, 10, 3)))

        self.assertEqual(listener.arguments(argumentIndex=0), [ItemChangedType.DATA])

    def testMarkerChanged(self):
        """Test sigItemChanged for markers"""
        self.plot.addMarker(10, 20, legend="test")
        marker = self.plot._getMarker("test")

        listener = SignalListener()
        marker.sigItemChanged.connect(listener)

        # Test signals in _BaseMarker
        marker.setPosition(10, 10)
        marker.setPosition(10, 10)  # Not sending event
        marker.setText("toto")
        self.assertEqual(
            listener.arguments(argumentIndex=0),
            [ItemChangedType.POSITION, ItemChangedType.TEXT],
        )

        # XMarker
        self.plot.addXMarker(10, legend="x")
        marker = self.plot._getMarker("x")

        listener = SignalListener()
        marker.sigItemChanged.connect(listener)
        marker.setPosition(20, 20)
        self.assertEqual(
            listener.arguments(argumentIndex=0), [ItemChangedType.POSITION]
        )

        # YMarker
        self.plot.addYMarker(10, legend="x")
        marker = self.plot._getMarker("x")

        listener = SignalListener()
        marker.sigItemChanged.connect(listener)
        marker.setPosition(20, 20)
        self.assertEqual(
            listener.arguments(argumentIndex=0), [ItemChangedType.POSITION]
        )

    def testScatterChanged(self):
        """Test sigItemChanged for scatter"""
        data = numpy.arange(10)
        self.plot.addScatter(data, data, data, legend="test")
        scatter = self.plot.getScatter("test")

        listener = SignalListener()
        scatter.sigItemChanged.connect(listener)

        # ColormapMixIn
        scatter.getColormap().setName("viridis")

        # Test of signals in Scatter class
        scatter.setData((0, 1, 2), (1, 0, 2), (0, 1, 2))

        # Visualization mode changed
        scatter.setVisualization(scatter.Visualization.SOLID)

        self.assertEqual(
            listener.arguments(),
            [
                (ItemChangedType.COLORMAP,),
                (ItemChangedType.DATA,),
                (ItemChangedType.COLORMAP,),
                (ItemChangedType.VISUALIZATION_MODE,),
            ],
        )

    def testShapeChanged(self):
        """Test sigItemChanged for shape"""
        data = numpy.array((1.0, 10.0))
        self.plot.addShape(data, data, legend="test", shape="rectangle")
        shape = self.plot._getItem(kind="item", legend="test")

        listener = SignalListener()
        shape.sigItemChanged.connect(listener)

        shape.setOverlay(True)
        shape.setPoints(((2.0, 2.0), (3.0, 3.0)))

        self.assertEqual(
            listener.arguments(), [(ItemChangedType.OVERLAY,), (ItemChangedType.DATA,)]
        )


class TestSymbol(PlotWidgetTestCase):
    """Test item's symbol"""

    def test(self):
        """Test sigItemChanged for curve"""
        self.plot.addCurve(numpy.arange(10), numpy.arange(10), legend="test")
        curve = self.plot.getCurve("test")

        # SymbolMixIn
        curve.setSymbol("o")
        name = curve.getSymbolName()
        self.assertEqual("Circle", name)

        name = curve.getSymbolName("d")
        self.assertEqual("Diamond", name)


class TestVisibleExtent(PlotWidgetTestCase):
    """Test item's visible extent feature"""

    def testGetVisibleBounds(self):
        """Test Item.getVisibleBounds"""

        # Create test items (with a bounding box of x: [1,3], y: [0,2])
        curve = items.Curve()
        curve.setData((1, 2, 3), (0, 1, 2))

        histogram = items.Histogram()
        histogram.setData((0, 1, 2), (1, 5 / 3, 7 / 3, 3))

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
                self.assertEqual(item.getVisibleBounds(), (1.0, 3.0, 0.0, 2.0))

                xaxis.setLimits(0.5, 2.5)
                self.assertEqual(item.getVisibleBounds(), (1, 2.5, 0.0, 2.0))

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
                    ymin, ymax = self.plot.getYAxis().getLimits()
                    self.plot.setLimits(
                        xmin - (xmax - xmin) / 2,
                        xmax + (xmax - xmin) / 2,
                        ymin - (ymax - ymin) / 2,
                        ymax + (ymax - ymin) / 2,
                    )
                    self.qapp.processEvents()


def testRegionOfInterestText():
    roi = RegionOfInterest()

    listener = SignalListener()
    roi.sigItemChanged.connect(listener)

    assert roi.getName() == roi.getText()

    roi.setText("some text")
    assert listener.arguments(argumentIndex=0) == [ItemChangedType.TEXT]
    listener.clear()
    assert roi.getText() == "some text"

    roi.setName("new_name")
    assert listener.arguments(argumentIndex=0) == [ItemChangedType.NAME]
    listener.clear()
    assert roi.getText() == "some text"

    roi.setText(None)
    assert listener.arguments(argumentIndex=0) == [ItemChangedType.TEXT]
    listener.clear()
    assert roi.getText() == "new_name"

    roi.setName("even_newer_name")
    assert listener.arguments(argumentIndex=0) == [
        ItemChangedType.NAME,
        ItemChangedType.TEXT,
    ]
    assert roi.getText() == "even_newer_name"


def testPlotAddItemsWithoutLegend(plotWidget):
    curve1 = items.Curve()
    curve1.setData([0, 10], [0, 20])
    plotWidget.addItem(curve1)

    curve2 = items.Curve()
    curve2.setData([0, -10], [0, -20])
    plotWidget.addItem(curve2)

    assert plotWidget.getItems() == (curve1, curve2)

    datarange = plotWidget.getDataRange()
    assert datarange.x == (-10, 10)
    assert datarange.y == (-20, 20)

    plotWidget.resetZoom()
    assert plotWidget.getXAxis().getLimits() == (-10, 10)
    assert plotWidget.getYAxis().getLimits() == (-20, 20)


def testPlotWidgetAddCurve(plotWidget):
    curve = plotWidget.addCurve(x=(0, 1), y=(1, 0), legend="test", symbol="s")
    assert isinstance(curve, items.Curve)
    assert numpy.array_equal(curve.getXData(copy=False), (0, 1))
    assert numpy.array_equal(curve.getYData(copy=False), (1, 0))
    assert curve.getName() == "test"
    assert curve.getSymbol() == "s"

    curveUpdated = plotWidget.addCurve(
        x=(0, 1, 2), y=(1, 0, 1), legend="test", symbol="o"
    )
    assert curveUpdated is curve
    assert numpy.array_equal(curveUpdated.getXData(copy=False), (0, 1, 2))
    assert numpy.array_equal(curveUpdated.getYData(copy=False), (1, 0, 1))
    assert curveUpdated.getName() == "test"
    assert curveUpdated.getSymbol() == "o"


def testPlotWidgetAddImage(plotWidget):
    image = plotWidget.addImage(((0, 1), (2, 3)), legend="test")
    assert isinstance(image, items.ImageData)
    assert numpy.array_equal(image.getData(copy=False), ((0, 1), (2, 3)))
    assert image.getName() == "test"

    imageUpdated = plotWidget.addImage([(0, 1)], legend="test")
    assert imageUpdated is image
    assert numpy.array_equal(image.getData(copy=False), [(0, 1)])
    assert image.getName() == "test"

    # Update with a 1pixel RGB image
    imageRgb = plotWidget.addImage([[(0.0, 0.0, 1.0)]], legend="test")
    assert isinstance(imageRgb, items.ImageRgba)
    assert numpy.array_equal(imageRgb.getData(copy=False), [[(0.0, 0.0, 1.0)]])
    assert imageRgb.getName() == "test"

    # Update with a 1pixel RGB image
    imageRgbUpdated = plotWidget.addImage([[(1.0, 0.0, 0.0)]], legend="test")
    assert imageRgbUpdated is imageRgb
    assert numpy.array_equal(imageRgbUpdated.getData(copy=False), [[(1.0, 0.0, 0.0)]])
    assert imageRgbUpdated.getName() == "test"


def testPlotWidgetAddScatter(plotWidget):
    scatter = plotWidget.addScatter(
        x=(0, 1), y=(0, 1), value=(0, 1), legend="test", symbol="s"
    )
    assert isinstance(scatter, items.Scatter)
    assert numpy.array_equal(scatter.getXData(copy=False), (0, 1))
    assert numpy.array_equal(scatter.getYData(copy=False), (0, 1))
    assert numpy.array_equal(scatter.getValueData(copy=False), (0, 1))
    assert scatter.getName() == "test"
    assert scatter.getSymbol() == "s"


def testPlotWidgetAddHistogram(plotWidget):
    histogram = plotWidget.addHistogram(
        histogram=[1], edges=(0, 1), legend="test", fill=True
    )
    assert isinstance(histogram, items.Histogram)
    assert numpy.array_equal(histogram.getBinEdgesData(copy=False), (0, 1))
    assert numpy.array_equal(histogram.getValueData(copy=False), [1])
    assert histogram.getName() == "test"
    assert histogram.isFill()


def testPlotWidgetAddMarker(plotWidget):
    marker = plotWidget.addMarker(x=0, y=1, legend="test")
    assert isinstance(marker, items.Marker)
    assert marker.getPosition() == (0, 1)
    assert marker.getName() == "test"
    assert plotWidget.getItems() == (marker,)

    xmarker = plotWidget.addXMarker(1, legend="test")
    assert isinstance(xmarker, items.XMarker)
    assert xmarker.getPosition() == (1, None)
    assert xmarker.getName() == "test"
    assert plotWidget.getItems() == (xmarker,)

    ymarker = plotWidget.addYMarker(2, legend="test")
    assert isinstance(ymarker, items.YMarker)
    assert ymarker.getPosition() == (None, 2)
    assert ymarker.getName() == "test"
    assert plotWidget.getItems() == (ymarker,)


def testPlotWidgetAddShape(plotWidget):
    shape = plotWidget.addShape(
        xdata=(0, 1), ydata=(0, 1), legend="test", shape="polygon"
    )
    assert isinstance(shape, items.Shape)
    assert numpy.array_equal(shape.getPoints(copy=False), ((0, 0), (1, 1)))
    assert shape.getName() == "test"
    assert shape.getType() == "polygon"


@pytest.mark.parametrize(
    "linestyle",
    (
        "",
        "-",
        "--",
        "-.",
        ":",
        (0.0, None),
        (0.5, ()),
        (0.0, (5.0, 5.0)),
        (4.0, (8.0, 4.0, 4.0, 4.0)),
    ),
)
@pytest.mark.parametrize("plotWidget", ("mpl", "gl"), indirect=True)
def testLineStyle(qapp_utils, plotWidget, linestyle):
    """Test different line styles for LineMixIn items"""
    plotWidget.setGraphTitle(f"Line style: {linestyle}")

    curve = plotWidget.addCurve((0, 1), (0, 1), linestyle=linestyle)
    assert curve.getLineStyle() == linestyle

    histogram = plotWidget.addHistogram((0.25, 0.75, 0.25), (0.0, 0.33, 0.66, 1.0))
    histogram.setLineStyle(linestyle)
    assert histogram.getLineStyle() == linestyle

    polylines = plotWidget.addShape(
        (0, 1), (1, 0), shape="polylines", linestyle=linestyle
    )
    assert polylines.getLineStyle() == linestyle

    rectangle = plotWidget.addShape(
        (0.4, 0.6), (0.4, 0.6), shape="rectangle", linestyle=linestyle
    )
    assert rectangle.getLineStyle() == linestyle

    xmarker = plotWidget.addXMarker(0.5)
    xmarker.setLineStyle(linestyle)
    assert xmarker.getLineStyle() == linestyle

    ymarker = plotWidget.addYMarker(0.5)
    ymarker.setLineStyle(linestyle)
    assert ymarker.getLineStyle() == linestyle

    plotWidget.replot()
    qapp_utils.qWait(100)
