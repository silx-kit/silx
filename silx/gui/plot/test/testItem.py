# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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

        # Test for signals in Points class
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
        data2 = data + 10

        # Test of signals in Scatter class
        scatter.setData(data2, data2, data2)

        self.assertEqual(listener.arguments(),
                         [(ItemChangedType.COLORMAP,),
                          (ItemChangedType.DATA,)])

    def testShapeChanged(self):
        """Test sigItemChanged for shape"""
        data = numpy.array((1., 10.))
        self.plot.addItem(data, data, legend='test', shape='rectangle')
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


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestSigItemChangedSignal))
    test_suite.addTest(loadTests(TestSymbol))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
