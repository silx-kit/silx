# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Basic tests for PlotWidget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/03/2016"


import unittest

import numpy

from silx.gui.test.utils import TestCaseQt

from silx.gui import qt
from silx.gui.plot import PlotWidget


SIZE = 1024
"""Size of the test image"""

DATA_2D = numpy.arange(SIZE ** 2).reshape(SIZE, SIZE)
"""Image data set"""


class _PlotWidgetTest(TestCaseQt):
    """Base class for tests of PlotWidget, not a TestCase in itself.

    plot attribute is the PlotWidget created for the test.
    """

    def setUp(self):
        super(_PlotWidgetTest, self).setUp()
        self.plot = PlotWidget()
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(_PlotWidgetTest, self).tearDown()


class TestPlotWidget(_PlotWidgetTest):
    """Basic tests for PlotWidget"""

    def testShow(self):
        """Most basic test"""
        pass

    def testSetTitleLabels(self):
        """Set title and axes labels"""

        title, xlabel, ylabel = 'the title', 'x label', 'y label'
        self.plot.setGraphTitle(title)
        self.plot.setGraphXLabel(xlabel)
        self.plot.setGraphYLabel(ylabel)
        self.qWait()

        self.assertEqual(self.plot.getGraphTitle(), title)
        self.assertEqual(self.plot.getGraphXLabel(), xlabel)
        self.assertEqual(self.plot.getGraphYLabel(), ylabel)


class TestPlotImage(_PlotWidgetTest):
    """Basic tests for addImage"""

    def setUp(self):
        super(TestPlotImage, self).setUp()

        self.plot.setGraphYLabel('Rows')
        self.plot.setGraphXLabel('Columns')

    def testPlotColormapTemperature(self):
        self.plot.setGraphTitle('Temp. Linear')

        colormap = {'name': 'temperature', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                    'colors': 256}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(0., 0.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()
        self.qWait()

    def testPlotColormapGray(self):
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setGraphTitle('Gray Linear')

        colormap = {'name': 'gray', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                    'colors': 256}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(0., 0.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()
        self.qWait()

    def testPlotColormapTemperatureLog(self):
        self.plot.setGraphTitle('Temp. Log')

        colormap = {'name': 'temperature', 'normalization': 'log',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                    'colors': 256}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(0., 0.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()
        self.qWait()

    def testPlotRgbRgba(self):
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setGraphTitle('RGB + RGBA')

        rgb = numpy.array(
            (((0, 0, 0), (128, 0, 0), (255, 0, 0)),
             ((0, 128, 0), (0, 128, 128), (0, 128, 256))),
            dtype=numpy.uint8)

        self.plot.addImage(rgb, legend="rgb",
                           origin=(0, 0), scale=(10, 10),
                           replace=False, resetzoom=False)

        rgba = numpy.array(
            (((0, 0, 0, .5), (.5, 0, 0, 1), (1, 0, 0, .5)),
             ((0, .5, 0, 1), (0, .5, .5, 1), (0, 1, 1, .5))),
            dtype=numpy.float32)

        self.plot.addImage(rgba, legend="rgba",
                           origin=(5, 5), scale=(10, 10),
                           replace=False, resetzoom=False)

        self.plot.resetZoom()
        self.qWait()


class TestPlotCurve(_PlotWidgetTest):
    """Basic tests for addCurve."""

    # Test data sets
    xData = numpy.arange(1000)
    yData = -500 + 100 * numpy.sin(xData)
    xData2 = xData + 1000
    yData2 = xData - 1000 + 200 * numpy.random.random(1000)

    def setUp(self):
        super(TestPlotCurve, self).setUp()
        self.plot.setGraphTitle('Curve')
        self.plot.setGraphYLabel('Rows')
        self.plot.setGraphXLabel('Columns')

        self.plot.setActiveCurveHandling(False)

    def testPlotCurveColorFloat(self):
        color = numpy.array(numpy.random.random(3 * 1000),
                            dtype=numpy.float32).reshape(1000, 3)

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve 1",
                           replace=False, resetzoom=False,
                           color=color,
                           linestyle="", symbol="s")
        self.plot.addCurve(self.xData2, self.yData2,
                           legend="curve 2",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        self.plot.resetZoom()
        self.qWait()

    def testPlotCurveColorByte(self):
        color = numpy.array(255 * numpy.random.random(3 * 1000),
                            dtype=numpy.uint8).reshape(1000, 3)

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve 1",
                           replace=False, resetzoom=False,
                           color=color,
                           linestyle="", symbol="s")
        self.plot.addCurve(self.xData2, self.yData2,
                           legend="curve 2",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        self.plot.resetZoom()
        self.qWait()

    def testPlotCurveColors(self):
        color = numpy.array(numpy.random.random(3 * 1000),
                            dtype=numpy.float32).reshape(1000, 3)

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve 2",
                           replace=False, resetzoom=False,
                           color=color, linestyle="-", symbol='o')
        self.plot.resetZoom()
        self.qWait()


class TestPlotMarker(_PlotWidgetTest):
    """Basic tests for add*Marker"""

    def setUp(self):
        super(TestPlotMarker, self).setUp()
        self.plot.setGraphYLabel('Rows')
        self.plot.setGraphXLabel('Columns')

        self.plot.setXAxisAutoScale(False)
        self.plot.setYAxisAutoScale(False)
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setLimits(0., 100., -100., 100.)

    def testPlotMarkerX(self):
        self.plot.setGraphTitle('Markers X')

        markers = [
            (10., 'blue', False, False),
            (20., 'red', False, False),
            (40., 'green', True, False),
            (60., 'gray', True, True),
            (80., 'black', False, True),
        ]

        for x, color, select, drag in markers:
            name = str(x)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addXMarker(x, name, name, color, select, drag)
        self.plot.resetZoom()
        self.qWait()

    def testPlotMarkerY(self):
        self.plot.setGraphTitle('Markers Y')

        markers = [
            (-50., 'blue', False, False),
            (-30., 'red', False, False),
            (0., 'green', True, False),
            (10., 'gray', True, True),
            (80., 'black', False, True),
        ]

        for y, color, select, drag in markers:
            name = str(y)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addYMarker(y, name, name, color, select, drag)
        self.plot.resetZoom()
        self.qWait()

    def testPlotMarkerPt(self):
        self.plot.setGraphTitle('Markers Pt')

        markers = [
            (10., -50., 'blue', False, False),
            (40., -30., 'red', False, False),
            (50., 0., 'green', True, False),
            (50., 20., 'gray', True, True),
            (70., 50., 'black', False, True),
        ]
        for x, y, color, select, drag in markers:
            name = "{0},{1}".format(x, y)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addMarker(x, y, name, name, color, select, drag)

        self.plot.resetZoom()
        self.qWait()


# TestPlotItem ################################################################

class TestPlotItem(_PlotWidgetTest):
    """Basic tests for addItem."""

    # Polygon coordinates and color
    polygons = [  # legend, x coords, y coords, color
        ('triangle', numpy.array((10, 30, 50)),
         numpy.array((55, 70, 55)), 'red'),
        ('square', numpy.array((10, 10, 50, 50)),
         numpy.array((10, 50, 50, 10)), 'green'),
        ('star', numpy.array((60, 70, 80, 60, 80)),
         numpy.array((25, 50, 25, 40, 40)), 'blue'),
    ]

    # Rectangle coordinantes and color
    rectangles = [  # legend, x coords, y coords, color
        ('square 1', numpy.array((1., 10.)),
         numpy.array((1., 10.)), 'red'),
        ('square 2', numpy.array((10., 20.)),
         numpy.array((10., 20.)), 'green'),
        ('square 3', numpy.array((20., 30.)),
         numpy.array((20., 30.)), 'blue'),
        ('rect 1', numpy.array((1., 30.)),
         numpy.array((35., 40.)), 'black'),
        ('line h', numpy.array((1., 30.)),
         numpy.array((45., 45.)), 'darkRed'),
    ]

    def setUp(self):
        super(TestPlotItem, self).setUp()

        self.plot.setGraphYLabel('Rows')
        self.plot.setGraphXLabel('Columns')
        self.plot.setXAxisAutoScale(False)
        self.plot.setYAxisAutoScale(False)
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setLimits(0., 100., -100., 100.)

    def testPlotItemPolygonFill(self):
        self.plot.setGraphTitle('Item Fill')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=True, color=color)
        self.plot.resetZoom()
        self.qWait()

    def testPlotItemPolygonNoFill(self):
        self.plot.setGraphTitle('Item No Fill')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=False, color=color)
        self.plot.resetZoom()
        self.qWait()

    def testPlotItemRectangleFill(self):
        self.plot.setGraphTitle('Rectangle Fill')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=True, color=color)
        self.plot.resetZoom()
        self.qWait()

    def testPlotItemRectangleNoFill(self):
        self.plot.setGraphTitle('Rectangle No Fill')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=False, color=color)
        self.plot.resetZoom()
        self.qWait()


##############################################################################
# Log
##############################################################################

class TestPlotEmptyLog(_PlotWidgetTest):
    """Basic tests for log plot"""
    def testEmptyPlotTitleLabelsLog(self):
        self.plot.setGraphTitle('Empty Log Log')
        self.plot.setGraphXLabel('X')
        self.plot.setGraphYLabel('Y')
        self.plot.setXAxisLogarithmic(True)
        self.plot.setYAxisLogarithmic(True)
        self.plot.resetZoom()
        self.qWait()


class TestPlotCurveLog(_PlotWidgetTest):
    """Basic tests for addCurve with log scale axes"""

    # Test data
    xData = numpy.arange(1000) + 1
    yData = xData ** 2

    def setUp(self):
        super(TestPlotCurveLog, self).setUp()

        self.plot.setGraphXLabel('X')
        self.plot.setGraphYLabel('X * X')

    def testPlotCurveLogX(self):
        self.plot.setXAxisLogarithmic(True)
        self.plot.setGraphTitle('Curve X: Log Y: Linear')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')
        self.plot.resetZoom()
        self.qWait()

    def testPlotCurveLogY(self):
        self.plot.setYAxisLogarithmic(True)

        self.plot.setGraphTitle('Curve X: Linear Y: Log')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')
        self.plot.resetZoom()
        self.qWait()

    def testPlotCurveLogXY(self):
        self.plot.setXAxisLogarithmic(True)
        self.plot.setYAxisLogarithmic(True)

        self.plot.setGraphTitle('Curve X: Log Y: Log')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')
        self.plot.resetZoom()
        self.qWait()


class TestPlotImageLog(_PlotWidgetTest):
    """Basic tests for addImage with log scale axes."""

    def setUp(self):
        super(TestPlotImageLog, self).setUp()

        self.plot.setGraphXLabel('Columns')
        self.plot.setGraphYLabel('Rows')

    def testPlotColormapGrayLogX(self):
        self.plot.setXAxisLogarithmic(True)
        self.plot.setGraphTitle('CMap X: Log Y: Linear')

        colormap = {'name': 'gray', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                    'colors': 256}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()
        self.qWait()

    def testPlotColormapGrayLogY(self):
        self.plot.setYAxisLogarithmic(True)
        self.plot.setGraphTitle('CMap X: Linear Y: Log')

        colormap = {'name': 'gray', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                    'colors': 256}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()
        self.qWait()

    def testPlotColormapGrayLogXY(self):
        self.plot.setXAxisLogarithmic(True)
        self.plot.setYAxisLogarithmic(True)
        self.plot.setGraphTitle('CMap X: Log Y: Log')

        colormap = {'name': 'gray', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                    'colors': 256}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()
        self.qWait()

    def testPlotRgbRgbaLogXY(self):
        self.plot.setXAxisLogarithmic(True)
        self.plot.setYAxisLogarithmic(True)
        self.plot.setGraphTitle('RGB + RGBA X: Log Y: Log')

        rgb = numpy.array(
            (((0, 0, 0), (128, 0, 0), (255, 0, 0)),
             ((0, 128, 0), (0, 128, 128), (0, 128, 256))),
            dtype=numpy.uint8)

        self.plot.addImage(rgb, legend="rgb",
                           origin=(1, 1), scale=(10, 10),
                           replace=False, resetzoom=False)

        rgba = numpy.array(
            (((0, 0, 0, .5), (.5, 0, 0, 1), (1, 0, 0, .5)),
             ((0, .5, 0, 1), (0, .5, .5, 1), (0, 1, 1, .5))),
            dtype=numpy.float32)

        self.plot.addImage(rgba, legend="rgba",
                           origin=(5., 5.), scale=(10., 10.),
                           replace=False, resetzoom=False)
        self.plot.resetZoom()
        self.qWait()


class TestPlotMarkerLog(_PlotWidgetTest):
    """Basic tests for markers on log scales"""

    # Test marker parameters
    markers = [  # x, y, color, selectable, draggable
        (10., 10., 'blue', False, False),
        (20., 20., 'red', False, False),
        (40., 100., 'green', True, False),
        (40., 500., 'gray', True, True),
        (60., 800., 'black', False, True),
    ]

    def setUp(self):
        super(TestPlotMarkerLog, self).setUp()

        self.plot.setGraphYLabel('Rows')
        self.plot.setGraphXLabel('Columns')
        self.plot.setXAxisAutoScale(False)
        self.plot.setYAxisAutoScale(False)
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setLimits(1., 100., 1., 1000.)
        self.plot.setXAxisLogarithmic(True)
        self.plot.setYAxisLogarithmic(True)

    def testPlotMarkerXLog(self):
        self.plot.setGraphTitle('Markers X, Log axes')

        for x, _, color, select, drag in self.markers:
            name = str(x)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addXMarker(x, name, name, color, select, drag)
        self.plot.resetZoom()
        self.qWait()

    def testPlotMarkerYLog(self):
        self.plot.setGraphTitle('Markers Y, Log axes')

        for _, y, color, select, drag in self.markers:
            name = str(y)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addYMarker(y, name, name, color, select, drag)
        self.plot.resetZoom()
        self.qWait()

    def testPlotMarkerPtLog(self):
        self.plot.setGraphTitle('Markers Pt, Log axes')

        for x, y, color, select, drag in self.markers:
            name = "{0},{1}".format(x, y)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addMarker(x, y, name, name, color, select, drag)
        self.plot.resetZoom()
        self.qWait()


class TestPlotItemLog(_PlotWidgetTest):
    """Basic tests for items with log scale axes"""

    # Polygon coordinates and color
    polygons = [  # legend, x coords, y coords, color
        ('triangle', numpy.array((10, 30, 50)),
         numpy.array((55, 70, 55)), 'red'),
        ('square', numpy.array((10, 10, 50, 50)),
         numpy.array((10, 50, 50, 10)), 'green'),
        ('star', numpy.array((60, 70, 80, 60, 80)),
         numpy.array((25, 50, 25, 40, 40)), 'blue'),
    ]

    # Rectangle coordinantes and color
    rectangles = [  # legend, x coords, y coords, color
        ('square 1', numpy.array((1., 10.)),
         numpy.array((1., 10.)), 'red'),
        ('square 2', numpy.array((10., 20.)),
         numpy.array((10., 20.)), 'green'),
        ('square 3', numpy.array((20., 30.)),
         numpy.array((20., 30.)), 'blue'),
        ('rect 1', numpy.array((1., 30.)),
         numpy.array((35., 40.)), 'black'),
        ('line h', numpy.array((1., 30.)),
         numpy.array((45., 45.)), 'darkRed'),
    ]

    def setUp(self):
        super(TestPlotItemLog, self).setUp()

        self.plot.setGraphYLabel('Rows')
        self.plot.setGraphXLabel('Columns')
        self.plot.setXAxisAutoScale(False)
        self.plot.setYAxisAutoScale(False)
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setLimits(1., 100., 1., 100.)
        self.plot.setXAxisLogarithmic(True)
        self.plot.setYAxisLogarithmic(True)

    def testPlotItemPolygonLogFill(self):
        self.plot.setGraphTitle('Item Fill Log')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=True, color=color)
        self.plot.resetZoom()
        self.qWait()

    def testPlotItemPolygonLogNoFill(self):
        self.plot.setGraphTitle('Item No Fill Log')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=False, color=color)
        self.plot.resetZoom()
        self.qWait()

    def testPlotItemRectangleLogFill(self):
        self.plot.setGraphTitle('Rectangle Fill Log')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=True, color=color)
        self.plot.resetZoom()
        self.qWait()

    def testPlotItemRectangleLogNoFill(self):
        self.plot.setGraphTitle('Rectangle No Fill Log')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=False, color=color)
        self.plot.resetZoom()
        self.qWait()


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotWidget))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotImage))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotCurve))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotMarker))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotItem))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotEmptyLog))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotCurveLog))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotImageLog))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotMarkerLog))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlotItemLog))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
