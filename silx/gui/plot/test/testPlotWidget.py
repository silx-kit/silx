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
__date__ = "15/09/2016"


import unittest

import numpy

from silx.testutils import ParametricTestCase
from silx.gui.testutils import TestCaseQt

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
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(_PlotWidgetTest, self).tearDown()


class TestPlotWidget(_PlotWidgetTest, ParametricTestCase):
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
        self.qapp.processEvents()

        self.assertEqual(self.plot.getGraphTitle(), title)
        self.assertEqual(self.plot.getGraphXLabel(), xlabel)
        self.assertEqual(self.plot.getGraphYLabel(), ylabel)

    def testChangeLimitsWithAspectRatio(self):
        def checkLimits(expectedXLim=None, expectedYLim=None,
                        expectedRatio=None):
            xlim = self.plot.getGraphXLimits()
            ylim = self.plot.getGraphYLimits()
            ratio = abs(xlim[1] - xlim[0]) / abs(ylim[1] - ylim[0])

            if expectedXLim is not None:
                self.assertEqual(expectedXLim, xlim)

            if expectedYLim is not None:
                self.assertEqual(expectedYLim, ylim)

            if expectedRatio is not None:
                self.assertTrue(numpy.allclose(expectedRatio, ratio))

        self.plot.setKeepDataAspectRatio()
        self.qapp.processEvents()
        xlim = self.plot.getGraphXLimits()
        ylim = self.plot.getGraphYLimits()
        defaultRatio = abs(xlim[1] - xlim[0]) / abs(ylim[1] - ylim[0])

        self.plot.setGraphXLimits(1., 10.)
        checkLimits(expectedXLim=(1., 10.), expectedRatio=defaultRatio)
        self.qapp.processEvents()
        checkLimits(expectedXLim=(1., 10.), expectedRatio=defaultRatio)

        self.plot.setGraphYLimits(1., 10.)
        checkLimits(expectedYLim=(1., 10.), expectedRatio=defaultRatio)
        self.qapp.processEvents()
        checkLimits(expectedYLim=(1., 10.), expectedRatio=defaultRatio)


class TestPlotImage(_PlotWidgetTest, ParametricTestCase):
    """Basic tests for addImage"""

    def setUp(self):
        super(TestPlotImage, self).setUp()

        self.plot.setGraphYLabel('Rows')
        self.plot.setGraphXLabel('Columns')

    def testPlotColormapTemperature(self):
        self.plot.setGraphTitle('Temp. Linear')

        colormap = {'name': 'temperature', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap)

    def testPlotColormapGray(self):
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setGraphTitle('Gray Linear')

        colormap = {'name': 'gray', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap)

    def testPlotColormapTemperatureLog(self):
        self.plot.setGraphTitle('Temp. Log')

        colormap = {'name': 'temperature', 'normalization': 'log',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap)

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

    def testPlotColormapCustom(self):
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setGraphTitle('Custom colormap')

        colormap = {'name': None, 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                    'colors': ((0., 0., 0.), (1., 0., 0.),
                               (0., 1., 0.), (0., 0., 1.))}
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap,
                           replace=False, resetzoom=False)

        colormap = {'name': None, 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                    'colors': numpy.array(
                        ((0, 0, 0, 0), (0, 0, 0, 128),
                         (128, 128, 128, 128), (255, 255, 255, 255)),
                        dtype=numpy.uint8)}
        self.plot.addImage(DATA_2D, legend="image 2", colormap=colormap,
                           origin=(DATA_2D.shape[0], 0),
                           replace=False, resetzoom=False)
        self.plot.resetZoom()

    def testImageOriginScale(self):
        """Test of image with different origin and scale"""
        self.plot.setGraphTitle('origin and scale')

        tests = [  # (origin, scale)
            ((10, 20), (1, 1)),
            ((10, 20), (-1, -1)),
            ((-10, 20), (2, 1)),
            ((10, -20), (-1, -2)),
            (100, 2),
            (-100, (1, 1)),
            ((10, 20), 2),
            ]

        for origin, scale in tests:
            with self.subTest(origin=origin, scale=scale):
                self.plot.addImage(DATA_2D, origin=origin, scale=scale)

                try:
                    ox, oy = origin
                except TypeError:
                    ox, oy = origin, origin
                try:
                    sx, sy = scale
                except TypeError:
                    sx, sy = scale, scale
                xbounds = ox, ox + DATA_2D.shape[1] * sx
                ybounds = oy, oy + DATA_2D.shape[0] * sy

                # Check limits without aspect ratio
                xmin, xmax = self.plot.getGraphXLimits()
                ymin, ymax = self.plot.getGraphYLimits()
                self.assertEqual(xmin, min(xbounds))
                self.assertEqual(xmax, max(xbounds))
                self.assertEqual(ymin, min(ybounds))
                self.assertEqual(ymax, max(ybounds))

                # Check limits with aspect ratio
                self.plot.setKeepDataAspectRatio(True)
                xmin, xmax = self.plot.getGraphXLimits()
                ymin, ymax = self.plot.getGraphYLimits()
                self.assertTrue(xmin <= min(xbounds))
                self.assertTrue(xmax >= max(xbounds))
                self.assertTrue(ymin <= min(ybounds))
                self.assertTrue(ymax >= max(ybounds))

                self.plot.setKeepDataAspectRatio(False)  # Reset aspect ratio
                self.plot.clear()
                self.plot.resetZoom()


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

    def testPlotCurveColors(self):
        color = numpy.array(numpy.random.random(3 * 1000),
                            dtype=numpy.float32).reshape(1000, 3)

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve 2",
                           replace=False, resetzoom=False,
                           color=color, linestyle="-", symbol='o')
        self.plot.resetZoom()


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

    def testPlotMarkerWithoutLegend(self):
        self.plot.setGraphTitle('Markers without legend')
        self.plot.setYAxisInverted(True)

        # Markers without legend
        self.plot.addMarker(10, 10)
        self.plot.addMarker(10, 20)
        self.plot.addMarker(40, 50, text='test', symbol=None)
        self.plot.addMarker(40, 50, text='test', symbol='+')
        self.plot.addXMarker(25)
        self.plot.addXMarker(35)
        self.plot.addXMarker(45, text='test')
        self.plot.addYMarker(55)
        self.plot.addYMarker(65)
        self.plot.addYMarker(75, text='test')

        self.plot.resetZoom()


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

    def testPlotItemPolygonNoFill(self):
        self.plot.setGraphTitle('Item No Fill')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=False, color=color)
        self.plot.resetZoom()

    def testPlotItemRectangleFill(self):
        self.plot.setGraphTitle('Rectangle Fill')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=True, color=color)
        self.plot.resetZoom()

    def testPlotItemRectangleNoFill(self):
        self.plot.setGraphTitle('Rectangle No Fill')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=False, color=color)
        self.plot.resetZoom()


class TestPlotActiveCurveImage(_PlotWidgetTest):
    """Basic tests for active image handling"""

    def testActiveCurveAndLabels(self):
        # Active curve handling off, no label change
        self.plot.setActiveCurveHandling(False)
        self.plot.setGraphXLabel('XLabel')
        self.plot.setGraphYLabel('YLabel')
        self.plot.addCurve((1, 2), (1, 2))
        self.assertEqual(self.plot.getGraphXLabel(), 'XLabel')
        self.assertEqual(self.plot.getGraphYLabel(), 'YLabel')

        self.plot.addCurve((1, 2), (2, 3), xlabel='x1', ylabel='y1')
        self.assertEqual(self.plot.getGraphXLabel(), 'XLabel')
        self.assertEqual(self.plot.getGraphYLabel(), 'YLabel')

        self.plot.clear()
        self.assertEqual(self.plot.getGraphXLabel(), 'XLabel')
        self.assertEqual(self.plot.getGraphYLabel(), 'YLabel')

        # Active curve handling on, label changes
        self.plot.setActiveCurveHandling(True)
        self.plot.setGraphXLabel('XLabel')
        self.plot.setGraphYLabel('YLabel')

        # labels changed as active curve
        self.plot.addCurve((1, 2), (1, 2), legend='1',
                           xlabel='x1', ylabel='y1')
        self.assertEqual(self.plot.getGraphXLabel(), 'x1')
        self.assertEqual(self.plot.getGraphYLabel(), 'y1')

        # labels not changed as not active curve
        self.plot.addCurve((1, 2), (2, 3), legend='2')
        self.assertEqual(self.plot.getGraphXLabel(), 'x1')
        self.assertEqual(self.plot.getGraphYLabel(), 'y1')

        # labels changed
        self.plot.setActiveCurve('2')
        self.assertEqual(self.plot.getGraphXLabel(), 'XLabel')
        self.assertEqual(self.plot.getGraphYLabel(), 'YLabel')

        self.plot.setActiveCurve('1')
        self.assertEqual(self.plot.getGraphXLabel(), 'x1')
        self.assertEqual(self.plot.getGraphYLabel(), 'y1')

        self.plot.clear()
        self.assertEqual(self.plot.getGraphXLabel(), 'XLabel')
        self.assertEqual(self.plot.getGraphYLabel(), 'YLabel')

    def testActiveImageAndLabels(self):
        # Active image handling always on, no API for toggling it
        self.plot.setGraphXLabel('XLabel')
        self.plot.setGraphYLabel('YLabel')

        # labels changed as active curve
        self.plot.addImage(numpy.arange(100).reshape(10, 10), replace=False,
                           legend='1', xlabel='x1', ylabel='y1')
        self.assertEqual(self.plot.getGraphXLabel(), 'x1')
        self.assertEqual(self.plot.getGraphYLabel(), 'y1')

        # labels not changed as not active curve
        self.plot.addImage(numpy.arange(100).reshape(10, 10), replace=False,
                           legend='2')
        self.assertEqual(self.plot.getGraphXLabel(), 'x1')
        self.assertEqual(self.plot.getGraphYLabel(), 'y1')

        # labels changed
        self.plot.setActiveImage('2')
        self.assertEqual(self.plot.getGraphXLabel(), 'XLabel')
        self.assertEqual(self.plot.getGraphYLabel(), 'YLabel')

        self.plot.setActiveImage('1')
        self.assertEqual(self.plot.getGraphXLabel(), 'x1')
        self.assertEqual(self.plot.getGraphYLabel(), 'y1')

        self.plot.clear()
        self.assertEqual(self.plot.getGraphXLabel(), 'XLabel')
        self.assertEqual(self.plot.getGraphYLabel(), 'YLabel')


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


class TestPlotCurveLog(_PlotWidgetTest, ParametricTestCase):
    """Basic tests for addCurve with log scale axes"""

    # Test data
    xData = numpy.arange(1000) + 1
    yData = xData ** 2

    def _setLabels(self):
        self.plot.setGraphXLabel('X')
        self.plot.setGraphYLabel('X * X')

    def testPlotCurveLogX(self):
        self._setLabels()
        self.plot.setXAxisLogarithmic(True)
        self.plot.setGraphTitle('Curve X: Log Y: Linear')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')

    def testPlotCurveLogY(self):
        self._setLabels()
        self.plot.setYAxisLogarithmic(True)

        self.plot.setGraphTitle('Curve X: Linear Y: Log')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')

    def testPlotCurveLogXY(self):
        self._setLabels()
        self.plot.setXAxisLogarithmic(True)
        self.plot.setYAxisLogarithmic(True)

        self.plot.setGraphTitle('Curve X: Log Y: Log')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')

    def testPlotCurveToggleLog(self):
        """Add a curve with negative data and toggle log axis"""
        arange = numpy.arange(1000) + 1
        tests = [  # name, xData, yData
            ('x>0, some negative y', arange, arange - 500),
            ('x>0, y<0', arange, -arange),
            ('some negative x, y>0', arange - 500, arange),
            ('x<0, y>0', -arange, arange),
            ('some negative x and y', arange - 500, arange - 500),
            ('x<0, y<0', -arange, -arange),
        ]

        for name, xData, yData in tests:
            with self.subTest(name):
                self.plot.addCurve(xData, yData, resetzoom=True)
                self.qapp.processEvents()

                # no log axis
                xLim = self.plot.getGraphXLimits()
                self.assertEqual(xLim, (min(xData), max(xData)))
                yLim = self.plot.getGraphYLimits()
                self.assertEqual(yLim, (min(yData), max(yData)))

                # x axis log
                self.plot.setXAxisLogarithmic(True)
                self.qapp.processEvents()

                xLim = self.plot.getGraphXLimits()
                yLim = self.plot.getGraphYLimits()
                positives = xData > 0
                if numpy.any(positives):
                    self.assertTrue(numpy.allclose(
                        xLim, (min(xData[positives]), max(xData[positives]))))
                    self.assertEqual(
                        yLim, (min(yData[positives]), max(yData[positives])))
                else:  # No positive x in the curve
                    self.assertEqual(xLim, (1., 100.))
                    self.assertEqual(yLim, (1., 100.))

                # x axis and y axis log
                self.plot.setYAxisLogarithmic(True)
                self.qapp.processEvents()

                xLim = self.plot.getGraphXLimits()
                yLim = self.plot.getGraphYLimits()
                positives = numpy.logical_and(xData > 0, yData > 0)
                if numpy.any(positives):
                    self.assertTrue(numpy.allclose(
                        xLim, (min(xData[positives]), max(xData[positives]))))
                    self.assertTrue(numpy.allclose(
                        yLim, (min(yData[positives]), max(yData[positives]))))
                else:  # No positive x and y in the curve
                    self.assertEqual(xLim, (1., 100.))
                    self.assertEqual(yLim, (1., 100.))

                # y axis log
                self.plot.setXAxisLogarithmic(False)
                self.qapp.processEvents()

                xLim = self.plot.getGraphXLimits()
                yLim = self.plot.getGraphYLimits()
                positives = yData > 0
                if numpy.any(positives):
                    self.assertEqual(
                        xLim, (min(xData[positives]), max(xData[positives])))
                    self.assertTrue(numpy.allclose(
                        yLim, (min(yData[positives]), max(yData[positives]))))
                else:  # No positive y in the curve
                    self.assertEqual(xLim, (1., 100.))
                    self.assertEqual(yLim, (1., 100.))

                # no log axis
                self.plot.setYAxisLogarithmic(False)
                self.qapp.processEvents()

                xLim = self.plot.getGraphXLimits()
                self.assertEqual(xLim, (min(xData), max(xData)))
                yLim = self.plot.getGraphYLimits()
                self.assertEqual(yLim, (min(yData), max(yData)))

                self.plot.clear()
                self.plot.resetZoom()
                self.qapp.processEvents()


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
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()

    def testPlotColormapGrayLogY(self):
        self.plot.setYAxisLogarithmic(True)
        self.plot.setGraphTitle('CMap X: Linear Y: Log')

        colormap = {'name': 'gray', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()

    def testPlotColormapGrayLogXY(self):
        self.plot.setXAxisLogarithmic(True)
        self.plot.setYAxisLogarithmic(True)
        self.plot.setGraphTitle('CMap X: Log Y: Log')

        colormap = {'name': 'gray', 'normalization': 'linear',
                    'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()

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

    def testPlotItemPolygonLogNoFill(self):
        self.plot.setGraphTitle('Item No Fill Log')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=False, color=color)
        self.plot.resetZoom()

    def testPlotItemRectangleLogFill(self):
        self.plot.setGraphTitle('Rectangle Fill Log')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=True, color=color)
        self.plot.resetZoom()

    def testPlotItemRectangleLogNoFill(self):
        self.plot.setGraphTitle('Rectangle No Fill Log')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=False, color=color)
        self.plot.resetZoom()


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
