# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
"""Basic tests for PlotWidget with 'none' backend"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import unittest
from functools import reduce
from silx.utils.testutils import ParametricTestCase

import numpy

from silx.gui.plot.PlotWidget import PlotWidget
from silx.gui.plot.items.histogram import _getHistogramCurve, _computeEdges


class TestPlot(unittest.TestCase):
    """Basic tests of Plot without backend"""

    def testPlotTitleLabels(self):
        """Create a Plot and set the labels"""

        plot = PlotWidget(backend='none')

        title, xlabel, ylabel = 'the title', 'x label', 'y label'
        plot.setGraphTitle(title)
        plot.getXAxis().setLabel(xlabel)
        plot.getYAxis().setLabel(ylabel)

        self.assertEqual(plot.getGraphTitle(), title)
        self.assertEqual(plot.getXAxis().getLabel(), xlabel)
        self.assertEqual(plot.getYAxis().getLabel(), ylabel)

    def testAddNoRemove(self):
        """add objects to the Plot"""

        plot = PlotWidget(backend='none')
        plot.addCurve(x=(1, 2, 3), y=(3, 2, 1))
        plot.addImage(numpy.arange(100.).reshape(10, -1))
        plot.addShape(numpy.array((1., 10.)),
                      numpy.array((10., 10.)),
                      shape="rectangle")
        plot.addXMarker(10.)


class TestPlotRanges(ParametricTestCase):
    """Basic tests of Plot data ranges without backend"""

    _getValidValues = {True: lambda ar: ar > 0,
                       False: lambda ar: numpy.ones(shape=ar.shape,
                                                    dtype=bool)}

    @staticmethod
    def _getRanges(arrays, are_logs):
        gen = (TestPlotRanges._getValidValues[is_log](ar)
               for (ar, is_log) in zip(arrays, are_logs))
        indices = numpy.where(reduce(numpy.logical_and, gen))[0]
        if len(indices) > 0:
            ranges = [(ar[indices[0]], ar[indices[-1]]) for ar in arrays]
        else:
            ranges = [None] * len(arrays)

        return ranges

    @staticmethod
    def _getRangesMinmax(ranges):
        # TODO : error if None in ranges.
        rangeMin = numpy.min([rng[0] for rng in ranges])
        rangeMax = numpy.max([rng[1] for rng in ranges])
        return rangeMin, rangeMax

    def testDataRangeNoPlot(self):
        """empty plot data range"""

        plot = PlotWidget(backend='none')

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.getXAxis()._setLogarithmic(logX)
                plot.getYAxis()._setLogarithmic(logY)
                dataRange = plot.getDataRange()
                self.assertIsNone(dataRange.x)
                self.assertIsNone(dataRange.y)
                self.assertIsNone(dataRange.yright)

    def testDataRangeLeft(self):
        """left axis range"""

        plot = PlotWidget(backend='none')

        xData = numpy.arange(10) - 4.9  # range : -4.9 , 4.1
        yData = numpy.arange(10) - 6.9  # range : -6.9 , 2.1

        plot.addCurve(x=xData,
                      y=yData,
                      legend='plot_0',
                      yaxis='left')

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.getXAxis()._setLogarithmic(logX)
                plot.getYAxis()._setLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRange, yRange = self._getRanges([xData, yData],
                                                 [logX, logY])
                self.assertSequenceEqual(dataRange.x, xRange)
                self.assertSequenceEqual(dataRange.y, yRange)
                self.assertIsNone(dataRange.yright)

    def testDataRangeRight(self):
        """right axis range"""

        plot = PlotWidget(backend='none')
        xData = numpy.arange(10) - 4.9  # range : -4.9 , 4.1
        yData = numpy.arange(10) - 6.9  # range : -6.9 , 2.1
        plot.addCurve(x=xData,
                      y=yData,
                      legend='plot_0',
                      yaxis='right')

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.getXAxis()._setLogarithmic(logX)
                plot.getYAxis()._setLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRange, yRange = self._getRanges([xData, yData],
                                                 [logX, logY])
                self.assertSequenceEqual(dataRange.x, xRange)
                self.assertIsNone(dataRange.y)
                self.assertSequenceEqual(dataRange.yright, yRange)

    def testDataRangeImage(self):
        """image data range"""

        origin = (-10, 25)
        scale = (3., 8.)
        image = numpy.arange(100.).reshape(20, 5)

        plot = PlotWidget(backend='none')
        plot.addImage(image,
                      origin=origin, scale=scale)

        xRange = numpy.array([0., image.shape[1] * scale[0]]) + origin[0]
        yRange = numpy.array([0., image.shape[0] * scale[1]]) + origin[1]

        ranges = {(False, False): (xRange, yRange),
                  (True, False): (None, None),
                  (True, True): (None, None),
                  (False, True): (None, None)}

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.getXAxis()._setLogarithmic(logX)
                plot.getYAxis()._setLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRange, yRange = ranges[logX, logY]
                self.assertTrue(numpy.array_equal(dataRange.x, xRange),
                                msg='{0} != {1}'.format(dataRange.x, xRange))
                self.assertTrue(numpy.array_equal(dataRange.y, yRange),
                                msg='{0} != {1}'.format(dataRange.y, yRange))
                self.assertIsNone(dataRange.yright)

    def testDataRangeLeftRight(self):
        """right+left axis range"""

        plot = PlotWidget(backend='none')

        xData_l = numpy.arange(10) - 0.9  # range : -0.9 , 8.1
        yData_l = numpy.arange(10) - 1.9  # range : -1.9 , 7.1
        plot.addCurve(x=xData_l,
                      y=yData_l,
                      legend='plot_l',
                      yaxis='left')

        xData_r = numpy.arange(10) - 4.9  # range : -4.9 , 4.1
        yData_r = numpy.arange(10) - 6.9  # range : -6.9 , 2.1
        plot.addCurve(x=xData_r,
                      y=yData_r,
                      legend='plot_r',
                      yaxis='right')

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.getXAxis()._setLogarithmic(logX)
                plot.getYAxis()._setLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRangeL, yRangeL = self._getRanges([xData_l, yData_l],
                                                   [logX, logY])
                xRangeR, yRangeR = self._getRanges([xData_r, yData_r],
                                                   [logX, logY])
                xRangeLR = self._getRangesMinmax([xRangeL, xRangeR])
                self.assertSequenceEqual(dataRange.x, xRangeLR)
                self.assertSequenceEqual(dataRange.y, yRangeL)
                self.assertSequenceEqual(dataRange.yright, yRangeR)

    def testDataRangeCurveImage(self):
        """right+left+image axis range"""

        # overlapping ranges :
        # image sets x min and y max
        # plot_left sets y min
        # plot_right sets x max (and yright)
        plot = PlotWidget(backend='none')

        origin = (-10, 5)
        scale = (3., 8.)
        image = numpy.arange(100.).reshape(20, 5)

        plot.addImage(image,
                      origin=origin, scale=scale, legend='image')

        xData_l = numpy.arange(10) - 0.9  # range : -0.9 , 8.1
        yData_l = numpy.arange(10) - 1.9  # range : -1.9 , 7.1
        plot.addCurve(x=xData_l,
                      y=yData_l,
                      legend='plot_l',
                      yaxis='left')

        xData_r = numpy.arange(10) + 4.1  # range : 4.1 , 13.1
        yData_r = numpy.arange(10) - 0.9  # range : -0.9 , 8.1
        plot.addCurve(x=xData_r,
                      y=yData_r,
                      legend='plot_r',
                      yaxis='right')

        imgXRange = numpy.array([0., image.shape[1] * scale[0]]) + origin[0]
        imgYRange = numpy.array([0., image.shape[0] * scale[1]]) + origin[1]

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.getXAxis()._setLogarithmic(logX)
                plot.getYAxis()._setLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRangeL, yRangeL = self._getRanges([xData_l, yData_l],
                                                   [logX, logY])
                xRangeR, yRangeR = self._getRanges([xData_r, yData_r],
                                                   [logX, logY])
                if logX or logY:
                    xRangeLR = self._getRangesMinmax([xRangeL, xRangeR])
                else:
                    xRangeLR = self._getRangesMinmax([xRangeL,
                                                      xRangeR,
                                                      imgXRange])
                    yRangeL = self._getRangesMinmax([yRangeL, imgYRange])
                self.assertSequenceEqual(dataRange.x, xRangeLR)
                self.assertSequenceEqual(dataRange.y, yRangeL)
                self.assertSequenceEqual(dataRange.yright, yRangeR)

    def testDataRangeImageNegativeScaleX(self):
        """image data range, negative scale"""

        origin = (-10, 25)
        scale = (-3., 8.)
        image = numpy.arange(100.).reshape(20, 5)

        plot = PlotWidget(backend='none')
        plot.addImage(image,
                      origin=origin, scale=scale)

        xRange = numpy.array([0., image.shape[1] * scale[0]]) + origin[0]
        xRange.sort()  # negative scale!
        yRange = numpy.array([0., image.shape[0] * scale[1]]) + origin[1]

        ranges = {(False, False): (xRange, yRange),
                  (True, False): (None, None),
                  (True, True): (None, None),
                  (False, True): (None, None)}

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.getXAxis()._setLogarithmic(logX)
                plot.getYAxis()._setLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRange, yRange = ranges[logX, logY]
                self.assertTrue(numpy.array_equal(dataRange.x, xRange),
                                msg='{0} != {1}'.format(dataRange.x, xRange))
                self.assertTrue(numpy.array_equal(dataRange.y, yRange),
                                msg='{0} != {1}'.format(dataRange.y, yRange))
                self.assertIsNone(dataRange.yright)

    def testDataRangeImageNegativeScaleY(self):
        """image data range, negative scale"""

        origin = (-10, 25)
        scale = (3., -8.)
        image = numpy.arange(100.).reshape(20, 5)

        plot = PlotWidget(backend='none')
        plot.addImage(image,
                      origin=origin, scale=scale)

        xRange = numpy.array([0., image.shape[1] * scale[0]]) + origin[0]
        yRange = numpy.array([0., image.shape[0] * scale[1]]) + origin[1]
        yRange.sort()  # negative scale!

        ranges = {(False, False): (xRange, yRange),
                  (True, False): (None, None),
                  (True, True): (None, None),
                  (False, True): (None, None)}

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.getXAxis()._setLogarithmic(logX)
                plot.getYAxis()._setLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRange, yRange = ranges[logX, logY]
                self.assertTrue(numpy.array_equal(dataRange.x, xRange),
                                msg='{0} != {1}'.format(dataRange.x, xRange))
                self.assertTrue(numpy.array_equal(dataRange.y, yRange),
                                msg='{0} != {1}'.format(dataRange.y, yRange))
                self.assertIsNone(dataRange.yright)

    def testDataRangeHiddenCurve(self):
        """curves with a hidden curve"""
        plot = PlotWidget(backend='none')
        plot.addCurve((0, 1), (0, 1), legend='shown')
        plot.addCurve((0, 1, 2), (5, 5, 5), legend='hidden')
        range1 = plot.getDataRange()
        self.assertEqual(range1.x, (0, 2))
        self.assertEqual(range1.y, (0, 5))
        plot.hideCurve('hidden')
        range2 = plot.getDataRange()
        self.assertEqual(range2.x, (0, 1))
        self.assertEqual(range2.y, (0, 1))


class TestPlotGetCurveImage(unittest.TestCase):
    """Test of plot getCurve and getImage methods"""

    def testGetCurve(self):
        """PlotWidget.getCurve and Plot.getActiveCurve tests"""

        plot = PlotWidget(backend='none')

        # No curve
        curve = plot.getCurve()
        self.assertIsNone(curve)  # No curve

        plot.setActiveCurveHandling(True)
        plot.addCurve(x=(0, 1), y=(0, 1), legend='curve 0')
        plot.addCurve(x=(0, 1), y=(0, 1), legend='curve 1')
        plot.addCurve(x=(0, 1), y=(0, 1), legend='curve 2')
        plot.setActiveCurve('curve 0')

        # Active curve
        active = plot.getActiveCurve()
        self.assertEqual(active.getName(), 'curve 0')
        curve = plot.getCurve()
        self.assertEqual(curve.getName(), 'curve 0')

        # No active curve and curves
        plot.setActiveCurveHandling(False)
        active = plot.getActiveCurve()
        self.assertIsNone(active)  # No active curve
        curve = plot.getCurve()
        self.assertEqual(curve.getName(), 'curve 2')  # Last added curve

        # Last curve hidden
        plot.hideCurve('curve 2', True)
        curve = plot.getCurve()
        self.assertEqual(curve.getName(), 'curve 1')  # Last added curve

        # All curves hidden
        plot.hideCurve('curve 1', True)
        plot.hideCurve('curve 0', True)
        curve = plot.getCurve()
        self.assertIsNone(curve)

    def testGetCurveOldApi(self):
        """old API PlotWidget.getCurve and Plot.getActiveCurve tests"""

        plot = PlotWidget(backend='none')

        # No curve
        curve = plot.getCurve()
        self.assertIsNone(curve)  # No curve

        plot.setActiveCurveHandling(True)
        x = numpy.arange(10.).astype(numpy.float32)
        y = x * x
        plot.addCurve(x=x, y=y, legend='curve 0', info=["whatever"])
        plot.addCurve(x=x, y=2*x, legend='curve 1', info="anything")
        plot.setActiveCurve('curve 0')

        # Active curve (4 elements)
        xOut, yOut, legend, info = plot.getActiveCurve()[:4]
        self.assertEqual(legend, 'curve 0')
        self.assertTrue(numpy.allclose(xOut, x), 'curve 0 wrong x data')
        self.assertTrue(numpy.allclose(yOut, y), 'curve 0 wrong y data')

        # Active curve (5 elements)
        xOut, yOut, legend, info, params = plot.getCurve("curve 1")
        self.assertEqual(legend, 'curve 1')
        self.assertEqual(info, 'anything')
        self.assertTrue(numpy.allclose(xOut, x), 'curve 1 wrong x data')
        self.assertTrue(numpy.allclose(yOut, 2 * x), 'curve 1 wrong y data')

    def testGetImage(self):
        """PlotWidget.getImage and PlotWidget.getActiveImage tests"""

        plot = PlotWidget(backend='none')

        # No image
        image = plot.getImage()
        self.assertIsNone(image)

        plot.addImage(((0, 1), (2, 3)), legend='image 0')
        plot.addImage(((0, 1), (2, 3)), legend='image 1')

        # Active image
        active = plot.getActiveImage()
        self.assertEqual(active.getName(), 'image 0')
        image = plot.getImage()
        self.assertEqual(image.getName(), 'image 0')

        # No active image
        plot.addImage(((0, 1), (2, 3)), legend='image 2')
        plot.setActiveImage(None)
        active = plot.getActiveImage()
        self.assertIsNone(active)
        image = plot.getImage()
        self.assertEqual(image.getName(), 'image 2')

        # Active image
        plot.setActiveImage('image 1')
        active = plot.getActiveImage()
        self.assertEqual(active.getName(), 'image 1')
        image = plot.getImage()
        self.assertEqual(image.getName(), 'image 1')

    def testGetImageOldApi(self):
        """PlotWidget.getImage and PlotWidget.getActiveImage old API tests"""

        plot = PlotWidget(backend='none')

        # No image
        image = plot.getImage()
        self.assertIsNone(image)

        image = numpy.arange(10).astype(numpy.float32)
        image.shape = 5, 2

        plot.addImage(image, legend='image 0', info=["Hi!"])

        # Active image
        data, legend, info, something, params = plot.getActiveImage()
        self.assertEqual(legend, 'image 0')
        self.assertEqual(info, ["Hi!"])
        self.assertTrue(numpy.allclose(data, image), "image 0 data not correct")

    def testGetAllImages(self):
        """PlotWidget.getAllImages test"""

        plot = PlotWidget(backend='none')

        # No image
        images = plot.getAllImages()
        self.assertEqual(len(images), 0)

        # 2 images
        data = numpy.arange(100).reshape(10, 10)
        plot.addImage(data, legend='1')
        plot.addImage(data, origin=(10, 10), legend='2')
        images = plot.getAllImages(just_legend=True)
        self.assertEqual(list(images), ['1', '2'])
        images = plot.getAllImages(just_legend=False)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].getName(), '1')
        self.assertEqual(images[1].getName(), '2')


class TestPlotAddScatter(unittest.TestCase):
    """Test of plot addScatter"""

    def testAddGetScatter(self):

        plot = PlotWidget(backend='none')

        # No curve
        scatter = plot._getItem(kind="scatter")
        self.assertIsNone(scatter)  # No curve

        plot.addScatter(x=(0, 1), y=(0, 1), value=(0, 1), legend='scatter 0')
        plot.addScatter(x=(0, 1), y=(0, 1), value=(0, 1), legend='scatter 1')
        plot.addScatter(x=(0, 1), y=(0, 1), value=(0, 1), legend='scatter 2')
        plot._setActiveItem('scatter', 'scatter 0')

        # Active scatter
        active = plot._getActiveItem(kind='scatter')
        self.assertEqual(active.getName(), 'scatter 0')

        # check default values
        self.assertAlmostEqual(active.getSymbolSize(), active._DEFAULT_SYMBOL_SIZE)
        self.assertEqual(active.getSymbol(), "o")
        self.assertAlmostEqual(active.getAlpha(), 1.0)

        # modify parameters
        active.setSymbolSize(20.5)
        active.setSymbol("d")
        active.setAlpha(0.777)

        s0 = plot.getScatter("scatter 0")

        self.assertAlmostEqual(s0.getSymbolSize(), 20.5)
        self.assertEqual(s0.getSymbol(), "d")
        self.assertAlmostEqual(s0.getAlpha(), 0.777)

        scatter1 = plot._getItem(kind='scatter', legend='scatter 1')
        self.assertEqual(scatter1.getName(), 'scatter 1')

    def testGetAllScatters(self):
        """PlotWidget.getAllImages test"""

        plot = PlotWidget(backend='none')

        items = plot.getItems()
        self.assertEqual(len(items), 0)

        plot.addScatter(x=(0, 1), y=(0, 1), value=(0, 1), legend='scatter 0')
        plot.addScatter(x=(0, 1), y=(0, 1), value=(0, 1), legend='scatter 1')
        plot.addScatter(x=(0, 1), y=(0, 1), value=(0, 1), legend='scatter 2')

        items = plot.getItems()
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].getName(), 'scatter 0')
        self.assertEqual(items[1].getName(), 'scatter 1')
        self.assertEqual(items[2].getName(), 'scatter 2')


class TestPlotHistogram(unittest.TestCase):
    """Basic tests for histogram."""

    def testEdges(self):
        x = numpy.array([0, 1, 2])
        edgesRight = numpy.array([0, 1, 2, 3])
        edgesLeft = numpy.array([-1, 0, 1, 2])
        edgesCenter = numpy.array([-0.5, 0.5, 1.5, 2.5])

        # testing x values for right
        edges = _computeEdges(x, 'right')
        numpy.testing.assert_array_equal(edges, edgesRight)

        edges = _computeEdges(x, 'center')
        numpy.testing.assert_array_equal(edges, edgesCenter)

        edges = _computeEdges(x, 'left')
        numpy.testing.assert_array_equal(edges, edgesLeft)

    def testHistogramCurve(self):
        y = numpy.array([3, 2, 5])
        edges = numpy.array([0, 1, 2, 3])

        xHisto, yHisto = _getHistogramCurve(y, edges)
        numpy.testing.assert_array_equal(
            yHisto, numpy.array([3, 3, 2, 2, 5, 5]))

        y = numpy.array([-3, 2, 5, 0])
        edges = numpy.array([-2, -1, 0, 1, 2])
        xHisto, yHisto = _getHistogramCurve(y, edges)
        numpy.testing.assert_array_equal(
            yHisto, numpy.array([-3, -3, 2, 2, 5, 5, 0, 0]))
