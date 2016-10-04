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
"""Basic tests for Plot"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/03/2016"


import unittest
from functools import reduce
from silx.testutils import ParametricTestCase

import numpy

from silx.gui.plot.Plot import Plot


class TestPlot(unittest.TestCase):
    """Basic tests of Plot without backend"""

    def testPlotTitleLabels(self):
        """Create a Plot and set the labels"""

        plot = Plot(backend='none')

        title, xlabel, ylabel = 'the title', 'x label', 'y label'
        plot.setGraphTitle(title)
        plot.setGraphXLabel(xlabel)
        plot.setGraphYLabel(ylabel)

        self.assertEqual(plot.getGraphTitle(), title)
        self.assertEqual(plot.getGraphXLabel(), xlabel)
        self.assertEqual(plot.getGraphYLabel(), ylabel)

    def testAddNoRemove(self):
        """add objects to the Plot"""

        plot = Plot(backend='none')
        plot.addCurve(x=(1, 2, 3), y=(3, 2, 1))
        plot.addImage(numpy.arange(100.).reshape(10, -1))
        plot.addItem(
            numpy.array((1., 10.)), numpy.array((10., 10.)), shape="rectangle")
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

        plot = Plot(backend='none')

        for logX, logY in ((False, False),
                           (True, False),
                           (True, True),
                           (False, True),
                           (False, False)):
            with self.subTest(logX=logX, logY=logY):
                plot.setXAxisLogarithmic(logX)
                plot.setYAxisLogarithmic(logY)
                dataRange = plot.getDataRange()
                self.assertIsNone(dataRange.x)
                self.assertIsNone(dataRange.y)
                self.assertIsNone(dataRange.yright)

    def testDataRangeLeft(self):
        """left axis range"""

        plot = Plot(backend='none')

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
                plot.setXAxisLogarithmic(logX)
                plot.setYAxisLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRange, yRange = self._getRanges([xData, yData],
                                                 [logX, logY])
                self.assertSequenceEqual(dataRange.x, xRange)
                self.assertSequenceEqual(dataRange.y, yRange)
                self.assertIsNone(dataRange.yright)

    def testDataRangeRight(self):
        """right axis range"""

        plot = Plot(backend='none')
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
                plot.setXAxisLogarithmic(logX)
                plot.setYAxisLogarithmic(logY)
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

        plot = Plot(backend='none')
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
                plot.setXAxisLogarithmic(logX)
                plot.setYAxisLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRange, yRange = ranges[logX, logY]
                self.assertTrue(numpy.array_equal(dataRange.x, xRange),
                                msg='{0} != {1}'.format(dataRange.x, xRange))
                self.assertTrue(numpy.array_equal(dataRange.y, yRange),
                                msg='{0} != {1}'.format(dataRange.y, yRange))
                self.assertIsNone(dataRange.yright)

    def testDataRangeLeftRight(self):
        """right+left axis range"""

        plot = Plot(backend='none')

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
                plot.setXAxisLogarithmic(logX)
                plot.setYAxisLogarithmic(logY)
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
        plot = Plot(backend='none')

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
                plot.setXAxisLogarithmic(logX)
                plot.setYAxisLogarithmic(logY)
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

        plot = Plot(backend='none')
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
                plot.setXAxisLogarithmic(logX)
                plot.setYAxisLogarithmic(logY)
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

        plot = Plot(backend='none')
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
                plot.setXAxisLogarithmic(logX)
                plot.setYAxisLogarithmic(logY)
                dataRange = plot.getDataRange()
                xRange, yRange = ranges[logX, logY]
                self.assertTrue(numpy.array_equal(dataRange.x, xRange),
                                msg='{0} != {1}'.format(dataRange.x, xRange))
                self.assertTrue(numpy.array_equal(dataRange.y, yRange),
                                msg='{0} != {1}'.format(dataRange.y, yRange))
                self.assertIsNone(dataRange.yright)


class TestPlotGetCurveImage(unittest.TestCase):
    """Test of plot getCurve and getImage methods"""

    def testGetCurve(self):
        """Plot.getCurve and Plot.getActiveCurve tests"""

        plot = Plot(backend='none')

        # No curve
        curve = plot.getCurve()
        self.assertIsNone(curve)  # No curve

        plot.setActiveCurveHandling(True)
        plot.addCurve(x=(0, 1), y=(0, 1), legend='curve 0')
        plot.addCurve(x=(0, 1), y=(0, 1), legend='curve 1')
        plot.addCurve(x=(0, 1), y=(0, 1), legend='curve 2')

        # Active curve
        active = plot.getActiveCurve()
        self.assertEqual(active[2], 'curve 0')  # Test curve legend
        curve = plot.getCurve()
        self.assertEqual(curve[2], 'curve 0')  # Test curve legend

        # No active curve and curves
        plot.setActiveCurveHandling(False)
        active = plot.getActiveCurve()
        self.assertIsNone(active)  # No active curve
        curve = plot.getCurve()
        self.assertEqual(curve[2], 'curve 2')  # Last added curve

        # Last curve hidden
        plot.hideCurve('curve 2', True)
        curve = plot.getCurve()
        self.assertEqual(curve[2], 'curve 1')  # Last added curve

        # All curves hidden
        plot.hideCurve('curve 1', True)
        plot.hideCurve('curve 0', True)
        curve = plot.getCurve()
        self.assertIsNone(curve)

    def testGetImage(self):
        """Plot.getImage and Plot.getActiveImage tests"""

        plot = Plot(backend='none')

        # No image
        image = plot.getImage()
        self.assertIsNone(image)

        plot.addImage(((0, 1), (2, 3)), legend='image 0', replace=False)
        plot.addImage(((0, 1), (2, 3)), legend='image 1', replace=False)

        # Active image
        active = plot.getActiveImage()
        self.assertEqual(active[1], 'image 0')  # Test image legend
        image = plot.getImage()
        self.assertEqual(image[1], 'image 0')  # Test image legend

        # No active image
        plot.addImage(((0, 1), (2, 3)), legend='image 2', replace=False)
        plot.setActiveImage(None)
        active = plot.getActiveImage()
        self.assertIsNone(active)
        image = plot.getImage()
        self.assertEqual(image[1], 'image 2')  # Test image legend

        # Active image
        plot.setActiveImage('image 1')
        active = plot.getActiveImage()
        self.assertEqual(active[1], 'image 1')
        image = plot.getImage()
        self.assertEqual(image[1], 'image 1')  # Test image legend


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestPlot, TestPlotRanges, TestPlotGetCurveImage):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
