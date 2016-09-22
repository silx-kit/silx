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

    def testDataRangeNoPlot(self):
        """empty plot data range"""

        plot = Plot(backend='none')
        dataRange = plot.getDataRange()
        self.assertIsNone(dataRange.x)
        self.assertIsNone(dataRange.y)
        self.assertIsNone(dataRange.yright)

    def testDataRangeLeft(self):
        """left axis range"""

        plot = Plot(backend='none')
        plot.addCurve(x=numpy.arange(10) - 5., y=numpy.arange(10) - 7.,
                      legend='plot_0', yaxis='left')

        dataRange = plot.getDataRange()
        self.assertEqual(dataRange.x, (-5., 4.))
        self.assertEqual(dataRange.y, (-7., 2.))
        self.assertIsNone(dataRange.yright)

    def testDataRangeRight(self):
        """right axis range"""

        plot = Plot(backend='none')
        plot.addCurve(x=numpy.arange(10) - 5., y=numpy.arange(10) - 7.,
                      legend='plot_0', yaxis='right')

        dataRange = plot.getDataRange()
        self.assertEqual(dataRange.x, (-5., 4.))
        self.assertIsNone(dataRange.y)
        self.assertEqual(dataRange.yright, (-7., 2.))

    def testDataRangeImage(self):
        """image data range"""

        plot = Plot(backend='none')
        plot.addImage(numpy.arange(100.).reshape(20, 5),
                      origin=(-10, 25), scale=(3., 8.))

        dataRange = plot.getDataRange()
        self.assertEqual(dataRange.x, (-10., 5.))
        self.assertEqual(dataRange.y, (25., 185.))
        self.assertIsNone(dataRange.yright)

    def testDataRangeLeftRight(self):
        """right+left axis range"""

        plot = Plot(backend='none')
        plot.addCurve(x=numpy.arange(10) - 1., y=numpy.arange(10) - 2.,
                      legend='plot_left', yaxis='left')
        plot.addCurve(x=numpy.arange(10) - 5., y=numpy.arange(10) - 7.,
                      legend='plot_right', yaxis='right')

        dataRange = plot.getDataRange()
        self.assertEqual(dataRange.x, (-5., 8.))
        self.assertEqual(dataRange.y, (-2, 7.))
        self.assertEqual(dataRange.yright, (-7., 2.))

    def testDataRangeCurveImage(self):
        """right+left+image axis range"""

        # overlapping ranges :
        # image sets x min and y max
        # plot_left sets y min
        # plot_right sets x max (and yright)
        plot = Plot(backend='none')
        plot.addImage(numpy.arange(100.).reshape(20, 5),
                      origin=(-10, 5), scale=(3., 8.), legend='image')
        plot.addCurve(x=numpy.arange(10) - 1., y=numpy.arange(10) - 2.,
                      legend='plot_left', yaxis='left')
        plot.addCurve(x=numpy.arange(10) + 5., y=numpy.arange(10) - 1.,
                      legend='plot_right', yaxis='right')

        dataRange = plot.getDataRange()
        self.assertEqual(dataRange.x, (-10., 14.))
        self.assertEqual(dataRange.y, (-2, 165.))
        self.assertEqual(dataRange.yright, (-1., 8.))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPlot))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
