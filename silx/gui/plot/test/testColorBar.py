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
"""Basic tests for ColorBar featues and sub widgets of Colorbar module"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "11/04/2017"

import unittest
from silx.gui.test.utils import TestCaseQt
from silx.gui.plot.ColorBar import _ColorScale
from silx.gui.plot.ColorBar import ColorBarWidget
from silx.gui.plot import Plot2D
import numpy


class TestColorScale(unittest.TestCase):
    """Test that interaction with the colorScale is correct"""
    def setUp(self):
        self.colorScaleWidget = _ColorScale(colormap=None, parent=None)

    def tearDown(self):
        self.colorScaleWidget.deleteLater()
        self.colorScaleWidget = None

    def testRelativePositionLinear(self):
        self.colorMapLin1 = { 'name': 'gray', 'normalization': 'linear',
                    'autoscale': False, 'vmin': 0.0, 'vmax': 1.0 }
        self.colorScaleWidget.setColormap(self.colorMapLin1)
        
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(0.25) == 0.25)
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(0.5) == 0.5)
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(1.0) == 1.0)

        self.colorMapLin2 = { 'name': 'viridis', 'normalization': 'linear',
                    'autoscale': False, 'vmin': -10, 'vmax': 0 }
        self.colorScaleWidget.setColormap(self.colorMapLin2)
        
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(0.25) == -7.5)
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(0.5) == -5.0)
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(1.0) == 0.0)

    def testRelativePositionLog(self):
        self.colorMapLog1 = { 'name': 'temperature', 'normalization': 'log',
                    'autoscale': False, 'vmin': 1.0, 'vmax': 100.0 }

        self.colorScaleWidget.setColormap(self.colorMapLog1)

        val = self.colorScaleWidget.getValueFromRelativePosition(1.0)
        self.assertTrue(val == 100.0)

        val = self.colorScaleWidget.getValueFromRelativePosition(0.5)
        self.assertTrue(val == 10.0)
        
        val = self.colorScaleWidget.getValueFromRelativePosition(0.0)
        self.assertTrue(val == 1.0)

    def testNegativeLogMin(self):
        colormap = { 'name': 'gray', 'normalization': 'log',
                    'autoscale': False, 'vmin': -1.0, 'vmax': 1.0 }

        with self.assertRaises(ValueError):
            self.colorScaleWidget.setColormap(colormap)

    def testNegativeLogMax(self):
        colormap = { 'name': 'gray', 'normalization': 'log',
                    'autoscale': False, 'vmin': 1.0, 'vmax': -1.0 }

        with self.assertRaises(ValueError):
            self.colorScaleWidget.setColormap(colormap)
        
class TestNoAutoscale(unittest.TestCase):
    """Test that ticks and color displayed are correct in the case of a colormap
    with no autoscale
    """

    def setUp(self):
        self.plot = Plot2D()
        self.colorBar = ColorBarWidget(parent=None, plot=self.plot)
        self.tickBar = self.colorBar.getColorScaleBar().getTickBar()
        self.colorScale = self.colorBar.getColorScaleBar().getColorScale()

    def tearDown(self):
        self.tickBar = None
        self.colorScale = None
        del self.colorBar
        self.plot.close()
        del self.plot

    def testLogNormNoAutoscale(self):
        colormapLog = { 'name': 'gray', 'normalization': 'log',
                    'autoscale': False, 'vmin': 1.0, 'vmax': 100.0 }

        data = numpy.linspace(10, 1e10, 9).reshape(3, 3)
        self.plot.addImage(data=data, colormap=colormapLog, legend='toto')
        self.plot.setActiveImage('toto')

        # test Ticks
        self.tickBar.setTicksNumber(10)
        self.tickBar.computeTicks()

        ticksTh = numpy.linspace(1.0, 100.0, 10)
        ticksTh = 10**ticksTh
        numpy.array_equal(self.tickBar.ticks, ticksTh)

        # test ColorScale
        val = self.colorScale.getValueFromRelativePosition(1.0)
        self.assertTrue(val == 100.0)

        val = self.colorScale.getValueFromRelativePosition(0.0)
        self.assertTrue(val == 1.0)

    def testLinearNormNoAutoscale(self):
        colormapLog = { 'name': 'gray', 'normalization': 'linear',
                    'autoscale': False, 'vmin': -4, 'vmax': 5 }

        data = numpy.linspace(1, 9, 9).reshape(3, 3)
        self.plot.addImage(data=data, colormap=colormapLog, legend='toto')
        self.plot.setActiveImage('toto')

        # test Ticks
        self.tickBar.setTicksNumber(10)
        self.tickBar.computeTicks()

        numpy.array_equal(self.tickBar.ticks, numpy.linspace(-4, 5, 10))

        # test ColorScale
        val = self.colorScale.getValueFromRelativePosition(1.0)
        self.assertTrue(val == 5.0)

        val = self.colorScale.getValueFromRelativePosition(0.0)
        self.assertTrue(val == -4.0)

class TestColorbarWidget(TestCaseQt):
    """Test interaction with the ColorScaleBar"""

    def setUp(self):
        super(TestColorbarWidget, self).setUp()
        self.plot = Plot2D()
        self.colorBar = ColorBarWidget(parent=None, plot=self.plot)

    def tearDown(self):
        del self.colorBar
        self.plot.close()
        del self.plot

        super(TestColorbarWidget, self).tearDown()

    def testEmptyColorBar(self):
        colorBar = ColorBarWidget(parent=None)
        colorBar.show()
        self.qWaitForWindowExposed(colorBar)

    def testNegativeColormaps(self):
        """test the behavior of the ColorBarWidget in the case of negative
        values

        Note : colorbar is modified by the Plot directly not ColorBarWidget
        """
        colormapLog = { 'name': 'gray', 'normalization': 'log',
                    'autoscale': True, 'vmin': -1.0, 'vmax': 1.0 }

        colormapLog2 = { 'name': 'gray', 'normalization': 'log',
                    'autoscale': False, 'vmin': -1.0, 'vmax': 1.0 }

        data = numpy.array([-5, -4, 0, 2, 3, 5, 10, 20, 30])
        data = data.reshape(3, 3)
        self.plot.addImage(data=data, colormap=colormapLog, legend='toto')
        self.plot.setActiveImage('toto')

        # default behavior when autoscale : set to minmal positive value
        data[data<1] = data.max()
        self.assertTrue(self.colorBar._colormap['vmin'] == data.min())
        self.assertTrue(self.colorBar._colormap['vmax'] == data.max())

        data = numpy.linspace(-9, -2, 100).reshape(10, 10)

        self.plot.addImage(data=data, colormap=colormapLog2, legend='toto')
        self.plot.setActiveImage('toto')
        # if negative values, changing bounds for default : 1, 10
        self.assertTrue(self.colorBar._colormap['vmin'] == 1)
        self.assertTrue(self.colorBar._colormap['vmax'] == 10)

    def testPlotAssocation(self):
        """Make sure the ColorBarWidget is proparly connected with the plot"""
        colormap = { 'name': 'gray', 'normalization': 'linear',
                    'autoscale': True, 'vmin': -1.0, 'vmax': 1.0 }

        # make sure that default settings are the same
        self.assertTrue(
            self.colorBar.getColormap() == self.plot.getDefaultColormap())

        data = numpy.linspace(0, 10, 100).reshape(10, 10)
        self.plot.addImage(data=data, colormap=colormap, legend='toto')
        self.plot.setActiveImage('toto')

        # make sure the modification of the colormap has been done
        self.assertFalse(
            self.colorBar.getColormap() == self.plot.getDefaultColormap())


def suite():
    test_suite = unittest.TestSuite()
    for ui in (TestColorScale, TestNoAutoscale, TestColorbarWidget):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(ui))

    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')