# coding: utf-8
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
"""Basic tests for ColorBar featues and sub widgets of Colorbar module"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "24/04/2018"

import unittest
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot.ColorBar import _ColorScale
from silx.gui.plot.ColorBar import ColorBarWidget
from silx.gui.colors import Colormap
from silx.gui import colors
from silx.gui.plot import Plot2D
from silx.gui import qt
import numpy


class TestColorScale(TestCaseQt):
    """Test that interaction with the colorScale is correct"""
    def setUp(self):
        super(TestColorScale, self).setUp()
        self.colorScaleWidget = _ColorScale(colormap=None, parent=None)
        self.colorScaleWidget.show()
        self.qWaitForWindowExposed(self.colorScaleWidget)

    def tearDown(self):
        self.qapp.processEvents()
        self.colorScaleWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.colorScaleWidget.close()
        del self.colorScaleWidget
        super(TestColorScale, self).tearDown()

    def testNoColormap(self):
        """Test _ColorScale without a colormap"""
        colormap = self.colorScaleWidget.getColormap()
        self.assertIsNone(colormap)

    def testRelativePositionLinear(self):
        self.colorMapLin1 = Colormap(name='gray',
                                     normalization=Colormap.LINEAR,
                                     vmin=0.0,
                                     vmax=1.0)
        self.colorScaleWidget.setColormap(self.colorMapLin1)

        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(0.25) == 0.25)
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(0.5) == 0.5)
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(1.0) == 1.0)

        self.colorMapLin2 = Colormap(name='viridis',
                                     normalization=Colormap.LINEAR,
                                     vmin=-10,
                                     vmax=0)
        self.colorScaleWidget.setColormap(self.colorMapLin2)

        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(0.25) == -7.5)
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(0.5) == -5.0)
        self.assertTrue(
            self.colorScaleWidget.getValueFromRelativePosition(1.0) == 0.0)

    def testRelativePositionLog(self):
        self.colorMapLog1 = Colormap(name='temperature',
                                     normalization=Colormap.LOGARITHM,
                                     vmin=1.0,
                                     vmax=100.0)

        self.colorScaleWidget.setColormap(self.colorMapLog1)

        val = self.colorScaleWidget.getValueFromRelativePosition(1.0)
        self.assertAlmostEqual(val, 100.0)

        val = self.colorScaleWidget.getValueFromRelativePosition(0.5)
        self.assertAlmostEqual(val, 10.0)

        val = self.colorScaleWidget.getValueFromRelativePosition(0.0)
        self.assertTrue(val == 1.0)


class TestNoAutoscale(TestCaseQt):
    """Test that ticks and color displayed are correct in the case of a colormap
    with no autoscale
    """

    def setUp(self):
        super(TestNoAutoscale, self).setUp()
        self.plot = Plot2D()
        self.colorBar = self.plot.getColorBarWidget()
        self.colorBar.setVisible(True)  # Makes sure the colormap is visible
        self.tickBar = self.colorBar.getColorScaleBar().getTickBar()
        self.colorScale = self.colorBar.getColorScaleBar().getColorScale()

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        self.qapp.processEvents()
        self.tickBar = None
        self.colorScale = None
        del self.colorBar
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestNoAutoscale, self).tearDown()

    def testLogNormNoAutoscale(self):
        colormapLog = Colormap(name='gray',
                               normalization=Colormap.LOGARITHM,
                               vmin=1.0,
                               vmax=100.0)

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
        self.assertAlmostEqual(val, 100.0)

        val = self.colorScale.getValueFromRelativePosition(0.0)
        self.assertTrue(val == 1.0)

    def testLinearNormNoAutoscale(self):
        colormapLog = Colormap(name='gray',
                               normalization=Colormap.LINEAR,
                               vmin=-4,
                               vmax=5)

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


class TestColorBarWidget(TestCaseQt):
    """Test interaction with the ColorBarWidget"""

    def setUp(self):
        super(TestColorBarWidget, self).setUp()
        self.plot = Plot2D()
        self.colorBar = self.plot.getColorBarWidget()
        self.colorBar.setVisible(True)  # Makes sure the colormap is visible

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        self.qapp.processEvents()
        del self.colorBar
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestColorBarWidget, self).tearDown()

    def testEmptyColorBar(self):
        colorBar = ColorBarWidget(parent=None)
        colorBar.show()
        self.qWaitForWindowExposed(colorBar)

    def testNegativeColormaps(self):
        """test the behavior of the ColorBarWidget in the case of negative
        values

        Note : colorbar is modified by the Plot directly not ColorBarWidget
        """
        colormapLog = Colormap(name='gray',
                               normalization=Colormap.LOGARITHM,
                               vmin=None,
                               vmax=None)

        data = numpy.array([-5, -4, 0, 2, 3, 5, 10, 20, 30])
        data = data.reshape(3, 3)
        self.plot.addImage(data=data, colormap=colormapLog, legend='toto')
        self.plot.setActiveImage('toto')

        # default behavior when with log and negative values: should set vmin
        # to 1 and vmax to 10
        self.assertTrue(self.colorBar.getColorScaleBar().minVal == 2)
        self.assertTrue(self.colorBar.getColorScaleBar().maxVal == 30)

        # if data is positive
        data[data < 1] = data.max()
        self.plot.addImage(data=data,
                           colormap=colormapLog,
                           legend='toto',
                           replace=True)
        self.plot.setActiveImage('toto')

        self.assertTrue(self.colorBar.getColorScaleBar().minVal == data.min())
        self.assertTrue(self.colorBar.getColorScaleBar().maxVal == data.max())

    def testPlotAssocation(self):
        """Make sure the ColorBarWidget is properly connected with the plot"""
        colormap = Colormap(name='gray',
                            normalization=Colormap.LINEAR,
                            vmin=None,
                            vmax=None)

        # make sure that default settings are the same (but a copy of the
        self.colorBar.setPlot(self.plot)
        self.assertTrue(
            self.colorBar.getColormap() is self.plot.getDefaultColormap())

        data = numpy.linspace(0, 10, 100).reshape(10, 10)
        self.plot.addImage(data=data, colormap=colormap, legend='toto')
        self.plot.setActiveImage('toto')

        # make sure the modification of the colormap has been done
        self.assertFalse(
            self.colorBar.getColormap() is self.plot.getDefaultColormap())
        self.assertTrue(
            self.colorBar.getColormap() is colormap)

        # test that colorbar is updated when default plot colormap changes
        self.plot.clear()
        plotColormap = Colormap(name='gray',
                                normalization=Colormap.LOGARITHM,
                                vmin=None,
                                vmax=None)
        self.plot.setDefaultColormap(plotColormap)
        self.assertTrue(self.colorBar.getColormap() is plotColormap)

    def testColormapWithoutRange(self):
        """Test with a colormap with vmin==vmax"""
        colormap = Colormap(name='gray',
                            normalization=Colormap.LINEAR,
                            vmin=1.0,
                            vmax=1.0)
        self.colorBar.setColormap(colormap)


class TestColorBarUpdate(TestCaseQt):
    """Test that the ColorBar is correctly updated when the signal 'sigChanged'
    of the colormap is emitted
    """

    def setUp(self):
        super(TestColorBarUpdate, self).setUp()
        self.plot = Plot2D()
        self.colorBar = self.plot.getColorBarWidget()
        self.colorBar.setVisible(True)  # Makes sure the colormap is visible
        self.colorBar.setPlot(self.plot)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)
        self.data = numpy.random.rand(9).reshape(3, 3)

    def tearDown(self):
        self.qapp.processEvents()
        del self.colorBar
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestColorBarUpdate, self).tearDown()

    def testUpdateColorMap(self):
        colormap = Colormap(name='gray',
                            normalization='linear',
                            vmin=0,
                            vmax=1)

        # check inital state
        self.plot.addImage(data=self.data, colormap=colormap, legend='toto')
        self.plot.setActiveImage('toto')

        self.assertTrue(self.colorBar.getColorScaleBar().minVal == 0)
        self.assertTrue(self.colorBar.getColorScaleBar().maxVal == 1)
        self.assertTrue(
            self.colorBar.getColorScaleBar().getTickBar()._vmin == 0)
        self.assertTrue(
            self.colorBar.getColorScaleBar().getTickBar()._vmax == 1)
        self.assertIsInstance(
            self.colorBar.getColorScaleBar().getTickBar()._normalizer,
            colors._LinearNormalization)

        # update colormap
        colormap.setVMin(0.5)
        self.assertTrue(self.colorBar.getColorScaleBar().minVal == 0.5)
        self.assertTrue(
            self.colorBar.getColorScaleBar().getTickBar()._vmin == 0.5)

        colormap.setVMax(0.8)
        self.assertTrue(self.colorBar.getColorScaleBar().maxVal == 0.8)
        self.assertTrue(
            self.colorBar.getColorScaleBar().getTickBar()._vmax == 0.8)

        colormap.setNormalization('log')
        self.assertIsInstance(
            self.colorBar.getColorScaleBar().getTickBar()._normalizer,
            colors._LogarithmicNormalization)

    # TODO : should also check that if the colormap is changing then values (especially in log scale)
    # should be coherent if in autoscale


def suite():
    test_suite = unittest.TestSuite()
    for ui in (TestColorScale, TestNoAutoscale, TestColorBarWidget,
               TestColorBarUpdate):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(ui))

    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
