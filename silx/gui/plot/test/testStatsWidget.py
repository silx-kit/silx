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
"""Basic tests for CurvesROIWidget"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "07/03/2018"


from silx.gui import qt
from silx.gui.test.utils import TestCaseQt
import unittest
import logging
import numpy
from silx.gui.plot import Plot1D, Plot2D
from silx.gui.plot import StatsWidget


_logger = logging.getLogger(__name__)


class TestStatsWidgetWithCurves(TestCaseQt):
    """Basic test for StatsWidget with curves"""
    def setUp(self):
        TestCaseQt.setUp(self)
        self.plot = Plot1D()
        x = range(20)
        y = range(20)
        self.plot.addCurve(x, y, legend='curve0')
        y = range(12, 32)
        self.plot.addCurve(x, y, legend='curve1')
        y = range(-2, 18)
        self.plot.addCurve(x, y, legend='curve2')
        self.widget = StatsWidget.StatsTable(plot=self.plot)

    def tearDown(self):
        del self.widget
        del self.plot
        TestCaseQt.tearDown(self)

    def testInit(self):
        """Make sure all the curves are registred on initialization"""
        self.assertTrue(self.widget.rowCount() is 3)

    def testRemoveCurve(self):
        """Make sure the Curves stats take into account the curve removal from
        plot"""
        self.plot.removeCurve('curve2')
        self.assertTrue(self.widget.rowCount() is 2)
        for iRow in range(2):
            self.assertTrue(self.widget.item(iRow, 0).text() in ('curve0', 'curve1'))

        self.plot.removeCurve('curve0')
        self.assertTrue(self.widget.rowCount() is 1)
        self.plot.removeCurve('curve1')
        self.assertTrue(self.widget.rowCount() is 0)

    def testAddCurve(self):
        """Make sure the Curves stats take into account the add curve action"""
        self.plot.addCurve(legend='curve3', x=range(10), y=range(10))
        self.assertTrue(self.widget.rowCount() is 4)

    def testUpdateCurveFrmAddCurve(self):
        """Make sure the stats of the cuve will be removed after updating a
        curve"""
        self.plot.addCurve(legend='curve0', x=range(10), y=range(10))
        self.assertTrue(self.widget.rowCount() is 3)
        itemMax = self.widget.item(self.widget._legendToItems['curve0'].row(),
                                   StatsWidget.StatsTable.COLUMNS_INDEX['max'])
        self.assertTrue(itemMax.text() == '9')

    def testUpdateCurveFrmCurveObj(self):
        self.plot.getCurve('curve0').setData(x=range(4), y=range(4))
        self.assertTrue(self.widget.rowCount() is 3)
        itemMax = self.widget.item(self.widget._legendToItems['curve0'].row(),
                                   StatsWidget.StatsTable.COLUMNS_INDEX['max'])
        self.assertTrue(itemMax.text() == '3')

    def testSetAnotherPlot(self):
        plot2 = Plot1D()
        plot2.addCurve(x=range(26), y=range(26), legend='new curve')
        self.widget.setPlot(plot2)
        self.assertTrue(self.widget.rowCount() is 1)


class TestStatsWidgetWithImages(TestCaseQt):
    """Basic test for StatsWidget with images"""
    def setUp(self):
        TestCaseQt.setUp(self)
        self.plot = Plot2D()

        self.plot.addImage(data=numpy.arange(128*128).reshape(128, 128),
                           legend='test1', replace=False)

        self.widget = StatsWidget.StatsTable(plot=self.plot)

    def tearDown(self):
        del self.widget
        del self.plot
        TestCaseQt.tearDown(self)

    def test(self):
        pass


class TestStatsWidgetWithScatter(TestCaseQt):
    def setUp(self):
        TestCaseQt.setUp(self)
        self.scatterPlot = Plot2D()
        self.scatterPlot.addScatter([0, 1, 2, 20, 50, 60],
                                    [2, 3, 4, 26, 69, 6],
                                    [5, 6, 7, 10, 90, 20],
                                    legend='scatter plot')
        self.widget = StatsWidget.StatsTable(plot=self.scatterPlot)

    def tearDown(self):
        del self.widget
        del self.scatterPlot
        TestCaseQt.tearDown(self)

    def testStats(self):
        columnsIndex = StatsWidget.StatsTable.COLUMNS_INDEX
        itemLegend = self.widget._legendToItems['scatter plot']
        itemMin = self.widget.item(itemLegend.row(), columnsIndex['min'])
        itemMax = self.widget.item(itemLegend.row(), columnsIndex['max'])
        itemDelta = self.widget.item(itemLegend.row(), columnsIndex['delta'])
        itemCoordsMin = self.widget.item(itemLegend.row(),
                                         columnsIndex['coords min'])
        itemCoordsMax = self.widget.item(itemLegend.row(),
                                         columnsIndex['coords max'])
        self.assertTrue(itemMin.text() == '5')
        self.assertTrue(itemMax.text() == '90')
        self.assertTrue(itemDelta.text() == '85')
        self.assertTrue(itemCoordsMin.text() == '(0, 2)')
        self.assertTrue(itemCoordsMax.text() == '(50, 69)')


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestStatsWidgetWithCurves, TestStatsWidgetWithImages,
                      TestStatsWidgetWithScatter):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
