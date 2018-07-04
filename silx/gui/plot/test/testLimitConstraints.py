# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""Test setLimitConstaints on the PlotWidget"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "30/08/2017"


import unittest
from silx.gui.plot import PlotWidget


class TestLimitConstaints(unittest.TestCase):
    """Tests setLimitConstaints class"""

    def setUp(self):
        self.plot = PlotWidget()

    def tearDown(self):
        self.plot = None

    def testApi(self):
        """Test availability of the API"""
        self.plot.getXAxis().setLimitsConstraints(minPos=1, maxPos=10)
        self.plot.getXAxis().setRangeConstraints(minRange=1, maxRange=1)
        self.plot.getYAxis().setLimitsConstraints(minPos=1, maxPos=10)
        self.plot.getYAxis().setRangeConstraints(minRange=1, maxRange=1)

    def testXMinMax(self):
        """Test limit constains on x-axis"""
        self.plot.getXAxis().setLimitsConstraints(minPos=0, maxPos=100)
        self.plot.setLimits(xmin=-1, xmax=101, ymin=-1, ymax=101)
        self.assertEqual(self.plot.getXAxis().getLimits(), (0, 100))
        self.assertEqual(self.plot.getYAxis().getLimits(), (-1, 101))

    def testYMinMax(self):
        """Test limit constains on y-axis"""
        self.plot.getYAxis().setLimitsConstraints(minPos=0, maxPos=100)
        self.plot.setLimits(xmin=-1, xmax=101, ymin=-1, ymax=101)
        self.assertEqual(self.plot.getXAxis().getLimits(), (-1, 101))
        self.assertEqual(self.plot.getYAxis().getLimits(), (0, 100))

    def testMinXRange(self):
        """Test min range constains on x-axis"""
        self.plot.getXAxis().setRangeConstraints(minRange=100)
        self.plot.setLimits(xmin=1, xmax=99, ymin=1, ymax=99)
        limits = self.plot.getXAxis().getLimits()
        self.assertEqual(limits[1] - limits[0], 100)
        limits = self.plot.getYAxis().getLimits()
        self.assertNotEqual(limits[1] - limits[0], 100)

    def testMaxXRange(self):
        """Test max range constains on x-axis"""
        self.plot.getXAxis().setRangeConstraints(maxRange=100)
        self.plot.setLimits(xmin=-1, xmax=101, ymin=-1, ymax=101)
        limits = self.plot.getXAxis().getLimits()
        self.assertEqual(limits[1] - limits[0], 100)
        limits = self.plot.getYAxis().getLimits()
        self.assertNotEqual(limits[1] - limits[0], 100)

    def testMinYRange(self):
        """Test min range constains on y-axis"""
        self.plot.getYAxis().setRangeConstraints(minRange=100)
        self.plot.setLimits(xmin=1, xmax=99, ymin=1, ymax=99)
        limits = self.plot.getXAxis().getLimits()
        self.assertNotEqual(limits[1] - limits[0], 100)
        limits = self.plot.getYAxis().getLimits()
        self.assertEqual(limits[1] - limits[0], 100)

    def testMaxYRange(self):
        """Test max range constains on y-axis"""
        self.plot.getYAxis().setRangeConstraints(maxRange=100)
        self.plot.setLimits(xmin=-1, xmax=101, ymin=-1, ymax=101)
        limits = self.plot.getXAxis().getLimits()
        self.assertNotEqual(limits[1] - limits[0], 100)
        limits = self.plot.getYAxis().getLimits()
        self.assertEqual(limits[1] - limits[0], 100)

    def testChangeOfConstraints(self):
        """Test changing of the constraints"""
        self.plot.getXAxis().setRangeConstraints(minRange=10, maxRange=10)
        # There is no more constraints on the range
        self.plot.getXAxis().setRangeConstraints(minRange=None, maxRange=None)
        self.plot.setLimits(xmin=-1, xmax=101, ymin=-1, ymax=101)
        self.assertEqual(self.plot.getXAxis().getLimits(), (-1, 101))

    def testSettingConstraints(self):
        """Test setting a constaint (setLimits first then the constaint)"""
        self.plot.setLimits(xmin=-1, xmax=101, ymin=-1, ymax=101)
        self.plot.getXAxis().setLimitsConstraints(minPos=0, maxPos=100)
        self.assertEqual(self.plot.getXAxis().getLimits(), (0, 100))


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestLimitConstaints))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
