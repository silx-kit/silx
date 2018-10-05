# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "28/06/2018"


import unittest
import numpy

from silx.gui import qt
from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot import PlotWindow
from silx.gui.plot.tools import profile
import silx.gui.plot.items.roi as roi_items


class TestScatterProfileToolBar(TestCaseQt, ParametricTestCase):
    """Tests for ScatterProfileToolBar class"""

    def setUp(self):
        super(TestScatterProfileToolBar, self).setUp()
        self.plot = PlotWindow()

        self.profile = profile.ScatterProfileToolBar(plot=self.plot)

        self.plot.addToolBar(self.profile)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        del self.profile
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestScatterProfileToolBar, self).tearDown()

    def testNoProfile(self):
        """Test ScatterProfileToolBar without profile"""
        self.assertEqual(self.profile.getPlotWidget(), self.plot)

        # Add a scatter plot
        self.plot.addScatter(
            x=(0., 1., 1., 0.), y=(0., 0., 1., 1.), value=(0., 1., 2., 3.))
        self.plot.resetZoom(dataMargins=(.1, .1, .1, .1))
        self.qapp.processEvents()

        # Check that there is no profile
        self.assertIsNone(self.profile.getProfileValues())
        self.assertIsNone(self.profile.getProfilePoints())

    def testHorizontalProfile(self):
        """Test ScatterProfileToolBar horizontal profile"""
        nPoints = 8
        self.profile.setNPoints(nPoints)
        self.assertEqual(self.profile.getNPoints(), nPoints)

        # Add a scatter plot
        self.plot.addScatter(
            x=(0., 1., 1., 0.), y=(0., 0., 1., 1.), value=(0., 1., 2., 3.))
        self.plot.resetZoom(dataMargins=(.1, .1, .1, .1))
        self.qapp.processEvents()

        # Activate Horizontal profile
        hlineAction = self.profile.actions()[0]
        hlineAction.trigger()
        self.qapp.processEvents()

        # Set a ROI profile
        roi = roi_items.HorizontalLineROI()
        roi.setPosition(0.5)
        self.profile._getRoiManager().addRoi(roi)

        # Wait for async interpolator init
        for _ in range(10):
            self.qWait(200)
            if not self.profile.hasPendingOperations():
                break

        self.assertIsNotNone(self.profile.getProfileValues())
        points = self.profile.getProfilePoints()
        self.assertEqual(len(points), nPoints)

        # Check that profile has same limits than Plot
        xLimits = self.plot.getXAxis().getLimits()
        self.assertEqual(points[0, 0], xLimits[0])
        self.assertEqual(points[-1, 0], xLimits[1])

        # Clear the profile
        clearAction = self.profile.actions()[-1]
        clearAction.trigger()
        self.qapp.processEvents()

        self.assertIsNone(self.profile.getProfileValues())
        self.assertIsNone(self.profile.getProfilePoints())
        self.assertEqual(self.profile.getProfileTitle(), '')

    def testVerticalProfile(self):
        """Test ScatterProfileToolBar vertical profile"""
        nPoints = 8
        self.profile.setNPoints(nPoints)
        self.assertEqual(self.profile.getNPoints(), nPoints)

        # Add a scatter plot
        self.plot.addScatter(
            x=(0., 1., 1., 0.), y=(0., 0., 1., 1.), value=(0., 1., 2., 3.))
        self.plot.resetZoom(dataMargins=(.1, .1, .1, .1))
        self.qapp.processEvents()

        # Activate vertical profile
        vlineAction = self.profile.actions()[1]
        vlineAction.trigger()
        self.qapp.processEvents()

        # Set a ROI profile
        roi = roi_items.VerticalLineROI()
        roi.setPosition(0.5)
        self.profile._getRoiManager().addRoi(roi)

        # Wait for async interpolator init
        for _ in range(10):
            self.qWait(200)
            if not self.profile.hasPendingOperations():
                break

        self.assertIsNotNone(self.profile.getProfileValues())
        points = self.profile.getProfilePoints()
        self.assertEqual(len(points), nPoints)

        # Check that profile has same limits than Plot
        yLimits = self.plot.getYAxis().getLimits()
        self.assertEqual(points[0, 1], yLimits[0])
        self.assertEqual(points[-1, 1], yLimits[1])

        # Check that profile limits are updated when changing limits
        self.plot.getYAxis().setLimits(yLimits[0] + 1, yLimits[1] + 10)
        self.qapp.processEvents()
        yLimits = self.plot.getYAxis().getLimits()
        points = self.profile.getProfilePoints()
        self.assertEqual(points[0, 1], yLimits[0])
        self.assertEqual(points[-1, 1], yLimits[1])

        # Clear the plot
        self.plot.clear()
        self.qapp.processEvents()
        self.assertIsNone(self.profile.getProfileValues())
        self.assertIsNone(self.profile.getProfilePoints())

    def testLineProfile(self):
        """Test ScatterProfileToolBar line profile"""
        nPoints = 8
        self.profile.setNPoints(nPoints)
        self.assertEqual(self.profile.getNPoints(), nPoints)

        # Activate line profile
        lineAction = self.profile.actions()[2]
        lineAction.trigger()
        self.qapp.processEvents()

        # Add a scatter plot
        self.plot.addScatter(
            x=(0., 1., 1., 0.), y=(0., 0., 1., 1.), value=(0., 1., 2., 3.))
        self.plot.resetZoom(dataMargins=(.1, .1, .1, .1))
        self.qapp.processEvents()

        # Set a ROI profile
        roi = roi_items.LineROI()
        roi.setEndPoints(numpy.array([0., 0.]), numpy.array([1., 1.]))
        self.profile._getRoiManager().addRoi(roi)

        # Wait for async interpolator init
        for _ in range(10):
            self.qWait(200)
            if not self.profile.hasPendingOperations():
                break

        self.assertIsNotNone(self.profile.getProfileValues())
        points = self.profile.getProfilePoints()
        self.assertEqual(len(points), nPoints)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestScatterProfileToolBar))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
