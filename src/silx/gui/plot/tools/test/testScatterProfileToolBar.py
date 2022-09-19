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
from silx.gui.plot.tools.profile import manager
from silx.gui.plot.tools.profile import core
from silx.gui.plot.tools.profile import rois


class TestScatterProfileToolBar(TestCaseQt, ParametricTestCase):
    """Tests for ScatterProfileToolBar class"""

    def setUp(self):
        super(TestScatterProfileToolBar, self).setUp()
        self.plot = PlotWindow()

        self.manager = manager.ProfileManager(plot=self.plot)
        self.manager.setItemType(scatter=True)
        self.manager.setActiveItemTracking(True)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        del self.manager
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestScatterProfileToolBar, self).tearDown()

    def testHorizontalProfile(self):
        """Test ScatterProfileToolBar horizontal profile"""

        roiManager = self.manager.getRoiManager()

        # Add a scatter plot
        self.plot.addScatter(
            x=(0., 1., 1., 0.), y=(0., 0., 1., 1.), value=(0., 1., 2., 3.))
        self.plot.resetZoom(dataMargins=(.1, .1, .1, .1))
        self.qapp.processEvents()

        # Set a ROI profile
        roi = rois.ProfileScatterHorizontalLineROI()
        roi.setPosition(0.5)
        roi.setNPoints(8)
        roiManager.addRoi(roi)

        # Wait for async interpolator init
        for _ in range(20):
            self.qWait(200)
            if not self.manager.hasPendingOperations():
                break
        self.qapp.processEvents()

        window = roi.getProfileWindow()
        self.assertIsNotNone(window)
        data = window.getProfile()
        self.assertIsInstance(data, core.CurveProfileData)
        self.assertEqual(len(data.coords), 8)

        # Check that profile has same limits than Plot
        xLimits = self.plot.getXAxis().getLimits()
        self.assertEqual(data.coords[0], xLimits[0])
        self.assertEqual(data.coords[-1], xLimits[1])

        # Clear the profile
        self.manager.clearProfile()
        self.qapp.processEvents()
        self.assertIsNone(roi.getProfileWindow())

    def testVerticalProfile(self):
        """Test ScatterProfileToolBar vertical profile"""

        roiManager = self.manager.getRoiManager()

        # Add a scatter plot
        self.plot.addScatter(
            x=(0., 1., 1., 0.), y=(0., 0., 1., 1.), value=(0., 1., 2., 3.))
        self.plot.resetZoom(dataMargins=(.1, .1, .1, .1))
        self.qapp.processEvents()

        # Set a ROI profile
        roi = rois.ProfileScatterVerticalLineROI()
        roi.setPosition(0.5)
        roi.setNPoints(8)
        roiManager.addRoi(roi)

        # Wait for async interpolator init
        for _ in range(10):
            self.qWait(200)
            if not self.manager.hasPendingOperations():
                break

        window = roi.getProfileWindow()
        self.assertIsNotNone(window)
        data = window.getProfile()
        self.assertIsInstance(data, core.CurveProfileData)
        self.assertEqual(len(data.coords), 8)

        # Check that profile has same limits than Plot
        yLimits = self.plot.getYAxis().getLimits()
        self.assertEqual(data.coords[0], yLimits[0])
        self.assertEqual(data.coords[-1], yLimits[1])

        # Check that profile limits are updated when changing limits
        self.plot.getYAxis().setLimits(yLimits[0] + 1, yLimits[1] + 10)

        # Wait for async interpolator init
        for _ in range(10):
            self.qWait(200)
            if not self.manager.hasPendingOperations():
                break

        yLimits = self.plot.getYAxis().getLimits()
        data = window.getProfile()
        self.assertEqual(data.coords[0], yLimits[0])
        self.assertEqual(data.coords[-1], yLimits[1])

        # Clear the profile
        self.manager.clearProfile()
        self.qapp.processEvents()
        self.assertIsNone(roi.getProfileWindow())

    def testLineProfile(self):
        """Test ScatterProfileToolBar line profile"""

        roiManager = self.manager.getRoiManager()

        # Add a scatter plot
        self.plot.addScatter(
            x=(0., 1., 1., 0.), y=(0., 0., 1., 1.), value=(0., 1., 2., 3.))
        self.plot.resetZoom(dataMargins=(.1, .1, .1, .1))
        self.qapp.processEvents()

        # Set a ROI profile
        roi = rois.ProfileScatterLineROI()
        roi.setEndPoints(numpy.array([0., 0.]), numpy.array([1., 1.]))
        roi.setNPoints(8)
        roiManager.addRoi(roi)

        # Wait for async interpolator init
        for _ in range(10):
            self.qWait(200)
            if not self.manager.hasPendingOperations():
                break

        window = roi.getProfileWindow()
        self.assertIsNotNone(window)
        data = window.getProfile()
        self.assertIsInstance(data, core.CurveProfileData)
        self.assertEqual(len(data.coords), 8)
