# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
"""Basic tests for Profile"""

__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "17/01/2018"

import numpy
import unittest

from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import (
    TestCaseQt, getQToolButtonFromAction)
from silx.gui import qt
from silx.gui.plot import PlotWindow, Plot1D, Plot2D, Profile
from silx.gui.plot.StackView import StackView
from silx.gui.plot.tools.profile import rois


class TestProfileToolBar(TestCaseQt, ParametricTestCase):
    """Tests for ProfileToolBar widget."""

    def setUp(self):
        super(TestProfileToolBar, self).setUp()
        profileWindow = PlotWindow()
        self.plot = PlotWindow()
        self.toolBar = Profile.ProfileToolBar(
            plot=self.plot, profileWindow=profileWindow)
        self.plot.addToolBar(self.toolBar)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)
        profileWindow.show()
        self.qWaitForWindowExposed(profileWindow)

        self.mouseMove(self.plot)  # Move to center
        self.qapp.processEvents()

    def tearDown(self):
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        del self.toolBar

        super(TestProfileToolBar, self).tearDown()

    def testAlignedProfile(self):
        """Test horizontal and vertical profile, without and with image"""
        # Use Plot backend widget to submit mouse events
        widget = self.plot.getWidgetHandle()
        for method in ('sum', 'mean'):
            with self.subTest(method=method):
                # 2 positions to use for mouse events
                pos1 = widget.width() * 0.4, widget.height() * 0.4
                pos2 = widget.width() * 0.6, widget.height() * 0.6

                for action in (self.toolBar.hLineAction, self.toolBar.vLineAction):
                    with self.subTest(mode=action.text()):
                        # Trigger tool button for mode
                        toolButton = getQToolButtonFromAction(action)
                        self.assertIsNot(toolButton, None)
                        self.mouseMove(toolButton)
                        self.mouseClick(toolButton, qt.Qt.LeftButton)

                        # Without image
                        self.mouseMove(widget, pos=pos1)
                        self.mouseClick(widget, qt.Qt.LeftButton, pos=pos1)

                        # with image
                        self.plot.addImage(
                            numpy.arange(100 * 100).reshape(100, -1))
                        self.mousePress(widget, qt.Qt.LeftButton, pos=pos1)
                        self.mouseMove(widget, pos=pos2)
                        self.mouseRelease(widget, qt.Qt.LeftButton, pos=pos2)

                        self.mouseMove(widget)
                        self.mouseClick(widget, qt.Qt.LeftButton)

                        manager = self.toolBar.getProfileManager()
                        for _ in range(20):
                            self.qWait(200)
                            if not manager.hasPendingOperations():
                                break

    def testDiagonalProfile(self):
        """Test diagonal profile, without and with image"""
        # Use Plot backend widget to submit mouse events
        widget = self.plot.getWidgetHandle()

        for method in ('sum', 'mean'):
            with self.subTest(method=method):
                # 2 positions to use for mouse events
                pos1 = widget.width() * 0.4, widget.height() * 0.4
                pos2 = widget.width() * 0.6, widget.height() * 0.6

                for image in (False, True):
                    with self.subTest(image=image):
                        if image:
                            self.plot.addImage(
                                numpy.arange(100 * 100).reshape(100, -1))

                        # Trigger tool button for diagonal profile mode
                        toolButton = getQToolButtonFromAction(
                            self.toolBar.lineAction)
                        self.assertIsNot(toolButton, None)
                        self.mouseMove(toolButton)
                        self.mouseClick(toolButton, qt.Qt.LeftButton)

                        # draw profile line
                        self.mouseMove(widget, pos=pos1)
                        self.mousePress(widget, qt.Qt.LeftButton, pos=pos1)
                        self.mouseMove(widget, pos=pos2)
                        self.mouseRelease(widget, qt.Qt.LeftButton, pos=pos2)

                        manager = self.toolBar.getProfileManager()
                        roi = manager.getSelectedRoi()
                        roi.setProfileLineWidth(3)
                        roi.setProfileMethod(method)

                        for _ in range(20):
                            self.qWait(200)
                            if not manager.hasPendingOperations():
                                break

                        if image is True:
                            profileCurve = self.toolBar.getProfilePlot().getAllCurves()[0]
                            if method == 'sum':
                                self.assertTrue(profileCurve.getData()[1].max() > 10000)
                            elif method == 'mean':
                                self.assertTrue(profileCurve.getData()[1].max() < 10000)


class TestProfile3DToolBar(TestCaseQt):
    """Tests for Profile3DToolBar widget.
    """
    def setUp(self):
        super(TestProfile3DToolBar, self).setUp()
        self.plot = StackView()
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

        self.plot.setStack(numpy.array([
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
            [[12, 13, 14], [15, 16, 17]]
        ]))

    def tearDown(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        self.plot = None

        super(TestProfile3DToolBar, self).tearDown()

    def testMethodProfile2D(self):
        """Test that the profile can have a different method if we want to
        compute then in 1D or in 2D"""

        toolBar = self.plot.getProfileToolbar()

        toolButton = getQToolButtonFromAction(toolBar.vLineAction)
        self.assertIsNot(toolButton, None)
        self.mouseMove(toolButton)
        self.mouseClick(toolButton, qt.Qt.LeftButton)
        plot2D = self.plot.getPlotWidget().getWidgetHandle()
        pos1 = plot2D.width() * 0.5, plot2D.height() * 0.5
        self.mouseClick(plot2D, qt.Qt.LeftButton, pos=pos1)

        manager = toolBar.getProfileManager()
        roi = manager.getSelectedRoi()
        roi.setProfileMethod("mean")
        roi.setProfileType("2D")
        roi.setProfileLineWidth(3)

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        # check 2D 'mean' profile
        profilePlot = toolBar.getProfilePlot()
        data = profilePlot.getActiveImage().getData()
        expected = numpy.array([[1, 4], [7, 10], [13, 16]])
        numpy.testing.assert_almost_equal(data, expected)

    def testMethodSumLine(self):
        """Simple interaction test to make sure the sum is correctly computed
        """
        toolBar = self.plot.getProfileToolbar()

        toolButton = getQToolButtonFromAction(toolBar.lineAction)
        self.assertIsNot(toolButton, None)
        self.mouseMove(toolButton)
        self.mouseClick(toolButton, qt.Qt.LeftButton)
        plot2D = self.plot.getPlotWidget().getWidgetHandle()
        pos1 = plot2D.width() * 0.5, plot2D.height() * 0.2
        pos2 = plot2D.width() * 0.5, plot2D.height() * 0.8

        self.mouseMove(plot2D, pos=pos1)
        self.mousePress(plot2D, qt.Qt.LeftButton, pos=pos1)
        self.mouseMove(plot2D, pos=pos2)
        self.mouseRelease(plot2D, qt.Qt.LeftButton, pos=pos2)

        manager = toolBar.getProfileManager()
        roi = manager.getSelectedRoi()
        roi.setProfileMethod("sum")
        roi.setProfileType("2D")
        roi.setProfileLineWidth(3)

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        # check 2D 'sum' profile
        profilePlot = toolBar.getProfilePlot()
        data = profilePlot.getActiveImage().getData()
        expected = numpy.array([[3, 12], [21, 30], [39, 48]])
        numpy.testing.assert_almost_equal(data, expected)


class TestGetProfilePlot(TestCaseQt):

    def testProfile1D(self):
        plot = Plot2D()
        plot.show()
        self.qWaitForWindowExposed(plot)
        plot.addImage([[0, 1], [2, 3]])

        toolBar = plot.getProfileToolbar()
        self.assertIsNone(toolBar.getProfileMainWindow())


        manager = toolBar.getProfileManager()
        roiManager = manager.getRoiManager()

        roi = rois.ProfileImageHorizontalLineROI()
        roi.setPosition(0.5)
        roiManager.addRoi(roi)
        roiManager.setSelectedRoi(roi)

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        self.assertIsInstance(toolBar.getProfileMainWindow(), qt.QMainWindow)
        self.assertIsInstance(plot.getProfilePlot(), Plot1D)
        plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        plot.close()
        del plot

    def testProfile2D(self):
        """Test that the profile plot associated to a stack view is either a
        Plot1D or a plot 2D instance."""
        plot = StackView()
        plot.show()
        self.qWaitForWindowExposed(plot)

        plot.setStack(numpy.array([[[0, 1], [2, 3]],
                                   [[4, 5], [6, 7]]]))

        toolBar = plot.getProfileToolbar()
        self.assertIsNone(toolBar.getProfileMainWindow())

        manager = toolBar.getProfileManager()
        roiManager = manager.getRoiManager()

        roi = rois.ProfileImageStackHorizontalLineROI()
        roi.setPosition(0.5)
        roi.setProfileType("2D")
        roiManager.addRoi(roi)
        roiManager.setSelectedRoi(roi)

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        self.assertIsInstance(toolBar.getProfileMainWindow(), qt.QMainWindow)
        self.assertIsInstance(toolBar.getProfilePlot(), Plot2D)

        roi.setProfileType("1D")

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        self.assertIsInstance(toolBar.getProfilePlot(), Plot1D)

        plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        plot.close()
        del plot


def suite():
    test_suite = unittest.TestSuite()
    for testClass in (TestProfileToolBar, TestGetProfilePlot,
                      TestProfile3DToolBar):
        test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
            testClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
