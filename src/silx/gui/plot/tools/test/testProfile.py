# /*##########################################################################
#
# Copyright (c) 2018-2021 European Synchrotron Radiation Facility
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
import contextlib
import numpy
import logging

from silx.gui import qt
from silx.utils import deprecation
from silx.utils import testutils

from silx.gui.utils.testutils import TestCaseQt
from silx.utils.testutils import ParametricTestCase
from silx.gui.plot import PlotWindow, Plot1D, Plot2D, Profile
from silx.gui.plot.StackView import StackView
from silx.gui.plot.tools.profile import rois
from silx.gui.plot.tools.profile import editors
from silx.gui.plot.items import roi as roi_items
from silx.gui.plot.tools.profile import manager
from silx.gui import plot as silx_plot

_logger = logging.getLogger(__name__)


class TestRois(TestCaseQt):

    def test_init(self):
        """Check that the constructor is not called twice"""
        roi = rois.ProfileImageVerticalLineROI()
        if qt.BINDING == "PyQt5":
            # the profile ROI + the shape
            self.assertEqual(roi.receivers(roi.sigRegionChanged), 2)


class TestInteractions(TestCaseQt):

    @contextlib.contextmanager
    def defaultPlot(self):
        try:
            widget = silx_plot.PlotWidget()
            widget.show()
            self.qWaitForWindowExposed(widget)
            yield widget
        finally:
            widget.close()
            widget = None
            self.qWait()

    @contextlib.contextmanager
    def imagePlot(self):
        try:
            widget = silx_plot.Plot2D()
            image = numpy.arange(10 * 10).reshape(10, -1)
            widget.addImage(image)
            widget.show()
            self.qWaitForWindowExposed(widget)
            yield widget
        finally:
            widget.close()
            widget = None
            self.qWait()

    @contextlib.contextmanager
    def scatterPlot(self):
        try:
            widget = silx_plot.ScatterView()

            nbX, nbY = 7, 5
            yy = numpy.atleast_2d(numpy.ones(nbY)).T
            xx = numpy.atleast_2d(numpy.ones(nbX))
            positionX = numpy.linspace(10, 50, nbX) * yy
            positionX = positionX.reshape(nbX * nbY)
            positionY = numpy.atleast_2d(numpy.linspace(20, 60, nbY)).T * xx
            positionY = positionY.reshape(nbX * nbY)
            values = numpy.arange(nbX * nbY)

            widget.setData(positionX, positionY, values)
            widget.resetZoom()
            widget.show()
            self.qWaitForWindowExposed(widget)
            yield widget.getPlotWidget()
        finally:
            widget.close()
            widget = None
            self.qWait()

    @contextlib.contextmanager
    def stackPlot(self):
        try:
            widget = silx_plot.StackView()
            image = numpy.arange(10 * 10).reshape(10, -1)
            cube = numpy.array([image, image, image])
            widget.setStack(cube)
            widget.resetZoom()
            widget.show()
            self.qWaitForWindowExposed(widget)
            yield widget.getPlotWidget()
        finally:
            widget.close()
            widget = None
            self.qWait()

    def waitPendingOperations(self, proflie):
        for _ in range(10):
            if not proflie.hasPendingOperations():
                return
            self.qWait(100)
        _logger.error("The profile manager still have pending operations")

    def genericRoiTest(self, plot, roiClass):
        profileManager = manager.ProfileManager(plot, plot)
        profileManager.setItemType(image=True, scatter=True)

        try:
            action = profileManager.createProfileAction(roiClass, plot)
            action.triggered[bool].emit(True)
            widget = plot.getWidgetHandle()

            # Do the mouse interaction
            pos1 = widget.width() * 0.4, widget.height() * 0.4
            self.mouseMove(widget, pos=pos1)
            self.mouseClick(widget, qt.Qt.LeftButton, pos=pos1)

            if issubclass(roiClass, roi_items.LineROI):
                pos2 = widget.width() * 0.6, widget.height() * 0.6
                self.mouseMove(widget, pos=pos2)
                self.mouseClick(widget, qt.Qt.LeftButton, pos=pos2)

            self.waitPendingOperations(profileManager)

            # Test that something was computed
            if issubclass(roiClass, rois._ProfileCrossROI):
                self.assertEqual(profileManager._computedProfiles, 2)
            elif issubclass(roiClass, roi_items.LineROI):
                self.assertGreaterEqual(profileManager._computedProfiles, 1)
            else:
                self.assertEqual(profileManager._computedProfiles, 1)

            # Test the created ROIs
            profileRois = profileManager.getRoiManager().getRois()
            if issubclass(roiClass, rois._ProfileCrossROI):
                self.assertEqual(len(profileRois), 3)
            else:
                self.assertEqual(len(profileRois), 1)
            # The first one should be the expected one
            roi = profileRois[0]

            # Test that something was displayed
            if issubclass(roiClass, rois._ProfileCrossROI):
                profiles = roi._getLines()
                window = profiles[0].getProfileWindow()
                self.assertIsNotNone(window)
                window = profiles[1].getProfileWindow()
                self.assertIsNotNone(window)
            else:
                window = roi.getProfileWindow()
                self.assertIsNotNone(window)
        finally:
            profileManager.clearProfile()

    def testImageActions(self):
        roiClasses = [
            rois.ProfileImageHorizontalLineROI,
            rois.ProfileImageVerticalLineROI,
            rois.ProfileImageLineROI,
            rois.ProfileImageCrossROI,
        ]
        with self.imagePlot() as plot:
            for roiClass in roiClasses:
                with self.subTest(roiClass=roiClass):
                    self.genericRoiTest(plot, roiClass)

    def testScatterActions(self):
        roiClasses = [
            rois.ProfileScatterHorizontalLineROI,
            rois.ProfileScatterVerticalLineROI,
            rois.ProfileScatterLineROI,
            rois.ProfileScatterCrossROI,
            rois.ProfileScatterHorizontalSliceROI,
            rois.ProfileScatterVerticalSliceROI,
            rois.ProfileScatterCrossSliceROI,
        ]
        with self.scatterPlot() as plot:
            for roiClass in roiClasses:
                with self.subTest(roiClass=roiClass):
                    self.genericRoiTest(plot, roiClass)

    def testStackActions(self):
        roiClasses = [
            rois.ProfileImageStackHorizontalLineROI,
            rois.ProfileImageStackVerticalLineROI,
            rois.ProfileImageStackLineROI,
            rois.ProfileImageStackCrossROI,
        ]
        with self.stackPlot() as plot:
            for roiClass in roiClasses:
                with self.subTest(roiClass=roiClass):
                    self.genericRoiTest(plot, roiClass)

    def genericEditorTest(self, plot, roi, editor):
        if isinstance(editor, editors._NoProfileRoiEditor):
            pass
        elif isinstance(editor, editors._DefaultImageStackProfileRoiEditor):
            # GUI to ROI
            editor._lineWidth.setValue(2)
            self.assertEqual(roi.getProfileLineWidth(), 2)
            editor._methodsButton.setMethod("sum")
            self.assertEqual(roi.getProfileMethod(), "sum")
            editor._profileDim.setDimension(1)
            self.assertEqual(roi.getProfileType(), "1D")
            # ROI to GUI
            roi.setProfileLineWidth(3)
            self.assertEqual(editor._lineWidth.value(), 3)
            roi.setProfileMethod("mean")
            self.assertEqual(editor._methodsButton.getMethod(), "mean")
            roi.setProfileType("2D")
            self.assertEqual(editor._profileDim.getDimension(), 2)
        elif isinstance(editor, editors._DefaultImageProfileRoiEditor):
            # GUI to ROI
            editor._lineWidth.setValue(2)
            self.assertEqual(roi.getProfileLineWidth(), 2)
            editor._methodsButton.setMethod("sum")
            self.assertEqual(roi.getProfileMethod(), "sum")
            # ROI to GUI
            roi.setProfileLineWidth(3)
            self.assertEqual(editor._lineWidth.value(), 3)
            roi.setProfileMethod("mean")
            self.assertEqual(editor._methodsButton.getMethod(), "mean")
        elif isinstance(editor, editors._DefaultScatterProfileRoiEditor):
            # GUI to ROI
            editor._nPoints.setValue(100)
            self.assertEqual(roi.getNPoints(), 100)
            # ROI to GUI
            roi.setNPoints(200)
            self.assertEqual(editor._nPoints.value(), 200)
        else:
            assert False

    def testEditors(self):
        roiClasses = [
            (rois.ProfileImageHorizontalLineROI, editors._DefaultImageProfileRoiEditor),
            (rois.ProfileImageVerticalLineROI, editors._DefaultImageProfileRoiEditor),
            (rois.ProfileImageLineROI, editors._DefaultImageProfileRoiEditor),
            (rois.ProfileImageCrossROI, editors._DefaultImageProfileRoiEditor),
            (rois.ProfileScatterHorizontalLineROI, editors._DefaultScatterProfileRoiEditor),
            (rois.ProfileScatterVerticalLineROI, editors._DefaultScatterProfileRoiEditor),
            (rois.ProfileScatterLineROI, editors._DefaultScatterProfileRoiEditor),
            (rois.ProfileScatterCrossROI, editors._DefaultScatterProfileRoiEditor),
            (rois.ProfileScatterHorizontalSliceROI, editors._NoProfileRoiEditor),
            (rois.ProfileScatterVerticalSliceROI, editors._NoProfileRoiEditor),
            (rois.ProfileScatterCrossSliceROI, editors._NoProfileRoiEditor),
            (rois.ProfileImageStackHorizontalLineROI, editors._DefaultImageStackProfileRoiEditor),
            (rois.ProfileImageStackVerticalLineROI, editors._DefaultImageStackProfileRoiEditor),
            (rois.ProfileImageStackLineROI, editors._DefaultImageStackProfileRoiEditor),
            (rois.ProfileImageStackCrossROI, editors._DefaultImageStackProfileRoiEditor),
        ]
        with self.defaultPlot() as plot:
            profileManager = manager.ProfileManager(plot, plot)
            editorAction = profileManager.createEditorAction(parent=plot)
            for roiClass, editorClass in roiClasses:
                with self.subTest(roiClass=roiClass):
                    roi = roiClass()
                    roi._setProfileManager(profileManager)
                    try:
                        # Force widget creation
                        menu = qt.QMenu(plot)
                        menu.addAction(editorAction)
                        widgets = editorAction.createdWidgets()
                        self.assertGreater(len(widgets), 0)

                        editorAction.setProfileRoi(roi)
                        editorWidget = editorAction._getEditor(widgets[0])
                        self.assertIsInstance(editorWidget, editorClass)
                        self.genericEditorTest(plot, roi, editorWidget)
                    finally:
                        editorAction.setProfileRoi(None)
                        menu.deleteLater()
                        menu = None
                        self.qapp.processEvents()


class TestProfileToolBar(TestCaseQt, ParametricTestCase):
    """Tests for ProfileToolBar widget."""

    def setUp(self):
        super(TestProfileToolBar, self).setUp()
        self.plot = PlotWindow()
        self.toolBar = Profile.ProfileToolBar(plot=self.plot)
        self.plot.addToolBar(self.toolBar)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

        self.mouseMove(self.plot)  # Move to center
        self.qapp.processEvents()
        deprecation.FORCE = True

    def tearDown(self):
        deprecation.FORCE = False
        self.qapp.processEvents()
        profileManager = self.toolBar.getProfileManager()
        profileManager.clearProfile()
        profileManager = None
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
                        action.trigger()
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

    @testutils.validate_logging(deprecation.depreclog.name, warning=4)
    def testDiagonalProfile(self):
        """Test diagonal profile, without and with image"""
        # Use Plot backend widget to submit mouse events
        widget = self.plot.getWidgetHandle()

        self.plot.addImage(
            numpy.arange(100 * 100).reshape(100, -1))

        for method in ('sum', 'mean'):
            with self.subTest(method=method):
                # 2 positions to use for mouse events
                pos1 = widget.width() * 0.4, widget.height() * 0.4
                pos2 = widget.width() * 0.6, widget.height() * 0.6

                # Trigger tool button for diagonal profile mode
                self.toolBar.lineAction.trigger()

                # draw profile line
                widget.setFocus(qt.Qt.OtherFocusReason)
                self.mouseMove(widget, pos=pos1)
                self.qWait(100)
                self.mousePress(widget, qt.Qt.LeftButton, pos=pos1)
                self.qWait(100)
                self.mouseMove(widget, pos=pos2)
                self.qWait(100)
                self.mouseRelease(widget, qt.Qt.LeftButton, pos=pos2)
                self.qWait(100)

                manager = self.toolBar.getProfileManager()

                for _ in range(20):
                    self.qWait(200)
                    if not manager.hasPendingOperations():
                        break

                roi = manager.getCurrentRoi()
                self.assertIsNotNone(roi)
                roi.setProfileLineWidth(3)
                roi.setProfileMethod(method)

                for _ in range(20):
                    self.qWait(200)
                    if not manager.hasPendingOperations():
                        break

                curveItem = self.toolBar.getProfilePlot().getAllCurves()[0]
                if method == 'sum':
                    self.assertTrue(curveItem.getData()[1].max() > 10000)
                elif method == 'mean':
                    self.assertTrue(curveItem.getData()[1].max() < 10000)

                # Remove the ROI so the profile window is also removed
                roiManager = manager.getRoiManager()
                roiManager.removeRoi(roi)
                self.qWait(100)


class TestDeprecatedProfileToolBar(TestCaseQt):
    """Tests old features of the ProfileToolBar widget."""

    def setUp(self):
        self.plot = None
        super(TestDeprecatedProfileToolBar, self).setUp()

    def tearDown(self):
        if self.plot is not None:
            self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
            self.plot.close()
            self.plot = None
            self.qWait()

        super(TestDeprecatedProfileToolBar, self).tearDown()

    @testutils.validate_logging(deprecation.depreclog.name, warning=2)
    def testCustomProfileWindow(self):
        from silx.gui.plot import ProfileMainWindow

        self.plot = PlotWindow()
        profileWindow = ProfileMainWindow.ProfileMainWindow(self.plot)
        toolBar = Profile.ProfileToolBar(parent=self.plot,
                                         plot=self.plot,
                                         profileWindow=profileWindow)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)
        profileWindow.show()
        self.qWaitForWindowExposed(profileWindow)
        self.qapp.processEvents()

        self.plot.addImage(numpy.arange(10 * 10).reshape(10, -1))
        profile = rois.ProfileImageHorizontalLineROI()
        profile.setPosition(5)
        toolBar.getProfileManager().getRoiManager().addRoi(profile)
        toolBar.getProfileManager().getRoiManager().setCurrentRoi(profile)

        for _ in range(20):
            self.qWait(200)
            if not toolBar.getProfileManager().hasPendingOperations():
                break

        # There is a displayed profile
        self.assertIsNotNone(profileWindow.getProfile())
        self.assertIs(toolBar.getProfileMainWindow(), profileWindow)

        # There is nothing anymore but the window is still there
        toolBar.getProfileManager().clearProfile()
        self.qapp.processEvents()
        self.assertIsNone(profileWindow.getProfile())


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
        deprecation.FORCE = True

    def tearDown(self):
        deprecation.FORCE = False
        profileManager = self.plot.getProfileToolbar().getProfileManager()
        profileManager.clearProfile()
        profileManager = None
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        self.plot = None

        super(TestProfile3DToolBar, self).tearDown()

    @testutils.validate_logging(deprecation.depreclog.name, warning=2)
    def testMethodProfile2D(self):
        """Test that the profile can have a different method if we want to
        compute then in 1D or in 2D"""

        toolBar = self.plot.getProfileToolbar()

        toolBar.vLineAction.trigger()
        plot2D = self.plot.getPlotWidget().getWidgetHandle()
        pos1 = plot2D.width() * 0.5, plot2D.height() * 0.5
        self.mouseClick(plot2D, qt.Qt.LeftButton, pos=pos1)

        manager = toolBar.getProfileManager()
        roi = manager.getCurrentRoi()
        roi.setProfileMethod("mean")
        roi.setProfileType("2D")
        roi.setProfileLineWidth(3)

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        # check 2D 'mean' profile
        profilePlot = toolBar.getProfilePlot()
        data = profilePlot.getAllImages()[0].getData()
        expected = numpy.array([[1, 4], [7, 10], [13, 16]])
        numpy.testing.assert_almost_equal(data, expected)

    @testutils.validate_logging(deprecation.depreclog.name, warning=2)
    def testMethodSumLine(self):
        """Simple interaction test to make sure the sum is correctly computed
        """
        toolBar = self.plot.getProfileToolbar()

        toolBar.lineAction.trigger()
        plot2D = self.plot.getPlotWidget().getWidgetHandle()
        pos1 = plot2D.width() * 0.5, plot2D.height() * 0.2
        pos2 = plot2D.width() * 0.5, plot2D.height() * 0.8

        self.mouseMove(plot2D, pos=pos1)
        self.mousePress(plot2D, qt.Qt.LeftButton, pos=pos1)
        self.mouseMove(plot2D, pos=pos2)
        self.mouseRelease(plot2D, qt.Qt.LeftButton, pos=pos2)

        manager = toolBar.getProfileManager()
        roi = manager.getCurrentRoi()
        roi.setProfileMethod("sum")
        roi.setProfileType("2D")
        roi.setProfileLineWidth(3)

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        # check 2D 'sum' profile
        profilePlot = toolBar.getProfilePlot()
        data = profilePlot.getAllImages()[0].getData()
        expected = numpy.array([[3, 12], [21, 30], [39, 48]])
        numpy.testing.assert_almost_equal(data, expected)


class TestGetProfilePlot(TestCaseQt):

    def setUp(self):
        self.plot = None
        super(TestGetProfilePlot, self).setUp()

    def tearDown(self):
        if self.plot is not None:
            manager = self.plot.getProfileToolbar().getProfileManager()
            manager.clearProfile()
            manager = None
            self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
            self.plot.close()
            self.plot = None

        super(TestGetProfilePlot, self).tearDown()

    def testProfile1D(self):
        self.plot = Plot2D()
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)
        self.plot.addImage([[0, 1], [2, 3]])

        toolBar = self.plot.getProfileToolbar()

        manager = toolBar.getProfileManager()
        roiManager = manager.getRoiManager()

        roi = rois.ProfileImageHorizontalLineROI()
        roi.setPosition(0.5)
        roiManager.addRoi(roi)
        roiManager.setCurrentRoi(roi)

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        profileWindow = roi.getProfileWindow()
        self.assertIsInstance(roi.getProfileWindow(), qt.QMainWindow)
        self.assertIsInstance(profileWindow.getCurrentPlotWidget(), Plot1D)

    def testProfile2D(self):
        """Test that the profile plot associated to a stack view is either a
        Plot1D or a plot 2D instance."""
        self.plot = StackView()
        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

        self.plot.setStack(numpy.array([[[0, 1], [2, 3]],
                                       [[4, 5], [6, 7]]]))

        toolBar = self.plot.getProfileToolbar()

        manager = toolBar.getProfileManager()
        roiManager = manager.getRoiManager()

        roi = rois.ProfileImageStackHorizontalLineROI()
        roi.setPosition(0.5)
        roi.setProfileType("2D")
        roiManager.addRoi(roi)
        roiManager.setCurrentRoi(roi)

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        profileWindow = roi.getProfileWindow()
        self.assertIsInstance(roi.getProfileWindow(), qt.QMainWindow)
        self.assertIsInstance(profileWindow.getCurrentPlotWidget(), Plot2D)

        roi.setProfileType("1D")

        for _ in range(20):
            self.qWait(200)
            if not manager.hasPendingOperations():
                break

        profileWindow = roi.getProfileWindow()
        self.assertIsInstance(roi.getProfileWindow(), qt.QMainWindow)
        self.assertIsInstance(profileWindow.getCurrentPlotWidget(), Plot1D)
