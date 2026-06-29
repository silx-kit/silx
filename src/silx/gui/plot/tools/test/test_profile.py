import numpy
import logging
import pytest

from silx.gui import qt
from silx.gui.utils.testutils import qWaitForWindowExposedAndActivate, QTest

from silx.gui.plot import PlotWindow, Plot1D, Plot2D, Profile
from silx.gui.plot.StackView import StackView
from silx.gui.plot.tools.profile import rois
from silx.gui.plot.tools.profile import editors
from silx.gui.plot.items import roi as roi_items
from silx.gui.plot.tools.profile import manager
from silx.gui import plot as silx_plot
from silx.gui.plot.tools.profile.manager import ProfileWindow

_logger = logging.getLogger(__name__)


@pytest.mark.skipif(qt.BINDING != "PyQt5", reason="Test only valid when using PyQt5")
def test_roi_init():
    roi = rois.ProfileImageVerticalLineROI()
    # the profile ROI + the shape
    assert roi.receivers(roi.sigRegionChanged) == 2


class TestInteractions:
    @pytest.fixture
    def defaultPlot(self, qWidgetFactory):
        widget = qWidgetFactory(silx_plot.PlotWidget)
        widget.show()
        yield widget

    @pytest.fixture
    def imagePlot(self, qWidgetFactory):
        widget = qWidgetFactory(silx_plot.Plot2D)
        image = numpy.arange(10 * 10).reshape(10, -1)
        widget.addImage(image)
        widget.show()
        yield widget

    @pytest.fixture
    def scatterPlot(self, qWidgetFactory):
        widget = qWidgetFactory(silx_plot.ScatterView)

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
        yield widget.getPlotWidget()

    @pytest.fixture
    def stackPlot(self, qWidgetFactory, qapp_utils):
        widget = qWidgetFactory(silx_plot.StackView)
        image = numpy.arange(10 * 10).reshape(10, -1)
        cube = numpy.array([image, image, image])
        widget.setStack(cube)
        widget.resetZoom()
        widget.show()
        yield widget.getPlotWidget()

    def waitPendingOperations(self, profile, qapp_utils):
        for _ in range(10):
            if not profile.hasPendingOperations():
                return
            qapp_utils.qWait(100)
        _logger.error("The profile manager still have pending operations")

    def genericRoiTest(self, plot, roiClass, qapp_utils):
        profileManager = manager.ProfileManager(plot, plot)
        profileManager.setItemType(image=True, scatter=True)

        try:
            action = profileManager.createProfileAction(roiClass, plot)
            action.triggered[bool].emit(True)
            widget = plot.getWidgetHandle()

            # Do the mouse interaction
            pos1 = widget.width() * 0.4, widget.height() * 0.4
            qapp_utils.mouseMove(widget, pos=pos1)
            qapp_utils.mouseClick(widget, qt.Qt.LeftButton, pos=pos1)

            if issubclass(roiClass, roi_items.LineROI):
                pos2 = widget.width() * 0.6, widget.height() * 0.6
                qapp_utils.mouseMove(widget, pos=pos2)
                qapp_utils.mouseClick(widget, qt.Qt.LeftButton, pos=pos2)

            self.waitPendingOperations(profileManager, qapp_utils)

            # Test that something was computed
            if issubclass(roiClass, rois._ProfileCrossROI):
                assert profileManager._computedProfiles == 2
            elif issubclass(roiClass, roi_items.LineROI):
                assert profileManager._computedProfiles >= 1
            else:
                assert profileManager._computedProfiles == 1

            # Test the created ROIs
            profileRois = profileManager.getRoiManager().getRois()
            if issubclass(roiClass, rois._ProfileCrossROI):
                assert len(profileRois) == 3
            else:
                assert len(profileRois) == 1
            # The first one should be the expected one
            roi = profileRois[0]

            # Test that something was displayed
            if issubclass(roiClass, rois._ProfileCrossROI):
                profiles = roi._getLines()
                window = profiles[0].getProfileWindow()
                assert window is not None
                window = profiles[1].getProfileWindow()
                assert window is not None
            else:
                window = roi.getProfileWindow()
                assert window is not None
        finally:
            profileManager.clearProfile()

    def testImageActions(self, imagePlot, subtests, qapp_utils):
        roiClasses = [
            rois.ProfileImageHorizontalLineROI,
            rois.ProfileImageVerticalLineROI,
            rois.ProfileImageLineROI,
            rois.ProfileImageCrossROI,
        ]
        for roiClass in roiClasses:
            with subtests.test(roiClass=roiClass):
                self.genericRoiTest(imagePlot, roiClass, qapp_utils)

    def testScatterActions(self, scatterPlot, subtests, qapp_utils):
        roiClasses = [
            rois.ProfileScatterHorizontalLineROI,
            rois.ProfileScatterVerticalLineROI,
            rois.ProfileScatterLineROI,
            rois.ProfileScatterCrossROI,
            rois.ProfileScatterHorizontalSliceROI,
            rois.ProfileScatterVerticalSliceROI,
            rois.ProfileScatterCrossSliceROI,
        ]
        for roiClass in roiClasses:
            with subtests.test(roiClass=roiClass):
                self.genericRoiTest(scatterPlot, roiClass, qapp_utils)

    def testStackActions(self, stackPlot, subtests, qapp_utils):
        roiClasses = [
            rois.ProfileImageStackHorizontalLineROI,
            rois.ProfileImageStackVerticalLineROI,
            rois.ProfileImageStackLineROI,
            rois.ProfileImageStackCrossROI,
        ]
        for roiClass in roiClasses:
            with subtests.test(roiClass=roiClass):
                self.genericRoiTest(stackPlot, roiClass, qapp_utils)

    def genericEditorTest(self, roi, editor):
        if isinstance(editor, editors._NoProfileRoiEditor):
            pass
        elif isinstance(editor, editors._DefaultImageStackProfileRoiEditor):
            # GUI to ROI
            editor._lineWidth.setValue(2)
            assert roi.getProfileLineWidth() == 2
            editor._methodsButton.setMethod("sum")
            assert roi.getProfileMethod() == "sum"
            editor._profileDim.setDimension(1)
            assert roi.getProfileType() == "1D"
            # ROI to GUI
            roi.setProfileLineWidth(3)
            assert editor._lineWidth.value() == 3
            roi.setProfileMethod("mean")
            assert editor._methodsButton.getMethod() == "mean"
            roi.setProfileType("2D")
            assert editor._profileDim.getDimension() == 2
        elif isinstance(editor, editors._DefaultImageProfileRoiEditor):
            # GUI to ROI
            editor._lineWidth.setValue(2)
            assert roi.getProfileLineWidth() == 2
            editor._methodsButton.setMethod("sum")
            assert roi.getProfileMethod() == "sum"
            # ROI to GUI
            roi.setProfileLineWidth(3)
            assert editor._lineWidth.value() == 3
            roi.setProfileMethod("mean")
            assert editor._methodsButton.getMethod() == "mean"
        elif isinstance(editor, editors._DefaultScatterProfileRoiEditor):
            # GUI to ROI
            editor._nPoints.setValue(100)
            assert roi.getNPoints() == 100
            # ROI to GUI
            roi.setNPoints(200)
            assert editor._nPoints.value() == 200
        else:
            assert False

    def testEditors(self, defaultPlot, subtests, qapp):
        roiClasses = [
            (rois.ProfileImageHorizontalLineROI, editors._DefaultImageProfileRoiEditor),
            (rois.ProfileImageVerticalLineROI, editors._DefaultImageProfileRoiEditor),
            (rois.ProfileImageLineROI, editors._DefaultImageProfileRoiEditor),
            (rois.ProfileImageCrossROI, editors._DefaultImageProfileRoiEditor),
            (
                rois.ProfileScatterHorizontalLineROI,
                editors._DefaultScatterProfileRoiEditor,
            ),
            (
                rois.ProfileScatterVerticalLineROI,
                editors._DefaultScatterProfileRoiEditor,
            ),
            (rois.ProfileScatterLineROI, editors._DefaultScatterProfileRoiEditor),
            (rois.ProfileScatterCrossROI, editors._DefaultScatterProfileRoiEditor),
            (rois.ProfileScatterHorizontalSliceROI, editors._NoProfileRoiEditor),
            (rois.ProfileScatterVerticalSliceROI, editors._NoProfileRoiEditor),
            (rois.ProfileScatterCrossSliceROI, editors._NoProfileRoiEditor),
            (
                rois.ProfileImageStackHorizontalLineROI,
                editors._DefaultImageStackProfileRoiEditor,
            ),
            (
                rois.ProfileImageStackVerticalLineROI,
                editors._DefaultImageStackProfileRoiEditor,
            ),
            (rois.ProfileImageStackLineROI, editors._DefaultImageStackProfileRoiEditor),
            (
                rois.ProfileImageStackCrossROI,
                editors._DefaultImageStackProfileRoiEditor,
            ),
        ]
        profileManager = manager.ProfileManager(defaultPlot, defaultPlot)
        editorAction = profileManager.createEditorAction(parent=defaultPlot)
        for roiClass, editorClass in roiClasses:
            with subtests.test(roiClass=roiClass):
                roi = roiClass()
                roi._setProfileManager(profileManager)
                try:
                    # Force widget creation
                    menu = qt.QMenu(defaultPlot)
                    menu.addAction(editorAction)
                    widgets = editorAction.createdWidgets()
                    assert len(widgets) > 0

                    editorAction.setProfileRoi(roi)
                    editorWidget = editorAction._getEditor(widgets[0])
                    assert isinstance(editorWidget, editorClass)
                    self.genericEditorTest(roi, editorWidget)
                finally:
                    editorAction.setProfileRoi(None)
                    menu.deleteLater()
                    menu = None
                    qapp.processEvents()


class TestProfileToolBar:
    """Tests for ProfileToolBar widget."""

    @pytest.fixture
    def plotAndToolBar(self, qWidgetFactory, qapp, qapp_utils):
        plot: PlotWindow = qWidgetFactory(PlotWindow)
        toolBar = Profile.ProfileToolBar(plot=plot)
        plot.addToolBar(toolBar)

        plot.show()

        qapp_utils.mouseMove(plot)  # Move to center
        qapp.processEvents()

        yield plot, toolBar

    def testAlignedProfile(self, plotAndToolBar, subtests, qapp_utils):
        """Test horizontal and vertical profile, without and with image"""
        plot, toolBar = plotAndToolBar
        # Use Plot backend widget to submit mouse events
        widget = plot.getWidgetHandle()
        for method in ("sum", "mean"):
            with subtests.test(method=method):
                # 2 positions to use for mouse events
                pos1 = widget.width() * 0.4, widget.height() * 0.4
                pos2 = widget.width() * 0.6, widget.height() * 0.6

                for action in (toolBar.hLineAction, toolBar.vLineAction):
                    with subtests.test(mode=action.text()):
                        # Trigger tool button for mode
                        action.trigger()
                        # Without image
                        qapp_utils.mouseMove(widget, pos=pos1)
                        qapp_utils.mouseClick(widget, qt.Qt.LeftButton, pos=pos1)

                        # with image
                        plot.addImage(numpy.arange(100 * 100).reshape(100, -1))
                        qapp_utils.mousePress(widget, qt.Qt.LeftButton, pos=pos1)
                        qapp_utils.mouseMove(widget, pos=pos2)
                        qapp_utils.mouseRelease(widget, qt.Qt.LeftButton, pos=pos2)

                        qapp_utils.mouseMove(widget)
                        qapp_utils.mouseClick(widget, qt.Qt.LeftButton)

                        manager = toolBar.getProfileManager()
                        for _ in range(20):
                            qapp_utils.qWait(200)
                            if not manager.hasPendingOperations():
                                break

    def testDiagonalProfile(
        self, plotAndToolBar, subtests: pytest.Subtests, qapp_utils
    ):
        """Test diagonal profile, without and with image"""
        # Use Plot backend widget to submit mouse events
        plot, toolBar = plotAndToolBar
        widget = plot.getWidgetHandle()

        plot.addImage(numpy.arange(100 * 100).reshape(100, -1))

        for method in ("sum", "mean"):
            with subtests.test(method=method):
                # 2 positions to use for mouse events
                pos1 = widget.width() * 0.4, widget.height() * 0.4
                pos2 = widget.width() * 0.6, widget.height() * 0.6

                # Trigger tool button for diagonal profile mode
                toolBar.lineAction.trigger()

                # draw profile line
                widget.setFocus(qt.Qt.OtherFocusReason)
                qapp_utils.mouseMove(widget, pos=pos1)
                qapp_utils.qWait(100)
                qapp_utils.mousePress(widget, qt.Qt.LeftButton, pos=pos1)
                qapp_utils.qWait(100)
                qapp_utils.mouseMove(widget, pos=pos2)
                qapp_utils.qWait(100)
                qapp_utils.mouseRelease(widget, qt.Qt.LeftButton, pos=pos2)
                qapp_utils.qWait(100)

                manager = toolBar.getProfileManager()

                for _ in range(20):
                    qapp_utils.qWait(200)
                    if not manager.hasPendingOperations():
                        break

                roi = manager.getCurrentRoi()
                assert roi is not None
                roi.setProfileLineWidth(3)
                roi.setProfileMethod(method)

                for _ in range(20):
                    qapp_utils.qWait(200)
                    if not manager.hasPendingOperations():
                        break

                curveItem = (
                    roi.getProfileWindow().getCurrentPlotWidget().getAllCurves()[0]
                )
                if method == "sum":
                    assert curveItem.getData()[1].max() > 10000
                elif method == "mean":
                    assert curveItem.getData()[1].max() < 10000

                # Remove the ROI so the profile window is also removed
                roiManager = manager.getRoiManager()
                roiManager.removeRoi(roi)
                qapp_utils.qWait(100)


class TestProfile3DToolBar:
    """Tests for Profile3DToolBar widget."""

    @pytest.fixture
    def plot(self, qWidgetFactory):
        plot = qWidgetFactory(StackView)
        plot.show()

        plot.setStack(
            numpy.array(
                [
                    [[0, 1, 2], [3, 4, 5]],
                    [[6, 7, 8], [9, 10, 11]],
                    [[12, 13, 14], [15, 16, 17]],
                ]
            )
        )
        yield plot

    def testMethodProfile2D(self, plot: StackView, qapp_utils):
        """Test that the profile can have a different method if we want to
        compute then in 1D or in 2D"""

        toolBar = plot.getProfileToolbar()

        toolBar.vLineAction.trigger()
        plot2D = plot.getPlotWidget().getWidgetHandle()
        pos1 = plot2D.width() * 0.5, plot2D.height() * 0.5
        qapp_utils.mouseClick(plot2D, qt.Qt.LeftButton, pos=pos1)

        manager = toolBar.getProfileManager()
        roi = manager.getCurrentRoi()
        roi.setProfileMethod("mean")
        roi.setProfileType("2D")
        roi.setProfileLineWidth(3)

        for _ in range(20):
            qapp_utils.qWait(200)
            if not manager.hasPendingOperations():
                break

        # check 2D 'mean' profile
        profilePlot = roi.getProfileWindow().getCurrentPlotWidget()
        data = profilePlot.getAllImages()[0].getData()
        expected = numpy.array([[1, 4], [7, 10], [13, 16]])
        numpy.testing.assert_almost_equal(data, expected)

    def testMethodSumLine(self, plot: StackView, qapp_utils):
        """Simple interaction test to make sure the sum is correctly computed"""
        toolBar = plot.getProfileToolbar()

        toolBar.lineAction.trigger()
        plot2D = plot.getPlotWidget().getWidgetHandle()
        pos1 = plot2D.width() * 0.5, plot2D.height() * 0.2
        pos2 = plot2D.width() * 0.5, plot2D.height() * 0.8

        qapp_utils.mouseMove(plot2D, pos=pos1)
        qapp_utils.mousePress(plot2D, qt.Qt.LeftButton, pos=pos1)
        qapp_utils.mouseMove(plot2D, pos=pos2)
        qapp_utils.mouseRelease(plot2D, qt.Qt.LeftButton, pos=pos2)

        manager = toolBar.getProfileManager()
        roi = manager.getCurrentRoi()
        roi.setProfileMethod("sum")
        roi.setProfileType("2D")
        roi.setProfileLineWidth(3)

        for _ in range(20):
            qapp_utils.qWait(200)
            if not manager.hasPendingOperations():
                break

        # check 2D 'sum' profile
        profilePlot = roi.getProfileWindow().getCurrentPlotWidget()
        data = profilePlot.getAllImages()[0].getData()
        expected = numpy.array([[3, 12], [21, 30], [39, 48]])
        numpy.testing.assert_almost_equal(data, expected)


@pytest.mark.parametrize("with_mask", (True, False))
def testProfile1D(qWidgetFactory, with_mask):
    """Test that the profile plot associated to a Plot2D is a 1D plot and that mask is take into account.

    Note: the mask; when applied; is at the center. As we have an od number of elements the expected result remains the same.
    """
    plot = qWidgetFactory(Plot2D)
    plot.show()
    qWaitForWindowExposedAndActivate(plot)

    plot.addImage([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    if with_mask:
        mask = numpy.zeros((5, 2))
        mask[2, :] = 1
        plot.setSelectionMask(mask)

    toolBar = plot.getProfileToolbar()

    manager = toolBar.getProfileManager()
    roiManager = manager.getRoiManager()

    roi = rois.ProfileImageHorizontalLineROI()
    roi.setPosition(2.5)
    roi.setProfileLineWidth(3)
    roiManager.addRoi(roi)
    roiManager.setCurrentRoi(roi)

    for _ in range(20):
        QTest.qWait(200)
        if not manager.hasPendingOperations():
            break

    profileWindow = roi.getProfileWindow()
    assert isinstance(roi.getProfileWindow(), ProfileWindow)
    plotWidget = profileWindow.getCurrentPlotWidget()
    assert isinstance(plotWidget, Plot1D)

    # check result
    curves = plotWidget.getAllCurves()
    assert len(curves) == 1
    profile = curves[0]
    numpy.testing.assert_almost_equal(profile.getYData(), numpy.array([4.0, 5.0]))


@pytest.mark.parametrize("with_mask", (True, False))
def testProfile2D(qWidgetFactory, with_mask):
    """Test that the profile plot associated to a stack view is either a
    Plot1D or a plot 2D instance.
    Make sure also that the mask is take into account.

    Note: the mask; when applied; is at the center. As we have an od number of elements the expected result remains the same.
    """
    plot = qWidgetFactory(StackView)
    plot.show()
    qWaitForWindowExposedAndActivate(plot)

    plot.setStack(
        stack=(
            numpy.arange(0, 10).reshape(5, 2),
            numpy.arange(10, 20).reshape(5, 2),
        )
    )
    if with_mask:
        mask = numpy.zeros((5, 2))
        mask[2, :] = 1
        plot.getPlotWidget().setSelectionMask(mask)

    toolBar = plot.getProfileToolbar()

    manager = toolBar.getProfileManager()
    roiManager = manager.getRoiManager()

    roi = rois.ProfileImageStackHorizontalLineROI()
    roi.setPosition(2.5)
    roi.setProfileLineWidth(3)
    roi.setProfileType("2D")
    roiManager.addRoi(roi)
    roiManager.setCurrentRoi(roi)

    for _ in range(20):
        QTest.qWait(200)
        if not manager.hasPendingOperations():
            break

    profileWindow = roi.getProfileWindow()
    assert isinstance(roi.getProfileWindow(), ProfileWindow)
    plotWidget = profileWindow.getCurrentPlotWidget()
    assert isinstance(plotWidget, Plot2D)
    images = plotWidget.getAllImages()
    assert len(images) == 1
    profile = images[0]
    numpy.testing.assert_almost_equal(
        profile.getData(), numpy.array([[4.0, 5.0], [14.0, 15.0]])
    )

    roi.setProfileType("1D")

    for _ in range(20):
        QTest.qWait(200)
        if not manager.hasPendingOperations():
            break

    profileWindow = roi.getProfileWindow()
    assert isinstance(roi.getProfileWindow(), ProfileWindow)

    plotWidget = profileWindow.getCurrentPlotWidget()
    assert isinstance(plotWidget, Plot1D)

    # check result
    curves = plotWidget.getAllCurves()
    assert len(curves) == 1
    profile = curves[0]
    numpy.testing.assert_almost_equal(profile.getYData(), numpy.array([4.0, 5.0]))
