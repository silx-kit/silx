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
import contextlib
import numpy
import logging

from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt
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
        if qt.BINDING not in ["PySide", "PySide2"]:
            self.assertEqual(roi.receivers(roi.sigRegionChanged), 1)

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
            editor = profileManager.createEditorAction(parent=plot)
            for roiClass, editorClass in roiClasses:
                with self.subTest(roiClass=roiClass):
                    roi = roiClass()
                    roi._setProfileManager(profileManager)
                    try:
                        editor.setProfileRoi(roi)
                        editorWidget = editor._getEditor()
                        self.assertIsInstance(editorWidget, editorClass)
                        self.genericEditorTest(plot, roi, editorWidget)
                    finally:
                        editor.setProfileRoi(None)

def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestRois))
    test_suite.addTest(loadTests(TestInteractions))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
