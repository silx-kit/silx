# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
import numpy.testing

from silx.gui import qt
from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt, SignalListener
from silx.gui.plot import PlotWindow
import silx.gui.plot.items.roi as roi_items
from silx.gui.plot.tools import roi


class TestRoiItems(TestCaseQt):

    def testLine_geometry(self):
        item = roi_items.LineROI()
        startPoint = numpy.array([1, 2])
        endPoint = numpy.array([3, 4])
        item.setEndPoints(startPoint, endPoint)
        numpy.testing.assert_allclose(item.getEndPoints()[0], startPoint)
        numpy.testing.assert_allclose(item.getEndPoints()[1], endPoint)

    def testHLine_geometry(self):
        item = roi_items.HorizontalLineROI()
        item.setPosition(15)
        self.assertEqual(item.getPosition(), 15)

    def testVLine_geometry(self):
        item = roi_items.VerticalLineROI()
        item.setPosition(15)
        self.assertEqual(item.getPosition(), 15)

    def testPoint_geometry(self):
        point = numpy.array([1, 2])
        item = roi_items.PointROI()
        item.setPosition(point)
        numpy.testing.assert_allclose(item.getPosition(), point)

    def testRectangle_originGeometry(self):
        origin = numpy.array([0, 0])
        size = numpy.array([10, 20])
        center = numpy.array([5, 10])
        item = roi_items.RectangleROI()
        item.setGeometry(origin=origin, size=size)
        numpy.testing.assert_allclose(item.getOrigin(), origin)
        numpy.testing.assert_allclose(item.getSize(), size)
        numpy.testing.assert_allclose(item.getCenter(), center)

    def testRectangle_centerGeometry(self):
        origin = numpy.array([0, 0])
        size = numpy.array([10, 20])
        center = numpy.array([5, 10])
        item = roi_items.RectangleROI()
        item.setGeometry(center=center, size=size)
        numpy.testing.assert_allclose(item.getOrigin(), origin)
        numpy.testing.assert_allclose(item.getSize(), size)
        numpy.testing.assert_allclose(item.getCenter(), center)

    def testRectangle_setCenterGeometry(self):
        origin = numpy.array([0, 0])
        size = numpy.array([10, 20])
        item = roi_items.RectangleROI()
        item.setGeometry(origin=origin, size=size)
        newCenter = numpy.array([0, 0])
        item.setCenter(newCenter)
        expectedOrigin = numpy.array([-5, -10])
        numpy.testing.assert_allclose(item.getOrigin(), expectedOrigin)
        numpy.testing.assert_allclose(item.getCenter(), newCenter)
        numpy.testing.assert_allclose(item.getSize(), size)

    def testRectangle_setOriginGeometry(self):
        origin = numpy.array([0, 0])
        size = numpy.array([10, 20])
        item = roi_items.RectangleROI()
        item.setGeometry(origin=origin, size=size)
        newOrigin = numpy.array([10, 10])
        item.setOrigin(newOrigin)
        expectedCenter = numpy.array([15, 20])
        numpy.testing.assert_allclose(item.getOrigin(), newOrigin)
        numpy.testing.assert_allclose(item.getCenter(), expectedCenter)
        numpy.testing.assert_allclose(item.getSize(), size)

    def testCircle_geometry(self):
        center = numpy.array([0, 0])
        radius = 10.
        item = roi_items.CircleROI()
        item.setGeometry(center=center, radius=radius)
        numpy.testing.assert_allclose(item.getCenter(), center)
        numpy.testing.assert_allclose(item.getRadius(), radius)

    def testCircle_setCenter(self):
        center = numpy.array([0, 0])
        radius = 10.
        item = roi_items.CircleROI()
        item.setGeometry(center=center, radius=radius)
        newCenter = numpy.array([-10, 0])
        item.setCenter(newCenter)
        numpy.testing.assert_allclose(item.getCenter(), newCenter)
        numpy.testing.assert_allclose(item.getRadius(), radius)

    def testCircle_setRadius(self):
        center = numpy.array([0, 0])
        radius = 10.
        item = roi_items.CircleROI()
        item.setGeometry(center=center, radius=radius)
        newRadius = 5.1
        item.setRadius(newRadius)
        numpy.testing.assert_allclose(item.getCenter(), center)
        numpy.testing.assert_allclose(item.getRadius(), newRadius)

    def testRectangle_isIn(self):
        origin = numpy.array([0, 0])
        size = numpy.array([10, 20])
        item = roi_items.RectangleROI()
        item.setGeometry(origin=origin, size=size)
        self.assertTrue(item.contains(position=(0, 0)))
        self.assertTrue(item.contains(position=(2, 14)))
        self.assertFalse(item.contains(position=(14, 12)))

    def testPolygon_emptyGeometry(self):
        points = numpy.empty((0, 2))
        item = roi_items.PolygonROI()
        item.setPoints(points)
        numpy.testing.assert_allclose(item.getPoints(), points)

    def testPolygon_geometry(self):
        points = numpy.array([[10, 10], [12, 10], [50, 1]])
        item = roi_items.PolygonROI()
        item.setPoints(points)
        numpy.testing.assert_allclose(item.getPoints(), points)

    def testPolygon_isIn(self):
        points = numpy.array([[0, 0], [0, 10], [5, 10]])
        item = roi_items.PolygonROI()
        item.setPoints(points)
        self.assertTrue(item.contains((0, 0)))
        self.assertFalse(item.contains((6, 2)))
        self.assertFalse(item.contains((-2, 5)))
        self.assertFalse(item.contains((2, -1)))
        self.assertFalse(item.contains((8, 1)))
        self.assertTrue(item.contains((1, 8)))

    def testArc_getToSetGeometry(self):
        """Test that we can use getGeometry as input to setGeometry"""
        item = roi_items.ArcROI()
        item.setFirstShapePoints(numpy.array([[5, 10], [50, 100]]))
        item.setGeometry(*item.getGeometry())

    def testArc_degenerated_point(self):
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 0, 0, 0, 0
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)

    def testArc_degenerated_line(self):
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 0, 100, numpy.pi, numpy.pi
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)

    def testArc_special_circle(self):
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 0, 100, numpy.pi, 3 * numpy.pi
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
        numpy.testing.assert_allclose(item.getCenter(), center)
        self.assertAlmostEqual(item.getInnerRadius(), innerRadius)
        self.assertAlmostEqual(item.getOuterRadius(), outerRadius)
        self.assertAlmostEqual(item.getStartAngle(), item.getEndAngle() - numpy.pi * 2.0)
        self.assertTrue(item.isClosed())

    def testArc_special_donut(self):
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 1, 100, numpy.pi, 3 * numpy.pi
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
        numpy.testing.assert_allclose(item.getCenter(), center)
        self.assertAlmostEqual(item.getInnerRadius(), innerRadius)
        self.assertAlmostEqual(item.getOuterRadius(), outerRadius)
        self.assertAlmostEqual(item.getStartAngle(), item.getEndAngle() - numpy.pi * 2.0)
        self.assertTrue(item.isClosed())

    def testArc_clockwiseGeometry(self):
        """Test that we can use getGeometry as input to setGeometry"""
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 1, 100, numpy.pi * 0.5, numpy.pi
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
        numpy.testing.assert_allclose(item.getCenter(), center)
        self.assertAlmostEqual(item.getInnerRadius(), innerRadius)
        self.assertAlmostEqual(item.getOuterRadius(), outerRadius)
        self.assertAlmostEqual(item.getStartAngle(), startAngle)
        self.assertAlmostEqual(item.getEndAngle(), endAngle)
        self.assertAlmostEqual(item.isClosed(), False)

    def testArc_anticlockwiseGeometry(self):
        """Test that we can use getGeometry as input to setGeometry"""
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 1, 100, numpy.pi * 0.5, -numpy.pi * 0.5
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
        numpy.testing.assert_allclose(item.getCenter(), center)
        self.assertAlmostEqual(item.getInnerRadius(), innerRadius)
        self.assertAlmostEqual(item.getOuterRadius(), outerRadius)
        self.assertAlmostEqual(item.getStartAngle(), startAngle)
        self.assertAlmostEqual(item.getEndAngle(), endAngle)
        self.assertAlmostEqual(item.isClosed(), False)

    def testHRange_geometry(self):
        item = roi_items.HorizontalRangeROI()
        vmin = 1
        vmax = 3
        item.setRange(vmin, vmax)
        self.assertAlmostEqual(item.getMin(), vmin)
        self.assertAlmostEqual(item.getMax(), vmax)
        self.assertAlmostEqual(item.getCenter(), 2)


class TestRegionOfInterestManager(TestCaseQt, ParametricTestCase):
    """Tests for RegionOfInterestManager class"""

    def setUp(self):
        super(TestRegionOfInterestManager, self).setUp()
        self.plot = PlotWindow()

        self.roiTableWidget = roi.RegionOfInterestTableWidget()
        dock = qt.QDockWidget()
        dock.setWidget(self.roiTableWidget)
        self.plot.addDockWidget(qt.Qt.BottomDockWidgetArea, dock)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        del self.roiTableWidget
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestRegionOfInterestManager, self).tearDown()

    def test(self):
        """Test ROI of different shapes"""
        tests = (  # shape, points=[list of (x, y), list of (x, y)]
            (roi_items.PointROI, numpy.array(([(10., 15.)], [(20., 25.)]))),
            (roi_items.RectangleROI,
                numpy.array((((1., 10.), (11., 20.)),
                            ((2., 3.), (12., 13.))))),
            (roi_items.PolygonROI,
                numpy.array((((0., 1.), (0., 10.), (10., 0.)),
                            ((5., 6.), (5., 16.), (15., 6.))))),
            (roi_items.LineROI,
                numpy.array((((10., 20.), (10., 30.)),
                            ((30., 40.), (30., 50.))))),
            (roi_items.HorizontalLineROI,
                numpy.array((((10., 20.), (10., 30.)),
                            ((30., 40.), (30., 50.))))),
            (roi_items.VerticalLineROI,
                numpy.array((((10., 20.), (10., 30.)),
                            ((30., 40.), (30., 50.))))),
            (roi_items.HorizontalLineROI,
                numpy.array((((10., 20.), (10., 30.)),
                            ((30., 40.), (30., 50.))))),
        )

        for roiClass, points in tests:
            with self.subTest(roiClass=roiClass):
                manager = roi.RegionOfInterestManager(self.plot)
                self.roiTableWidget.setRegionOfInterestManager(manager)
                manager.start(roiClass)

                self.assertEqual(manager.getRois(), ())

                finishListener = SignalListener()
                manager.sigInteractiveModeFinished.connect(finishListener)

                changedListener = SignalListener()
                manager.sigRoiChanged.connect(changedListener)

                # Add a point
                r = roiClass()
                r.setFirstShapePoints(points[0])
                manager.addRoi(r)
                self.qapp.processEvents()
                self.assertTrue(len(manager.getRois()), 1)
                self.assertEqual(changedListener.callCount(), 1)

                # Remove it
                manager.removeRoi(manager.getRois()[0])
                self.assertEqual(manager.getRois(), ())
                self.assertEqual(changedListener.callCount(), 2)

                # Add two point
                r = roiClass()
                r.setFirstShapePoints(points[0])
                manager.addRoi(r)
                self.qapp.processEvents()
                r = roiClass()
                r.setFirstShapePoints(points[1])
                manager.addRoi(r)
                self.qapp.processEvents()
                self.assertTrue(len(manager.getRois()), 2)
                self.assertEqual(changedListener.callCount(), 4)

                # Reset it
                result = manager.clear()
                self.assertTrue(result)
                self.assertEqual(manager.getRois(), ())
                self.assertEqual(changedListener.callCount(), 5)

                changedListener.clear()

                # Add two point
                r = roiClass()
                r.setFirstShapePoints(points[0])
                manager.addRoi(r)
                self.qapp.processEvents()
                r = roiClass()
                r.setFirstShapePoints(points[1])
                manager.addRoi(r)
                self.qapp.processEvents()
                self.assertTrue(len(manager.getRois()), 2)
                self.assertEqual(changedListener.callCount(), 2)

                # stop
                result = manager.stop()
                self.assertTrue(result)
                self.assertTrue(len(manager.getRois()), 1)
                self.qapp.processEvents()
                self.assertEqual(finishListener.callCount(), 1)

                manager.clear()

    def testRoiDisplay(self):
        rois = []

        # Line
        item = roi_items.LineROI()
        startPoint = numpy.array([1, 2])
        endPoint = numpy.array([3, 4])
        item.setEndPoints(startPoint, endPoint)
        rois.append(item)
        # Horizontal line
        item = roi_items.HorizontalLineROI()
        item.setPosition(15)
        rois.append(item)
        # Vertical line
        item = roi_items.VerticalLineROI()
        item.setPosition(15)
        rois.append(item)
        # Point
        item = roi_items.PointROI()
        point = numpy.array([1, 2])
        item.setPosition(point)
        rois.append(item)
        # Rectangle
        item = roi_items.RectangleROI()
        origin = numpy.array([0, 0])
        size = numpy.array([10, 20])
        item.setGeometry(origin=origin, size=size)
        rois.append(item)
        # Polygon
        item = roi_items.PolygonROI()
        points = numpy.array([[10, 10], [12, 10], [50, 1]])
        item.setPoints(points)
        rois.append(item)
        # Degenerated polygon: No points
        item = roi_items.PolygonROI()
        points = numpy.empty((0, 2))
        item.setPoints(points)
        rois.append(item)
        # Degenerated polygon: A single point
        item = roi_items.PolygonROI()
        points = numpy.array([[5, 10]])
        item.setPoints(points)
        rois.append(item)
        # Degenerated arc: it's a point
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 0, 0, 0, 0
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
        rois.append(item)
        # Degenerated arc: it's a line
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 0, 100, numpy.pi, numpy.pi
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
        rois.append(item)
        # Special arc: it's a donut
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 1, 100, numpy.pi, 3 * numpy.pi
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
        rois.append(item)
        # Arc
        item = roi_items.ArcROI()
        center = numpy.array([10, 20])
        innerRadius, outerRadius, startAngle, endAngle = 1, 100, numpy.pi * 0.5, numpy.pi
        item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
        rois.append(item)
        # Horizontal Range
        item = roi_items.HorizontalRangeROI()
        item.setRange(-1, 3)
        rois.append(item)

        manager = roi.RegionOfInterestManager(self.plot)
        self.roiTableWidget.setRegionOfInterestManager(manager)
        for item in rois:
            with self.subTest(roi=str(item)):
                manager.addRoi(item)
                self.qapp.processEvents()
                item.setEditable(True)
                self.qapp.processEvents()
                item.setEditable(False)
                self.qapp.processEvents()
                manager.removeRoi(item)
                self.qapp.processEvents()

    def testSelectionProxy(self):
        item1 = roi_items.PointROI()
        item1.setSelectable(True)
        item2 = roi_items.PointROI()
        item2.setSelectable(True)
        item1.setFocusProxy(item2)
        manager = roi.RegionOfInterestManager(self.plot)
        manager.setCurrentRoi(item1)
        self.assertIs(manager.getCurrentRoi(), item2)

    def testRemovedSelection(self):
        item1 = roi_items.PointROI()
        item1.setSelectable(True)
        manager = roi.RegionOfInterestManager(self.plot)
        manager.addRoi(item1)
        manager.setCurrentRoi(item1)
        manager.removeRoi(item1)
        self.assertIs(manager.getCurrentRoi(), None)

    def testMaxROI(self):
        """Test Max ROI"""
        origin1 = numpy.array([1., 10.])
        size1 = numpy.array([10., 10.])
        origin2 = numpy.array([2., 3.])
        size2 = numpy.array([10., 10.])

        manager = roi.InteractiveRegionOfInterestManager(self.plot)
        self.roiTableWidget.setRegionOfInterestManager(manager)
        self.assertEqual(manager.getRois(), ())

        changedListener = SignalListener()
        manager.sigRoiChanged.connect(changedListener)

        # Add two point
        item = roi_items.RectangleROI()
        item.setGeometry(origin=origin1, size=size1)
        manager.addRoi(item)
        item = roi_items.RectangleROI()
        item.setGeometry(origin=origin2, size=size2)
        manager.addRoi(item)
        self.qapp.processEvents()
        self.assertEqual(changedListener.callCount(), 2)
        self.assertEqual(len(manager.getRois()), 2)

        # Try to set max ROI to 1 while there is 2 ROIs
        with self.assertRaises(ValueError):
            manager.setMaxRois(1)

        manager.clear()
        self.assertEqual(len(manager.getRois()), 0)
        self.assertEqual(changedListener.callCount(), 3)

        # Set max limit to 1
        manager.setMaxRois(1)

        # Add a point
        item = roi_items.RectangleROI()
        item.setGeometry(origin=origin1, size=size1)
        manager.addRoi(item)
        self.qapp.processEvents()
        self.assertEqual(changedListener.callCount(), 4)

        # Add a 2nd point while max ROI is 1
        item = roi_items.RectangleROI()
        item.setGeometry(origin=origin1, size=size1)
        manager.addRoi(item)
        self.qapp.processEvents()
        self.assertEqual(changedListener.callCount(), 6)
        self.assertEqual(len(manager.getRois()), 1)

    def testChangeInteractionMode(self):
        """Test change of interaction mode"""
        manager = roi.RegionOfInterestManager(self.plot)
        self.roiTableWidget.setRegionOfInterestManager(manager)
        manager.start(roi_items.PointROI)

        interactiveModeToolBar = self.plot.getInteractiveModeToolBar()
        panAction = interactiveModeToolBar.getPanModeAction()

        for roiClass in manager.getSupportedRoiClasses():
            with self.subTest(roiClass=roiClass):
                # Change to pan mode
                panAction.trigger()

                # Change to interactive ROI mode
                action = manager.getInteractionModeAction(roiClass)
                action.trigger()

                self.assertEqual(roiClass, manager.getCurrentInteractionModeRoiClass())

        manager.clear()

    def testLineInteraction(self):
        """This test make sure that a ROI based on handles can be edited with
        the mouse."""
        xlimit = self.plot.getXAxis().getLimits()
        ylimit = self.plot.getYAxis().getLimits()
        points = numpy.array([xlimit, ylimit]).T
        center = numpy.mean(points, axis=0)

        # Create the line
        manager = roi.RegionOfInterestManager(self.plot)
        item = roi_items.LineROI()
        item.setEndPoints(points[0], points[1])
        item.setEditable(True)
        manager.addRoi(item)
        self.qapp.processEvents()

        # Drag the center
        widget = self.plot.getWidgetHandle()
        mx, my = self.plot.dataToPixel(*center)
        self.mouseMove(widget, pos=(mx, my))
        self.mousePress(widget, qt.Qt.LeftButton, pos=(mx, my))
        self.mouseMove(widget, pos=(mx, my+50))
        self.mouseRelease(widget, qt.Qt.LeftButton, pos=(mx, my))

        result = numpy.array(item.getEndPoints())
        # x location is still the same
        numpy.testing.assert_allclose(points[:, 0], result[:, 0], atol=0.5)
        # size is still the same
        numpy.testing.assert_allclose(points[1] - points[0],
                                      result[1] - result[0], atol=0.5)
        # But Y is not the same
        self.assertNotEqual(points[0, 1], result[0, 1])
        self.assertNotEqual(points[1, 1], result[1, 1])
        item = None
        manager.clear()
        self.qapp.processEvents()

    def testPlotWhenCleared(self):
        """PlotWidget.clear should clean up the available ROIs"""
        manager = roi.RegionOfInterestManager(self.plot)
        item = roi_items.LineROI()
        item.setEndPoints((0, 0), (1, 1))
        item.setEditable(True)
        manager.addRoi(item)
        self.qWait()
        try:
            # Make sure the test setup is fine
            self.assertNotEqual(len(manager.getRois()), 0)
            self.assertNotEqual(len(self.plot.getItems()), 0)

            # Call clear and test the expected state
            self.plot.clear()
            self.assertEqual(len(manager.getRois()), 0)
            self.assertEqual(len(self.plot.getItems()), 0)
        finally:
            # Clean up
            manager.clear()

    def testPlotWhenRoiRemoved(self):
        """Make sure there is no remaining items in the plot when a ROI is removed"""
        manager = roi.RegionOfInterestManager(self.plot)
        item = roi_items.LineROI()
        item.setEndPoints((0, 0), (1, 1))
        item.setEditable(True)
        manager.addRoi(item)
        self.qWait()
        try:
            # Make sure the test setup is fine
            self.assertNotEqual(len(manager.getRois()), 0)
            self.assertNotEqual(len(self.plot.getItems()), 0)

            # Call clear and test the expected state
            manager.removeRoi(item)
            self.assertEqual(len(manager.getRois()), 0)
            self.assertEqual(len(self.plot.getItems()), 0)
        finally:
            # Clean up
            manager.clear()



def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestRoiItems))
    test_suite.addTest(loadTests(TestRegionOfInterestManager))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
