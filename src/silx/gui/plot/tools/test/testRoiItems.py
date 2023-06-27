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


import numpy.testing

from silx.gui.utils.testutils import TestCaseQt
import silx.gui.plot.items.roi as roi_items


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

    def testCircle_contains(self):
        center = numpy.array([2, -1])
        radius = 1.
        item = roi_items.CircleROI()
        item.setGeometry(center=center, radius=radius)
        self.assertTrue(item.contains([1, -1]))
        self.assertFalse(item.contains([0, 0]))
        self.assertTrue(item.contains([2, 0]))
        self.assertFalse(item.contains([3.01, -1]))

    def testEllipse_contains(self):
        center = numpy.array([-2, 0])
        item = roi_items.EllipseROI()
        item.setCenter(center)
        item.setOrientation(numpy.pi / 4.0)
        item.setMajorRadius(2)
        item.setMinorRadius(1)
        print(item.getMinorRadius(), item.getMajorRadius())
        self.assertFalse(item.contains([0, 0]))
        self.assertTrue(item.contains([-1, 1]))
        self.assertTrue(item.contains([-3, 0]))
        self.assertTrue(item.contains([-2, 0]))
        self.assertTrue(item.contains([-2, 1]))
        self.assertFalse(item.contains([-4, 1]))

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

    def testBand_getToSetGeometry(self):
        """Test that we can use getGeometry as input to setGeometry"""
        item = roi_items.BandROI()
        item.setFirstShapePoints(numpy.array([[5, 10], [50, 100]]))
        item.setGeometry(*item.getGeometry())
