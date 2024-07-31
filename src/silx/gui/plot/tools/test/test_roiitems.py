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


import pytest
import numpy.testing

import silx.gui.plot.items.roi as roi_items


def testLine_geometry(qapp):
    item = roi_items.LineROI()
    startPoint = numpy.array([1, 2])
    endPoint = numpy.array([3, 4])
    item.setEndPoints(startPoint, endPoint)
    numpy.testing.assert_allclose(item.getEndPoints()[0], startPoint)
    numpy.testing.assert_allclose(item.getEndPoints()[1], endPoint)


def testHLine_geometry(qapp):
    item = roi_items.HorizontalLineROI()
    item.setPosition(15)
    assert item.getPosition() == 15


def testVLine_geometry(qapp):
    item = roi_items.VerticalLineROI()
    item.setPosition(15)
    assert item.getPosition() == 15


def testPoint_geometry(qapp):
    point = numpy.array([1, 2])
    item = roi_items.PointROI()
    item.setPosition(point)
    numpy.testing.assert_allclose(item.getPosition(), point)


def testRectangle_originGeometry(qapp):
    origin = numpy.array([0, 0])
    size = numpy.array([10, 20])
    center = numpy.array([5, 10])
    item = roi_items.RectangleROI()
    item.setGeometry(origin=origin, size=size)
    numpy.testing.assert_allclose(item.getOrigin(), origin)
    numpy.testing.assert_allclose(item.getSize(), size)
    numpy.testing.assert_allclose(item.getCenter(), center)


def testRectangle_centerGeometry(qapp):
    origin = numpy.array([0, 0])
    size = numpy.array([10, 20])
    center = numpy.array([5, 10])
    item = roi_items.RectangleROI()
    item.setGeometry(center=center, size=size)
    numpy.testing.assert_allclose(item.getOrigin(), origin)
    numpy.testing.assert_allclose(item.getSize(), size)
    numpy.testing.assert_allclose(item.getCenter(), center)


def testRectangle_setCenterGeometry(qapp):
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


def testRectangle_setOriginGeometry(qapp):
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


def testCircle_geometry(qapp):
    center = numpy.array([0, 0])
    radius = 10.0
    item = roi_items.CircleROI()
    item.setGeometry(center=center, radius=radius)
    numpy.testing.assert_allclose(item.getCenter(), center)
    numpy.testing.assert_allclose(item.getRadius(), radius)


def testCircle_setCenter(qapp):
    center = numpy.array([0, 0])
    radius = 10.0
    item = roi_items.CircleROI()
    item.setGeometry(center=center, radius=radius)
    newCenter = numpy.array([-10, 0])
    item.setCenter(newCenter)
    numpy.testing.assert_allclose(item.getCenter(), newCenter)
    numpy.testing.assert_allclose(item.getRadius(), radius)


def testCircle_setRadius(qapp):
    center = numpy.array([0, 0])
    radius = 10.0
    item = roi_items.CircleROI()
    item.setGeometry(center=center, radius=radius)
    newRadius = 5.1
    item.setRadius(newRadius)
    numpy.testing.assert_allclose(item.getCenter(), center)
    numpy.testing.assert_allclose(item.getRadius(), newRadius)


def testCircle_contains(qapp):
    center = numpy.array([2, -1])
    radius = 1.0
    item = roi_items.CircleROI()
    item.setGeometry(center=center, radius=radius)
    assert item.contains([1, -1])
    assert not item.contains([0, 0])
    assert item.contains([2, 0])
    assert not item.contains([3.01, -1])


def testEllipse_contains(qapp):
    center = numpy.array([-2, 0])
    item = roi_items.EllipseROI()
    item.setCenter(center)
    item.setOrientation(numpy.pi / 4.0)
    item.setMajorRadius(2)
    item.setMinorRadius(1)
    print(item.getMinorRadius(), item.getMajorRadius())
    assert not item.contains([0, 0])
    assert item.contains([-1, 1])
    assert item.contains([-3, 0])
    assert item.contains([-2, 0])
    assert item.contains([-2, 1])
    assert not item.contains([-4, 1])


def testRectangle_isIn(qapp):
    origin = numpy.array([0, 0])
    size = numpy.array([10, 20])
    item = roi_items.RectangleROI()
    item.setGeometry(origin=origin, size=size)
    assert item.contains(position=(0, 0))
    assert item.contains(position=(2, 14))
    assert not item.contains(position=(14, 12))


def testPolygon_emptyGeometry(qapp):
    points = numpy.empty((0, 2))
    item = roi_items.PolygonROI()
    item.setPoints(points)
    numpy.testing.assert_allclose(item.getPoints(), points)


def testPolygon_geometry(qapp):
    points = numpy.array([[10, 10], [12, 10], [50, 1]])
    item = roi_items.PolygonROI()
    item.setPoints(points)
    numpy.testing.assert_allclose(item.getPoints(), points)


def testPolygon_isIn(qapp):
    points = numpy.array([[0, 0], [0, 10], [5, 10]])
    item = roi_items.PolygonROI()
    item.setPoints(points)
    assert item.contains((0, 0))
    assert not item.contains((6, 2))
    assert not item.contains((-2, 5))
    assert not item.contains((2, -1))
    assert not item.contains((8, 1))
    assert item.contains((1, 8))


def testArc_getToSetGeometry(qapp):
    """Test that we can use getGeometry as input to setGeometry"""
    item = roi_items.ArcROI()
    item.setFirstShapePoints(numpy.array([[5, 10], [50, 100]]))
    item.setGeometry(*item.getGeometry())


def testArc_degenerated_point(qapp):
    item = roi_items.ArcROI()
    center = numpy.array([10, 20])
    innerRadius, outerRadius, startAngle, endAngle = 0, 0, 0, 0
    item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)


def testArc_degenerated_line(qapp):
    item = roi_items.ArcROI()
    center = numpy.array([10, 20])
    innerRadius, outerRadius, startAngle, endAngle = 0, 100, numpy.pi, numpy.pi
    item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)


def testArc_special_circle(qapp):
    item = roi_items.ArcROI()
    center = numpy.array([10, 20])
    innerRadius, outerRadius, startAngle, endAngle = 0, 100, numpy.pi, 3 * numpy.pi
    item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
    numpy.testing.assert_allclose(item.getCenter(), center)
    assert item.getInnerRadius() == pytest.approx(innerRadius)
    assert item.getOuterRadius() == pytest.approx(outerRadius)
    assert item.getStartAngle() == pytest.approx(item.getEndAngle() - numpy.pi * 2.0)
    assert item.isClosed()


def testArc_special_donut(qapp):
    item = roi_items.ArcROI()
    center = numpy.array([10, 20])
    innerRadius, outerRadius, startAngle, endAngle = 1, 100, numpy.pi, 3 * numpy.pi
    item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
    numpy.testing.assert_allclose(item.getCenter(), center)
    assert item.getInnerRadius() == pytest.approx(innerRadius)
    assert item.getOuterRadius() == pytest.approx(outerRadius)
    assert item.getStartAngle() == pytest.approx(item.getEndAngle() - numpy.pi * 2.0)
    assert item.isClosed()


def testArc_clockwiseGeometry(qapp):
    """Test that we can use getGeometry as input to setGeometry"""
    item = roi_items.ArcROI()
    center = numpy.array([10, 20])
    innerRadius, outerRadius, startAngle, endAngle = 1, 100, numpy.pi * 0.5, numpy.pi
    item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
    numpy.testing.assert_allclose(item.getCenter(), center)
    assert item.getInnerRadius() == pytest.approx(innerRadius)
    assert item.getOuterRadius() == pytest.approx(outerRadius)
    assert item.getStartAngle() == pytest.approx(startAngle)
    assert item.getEndAngle() == pytest.approx(endAngle)
    assert not item.isClosed()


def testArc_anticlockwiseGeometry(qapp):
    """Test that we can use getGeometry as input to setGeometry"""
    item = roi_items.ArcROI()
    center = numpy.array([10, 20])
    innerRadius, outerRadius, startAngle, endAngle = (
        1,
        100,
        numpy.pi * 0.5,
        -numpy.pi * 0.5,
    )
    item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
    numpy.testing.assert_allclose(item.getCenter(), center)
    assert item.getInnerRadius() == pytest.approx(innerRadius)
    assert item.getOuterRadius() == pytest.approx(outerRadius)
    assert item.getStartAngle() == pytest.approx(startAngle)
    assert item.getEndAngle() == pytest.approx(endAngle)
    assert not item.isClosed()


def testArc_position(qapp):
    """Test validity of getPosition"""
    item = roi_items.ArcROI()
    center = numpy.array([10, 20])
    innerRadius, outerRadius, startAngle, endAngle = 1, 100, numpy.pi * 0.5, numpy.pi
    item.setGeometry(center, innerRadius, outerRadius, startAngle, endAngle)
    assert item.getPosition(roi_items.ArcROI.Role.START) == pytest.approx((10.0, 70.5))
    assert item.getPosition(roi_items.ArcROI.Role.STOP) == pytest.approx((-40.5, 20.0))
    assert item.getPosition(roi_items.ArcROI.Role.MIDDLE) == pytest.approx(
        (-25.71, 55.71), abs=0.1
    )
    assert item.getPosition(roi_items.ArcROI.Role.CENTER) == pytest.approx(
        (10.0, 20), abs=0.1
    )


def testHRange_geometry(qapp):
    item = roi_items.HorizontalRangeROI()
    vmin = 1
    vmax = 3
    item.setRange(vmin, vmax)
    assert item.getMin() == pytest.approx(vmin)
    assert item.getMax() == pytest.approx(vmax)
    assert item.getCenter() == pytest.approx(2)


def testBand_getToSetGeometry(qapp):
    """Test that we can use getGeometry as input to setGeometry"""
    item = roi_items.BandROI()
    item.setFirstShapePoints(numpy.array([[5, 10], [50, 100]]))
    item.setGeometry(*item.getGeometry())
