# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Basic tests for Bounding box"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "27/09/2019"


import unittest
import numpy
from silx.image._boundingbox import _BoundingBox


class TestBB(unittest.TestCase):
    """Some simple test on the bounding box class"""
    def test_creation(self):
        """test some constructors"""
        pts = numpy.array([(0, 0), (10, 20), (20, 0)])
        bb = _BoundingBox.from_points(pts)
        self.assertTrue(bb.bottom_left == (0, 0))
        self.assertTrue(bb.top_right == (20, 20))
        pts = numpy.array([(0, 10), (10, 20), (45, 30), (35, 0)])
        bb = _BoundingBox.from_points(pts)
        self.assertTrue(bb.bottom_left == (0, 0))
        print(bb.top_right)
        self.assertTrue(bb.top_right == (45, 30))

    def test_isIn_pt(self):
        """test the isIn function with points"""
        bb = _BoundingBox(bottom_left=(6, 2), top_right=(12, 6))
        self.assertTrue(bb.contains((10, 4)))
        self.assertTrue(bb.contains((6, 2)))
        self.assertTrue(bb.contains((12, 2)))
        self.assertFalse(bb.contains((0, 0)))
        self.assertFalse(bb.contains((20, 0)))
        self.assertFalse(bb.contains((10, 0)))

    def test_collide(self):
        """test the collide function"""
        bb1 = _BoundingBox(bottom_left=(6, 2), top_right=(12, 6))
        self.assertTrue(bb1.collide(_BoundingBox(bottom_left=(6, 2), top_right=(12, 6))))
        bb1 = _BoundingBox(bottom_left=(6, 2), top_right=(12, 6))
        self.assertFalse(bb1.collide(_BoundingBox(bottom_left=(12, 2), top_right=(12, 2))))

    def test_isIn_bb(self):
        """test the isIn function with other bounding box"""
        bb1 = _BoundingBox(bottom_left=(6, 2), top_right=(12, 6))
        self.assertTrue(bb1.contains(_BoundingBox(bottom_left=(6, 2), top_right=(12, 6))))
        bb1 = _BoundingBox(bottom_left=(6, 2), top_right=(12, 6))
        self.assertTrue(bb1.contains(_BoundingBox(bottom_left=(12, 2), top_right=(12, 2))))
        self.assertFalse(_BoundingBox(bottom_left=(12, 2), top_right=(12, 2)).contains(bb1))
