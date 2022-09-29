#
#    Project: silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2016  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/04/2018"

import unittest
import numpy
import silx.image.marchingsquares


class MockMarchingSquares(object):

    last = None

    def __init__(self, image, mask=None):
        MockMarchingSquares.last = self
        self.events = []
        self.events.append(("image", image))
        self.events.append(("mask", mask))

    def find_pixels(self, level):
        self.events.append(("find_pixels", level))
        return None

    def find_contours(self, level):
        self.events.append(("find_contours", level))
        return None


class TestFunctionalApi(unittest.TestCase):
    """Test that the default functional API is called using the right
    parameters to the right location."""

    def setUp(self):
        self.old_impl = silx.image.marchingsquares.MarchingSquaresMergeImpl
        silx.image.marchingsquares.MarchingSquaresMergeImpl = MockMarchingSquares

    def tearDown(self):
        silx.image.marchingsquares.MarchingSquaresMergeImpl = self.old_impl
        del self.old_impl

    def test_default_find_contours(self):
        image = numpy.ones((2, 2), dtype=numpy.float32)
        mask = numpy.zeros((2, 2), dtype=numpy.int32)
        level = 2.5
        silx.image.marchingsquares.find_contours(image=image, level=level, mask=mask)
        events = MockMarchingSquares.last.events
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0][0], "image")
        self.assertEqual(events[0][1][0, 0], 1)
        self.assertEqual(events[1][0], "mask")
        self.assertEqual(events[1][1][0, 0], 0)
        self.assertEqual(events[2][0], "find_contours")
        self.assertEqual(events[2][1], level)

    def test_default_find_pixels(self):
        image = numpy.ones((2, 2), dtype=numpy.float32)
        mask = numpy.zeros((2, 2), dtype=numpy.int32)
        level = 3.5
        silx.image.marchingsquares.find_pixels(image=image, level=level, mask=mask)
        events = MockMarchingSquares.last.events
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0][0], "image")
        self.assertEqual(events[0][1][0, 0], 1)
        self.assertEqual(events[1][0], "mask")
        self.assertEqual(events[1][1][0, 0], 0)
        self.assertEqual(events[2][0], "find_pixels")
        self.assertEqual(events[2][1], level)
