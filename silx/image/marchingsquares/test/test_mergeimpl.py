# -*- coding: utf-8 -*-
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
__date__ = "18/04/2018"

import unittest
import numpy
from .._mergeimpl import MarchingSquaresMergeImpl


class TestMergeImplApi(unittest.TestCase):

    def test_image_not_an_array(self):
        bad_image = 1
        self.assertRaises(ValueError, MarchingSquaresMergeImpl, bad_image)

    def test_image_bad_dim(self):
        bad_image = numpy.array([[[1.0]]])
        self.assertRaises(ValueError, MarchingSquaresMergeImpl, bad_image)

    def test_image_not_big_enough(self):
        bad_image = numpy.array([[1.0, 1.0, 1.0, 1.0]])
        self.assertRaises(ValueError, MarchingSquaresMergeImpl, bad_image)

    def test_mask_not_an_array(self):
        image = numpy.array([[1.0, 1.0], [1.0, 1.0]])
        bad_mask = 1
        self.assertRaises(ValueError, MarchingSquaresMergeImpl, image, bad_mask)

    def test_mask_not_match(self):
        image = numpy.array([[1.0, 1.0], [1.0, 1.0]])
        bad_mask = numpy.array([[1.0, 1.0]])
        self.assertRaises(ValueError, MarchingSquaresMergeImpl, image, bad_mask)

    def test_ok_anyway_bad_type(self):
        image = numpy.array([[1.0, 1.0], [1.0, 1.0]], dtype=numpy.int32)
        mask = numpy.array([[1.0, 1.0], [1.0, 1.0]], dtype=numpy.float32)
        MarchingSquaresMergeImpl(image, mask)

    def test_find_contours_result(self):
        image = numpy.zeros((2, 2))
        image[0, 0] = 1
        ms = MarchingSquaresMergeImpl(image)
        polygons = ms.find_contours(0.5)
        self.assertIsInstance(polygons, list)
        self.assertTrue(len(polygons), 1)
        self.assertIsInstance(polygons[0], numpy.ndarray)
        self.assertEqual(polygons[0].shape[1], 2)
        self.assertEqual(polygons[0].dtype.kind, "f")

    def test_find_pixels_result(self):
        image = numpy.zeros((2, 2))
        image[0, 0] = 1
        ms = MarchingSquaresMergeImpl(image)
        pixels = ms.find_pixels(0.5)
        self.assertIsInstance(pixels, numpy.ndarray)
        self.assertEqual(pixels.shape[1], 2)
        self.assertEqual(pixels.dtype.kind, "i")

    def test_find_contours_empty_result(self):
        image = numpy.zeros((2, 2))
        ms = MarchingSquaresMergeImpl(image)
        polygons = ms.find_contours(0.5)
        self.assertIsInstance(polygons, list)
        self.assertEqual(len(polygons), 0)

    def test_find_pixels_empty_result(self):
        image = numpy.zeros((2, 2))
        ms = MarchingSquaresMergeImpl(image)
        pixels = ms.find_pixels(0.5)
        self.assertIsInstance(pixels, numpy.ndarray)
        self.assertEqual(pixels.shape[1], 2)
        self.assertEqual(pixels.shape[0], 0)
        self.assertEqual(pixels.dtype.kind, "i")

    def test_find_contours_yx_result(self):
        image = numpy.zeros((2, 2))
        image[1, 0] = 1
        ms = MarchingSquaresMergeImpl(image)
        polygons = ms.find_contours(0.5)
        polygon = polygons[0]
        self.assertTrue((polygon == (0.5, 0)).any())
        self.assertTrue((polygon == (1, 0.5)).any())

    def test_find_pixels_yx_result(self):
        image = numpy.zeros((2, 2))
        image[1, 0] = 1
        ms = MarchingSquaresMergeImpl(image)
        pixels = ms.find_pixels(0.5)
        self.assertTrue((pixels == (1, 0)).any())


class TestMergeImplContours(unittest.TestCase):

    def test_merge_segments(self):
        image = numpy.zeros((4, 4))
        image[(2, 3), :] = 1
        ms = MarchingSquaresMergeImpl(image)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 1)

    def test_merge_segments_2(self):
        image = numpy.zeros((4, 4))
        image[(2, 3), :] = 1
        image[2, 2] = 0
        ms = MarchingSquaresMergeImpl(image)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 1)

    def test_merge_tiles(self):
        image = numpy.zeros((4, 4))
        image[(2, 3), :] = 1
        ms = MarchingSquaresMergeImpl(image, group_size=2)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 1)

    def test_fully_masked(self):
        image = numpy.zeros((5, 5))
        image[(2, 3), :] = 1
        mask = numpy.ones((5, 5))
        ms = MarchingSquaresMergeImpl(image, mask)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 0)

    def test_fully_masked_minmax(self):
        """This invalidates all the tiles. The route is not the same."""
        image = numpy.zeros((5, 5))
        image[(2, 3), :] = 1
        mask = numpy.ones((5, 5))
        ms = MarchingSquaresMergeImpl(image, mask, group_size=2, use_minmax_cache=True)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 0)

    def test_masked_segments(self):
        image = numpy.zeros((5, 5))
        image[(2, 3, 4), :] = 1
        mask = numpy.zeros((5, 5))
        mask[:, 2] = 1
        ms = MarchingSquaresMergeImpl(image, mask)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 2)

    def test_closed_polygon(self):
        image = numpy.zeros((5, 5))
        image[2, 2] = 1
        image[1, 2] = 1
        image[3, 2] = 1
        image[2, 1] = 1
        image[2, 3] = 1
        mask = None
        ms = MarchingSquaresMergeImpl(image, mask)
        polygons = ms.find_contours(0.9)
        self.assertEqual(len(polygons), 1)
        self.assertEqual(list(polygons[0][0]), list(polygons[0][-1]))

    def test_closed_polygon_between_tiles(self):
        image = numpy.zeros((5, 5))
        image[2, 2] = 1
        image[1, 2] = 1
        image[3, 2] = 1
        image[2, 1] = 1
        image[2, 3] = 1
        mask = None
        ms = MarchingSquaresMergeImpl(image, mask, group_size=2)
        polygons = ms.find_contours(0.9)
        self.assertEqual(len(polygons), 1)
        self.assertEqual(list(polygons[0][0]), list(polygons[0][-1]))

    def test_open_polygon(self):
        image = numpy.zeros((5, 5))
        image[2, 2] = 1
        image[1, 2] = 1
        image[3, 2] = 1
        image[2, 1] = 1
        image[2, 3] = 1
        mask = numpy.zeros((5, 5))
        mask[1, 1] = 1
        ms = MarchingSquaresMergeImpl(image, mask)
        polygons = ms.find_contours(0.9)
        self.assertEqual(len(polygons), 1)
        self.assertNotEqual(list(polygons[0][0]), list(polygons[0][-1]))

    def test_ambiguous_pattern(self):
        image = numpy.zeros((6, 8))
        image[(3, 4), :] = 1
        image[:, (0, -1)] = 0
        image[3, 3] = -0.001
        image[4, 4] = 0.0
        mask = None
        ms = MarchingSquaresMergeImpl(image, mask)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 2)

    def test_ambiguous_pattern_2(self):
        image = numpy.zeros((6, 8))
        image[(3, 4), :] = 1
        image[:, (0, -1)] = 0
        image[3, 3] = +0.001
        image[4, 4] = 0.0
        mask = None
        ms = MarchingSquaresMergeImpl(image, mask)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 1)

    def count_closed_polygons(self, polygons):
        closed = 0
        for polygon in polygons:
            if list(polygon[0]) == list(polygon[-1]):
                closed += 1
        return closed

    def test_image(self):
        # example from skimage
        x, y = numpy.ogrid[-numpy.pi:numpy.pi:100j, -numpy.pi:numpy.pi:100j]
        image = numpy.sin(numpy.exp((numpy.sin(x)**3 + numpy.cos(y)**2)))
        mask = None
        ms = MarchingSquaresMergeImpl(image, mask)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 11)
        self.assertEqual(self.count_closed_polygons(polygons), 3)

    def test_image_tiled(self):
        # example from skimage
        x, y = numpy.ogrid[-numpy.pi:numpy.pi:100j, -numpy.pi:numpy.pi:100j]
        image = numpy.sin(numpy.exp((numpy.sin(x)**3 + numpy.cos(y)**2)))
        mask = None
        ms = MarchingSquaresMergeImpl(image, mask, group_size=50)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 11)
        self.assertEqual(self.count_closed_polygons(polygons), 3)

    def test_image_tiled_minmax(self):
        # example from skimage
        x, y = numpy.ogrid[-numpy.pi:numpy.pi:100j, -numpy.pi:numpy.pi:100j]
        image = numpy.sin(numpy.exp((numpy.sin(x)**3 + numpy.cos(y)**2)))
        mask = None
        ms = MarchingSquaresMergeImpl(image, mask, group_size=50, use_minmax_cache=True)
        polygons = ms.find_contours(0.5)
        self.assertEqual(len(polygons), 11)
        self.assertEqual(self.count_closed_polygons(polygons), 3)


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestMergeImplApi))
    test_suite.addTest(loadTests(TestMergeImplContours))
    return test_suite
