# coding: utf-8
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
# ############################################################################*/
"""Tests for polygon functions
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "31/05/2016"


import logging
import unittest
import numpy

from silx.testutils import ParametricTestCase
from silx.image import polygon

logging.basicConfig()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARNING)


class TestPolygonFill(ParametricTestCase):
    """basic poylgon test"""

# tests: non square mask, y=y, x=x, point outside mask
# with and without border

    def test_squares(self):
        """Test polygon fill for a square polygons"""
        mask_shape = 4, 4
        tests = {
            # test name: [(row min, row max), (col min, col max)]
            'square in': [(1, 3), (1, 3)],
            'square out': [(1, 3), (1, 10)],
            'square around': [(-1, 5), (-1, 5)],
            }

        for test_name, (rows, cols) in tests.items():
            with self.subTest(msg=test_name, rows=rows, cols=cols,
                              mask_shape=mask_shape):
                ref_mask = numpy.zeros(mask_shape, dtype=numpy.uint8)
                ref_mask[max(0, rows[0]):rows[1],
                         max(0, cols[0]):cols[1]] = True

                vertices = [(rows[0], cols[0]), (rows[1], cols[0]),
                            (rows[1], cols[1]), (rows[0], cols[1])]
                mask = polygon.polygon_fill(vertices, ref_mask.shape)
                if not numpy.all(numpy.equal(ref_mask, mask)):
                    _logger.debug('%s failed with mask != ref_mask:',
                                  test_name)
                    _logger.debug('result:\n%s', str(mask))
                    _logger.debug('ref:\n%s', str(ref_mask))
                self.assertTrue(numpy.all(numpy.equal(ref_mask, mask)))

    def test_eight(self):
        """Tests with eight shape with different rotation and direction"""
        ref_mask = numpy.array((
            (1, 1, 1, 1, 1, 0),
            (0, 1, 1, 1, 0, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 1, 0, 0, 0),
            (0, 1, 1, 1, 0, 0),
            (0, 0, 0, 0, 0, 0)), dtype=numpy.uint8)
        ref_mask_rot = numpy.asarray(numpy.logical_not(ref_mask),
                                     dtype=numpy.uint8)
        ref_mask_rot[:, -1] = 0
        ref_mask_rot[-1, :] = 0

        tests = {
            'dir 1': ([(0, 0), (5, 5), (5, 0), (0, 5)], ref_mask),
            'dir 1, rot 90': ([(5, 0), (0, 5), (5, 5), (0, 0)], ref_mask_rot),
            'dir 1, rot 180': ([(5, 5), (0, 0), (0, 5), (5, 0)], ref_mask),
            'dir 1, rot -90': ([(0, 5), (5, 0), (0, 0), (5, 5)], ref_mask_rot),
            'dir 2': ([(0, 0), (0, 5), (5, 0), (5, 5)], ref_mask),
            'dir 2, rot 90': ([(5, 0), (0, 0), (5, 5), (0, 5)], ref_mask_rot),
            'dir 2, rot 180': ([(5, 5), (5, 0), (0, 5), (0, 0)], ref_mask),
            'dir 2, rot -90': ([(0, 5), (5, 5), (0, 0), (5, 0)], ref_mask_rot),
        }

        for test_name, (vertices, ref_mask) in tests.items():
            with self.subTest(msg=test_name):
                mask = polygon.polygon_fill(vertices, ref_mask.shape)
                if not numpy.all(numpy.equal(ref_mask, mask)):
                    _logger.debug('%s failed with mask != ref_mask:',
                                  test_name)
                    _logger.debug('result:\n%s', str(mask))
                    _logger.debug('ref:\n%s', str(ref_mask))
                self.assertTrue(numpy.all(numpy.equal(ref_mask, mask)))

    def test_shapes(self):
        """Tests with shapes and reference mask"""
        tests = {
            # name: (
            #     polygon corners as a list of (row, col),
            #     ref_mask)
            'concave polygon': (
                [(1, 1), (4, 3), (1, 5), (2, 3)],
                numpy.array((
                    (0, 0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 1, 1, 0),
                    (0, 0, 0, 1, 0, 0),
                    (0, 0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, 0)), dtype=numpy.uint8)),
            'concave polygon partly outside mask': (
                [(-1, -1), (4, 3), (1, 5), (2, 3)],
                numpy.array((
                    (1, 0, 0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 0),
                    (0, 0, 1, 1, 1, 0),
                    (0, 0, 0, 1, 0, 0),
                    (0, 0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, 0)), dtype=numpy.uint8)),
            'polygon surrounding mask': (
                [(-1, -1), (-1, 7), (7, 7), (7, -1), (0, -1),
                 (8, -2), (8, 8), (-2, 8)],
                numpy.zeros((6, 6), dtype=numpy.uint8))
            }

        for test_name, (vertices, ref_mask) in tests.items():
            with self.subTest(msg=test_name):
                mask = polygon.polygon_fill(vertices, ref_mask.shape)
                if not numpy.all(numpy.equal(ref_mask, mask)):
                    _logger.debug('%s failed with mask != ref_mask:',
                                  test_name)
                    _logger.debug('result:\n%s', str(mask))
                    _logger.debug('ref:\n%s', str(ref_mask))
                self.assertTrue(numpy.all(numpy.equal(ref_mask, mask)))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPolygonFill))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
