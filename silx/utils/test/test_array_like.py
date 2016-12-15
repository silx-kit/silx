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
# ###########################################################################*/
"""Tests for array_like module"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "15/12/2016"

try:
    import h5py
except ImportError:
    h5py = None

import numpy
import os
import tempfile
import unittest

from ..array_like import TransposedDatasetView


@unittest.skipIf(h5py is None,
                 "h5py is needed to test TransposedDatasetView")
class TestTransposedDatasetView(unittest.TestCase):

    def setUp(self):
        # dataset attributes
        self.ndim = 3
        self.original_shape = (5, 10, 20)
        self.size = 1
        for dim in self.original_shape:
            self.size *= dim

        volume = numpy.arange(self.size).reshape(self.original_shape)

        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "tempfile.h5")
        with h5py.File(self.h5_fname, "w") as f:
            f["volume"] = volume

        self.h5f = h5py.File(self.h5_fname, "r")

    def tearDown(self):
        self.h5f.close()
        os.unlink(self.h5_fname)
        os.rmdir(self.tempdir)

    def _testSize(self, obj):
        """These assertions apply to all following test cases"""
        self.assertEqual(obj.ndim, self.ndim)
        self.assertEqual(obj.size, self.size)
        size_from_shape = 1
        for dim in obj.shape:
            size_from_shape *= dim
        self.assertEqual(size_from_shape, self.size)

        for dim in self.original_shape:
            self.assertIn(dim, obj.shape)

    def testNoTransposition(self):
        """no transposition (transposition = (0, 1, 2))"""
        a = TransposedDatasetView(self.h5f["volume"])

        self.assertEqual(a.shape, self.original_shape)
        self._testSize(a)

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    self.assertEqual(self.h5f["volume"][i, j, k],
                                     a[i, j, k])

    def _testTransposition(self, transposition):
        """test transposed dataset

        :param tuple transposition: List of dimensions (0... n-1) sorted
            in the desired order
        """
        a = TransposedDatasetView(self.h5f["volume"],
                                  transposition=transposition)
        self._testSize(a)

        # sort shape of transposed object, to hopefully find the original shape
        sorted_shape = tuple(dim_size for (_, dim_size) in
                             sorted(zip(transposition, a.shape)))
        self.assertEqual(sorted_shape, self.original_shape)

        # test the TransposedDatasetView.__array__ method
        self.assertTrue(numpy.array_equal(
                numpy.array(a),
                numpy.array(self.h5f["volume"]).transpose(transposition)))

        # test the TransposedDatasetView.__getitem__
        # (step adjusted to test at least 3 indices in each dimension)
        for i in range(0, a.shape[0], a.shape[0] // 3):
            for j in range(0, a.shape[1], a.shape[1] // 3):
                for k in range(0, a.shape[2], a.shape[2] // 3):
                    sorted_indices = tuple(idx for (_, idx) in
                                           sorted(zip(transposition, [i, j, k])))
                    viewed_value = a[i, j, k]
                    corresponding_original_value = self.h5f["volume"][sorted_indices]
                    self.assertEqual(viewed_value,
                                     corresponding_original_value)

    def testTransposition012(self):
        """transposition = (0, 1, 2)
        (should be the same as testNoTransposition)"""
        self._testTransposition((0, 1, 2))

    def testTransposition021(self):
        """transposition = (0, 2, 1)"""
        self._testTransposition((0, 2, 1))

    def testTransposition102(self):
        """transposition = (1, 0, 2)"""
        self._testTransposition((1, 0, 2))

    def testTransposition120(self):
        """transposition = (1, 2, 0)"""
        self._testTransposition((1, 2, 0))

    def testTransposition201(self):
        """transposition = (2, 0, 1)"""
        self._testTransposition((2, 0, 1))

    def testTransposition210(self):
        """transposition = (2, 1, 0)"""
        self._testTransposition((2, 1, 0))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestTransposedDatasetView))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
