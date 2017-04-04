# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
"""Tests of the combo module"""

from __future__ import division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "20/12/2016"


import unittest

import numpy

from silx.test.utils import ParametricTestCase

from silx.math.combo import min_max


class TestMinMax(ParametricTestCase):
    """Tests of min max combo"""

    FLOATING_DTYPES = 'float32', 'float64'
    SIGNED_INT_DTYPES = 'uint8', 'uint16', 'uint32', 'uint64'
    UNSIGNED_INT_DTYPES = 'uint8', 'uint16', 'uint32', 'uint64'
    DTYPES = FLOATING_DTYPES + SIGNED_INT_DTYPES + UNSIGNED_INT_DTYPES

    def _test_min_max(self, data, min_positive):
        """Compare min_max with numpy for the given dataset

        :param numpy.ndarray data: Data set to use for test
        :param bool min_positive: True to test with positive min
        """
        result = min_max(data, min_positive)

        minimum = numpy.nanmin(data)
        if numpy.isnan(minimum):  # All NaNs
            self.assertTrue(numpy.isnan(result.minimum))
            self.assertEqual(result.argmin, 0)

        else:
            self.assertEqual(result.minimum, minimum)

            argmin = numpy.where(data == minimum)[0][0]
            self.assertEqual(result.argmin, argmin)

        maximum = numpy.nanmax(data)
        if numpy.isnan(maximum):  # All NaNs
            self.assertTrue(numpy.isnan(result.maximum))
            self.assertEqual(result.argmax, 0)

        else:
            self.assertEqual(result.maximum, maximum)

            argmax = numpy.where(data == maximum)[0][0]
            self.assertEqual(result.argmax, argmax)

        if min_positive:
            pos_data = data[data > 0]
            if len(pos_data) > 0:
                min_pos = numpy.min(pos_data)
                argmin_pos = numpy.where(data == min_pos)[0][0]
            else:
                min_pos = None
                argmin_pos = None
            self.assertEqual(result.min_positive, min_pos)
            self.assertEqual(result.argmin_positive, argmin_pos)

    def test_different_datasets(self):
        """Test min_max with different numpy.arange datasets."""
        size = 1000

        for dtype in self.DTYPES:

            tests = {
                '0 to N': (0, 1),
                'N-1 to 0': (size - 1, -1)}
            if dtype not in self.UNSIGNED_INT_DTYPES:
                tests['N/2 to -N/2'] = size // 2, -1
                tests['0 to -N'] = 0, -1

            for name, (start, step) in tests.items():
                for min_positive in (True, False):
                    with self.subTest(dtype=dtype,
                                      min_positive=min_positive,
                                      data=name):
                        data = numpy.arange(
                            start, start + step * size, step, dtype=dtype)

                        self._test_min_max(data, min_positive)

    def test_nodata(self):
        """Test min_max with None and empty array"""
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    min_max(None)
                
                data = numpy.array((), dtype=dtype)
                with self.assertRaises(ValueError):
                    min_max(data)

    def test_nandata(self):
        """Test min_max with NaN in data"""
        tests = [
            (float('nan'), float('nan')),  # All NaNs
            (float('nan'), 1.0),  # NaN first and positive
            (float('nan'), -1.0),  # NaN first and negative
            (1.0, 2.0, float('nan')),  # NaN last and positive
            (-1.0, -2.0, float('nan')),  # NaN last and negative
            (1.0, float('nan'), -1.0),  # Some NaN
        ]

        for dtype in self.FLOATING_DTYPES:
            for data in tests:
                with self.subTest(dtype=dtype, data=data):
                    data = numpy.array(data, dtype=dtype)
                    self._test_min_max(data, min_positive=True)

    def test_infdata(self):
        """Test min_max with inf."""
        tests = [
            [float('inf')] * 3,  # All +inf
            [float('inf')] * 3,  # All -inf
            (float('inf'), float('-inf')),  # + and - inf
            (float('inf'), float('-inf'), float('nan')),  # +/-inf, nan last
            (float('nan'), float('-inf'), float('inf')),  # +/-inf, nan first
            (float('inf'), float('nan'), float('-inf')),  # +/-inf, nan center
        ]

        for dtype in self.FLOATING_DTYPES:
            for data in tests:
                with self.subTest(dtype=dtype, data=data):
                    data = numpy.array(data, dtype=dtype)
                    self._test_min_max(data, min_positive=True)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestMinMax))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
