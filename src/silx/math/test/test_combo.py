# /*##########################################################################
# Copyright (C) 2016-2020 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import unittest

import numpy

from silx.utils.testutils import ParametricTestCase

from silx.math.combo import min_max


class TestMinMax(ParametricTestCase):
    """Tests of min max combo"""

    FLOATING_DTYPES = 'float32', 'float64'
    if hasattr(numpy, "float128"):
        FLOATING_DTYPES += ('float128',)
    SIGNED_INT_DTYPES = 'int8', 'int16', 'int32', 'int64'
    UNSIGNED_INT_DTYPES = 'uint8', 'uint16', 'uint32', 'uint64'
    DTYPES = FLOATING_DTYPES + SIGNED_INT_DTYPES + UNSIGNED_INT_DTYPES

    def _numpy_min_max(self, data, min_positive=False, finite=False):
        """Reference numpy implementation of min_max

        :param numpy.ndarray data: Data set to use for test
        :param bool min_positive: True to test with positive min
        :param bool finite: True to only test finite values
        """
        data = numpy.array(data, copy=False)
        if data.size == 0:
            raise ValueError('Zero-sized array')

        minimum = None
        argmin = None
        maximum = None
        argmax = None
        min_pos = None
        argmin_pos = None

        if finite:
            filtered_data = data[numpy.isfinite(data)]
        else:
            filtered_data = data

        if filtered_data.size > 0:
            if numpy.all(numpy.isnan(filtered_data)):
                minimum = numpy.nan
                argmin = 0
                maximum = numpy.nan
                argmax = 0
            else:
                minimum = numpy.nanmin(filtered_data)
                # nanargmin equivalent
                argmin = numpy.where(data == minimum)[0][0]
                maximum = numpy.nanmax(filtered_data)
                # nanargmax equivalent
                argmax = numpy.where(data == maximum)[0][0]

            if min_positive:
                with numpy.errstate(invalid='ignore'):
                    # Ignore invalid value encountered in greater
                    pos_data = filtered_data[filtered_data > 0]
                if pos_data.size > 0:
                    min_pos = numpy.min(pos_data)
                    argmin_pos = numpy.where(data == min_pos)[0][0]

        return minimum, min_pos, maximum, argmin, argmin_pos, argmax

    def _test_min_max(self, data, min_positive, finite=False):
        """Compare min_max with numpy for the given dataset

        :param numpy.ndarray data: Data set to use for test
        :param bool min_positive: True to test with positive min
        :param bool finite: True to only test finite values
        """
        minimum, min_pos, maximum, argmin, argmin_pos, argmax = \
            self._numpy_min_max(data, min_positive, finite)

        result = min_max(data, min_positive, finite)

        self.assertSimilar(minimum, result.minimum)
        self.assertSimilar(min_pos, result.min_positive)
        self.assertSimilar(maximum, result.maximum)
        self.assertSimilar(argmin, result.argmin)
        self.assertSimilar(argmin_pos, result.argmin_positive)
        self.assertSimilar(argmax, result.argmax)

    def assertSimilar(self, a, b):
        """Assert that a and b are both None or NaN or that a == b."""
        self.assertTrue((a is None and b is None) or
                        (numpy.isnan(a) and numpy.isnan(b)) or
                        a == b)

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

    NAN_TEST_DATA = [
        (float('nan'), float('nan')),  # All NaNs
        (float('nan'), 1.0),  # NaN first and positive
        (float('nan'), -1.0),  # NaN first and negative
        (1.0, 2.0, float('nan')),  # NaN last and positive
        (-1.0, -2.0, float('nan')),  # NaN last and negative
        (1.0, float('nan'), -1.0),  # Some NaN
    ]

    def test_nandata(self):
        """Test min_max with NaN in data"""
        for dtype in self.FLOATING_DTYPES:
            for data in self.NAN_TEST_DATA:
                with self.subTest(dtype=dtype, data=data):
                    data = numpy.array(data, dtype=dtype)
                    self._test_min_max(data, min_positive=True)

    INF_TEST_DATA = [
        [float('inf')] * 3,  # All +inf
        [float('-inf')] * 3,  # All -inf
        (float('inf'), float('-inf')),  # + and - inf
        (float('inf'), float('-inf'), float('nan')),  # +/-inf, nan last
        (float('nan'), float('-inf'), float('inf')),  # +/-inf, nan first
        (float('inf'), float('nan'), float('-inf')),  # +/-inf, nan center
    ]

    def test_infdata(self):
        """Test min_max with inf."""
        for dtype in self.FLOATING_DTYPES:
            for data in self.INF_TEST_DATA:
                with self.subTest(dtype=dtype, data=data):
                    data = numpy.array(data, dtype=dtype)
                    self._test_min_max(data, min_positive=True)

    def test_finite(self):
        """Test min_max with finite=True"""
        tests = [
            (-1., 2., 0.),  # Basic test
            (float('nan'), float('inf'), float('-inf')),  # NaN + Inf
            (float('nan'), float('inf'), -2, float('-inf')),  # NaN + Inf + 1 value
            (float('inf'), -3, -2), # values + inf
        ]
        tests += self.INF_TEST_DATA
        tests += self.NAN_TEST_DATA

        for dtype in self.FLOATING_DTYPES:
            for data in tests:
                with self.subTest(dtype=dtype, data=data):
                    data = numpy.array(data, dtype=dtype)
                    self._test_min_max(data, min_positive=True, finite=True)
