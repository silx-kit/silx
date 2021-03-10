# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
"""Test for interpolate module"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "11/07/2019"


import unittest

import numpy
try:
    from scipy.interpolate import interpn
except ImportError:
    interpn = None

from silx.utils.testutils import ParametricTestCase
from silx.math import interpolate


@unittest.skipUnless(interpn is not None, "scipy missing")
class TestInterp3d(ParametricTestCase):
    """Test silx.math.interpolate.interp3d"""

    @staticmethod
    def ref_interp3d(data, points):
        """Reference implementation of interp3d based on scipy

        :param numpy.ndarray data: 3D floating dataset
        :param numpy.ndarray points: Array of points of shape (N, 3)
        """
        return interpn(
            [numpy.arange(dim, dtype=data.dtype) for dim in data.shape],
            data,
            points,
            method='linear')

    def test_random_data(self):
        """Test interp3d with random data"""
        size = 32
        npoints = 10

        ref_data = numpy.random.random((size, size, size))
        ref_points = numpy.random.random(npoints*3).reshape(npoints, 3) * (size -1)

        for dtype in (numpy.float32, numpy.float64):
            data = ref_data.astype(dtype)
            points = ref_points.astype(dtype)
            ref_result = self.ref_interp3d(data, points)

            for method in (u'linear', u'linear_omp'):
                with self.subTest(method=method):
                    result = interpolate.interp3d(data, points, method=method)
                    self.assertTrue(numpy.allclose(ref_result, result))

    def test_notfinite_data(self):
        """Test interp3d with NaN and inf"""
        data = numpy.ones((3, 3, 3), dtype=numpy.float64)
        data[0, 0, 0] = numpy.nan
        data[2, 2, 2] = numpy.inf
        points = numpy.array([(0.5, 0.5, 0.5),
                              (1.5, 1.5, 1.5)])

        for method in (u'linear', u'linear_omp'):
            with self.subTest(method=method):
                result = interpolate.interp3d(
                    data, points, method=method)
                self.assertTrue(numpy.isnan(result[0]))
                self.assertTrue(result[1] == numpy.inf)

    def test_points_outside(self):
        """Test interp3d with points outside the volume"""
        data = numpy.ones((4, 4, 4), dtype=numpy.float64)
        points = numpy.array([(-0.1, -0.1, -0.1),
                              (3.1, 3.1, 3.1),
                              (-0.1, 1., 1.),
                              (1., 1., 3.1)])

        for method in (u'linear', u'linear_omp'):
            for fill_value in (numpy.nan, 0., -1.):
                with self.subTest(method=method):
                    result = interpolate.interp3d(
                        data, points, method=method, fill_value=fill_value)
                    if numpy.isnan(fill_value):
                        self.assertTrue(numpy.all(numpy.isnan(result)))
                    else:
                        self.assertTrue(numpy.all(numpy.equal(result, fill_value)))

    def test_integer_points(self):
        """Test interp3d with integer points coord"""
        data = numpy.arange(4**3, dtype=numpy.float64).reshape(4, 4, 4)
        points = numpy.array([(0., 0., 0.),
                              (0., 0., 1.),
                              (2., 3., 0.),
                              (3., 3., 3.)])

        ref_result = data[tuple(points.T.astype(numpy.int32))]

        for method in (u'linear', u'linear_omp'):
            with self.subTest(method=method):
                result = interpolate.interp3d(data, points, method=method)
                self.assertTrue(numpy.allclose(ref_result, result))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestInterp3d))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
