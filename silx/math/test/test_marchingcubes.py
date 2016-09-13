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
"""Tests of the marchingcubes module"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "07/09/2016"

import unittest

import numpy

from silx.math import marchingcubes


class TestMarchingCubes(unittest.TestCase):
    """Tests of marching cubes"""

    def assertAllClose(self, array1, array2, msg=None,
                       rtol=1e-05, atol=1e-08):
        """Assert that the 2 numpy.ndarrays are almost equal.

        :param str msg: Message to provide when assert fails
        :param float rtol: Relative tolerance, see :func:`numpy.allclose`
        :param float atol: Absolute tolerance, see :func:`numpy.allclose`
        """
        if not numpy.allclose(array1, array2, rtol, atol):
            raise self.failureException(msg)

    def test_cube(self):
        """Unit tests with a single cube"""

        # No isosurface
        cube_zero = numpy.zeros((2, 2, 2), dtype=numpy.float32)

        result = marchingcubes.MarchingCubes(cube_zero, 1.)
        self.assertEqual(result.shape, cube_zero.shape)
        self.assertEqual(result.isolevel, 1.)
        self.assertEqual(result.invert_normals, True)

        vertices, normals, indices = result
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(normals), 0)
        self.assertEqual(len(indices), 0)

        # isosurface perpendicular to Z
        cube = numpy.array(
            (((0., 0.), (0., 0.)),
             ((1., 1.), (1., 1.))), dtype=numpy.float32)
        level = 0.5
        vertices, normals, indices = marchingcubes.MarchingCubes(
            cube, level, invert_normals=False)
        self.assertAllClose(vertices[:, 2], level)
        self.assertAllClose(normals, (0., 0., 1.))
        self.assertEqual(len(indices), 2)

        # isosurface perpendicular to Y
        cube = numpy.array(
            (((0., 0.), (1., 1.)),
             ((0., 0.), (1., 1.))), dtype=numpy.float32)
        level = 0.2
        vertices, normals, indices = marchingcubes.MarchingCubes(cube, level)
        self.assertAllClose(vertices[:, 1], level)
        self.assertAllClose(normals, (0., -1., 0.))
        self.assertEqual(len(indices), 2)

        # isosurface perpendicular to X
        cube = numpy.array(
            (((0., 1.), (0., 1.)),
             ((0., 1.), (0., 1.))), dtype=numpy.float32)
        level = 0.9
        vertices, normals, indices = marchingcubes.MarchingCubes(
            cube, level, invert_normals=False)
        self.assertAllClose(vertices[:, 0], level)
        self.assertAllClose(normals, (1., 0., 0.))
        self.assertEqual(len(indices), 2)

        # isosurface normal in Y, Z
        cube = numpy.array(
            (((0., 0.), (0., 0.)),
             ((0., 0.), (1., 1.))), dtype=numpy.float32)
        level = 0.5
        vertices, normals, indices = marchingcubes.MarchingCubes(cube, level)
        self.assertAllClose(normals[:, 0], 0.)
        self.assertEqual(len(indices), 2)


test_cases = (TestMarchingCubes,)


def suite():
    test_suite = unittest.TestSuite()
    for test_class in test_cases:
        test_suite.addTests(
            unittest.defaultTestLoader.loadTestsFromTestCase(test_class))
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
