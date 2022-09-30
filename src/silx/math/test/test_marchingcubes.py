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
__date__ = "17/01/2018"

import unittest

import numpy

from silx.utils.testutils import ParametricTestCase

from silx.math import marchingcubes


class TestMarchingCubes(ParametricTestCase):
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

        # Cube array dimensions: shape = (dim 0, dim 1, dim2)
        #
        #      dim 0 (Z)
        #        ^
        #        |
        #      4 +------+ 5
        #       /|     /|
        #      / |    / |
        #   6 +------+ 7|
        #     |  |   |  |
        #     |0 +---|--+ 1 -> dim 2 (X)
        #     | /    | /
        #     |/     |/
        #   2 +------+ 3
        #    /
        #   dim 1 (Y)

        # isosurface perpendicular to dim 0 (Z)
        cube = numpy.array(
            (((0., 0.), (0., 0.)),
             ((1., 1.), (1., 1.))), dtype=numpy.float32)
        level = 0.5
        vertices, normals, indices = marchingcubes.MarchingCubes(
            cube, level, invert_normals=False)
        self.assertAllClose(vertices[:, 0], level)
        self.assertAllClose(normals, (1., 0., 0.))
        self.assertEqual(len(indices), 2)

        # isosurface perpendicular to dim 1 (Y)
        cube = numpy.array(
            (((0., 0.), (1., 1.)),
             ((0., 0.), (1., 1.))), dtype=numpy.float32)
        level = 0.2
        vertices, normals, indices = marchingcubes.MarchingCubes(cube, level)
        self.assertAllClose(vertices[:, 1], level)
        self.assertAllClose(normals, (0., -1., 0.))
        self.assertEqual(len(indices), 2)

        # isosurface perpendicular to dim 2 (X)
        cube = numpy.array(
            (((0., 1.), (0., 1.)),
             ((0., 1.), (0., 1.))), dtype=numpy.float32)
        level = 0.9
        vertices, normals, indices = marchingcubes.MarchingCubes(
            cube, level, invert_normals=False)
        self.assertAllClose(vertices[:, 2], level)
        self.assertAllClose(normals, (0., 0., 1.))
        self.assertEqual(len(indices), 2)

        # isosurface normal in dim1, dim 0 (Y, Z) plane
        cube = numpy.array(
            (((0., 0.), (0., 0.)),
             ((0., 0.), (1., 1.))), dtype=numpy.float32)
        level = 0.5
        vertices, normals, indices = marchingcubes.MarchingCubes(cube, level)
        self.assertAllClose(normals[:, 2], 0.)
        self.assertEqual(len(indices), 2)

    def test_sampling(self):
        """Test different sampling, comparing to reference without sampling"""
        isolevel = 0.5
        size = 9
        chessboard = numpy.zeros((size, size, size), dtype=numpy.float32)
        chessboard.reshape(-1)[::2] = 1  # OK as long as dimensions are odd

        ref_result = marchingcubes.MarchingCubes(chessboard, isolevel)

        samplings = [
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 2, 2),
            (3, 3, 3),
            (1, 3, 1),
            (1, 1, 3),
        ]

        for sampling in samplings:
            with self.subTest(sampling=sampling):
                sampling = numpy.array(sampling)

                data = 1e6 * numpy.ones(
                    sampling * size, dtype=numpy.float32)
                # Copy ref chessboard in data according to sampling
                data[::sampling[0], ::sampling[1], ::sampling[2]] = chessboard

                result = marchingcubes.MarchingCubes(data, isolevel,
                                                     sampling=sampling)
                # Compare vertices normalized with shape
                self.assertAllClose(
                    ref_result.get_vertices() / ref_result.shape,
                    result.get_vertices() / result.shape,
                    atol=0., rtol=0.)

                # Compare normals
                # This comparison only works for normals aligned with axes
                # otherwise non uniform sampling would make different normals
                self.assertAllClose(ref_result.get_normals(),
                                    result.get_normals(),
                                    atol=0., rtol=0.)

                self.assertAllClose(ref_result.get_indices(),
                                    result.get_indices(),
                                    atol=0., rtol=0.)
