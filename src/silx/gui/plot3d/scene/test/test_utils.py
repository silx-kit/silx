# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import unittest
from silx.utils.testutils import ParametricTestCase

import numpy

from silx.gui.plot3d.scene import utils


# angleBetweenVectors #########################################################

class TestAngleBetweenVectors(ParametricTestCase):

    TESTS = {  # name: (refvector, vectors, norm, refangles)
        'single vector':
            ((1., 0., 0.), (1., 0., 0.), (0., 0., 1.), 0.),
        'single vector, no norm':
            ((1., 0., 0.), (1., 0., 0.), None, 0.),

        'with orthogonal norm':
            ((1., 0., 0.),
             ((1., 0., 0.), (0., 1., 0.), (-1., 0., 0.), (0., -1., 0.)),
             (0., 0., 1.),
             (0., 90., 180., 270.)),

        'with coplanar norm':  # = similar to no norm
            ((1., 0., 0.),
             ((1., 0., 0.), (0., 1., 0.), (-1., 0., 0.), (0., -1., 0.)),
             (1., 0., 0.),
             (0., 90., 180., 90.)),

        'without norm':
            ((1., 0., 0.),
             ((1., 0., 0.), (0., 1., 0.), (-1., 0., 0.), (0., -1., 0.)),
             None,
             (0., 90., 180., 90.)),

        'not unit vectors':
            ((2., 2., 0.), ((1., 1., 0.), (1., -1., 0.)), None, (0., 90.)),
    }

    def testAngleBetweenVectorsFunction(self):
        for name, params in self.TESTS.items():
            refvector, vectors, norm, refangles = params
            with self.subTest(name):
                refangles = numpy.radians(refangles)

                refvector = numpy.array(refvector)
                vectors = numpy.array(vectors)
                if norm is not None:
                    norm = numpy.array(norm)

                testangles = utils.angleBetweenVectors(
                    refvector, vectors, norm)

                self.assertTrue(
                    numpy.allclose(testangles, refangles, atol=1e-5))


# Plane #######################################################################

class AssertNotificationContext(object):
    """Context that checks if an event.Notifier is sending events."""

    def __init__(self, notifier, count=1):
        """Initializer.

        :param event.Notifier notifier: The notifier to test.
        :param int count: The expected number of calls.
        """
        self._notifier = notifier
        self._callCount = None
        self._count = count

    def __enter__(self):
        self._callCount = 0
        self._notifier.addListener(self._callback)

    def __exit__(self, exc_type, exc_value, traceback):
        # Do not return True so exceptions are propagated
        self._notifier.removeListener(self._callback)
        assert self._callCount == self._count
        self._callCount = None

    def _callback(self, *args, **kwargs):
        self._callCount += 1


class TestPlaneParameters(ParametricTestCase):
    """Test Plane.parameters read/write and notifications."""

    PARAMETERS = {
        'unit normal': (1., 0., 0., 1.),
        'not unit normal': (1., 1., 0., 1.),
        'd = 0': (1., 0., 0., 0.)
    }

    def testParameters(self):
        """Check parameters read/write and notification."""
        plane = utils.Plane()

        for name, parameters in self.PARAMETERS.items():
            with self.subTest(name, parameters=parameters):
                with AssertNotificationContext(plane):
                    plane.parameters = parameters

                # Plane parameters are converted to have a unit normal
                normparams = parameters / numpy.linalg.norm(parameters[:3])
                self.assertTrue(numpy.allclose(plane.parameters, normparams))

    ZEROS_PARAMETERS = (
        (0., 0., 0., 0.),
        (0., 0., 0., 1.)
    )

    ZEROS = 0., 0., 0., 0.

    def testParametersNoPlane(self):
        """Test Plane.parameters with ||normal|| == 0 ."""
        plane = utils.Plane()
        plane.parameters = self.ZEROS

        for parameters in self.ZEROS_PARAMETERS:
            with self.subTest(parameters=parameters):
                with AssertNotificationContext(plane, count=0):
                    plane.parameters = parameters
                self.assertTrue(
                    numpy.allclose(plane.parameters, self.ZEROS, 0., 0.))


# unindexArrays ###############################################################

class TestUnindexArrays(ParametricTestCase):
    """Test unindexArrays function."""

    def testBasicModes(self):
        """Test for modes: points, lines and triangles"""
        indices = numpy.array((1, 2, 0))
        arrays = (numpy.array((0., 1., 2.)),
                  numpy.array(((0, 0), (1, 1), (2, 2))))
        refresults = (numpy.array((1., 2., 0.)),
                      numpy.array(((1, 1), (2, 2), (0, 0))))

        for mode in ('points', 'lines', 'triangles'):
            with self.subTest(mode=mode):
                testresults = utils.unindexArrays(mode, indices, *arrays)
                for ref, test in zip(refresults, testresults):
                    self.assertTrue(numpy.equal(ref, test).all())

    def testPackedLines(self):
        """Test for modes: line_strip, loop"""
        indices = numpy.array((1, 2, 0))
        arrays = (numpy.array((0., 1., 2.)),
                  numpy.array(((0, 0), (1, 1), (2, 2))))
        results = {
            'line_strip': (
                numpy.array((1., 2., 2., 0.)),
                numpy.array(((1, 1), (2, 2), (2, 2), (0, 0)))),
            'loop': (
                numpy.array((1., 2., 2., 0., 0., 1.)),
                numpy.array(((1, 1), (2, 2), (2, 2), (0, 0), (0, 0), (1, 1)))),
        }

        for mode, refresults in results.items():
            with self.subTest(mode=mode):
                testresults = utils.unindexArrays(mode, indices, *arrays)
                for ref, test in zip(refresults, testresults):
                    self.assertTrue(numpy.equal(ref, test).all())

    def testPackedTriangles(self):
        """Test for modes: triangle_strip, fan"""
        indices = numpy.array((1, 2, 0, 3))
        arrays = (numpy.array((0., 1., 2., 3.)),
                  numpy.array(((0, 0), (1, 1), (2, 2), (3, 3))))
        results = {
            'triangle_strip': (
                numpy.array((1., 2., 0., 2., 0., 3.)),
                numpy.array(((1, 1), (2, 2), (0, 0), (2, 2), (0, 0), (3, 3)))),
            'fan': (
                numpy.array((1., 2., 0., 1., 0., 3.)),
                numpy.array(((1, 1), (2, 2), (0, 0), (1, 1), (0, 0), (3, 3)))),
        }

        for mode, refresults in results.items():
            with self.subTest(mode=mode):
                testresults = utils.unindexArrays(mode, indices, *arrays)
                for ref, test in zip(refresults, testresults):
                    self.assertTrue(numpy.equal(ref, test).all())

    def testBadIndices(self):
        """Test with negative indices and indices higher than array length"""
        arrays = numpy.array((0, 1)), numpy.array((0, 1, 2))

        # negative indices
        with self.assertRaises(AssertionError):
            utils.unindexArrays('points', (-1, 0), *arrays)

        # Too high indices
        with self.assertRaises(AssertionError):
            utils.unindexArrays('points', (0, 10), *arrays)


# triangleNormals #############################################################

class TestTriangleNormals(ParametricTestCase):
    """Test triangleNormals function."""

    def test(self):
        """Test for modes: points, lines and triangles"""
        positions = numpy.array(
            ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.),  # normal = Z
             (1., 1., 1.), (1., 2., 3.), (4., 5., 6.),  # Random triangle
             # Degenerated triangles:
             (0., 0., 0.), (1., 0., 0.), (2., 0., 0.),  # Colinear points
             (1., 1., 1.), (1., 1., 1.), (1., 1., 1.),  # All same point
             ),
            dtype='float32')

        normals = numpy.array(
            ((0., 0., 1.),
             (-0.40824829,  0.81649658, -0.40824829),
             (0., 0., 0.),
             (0., 0., 0.)),
            dtype='float32')

        testnormals = utils.trianglesNormal(positions)
        self.assertTrue(numpy.allclose(testnormals, normals))
