# /*##########################################################################
# Copyright (C) 2018-2026 European Synchrotron Radiation Facility
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
"""Tests of the calibration module"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "06/02/2026"


import numpy
import pytest

from silx.math.calibration import (
    NoCalibration,
    LinearCalibration,
    ArrayCalibration,
    FunctionCalibration,
)


def testNoCalibration():
    calib = NoCalibration()
    assert calib.is_affine()
    assert calib.get_slope() == 1.0
    assert calib(0.0) == 0.0

    values = numpy.array([3.14, 2.73, 1337])
    assert numpy.array_equal(calib(values), values)


def testLinearCalibration():
    y_intercept = 1.5
    slope = 2.5
    calib = LinearCalibration(y_intercept=y_intercept, slope=slope)

    assert calib.is_affine()
    assert calib.get_slope() == slope
    assert calib(0.0) == y_intercept
    values = numpy.array([3.14, 2.73, 1337])
    assert numpy.array_equal(calib(values), y_intercept + slope * values)


class TestArrayCalibration:
    def testAffineArray(self):
        calib = ArrayCalibration([0.1, 0.2, 0.3])
        assert calib.is_affine()
        assert calib.get_slope() == 0.1

    def testAffineArrayWithHighDynamicRange(self):
        array = numpy.linspace(1e-12, 4, 100000, dtype=numpy.float32)
        calib = ArrayCalibration(array)
        assert calib.is_affine()
        assert numpy.isclose(calib.get_slope(), numpy.mean(numpy.diff(array)))

    def testNotAffineArray(self):
        array = numpy.array([45.2, 25.3, 666.0, -8.0])
        calib = ArrayCalibration(array)

        assert not calib.is_affine()

        with pytest.raises(AttributeError):
            calib.get_slope()

        assert calib(0) == array[0]

        values = numpy.array([3.14, 2.73, 1337])
        with pytest.raises(ValueError):
            # values is an array with a different shape
            calib(values)

        with pytest.raises(ValueError):
            # floats are not valid indices
            calib(3.14)

        assert numpy.array_equal(calib([1, 2, 3, 4]), array)

        for idx, value in enumerate(array):
            assert calib(idx) == value

    def testEmptyArray(self):
        with pytest.raises(ValueError):
            ArrayCalibration(numpy.array([]))

    def testSingleElementArray(self):
        calib = ArrayCalibration(numpy.array([1]))
        assert not calib.is_affine()


class TestFunctionCalibration:
    def testAffineFunction(self):
        def affine_function(x):
            return 52.0 * x + 0.01

        calib = FunctionCalibration(affine_function, is_affine=True)

        assert calib.is_affine()
        assert numpy.isclose(calib.get_slope(), 52.0)
        for value in numpy.array([3.14, 2.73, 1337]):
            assert numpy.isclose(calib(value), affine_function(value))

    def testNotAffineFunction(self):
        non_affine_function = numpy.sin
        calib = FunctionCalibration(non_affine_function)

        assert not calib.is_affine()
        with pytest.raises(AttributeError):
            calib.get_slope()
        for x in numpy.array([3.14, 2.73, 1337]):
            assert numpy.isclose(calib(x), non_affine_function(x))
