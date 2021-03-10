# coding: utf-8
# /*##########################################################################
# Copyright (C) 2018 European Synchrotron Radiation Facility
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

from __future__ import division

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "14/05/2018"


import unittest

import numpy

from silx.math.calibration import NoCalibration, LinearCalibration, \
    ArrayCalibration, FunctionCalibration


X = numpy.array([3.14, 2.73, 1337])


class TestNoCalibration(unittest.TestCase):
    def setUp(self):
        self.calib = NoCalibration()

    def testIsAffine(self):
        self.assertTrue(self.calib.is_affine())

    def testSlope(self):
        self.assertEqual(self.calib.get_slope(), 1.)

    def testYIntercept(self):
        self.assertEqual(self.calib(0.),
                         0.)

    def testCall(self):
        self.assertTrue(numpy.array_equal(self.calib(X), X))


class TestLinearCalibration(unittest.TestCase):
    def setUp(self):
        self.y_intercept = 1.5
        self.slope = 2.5
        self.calib = LinearCalibration(y_intercept=self.y_intercept,
                                       slope=self.slope)

    def testIsAffine(self):
        self.assertTrue(self.calib.is_affine())

    def testSlope(self):
        self.assertEqual(self.calib.get_slope(), self.slope)

    def testYIntercept(self):
        self.assertEqual(self.calib(0.),
                         self.y_intercept)

    def testCall(self):
        self.assertTrue(numpy.array_equal(self.calib(X),
                                          self.y_intercept + self.slope * X))


class TestArrayCalibration(unittest.TestCase):
    def setUp(self):
        self.arr = numpy.array([45.2, 25.3, 666., -8.])
        self.calib = ArrayCalibration(self.arr)
        self.affine_calib = ArrayCalibration([0.1, 0.2, 0.3])

    def testIsAffine(self):
        self.assertFalse(self.calib.is_affine())
        self.assertTrue(self.affine_calib.is_affine())

    def testSlope(self):
        with self.assertRaises(AttributeError):
            self.calib.get_slope()
        self.assertEqual(self.affine_calib.get_slope(),
                         0.1)

    def testYIntercept(self):
        self.assertEqual(self.calib(0),
                         self.arr[0])

    def testCall(self):
        with self.assertRaises(ValueError):
            # X is an array with a different shape
            self.calib(X)

        with self.assertRaises(ValueError):
            # floats are not valid indices
            self.calib(3.14)

        self.assertTrue(
            numpy.array_equal(self.calib([1, 2, 3, 4]),
                              self.arr))

        for idx, value in enumerate(self.arr):
            self.assertEqual(self.calib(idx), value)


class TestFunctionCalibration(unittest.TestCase):
    def setUp(self):
        self.non_affine_fun = numpy.sin
        self.non_affine_calib = FunctionCalibration(self.non_affine_fun)

        self.affine_fun = lambda x: 52. * x + 0.01
        self.affine_calib = FunctionCalibration(self.affine_fun,
                                                is_affine=True)

    def testIsAffine(self):
        self.assertFalse(self.non_affine_calib.is_affine())
        self.assertTrue(self.affine_calib.is_affine())

    def testSlope(self):
        with self.assertRaises(AttributeError):
            self.non_affine_calib.get_slope()
        self.assertAlmostEqual(self.affine_calib.get_slope(),
                               52.)

    def testCall(self):
        for x in X:
            self.assertAlmostEqual(self.non_affine_calib(x),
                                   self.non_affine_fun(x))
            self.assertAlmostEqual(self.affine_calib(x),
                                   self.affine_fun(x))


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestNoCalibration))
    test_suite.addTest(loadTests(TestArrayCalibration))
    test_suite.addTest(loadTests(TestLinearCalibration))
    test_suite.addTest(loadTests(TestFunctionCalibration))
    return test_suite

if __name__ == '__main__':
    unittest.main(defaultTest="suite")
