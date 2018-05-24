# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
"""Tests for silx.uitls.number module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "24/05/2018"

import logging
import numpy
import unittest
from silx.utils import number

_logger = logging.getLogger(__name__)


class TestConversionTypes(unittest.TestCase):

    def testEmptyFail(self):
        self.assertRaises(ValueError, number.min_numerical_convertible_type, "")

    def testStringFail(self):
        self.assertRaises(ValueError, number.min_numerical_convertible_type, "a")

    def testInteger(self):
        dtype = number.min_numerical_convertible_type("1456")
        self.assertTrue(numpy.issubdtype(dtype, numpy.unsignedinteger))

    def testTrailledInteger(self):
        dtype = number.min_numerical_convertible_type(" \t\n\r1456\t\n\r")
        self.assertTrue(numpy.issubdtype(dtype, numpy.unsignedinteger))

    def testPositiveInteger(self):
        dtype = number.min_numerical_convertible_type("+1456")
        self.assertTrue(numpy.issubdtype(dtype, numpy.unsignedinteger))

    def testNegativeInteger(self):
        dtype = number.min_numerical_convertible_type("-1456")
        self.assertTrue(numpy.issubdtype(dtype, numpy.signedinteger))

    def testIntegerExponential(self):
        dtype = number.min_numerical_convertible_type("14e10")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testIntegerPositiveExponential(self):
        dtype = number.min_numerical_convertible_type("14e+10")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testIntegerNegativeExponential(self):
        dtype = number.min_numerical_convertible_type("14e-10")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testNumberDecimal(self):
        dtype = number.min_numerical_convertible_type("14.5")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testPositiveNumberDecimal(self):
        dtype = number.min_numerical_convertible_type("+14.5")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testNegativeNumberDecimal(self):
        dtype = number.min_numerical_convertible_type("-14.5")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testDecimal(self):
        dtype = number.min_numerical_convertible_type(".50")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testPositiveDecimal(self):
        dtype = number.min_numerical_convertible_type("+.5")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testNegativeDecimal(self):
        dtype = number.min_numerical_convertible_type("-.5")
        self.assertTrue(numpy.issubdtype(dtype, numpy.floating))

    def testFloat16(self):
        dtype = number.min_numerical_convertible_type("1.50")
        self.assertEqual(dtype, numpy.float16)

    def testFloat32(self):
        dtype = number.min_numerical_convertible_type("1400.50")
        self.assertEqual(dtype, numpy.float32)

    def testFloat64(self):
        dtype = number.min_numerical_convertible_type("10000.000010")
        self.assertEqual(dtype, numpy.float64)

    def testFloat80(self):
        if not hasattr(numpy, "float128"):
            self.skipTest("float-80bits not supported")
        dtype = number.min_numerical_convertible_type("1000000000.0000101")
        self.assertIn(dtype, (numpy.float128, numpy.longdouble))

    def testExponentialBiggerThanDecimal(self):
        dtype = number.min_numerical_convertible_type("14000.0e5")
        self.assertEqual(dtype, numpy.float64)

    def testExponentialSmallerThanNumber(self):
        dtype = number.min_numerical_convertible_type("14000.0e-5")
        self.assertEqual(dtype, numpy.float32)

    def testExponentialSmallerThanDecimal(self):
        dtype = number.min_numerical_convertible_type("0.00001e5")
        self.assertEqual(dtype, numpy.float32)

    def testExponentialBiggerThanNumber(self):
        dtype = number.min_numerical_convertible_type("0.00001e-5")
        self.assertEqual(dtype, numpy.float64)


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestConversionTypes))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
