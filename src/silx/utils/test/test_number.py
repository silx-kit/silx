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
__date__ = "05/06/2018"

import logging
import numpy
import unittest
import pkg_resources
from silx.utils import number
from silx.utils import testutils

_logger = logging.getLogger(__name__)


class TestConversionTypes(testutils.ParametricTestCase):

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

    def testMantissa16(self):
        dtype = number.min_numerical_convertible_type("1.50")
        self.assertEqual(dtype, numpy.float16)

    def testFloat32(self):
        dtype = number.min_numerical_convertible_type("-23.172")
        self.assertEqual(dtype, numpy.float32)

    def testMantissa32(self):
        dtype = number.min_numerical_convertible_type("1400.50")
        self.assertEqual(dtype, numpy.float32)

    def testMantissa64(self):
        dtype = number.min_numerical_convertible_type("10000.000010")
        self.assertEqual(dtype, numpy.float64)

    def testMantissa80(self):
        self.skipIfFloat80NotSupported()
        dtype = number.min_numerical_convertible_type("1000000000.00001013")

        if pkg_resources.parse_version(numpy.version.version) <= pkg_resources.parse_version("1.10.4"):
            # numpy 1.8.2 -> Debian 8
            # Checking a float128 precision with numpy 1.8.2 using abs(diff) is not working.
            # It looks like the difference is done using float64 (diff == 0.0)
            expected = (numpy.longdouble, numpy.float64)
        else:
            expected = (numpy.longdouble, )
        self.assertIn(dtype, expected)

    def testExponent32(self):
        dtype = number.min_numerical_convertible_type("14.0e30")
        self.assertEqual(dtype, numpy.float32)

    def testExponent64(self):
        dtype = number.min_numerical_convertible_type("14.0e300")
        self.assertEqual(dtype, numpy.float64)

    def testExponent80(self):
        self.skipIfFloat80NotSupported()
        dtype = number.min_numerical_convertible_type("14.0e3000")
        self.assertEqual(dtype, numpy.longdouble)

    def testFloat32ToString(self):
        value = str(numpy.float32(numpy.pi))
        dtype = number.min_numerical_convertible_type(value)
        self.assertIn(dtype, (numpy.float32, numpy.float64))

    def skipIfFloat80NotSupported(self):
        if number.is_longdouble_64bits():
            self.skipTest("float-80bits not supported")

    def testLosePrecisionUsingFloat80(self):
        self.skipIfFloat80NotSupported()
        if pkg_resources.parse_version(numpy.version.version) <= pkg_resources.parse_version("1.10.4"):
            self.skipTest("numpy > 1.10.4 expected")
        # value does not fit even in a 128 bits mantissa
        value = "1.0340282366920938463463374607431768211456"
        func = testutils.test_logging(number._logger.name, warning=1)
        func = func(number.min_numerical_convertible_type)
        dtype = func(value)
        self.assertIn(dtype, (numpy.longdouble, ))

    def testMillisecondEpochTime(self):
        datetimes = ['1465803236.495412',
                     '1465803236.999362',
                     '1465803237.504311',
                     '1465803238.009261',
                     '1465803238.512211',
                     '1465803239.016160',
                     '1465803239.520110',
                     '1465803240.026059',
                     '1465803240.529009']
        for datetime in datetimes:
            with self.subTest(datetime=datetime):
                dtype = number.min_numerical_convertible_type(datetime)
                self.assertEqual(dtype, numpy.float64)


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestConversionTypes))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
