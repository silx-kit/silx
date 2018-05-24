# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""Utilitary functions dealing with numbers.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "24/05/2018"

import numpy
import re


_parse_numeric_value = re.compile("^\s*[-+]?0*(\d+?)?(?:\.(\d+?)0*)?(?:[eE]([-+]?\d+))?\s*$")
"""Match integer or floating-point numbers"""


def min_numerical_convertible_type(string):
    """
    Parse the string and return the minimal numerical type which fit for a
    convertion.

    :param str string: Representation of a float/integer with text
    :raise ValueError: When the string is not a numerical value
    :retrun: A numpy numerical type
    """
    if string == "":
        raise ValueError("Not a numerical value")
    match = _parse_numeric_value.match(string)
    if match is None:
        raise ValueError("Not a numerical value")
    number, decimal, exponent = match.groups()
    if decimal is None and exponent is None:
        # TODO: We could find the int type without converting the number
        value = int(string)
        return numpy.min_scalar_type(value).type
    else:
        if number is None:
            number = ""
        if decimal is None:
            decimal = ""
        significante_size = len(number) + len(decimal)
        if exponent is None:
            if significante_size <= 3:
                # A float16 is accurate with about 3.311 digits
                return numpy.float16
            exponent = 0
        else:
            exponent = abs(int(exponent))

        if significante_size <= 7:
            # Expect at least float 32-bits
            expected_mantissa = 32
        elif significante_size <= 15:
            # Expect at least float 64-bits
            expected_mantissa = 64
        elif significante_size <= 19:
            # Expect at least float 80-bits
            expected_mantissa = 80
        elif significante_size <= 34:
            # Expect at least float 128-bits (real 128-bits, referenced as binary128)
            # Unsupported by numpy
            expected_mantissa = 128
        else:
            expected_mantissa = 999

        if exponent <= 37:
            # Up to 3.402823 * 10**38
            expected_exponent = 32
        elif exponent <= 307:
            # Up to 10**308
            expected_exponent = 64
        elif exponent <= 4932:
            # Up to 10**4932
            expected_exponent = 80
        else:
            expected_exponent = 999

        expected = max(expected_mantissa, expected_exponent)
        if expected >= 128:
            # Here we lose precision
            if hasattr(numpy, "longdouble"):
                return numpy.longdouble
            else:
                return numpy.float64

        if expected == 32:
            return numpy.float32
        elif expected == 64:
            return numpy.float64
        elif expected == 80:
            # A float 80-bits if available (padded using 96 or 128 bits)
            if hasattr(numpy, "longdouble"):
                return numpy.longdouble
            else:
                # Here we lose precision
                return numpy.float64
