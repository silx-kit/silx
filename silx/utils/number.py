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


_parse_numeric_value = re.compile("[-+]?(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?")
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
    if match.end() != len(string):
        raise ValueError("Not a numerical value")
    number, decimal, exponent = match.groups()
    if decimal is None and exponent is None:
        # TODO: We could find the int type without converting the number
        value = int(string)
        return numpy.min_scalar_type(value).type
    else:
        if number is None:
            number = 0
        else:
            number = len(number)
        if decimal is None:
            decimal = 0
        else:
            decimal = len(decimal)
        if exponent is None:
            exponent = 0
        else:
            exponent = int(exponent)

        if exponent > 0:
            digits = number + max(decimal, exponent)
        elif exponent < 0:
            exponent = -exponent
            digits = max(number, exponent) + decimal
        else:
            digits = number + decimal

        if digits <= 3:
            # A float16 is accurate with about 3.311 digits
            return numpy.float16
        if digits <= 7:
            # A float32 is accurate with about 7 digits
            return numpy.float32
        elif digits <= 16:
            # A float64 is accurate with about 16 digits
            return numpy.float64
        else:
            # A float 80-bits if available (padded using 96 or 128 bits)
            if hasattr(numpy, "longdouble"):
                return numpy.longdouble
            else:
                return numpy.float64
