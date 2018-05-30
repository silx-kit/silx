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
__date__ = "30/05/2018"

import numpy
import re
import logging


_logger = logging.getLogger(__name__)


if hasattr(numpy, "longdouble"):
    _biggest_float = numpy.longdouble
    _float_types = (numpy.float16, numpy.float32, numpy.float64, numpy.longdouble)
else:
    _biggest_float = numpy.float64
    _float_types = (numpy.float16, numpy.float32, numpy.float64)


_parse_numeric_value = re.compile("^\s*[-+]?0*(\d+?)?(?:\.(\d+))?(?:[eE]([-+]?\d+))?\s*$")


def min_numerical_convertible_type(string, check_accuracy=True):
    """
    Parse the string and return the minimal numerical type which fit for a
    convertion.

    :param str string: Representation of a float/integer with text
    :param bool check_accuracy: If true, a warning is pushed on the logger
        in case there is a lose of accuracy.
    :raise ValueError: When the string is not a numerical value
    :retrun: A numpy numerical type
    """
    # Try integer
    try:
        value = int(string)
    except ValueError:
        pass
    else:
        return numpy.min_scalar_type(value).type

    # Try floating-point
    try:
        value = _biggest_float(string)
    except ValueError:
        pass
    else:
        for numpy_type in _float_types:
            if numpy.can_cast(value, numpy_type, casting="safe"):
                break

        if check_accuracy and numpy_type == _biggest_float:
            match = _parse_numeric_value.match(string)
            if match is None:
                assert(False)
            number, decimal, _exponent = match.groups()
            if number is None:
                number = ""
            if decimal is None:
                decimal = ""
            expected = number + decimal
            # This format the number without python convertion to float64
            result = numpy.array2string(value, precision=len(number) + len(decimal), floatmode="fixed")
            result = result.replace(".", "").replace("-", "")
            if not result.startswith(expected):
                _logger.warning("Not able to convert '%s' using %s without losing precision", string, _biggest_float)

        return numpy_type

    raise ValueError("Not a numerical value")
