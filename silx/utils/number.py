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
__date__ = "05/06/2018"

import numpy
import re
import logging


_logger = logging.getLogger(__name__)


_biggest_float = None

if hasattr(numpy, "longdouble"):
    finfo = numpy.finfo(numpy.longdouble)
    # The bit for sign is missing here
    bits = finfo.nexp + finfo.nmant
    if bits > 64:
        _biggest_float = numpy.longdouble
        # From bigger to smaller
        _float_types = (numpy.longdouble, numpy.float64, numpy.float32, numpy.float16)
if _biggest_float is None:
    _biggest_float = numpy.float64
    # From bigger to smaller
    _float_types = (numpy.float64, numpy.float32, numpy.float16)


_parse_numeric_value = re.compile(r"^\s*[-+]?0*(\d+?)?(?:\.(\d+))?(?:[eE]([-+]?\d+))?\s*$")


def is_longdouble_64bits():
    """Returns true if the system uses floating-point 64bits for it's
    long double type.

    .. note:: Comparing `numpy.longdouble` and `numpy.float64` on Windows is not
        possible (or at least not will all the numpy version)
    """
    return _biggest_float == numpy.float64


def min_numerical_convertible_type(string, check_accuracy=True):
    """
    Parse the string and try to return the smallest numerical type to use for
    a safe conversion. It has some known issues: precission loss.

    :param str string: Representation of a float/integer with text
    :param bool check_accuracy: If true, a warning is pushed on the logger
        in case there is a loss of accuracy.
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
        # It's an integer
        # TODO: We could find the int type without converting the number
        value = int(string)
        return numpy.min_scalar_type(value).type

    # Try floating-point
    try:
        value = _biggest_float(string)
    except ValueError:
        raise ValueError("Not a numerical value")

    if number is None:
        number = ""
    if decimal is None:
        decimal = ""
    if exponent is None:
        exponent = "0"

    nb_precision_digits = int(exponent) - len(decimal) - 1
    precision = _biggest_float(10) ** nb_precision_digits * 1.2
    previous_type = _biggest_float
    for numpy_type in _float_types:
        if numpy_type == _biggest_float:
            # value was already casted using the bigger type
            continue
        reduced_value = numpy_type(value)
        if not numpy.isfinite(reduced_value):
            break
        # numpy isclose(atol=is not accurate enough)
        diff = value - reduced_value
        # numpy 1.8.2 looks to do the substraction using float64...
        # we lose precision here
        diff = numpy.abs(diff)
        if diff > precision:
            break
        previous_type = numpy_type

    # It's the smaller float type which fit with enougth precision
    numpy_type = previous_type

    if check_accuracy and numpy_type == _biggest_float:
        # Check the precision using the original string
        expected = number + decimal
        # This format the number without python convertion
        try:
            result = numpy.array2string(value, precision=len(number) + len(decimal), floatmode="fixed")
        except TypeError:
            # numpy 1.8.2 do not have floatmode argument
            _logger.warning("Not able to check accuracy of the conversion of '%s' using %s", string, _biggest_float)
            return numpy_type

        result = result.replace(".", "").replace("-", "")
        if not result.startswith(expected):
            _logger.warning("Not able to convert '%s' using %s without losing precision", string, _biggest_float)

    return numpy_type
