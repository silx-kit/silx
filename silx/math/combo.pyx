# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""This module provides combination of statistics as single operation.

For now it provides min/max (and optionally positive min) and indices
of first occurences in a single pass.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "20/12/2016"

cimport cython
from libc.math cimport isnan

import numpy


ctypedef fused _number:
    float
    double
    char
    short
    int
    long
    unsigned char
    unsigned short
    unsigned int
    unsigned long


class _MinMaxResult(object):
    """Object storing result from :func:`min_max`"""

    def __init__(self, minimum, min_pos, maximum,
                 argmin, argmin_pos, argmax):
        self._minimum = minimum
        self._min_positive = min_pos
        self._maximum = maximum

        self._argmin = argmin
        self._argmin_positive = argmin_pos
        self._argmax = argmax

    minimum = property(
        lambda self: self._minimum,
        doc="Minimum value of the array")
    maximum = property(
        lambda self: self._maximum,
        doc="Maximum value of the array")

    argmin = property(
        lambda self: self._argmin,
        doc="Index of the first occurence of the minimum value")
    argmax = property(
        lambda self: self._argmax,
        doc="Index of the first occurence of the maximum value")

    min_positive = property(
        lambda self: self._min_positive,
        doc="""Strictly positive minimum value
        
        It is None if no value is strictly positive.
        """)
    argmin_positive = property(
        lambda self: self._argmin_positive,
        doc="""Index of the strictly positive minimum value.

        It is None if no value is strictly positive.
        It is the index of the first occurence.""")

    def __getitem__(self, key):
        if key == 0:
            return self.minimum
        elif key == 1:
            return self.maximum
        else:
            raise IndexError("Index out of range")


@cython.boundscheck(False)
@cython.wraparound(False)
def _min_max(_number[:] data, bint min_positive=False):
    """See :func:`min_max` for documentation."""
    cdef:
        _number value, minimum, minpos, maximum
        unsigned int length
        unsigned int index = 0
        unsigned int min_index = 0
        unsigned int min_pos_index = 0
        unsigned int max_index = 0

    length = len(data)

    if length == 0:
        raise ValueError('Zero-size array')

    with nogil:
        # Init starting values
        value = data[0]
        minimum = value
        maximum = value
        if min_positive and value > 0:
            min_pos = value
        else:
            min_pos = 0

        if _number in cython.floating:
            # For floating, loop until first not NaN value
            for index in range(length):
                value = data[index]
                if not isnan(value):
                    minimum = value
                    min_index = index
                    maximum = value
                    max_index = index
                    break

        if not min_positive:
            for index in range(index, length):
                value = data[index]
                if value > maximum:
                    maximum = value
                    max_index = index
                elif value < minimum:
                    minimum = value
                    min_index = index

        else:
            # Loop until min_pos is defined
            for index in range(index, length):
                value = data[index]
                if value > maximum:
                    maximum = value
                    max_index = index
                elif value < minimum:
                    minimum = value
                    min_index = index

                if value > 0:
                    min_pos = value
                    min_pos_index = index
                    break

            # Loop until the end
            for index in range(index+1, length):
                value = data[index]
                if value > maximum:
                    maximum = value
                    max_index = index
                else:
                    if value < minimum:
                        minimum = value
                        min_index = index

                    if value > 0 and value < min_pos:
                        min_pos = value
                        min_pos_index = index

    return _MinMaxResult(minimum,
                         min_pos if min_pos > 0 else None,
                         maximum,
                         min_index,
                         min_pos_index if min_pos > 0 else None,
                         max_index)


@cython.embedsignature(True)
def min_max(data not None, bint min_positive=False):
    """Returns min, max and optionally strictly positive min of data.

    It also computes the indices of first occurence of min/max.

    NaNs are ignored while computing min/max unless all data is NaNs,
    in which case returned min/max are NaNs.

    Examples:

    >>> import numpy
    >>> data = numpy.arange(10)

    Usage as a function returning min and max:

    >>> min_, max_ = min_max(data)

    Usage as a function returning a result object to access all information:

    >>> result = min_max(data)  # Do not get positive min
    >>> result.minimum, result.argmin
    0, 0
    >>> result.maximum, result.argmax
    9, 10
    >>> result.min_positive, result.argmin_positive  # Not computed
    None, None

    Getting strictly positive min information:

    >>> result = min_max(data, min_positive=True)
    >>> result.min_positive, result.argmin_positive  # Computed
    1, 1

    :param data: Array-like dataset
    :param bool min_positive: True to compute the positive min and argmin
                              Default: False.
    :returns: An object with minimum, maximum and min_positive attributes
              and the indices of first occurence in the flattened data:
              argmin, argmax and argmin_positive attributes.
              If all data is <= 0 or min_positive argument is False, then
              min_positive and argmin_positive are None.
    :raises: ValueError if data is empty
    """
    return _min_max(numpy.asanyarray(data).ravel(), min_positive)

