# /*##########################################################################
#
# Copyright (c) 2016-2025 European Synchrotron Radiation Facility
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
of first occurrences (i.e., argmin/argmax) in a single pass.
"""

__authors__ = ["T. Vincent", "Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "09/09/2025"

cimport cython
from libc.math cimport isnan, isfinite, INFINITY, fabs, sqrt
from typing import TypeVar, Generic

import numpy


# All supported types
ctypedef fused _number:
    float
    double
    long double
    signed char
    signed short
    signed int
    signed long
    signed long long
    unsigned char
    unsigned short
    unsigned int
    unsigned long
    unsigned long long

# All supported floating types:
# cython.floating + long double
ctypedef fused _floating:
    float
    double
    long double


T = TypeVar('T')


class _MinMaxResult(Generic[T]):
    """Result from :func:`min_max`"""

    def __init__(
        self, minimum, min_pos, maximum,
        argmin, argmin_pos, argmax
    ):
        self._minimum = minimum
        self._min_positive = min_pos
        self._maximum = maximum

        self._argmin = argmin
        self._argmin_positive = argmin_pos
        self._argmax = argmax

    @property
    def  minimum(self) -> T:
        """Minimum value of the array"""
        return self._minimum

    @property
    def maximum(self) -> T:
        """Maximum value of the array"""
        return self._maximum

    @property
    def argmin(self) -> int:
        """Index of the first occurrence of the minimum value"""
        return self._argmin

    @property
    def argmax(self) -> int:
        """Index of the first occurrence of the maximum value"""
        return self._argmax

    @property
    def min_positive(self) -> T | None:
        """
        Strictly positive minimum value

        It is None if no value is strictly positive.
        """
        return self._min_positive

    @property
    def argmin_positive(self) -> int | None:
        """
        Index of the strictly positive minimum value.

        It is None if no value is strictly positive.
        It is the index of the first occurrence.
        """
        return self._argmin_positive

    def __getitem__(self, key: int):
        if key == 0:
            return self.minimum
        elif key == 1:
            return self.maximum
        else:
            raise IndexError("Index out of range")


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def _min_max(_number[::1] data, bint min_positive=False):
    """:func:`min_max` implementation including infinite values

    See :func:`min_max` for documentation.
    """
    cdef:
        _number value, minimum, min_pos, maximum
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

        if _number in _floating:
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
            for index in range(index + 1, length):
                value = data[index]
                if value > maximum:
                    maximum = value
                    max_index = index
                else:
                    if value < minimum:
                        minimum = value
                        min_index = index

                    if 0 < value < min_pos:
                        min_pos = value
                        min_pos_index = index

    return _MinMaxResult(
        minimum,
        min_pos if min_pos > 0 else None,
        maximum,
        min_index,
        min_pos_index if min_pos > 0 else None,
        max_index)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def _finite_min_max(_floating[::1] data, bint min_positive=False):
    """:func:`min_max` implementation for floats skipping infinite values

    See :func:`min_max` for documentation.
    """
    cdef:
        _floating value, minimum, min_pos, maximum
        unsigned int length
        unsigned int index = 0
        unsigned int min_index = 0
        unsigned int min_pos_index = 0
        unsigned int max_index = 0

    length = len(data)

    if length == 0:
        raise ValueError('Zero-size array')

    with nogil:
        minimum = INFINITY
        maximum = -INFINITY
        min_pos = INFINITY

        if not min_positive:
            for index in range(length):
                value = data[index]
                if isfinite(value):
                    if value > maximum:
                        maximum = value
                        max_index = index
                    if value < minimum:
                        minimum = value
                        min_index = index

        else:
            for index in range(index, length):
                value = data[index]
                if isfinite(value):
                    if value > maximum:
                        maximum = value
                        max_index = index
                    if value < minimum:
                        minimum = value
                        min_index = index

                    if 0. < value < min_pos:
                        min_pos = value
                        min_pos_index = index

    return _MinMaxResult(
        minimum if isfinite(minimum) else None,
        min_pos if isfinite(min_pos) else None,
        maximum if isfinite(maximum) else None,
        min_index if isfinite(minimum) else None,
        min_pos_index if isfinite(min_pos) else None,
        max_index if isfinite(maximum) else None)


def min_max(data not None, bint min_positive=False, bint finite=False) -> _MinMaxResult[int] | _MinMaxResult[float]:
    """Returns min, max and optionally strictly positive min of data.

    It also computes the indices of first occurrence of min/max.

    NaNs are ignored while computing min/max unless all data is NaNs,
    in which case returned min/max are NaNs.

    The result data type is that of the input data, except for the following cases.
    For input using non-native bytes order, the result is returned as native
    floating-point or integers. For input using 16-bits floating-point,
    the result is returned as 32-bits floating-point.

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

    If *finite* is True, min/max information is computed only from finite data.
    Then, all result fields (include minimum and maximum) can be None
    when all data is infinity or NaN.

    :param data: Array-like dataset
    :param bool min_positive: True to compute the positive min and argmin
        Default: False.
    :param bool finite: True to compute min/max from finite data only
        Default: False.
    :returns: An object with minimum, maximum and min_positive attributes
        and the indices of first occurrence in the flattened data:
        argmin, argmax and argmin_positive attributes.
        If all data is <= 0 or min_positive argument is False, then
        min_positive and argmin_positive are None.
    :raises: ValueError if data is empty
    """
    data = numpy.asarray(data)
    native_endian_dtype = data.dtype.newbyteorder('N')
    if native_endian_dtype.kind == 'f' and native_endian_dtype.itemsize == 2:
        # Use native float32 instead of float16
        native_endian_dtype = "=f4"
    data = numpy.ascontiguousarray(data, dtype=native_endian_dtype).ravel()
    if finite and data.dtype.kind == 'f':
        return _finite_min_max(data, min_positive)
    else:
        return _min_max(data, min_positive)


cdef inline bint _is_valid(double value,
                           char mask_value,
                           bint do_dummy,
                           double dummy,
                           double delta_dummy) noexcept nogil:
    """return True if the value is valid"""
    cdef:
        bint rval=isfinite(value) and not mask_value
    if do_dummy:
        if delta_dummy:
            rval &= fabs(value-dummy)>delta_dummy
        else:
            rval &= value!=dummy

    return rval


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mean_std(data,
             ddof=0,
             mask=None,
             dummy=None,
             delta_dummy=0):
    """Computes mean and estimation of std in a single pass.

    Based on formula #12, #13 and #28 from :
    https://ds.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf

    :param data: Array-like dataset
    :param int ddof:
       Means Delta Degrees of Freedom.
       The divisor used in calculations is data.size - ddof.
       Default: 0 (as in numpy.std).
    :param mask: array with 0 for valid values, same size as data
    :param dummy: dynamic mask for value=dummy
    :param delta_dummy: dynamic mask for abs(value-dummy)<=delta_dummy
    :returns: A tuple: (mean, std)
    :raises: ValueError if data is empty"""

    cdef:
        unsigned int length, index
        bint do_mask, do_dummy
        double value, delta, X, XX, cnt, new_cnt
        double mean, variance, standard_deviation
        double cdummy, cdelta_dummy
        char[::1] cmask
        double[::1] cdata = numpy.ascontiguousarray(data, dtype=numpy.float64).ravel()

    length = cdata.shape[0]
    if mask is None:
        do_mask = False
    else:
        do_mask = True
        cmask = numpy.ascontiguousarray(mask, dtype=numpy.int8).ravel()
    if dummy is None:
        do_dummy = False
        cdummy = delta_dummy = 0.0
    else:
        do_dummy = True
        cdummy = float(dummy)
        cdelta_dummy = float(delta_dummy)

    if length == 0:
        raise ValueError('Zero-size array')

    X = 0.0
    XX = 0.0
    cnt = 0.0

    with nogil:
        for index in range(length):
            value = cdata[index]
            if _is_valid(value, cmask[index] if do_mask else 0, do_dummy, cdummy, cdelta_dummy):
                new_cnt = cnt + 1.0
                if cnt:
                    delta = X-cnt*value
                    XX += delta*delta/(cnt*new_cnt)
                X += value
                cnt = new_cnt

    mean = X / cnt
    if length <= ddof:
        standard_deviation = float('nan')
    else:
        variance =  XX / (cnt - ddof)
        standard_deviation = sqrt(variance)

    return (mean, standard_deviation)
