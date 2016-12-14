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
"""Numpy-array-like object, implementing common numpy array features
for datasets or nested sequences, while trying to avoid copying data."""

from __future__ import absolute_import, print_function, division
import numpy
import sys

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "13/12/2016"



def is_array(obj):
    """Return True if object implements necessary attributes to be
    considered similar to a numpy array.

    Attributes needed are "shape" and "dtype".

    :param obj: Array-like object (numpy array, h5py dataset...)
    :return: boolean
    """
    # add more required attribute if necessary
    for attr in ("shape", "dtype"):
        if not hasattr(obj, attr):
            return False
    return True


def is_nested_sequence(obj):
    """Return True if object is a nested sequence.

    Numpy arrays, h5py datasets, lists/tuples of lists/tuples are
    considered to be nested sequences.

    :param obj: nested sequence (numpy array, h5py dataset...)
    :return: boolean"""
    if is_array(obj):
        return True
    if hasattr(obj, "__len__"):
        return True
    return False


def get_shape(array_like):
    """Return shape of an array like object.

    In case the object is a nested sequence (list of lists, tuples...),
    the size of each dimension is assumed to be uniform, and is deduced from
    the length of the first sequence.

    :param array_like: Array like object: numpy array, hdf5 dataset,
        multi-dimensional sequence
    :return: Shape of array, as a tuple of integers
    """
    if hasattr(array_like, "shape"):
        return array_like.shape

    shape = []
    subsequence = array_like
    while hasattr(subsequence, "__len__"):
        shape.append(len(subsequence))
        subsequence = subsequence[0]

    return tuple(shape)


def get_dtype(array_like):
    """Return dtype of an array like object.

    In the case of a nested sequence, the type of the first value
    is inspected.

    :param array_like: Array like object: numpy array, hdf5 dataset,
        multi-dimensional nested sequence
    :return: numpy dtype of object
    """
    if hasattr(array_like, "dtype"):
        return array_like.dtype

    # don't try the while loop with strings or bytes (infinite loop)
    string_types =  (str,) if sys.version_info[0] == 3 else (basestring,)
    binary_types = (bytes,) if sys.version_info[0] == 3 else (str,)
    if isinstance(array_like, string_types + binary_types):
        return numpy.dtype(array_like)

    subsequence = array_like
    while hasattr(subsequence, "__len__"):
        subsequence = subsequence[0]

    return numpy.dtype(type(subsequence))


def get_array_type(array_like):
    """Return "numpy array", "h5py dataset" or "sequence"

    :param array_like: Array like object: numpy array, hdf5 dataset,
        multi-dimensional nested sequence
    :return: Type of array
    """
    if isinstance(array_like, numpy.ndarray):
        return "numpy array"
    if is_array(array_like):
        if hasattr(array_like, "file"):
            return "h5py dataset"
    if is_nested_sequence(array_like):
        return "sequence"

    subsequence = array_like
    while hasattr(subsequence, "__len__"):
        subsequence = subsequence[0]

    return numpy.dtype(type(subsequence))


class ArrayLike(object):
    """

    :param array_like: Array, dataset or nested sequence.
         Nested sequences must be rectangular and of homogeneous type.
    """
    def __init__(self, array_like):
        """

        """
        super(ArrayLike, self).__init__()
        self.array_like = array_like
        """original object"""
        self._cache = None
        """data as a numpy array (created when/if needed)"""

        if is_array(array_like):
            self.shape = array_like.shape
            self.dtype = array_like.dtype
        elif is_nested_sequence(array_like):
            self.shape = get_shape(array_like)
            self.dtype = get_dtype(array_like)

        self.array_type = get_array_type(array_like)

    @property
    def cached_array(self):
        """data as a numpy array (created when/if needed)"""
        if self._cache is None:
            self._cache = numpy.asarray(self.array_like)
        return self._cache

    def __len__(self):
        return len(self.array_like)

    def __getitem__(self, item):
        """Implement slicing and fancy indexing for sequences"""
        # arrays and datasets already support slicing and fancy indexing
        if self.array_type in ["numpy array", "h5py dataset"]:
            return self.array_like[item]

        # From now on, we assume array_like is a nested sequence.
        # Regular int indexing or simple slice
        if isinstance(item, (int, slice)):
            return self.array_like[item]

        # multidimensional/fancy slicing: numpy array needed
        if hasattr(item, "__len__"):
            return self.cached_array[item]
            # TODO: implement nD slicing without array casting



