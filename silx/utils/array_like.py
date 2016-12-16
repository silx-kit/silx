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
"""Functions and classes for array-like objects, implementing common numpy
array features for datasets or nested sequences, while trying to avoid copying
data.

Classes:

    - :class:`TransposedDatasetView`: Similar to a numpy view, to access
      a h5py dataset as if it was transposed, but without having to cast
      it into a numpy array (this lets h5py handle reading the data from the
      file into memory, as needed).

Functions:

    - :func:`is_array`
    - :func:`is_nested_sequence`
    - :func:`get_shape`
    - :func:`get_dtype`
    - :func:`get_array_type`

"""

from __future__ import absolute_import, print_function, division
import numpy
import sys

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "13/12/2016"


def is_array(obj):
    """Return True if object implements necessary attributes to be
    considered similar to a numpy array.

    Attributes needed are "shape", "dtype", "__getitem__"
    and "__array__".

    :param obj: Array-like object (numpy array, h5py dataset...)
    :return: boolean
    """
    # add more required attribute if necessary
    for attr in ("shape", "dtype", "__array__", "__getitem__"):
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

#
# class ArrayLike(object):
#     """
#
#     :param array_like: Array, dataset or nested sequence.
#          Nested sequences must be rectangular and of homogeneous type.
#     """
#     def __init__(self, array_like):
#         """
#
#         """
#         super(ArrayLike, self).__init__()
#         self.array_like = array_like
#         """original object"""
#         self._cache = None
#         """data as a numpy array (created when/if needed)"""
#
#         if is_array(array_like):
#             self.shape = array_like.shape
#             self.dtype = array_like.dtype
#         elif is_nested_sequence(array_like):
#             self.shape = get_shape(array_like)
#             self.dtype = get_dtype(array_like)
#
#         self.array_type = get_array_type(array_like)
#
#     @property
#     def cached_array(self):
#         """data as a numpy array (created when/if needed)"""
#         if self._cache is None:
#             self._cache = numpy.asarray(self.array_like)
#         return self._cache
#
#     def __len__(self):
#         return len(self.array_like)
#
#     def __getitem__(self, item):
#         """Implement slicing and fancy indexing for sequences"""
#         # arrays and datasets already support slicing and fancy indexing
#         if self.array_type in ["numpy array", "h5py dataset"]:
#             return self.array_like[item]
#
#         # From now on, we assume array_like is a nested sequence.
#         # Regular int indexing or simple slice
#         if isinstance(item, (int, slice)):
#             return self.array_like[item]
#
#         # multidimensional/fancy slicing: numpy array needed
#         if hasattr(item, "__len__"):
#             #
#             return self.cached_array[item]
#             # TODO: implement nD slicing without array casting


class TransposedDatasetView(object):
    """
    This class provides a way to transpose a dataset without
    casting it into a numpy array. This way, the dataset in a file need not
    necessarily be integrally read into memory to view it in a different
    transposition.

    .. note::
        The performances depend a lot on the way the dataset was written
        to file. Depending on the chunking strategy, reading a complete 2D slice
        in an unfavorable direction may still require the entire dataset to
        be read from disk.

    :param dataset: h5py dataset
    :param transposition: List of dimension numbers in the wanted order
    """
    def __init__(self, dataset, transposition=None):
        """

        """
        super(TransposedDatasetView, self).__init__()
        self.dataset = dataset
        """original dataset"""

        self.shape = dataset.shape
        """Tuple of array dimensions"""
        self.dtype = dataset.dtype
        """Data-type of the array’s element"""
        self.ndim = len(dataset.shape)
        """Number of array dimensions"""

        size = 0
        if self.ndim:
            size = 1
            for dimsize in self.shape:
                size *= dimsize
        self.size = size
        """Number of elements in the array."""

        self.transposition = list(range(self.ndim))
        """List of dimension indices, in an order depending on the
        specified transposition. By default this is simply
        [0, ..., self.ndim], but it can be changed by specifying a different
        `transposition` parameter at initialization.

        Use :meth:`transpose`, to create a new :class:`TransposedDatasetView`
        with a different :attr:`transposition`.
        """

        if transposition is not None:
            assert len(transposition) == self.ndim
            assert set(transposition) == set(list(range(self.ndim))), \
                "Transposition must be a list containing all dimensions"
            self.transposition = transposition
            self.__sort_shape()

    def __sort_shape(self):
        """Sort shape in the order defined in :attr:`transposition`
        """
        new_shape = tuple(self.shape[dim] for dim in self.transposition)
        self.shape = new_shape

    def __sort_indices(self, indices):
        """Return array indices sorted in the order needed
        to access data in the original non-transposed dataset.

        :param indices: Tuple of ndim indices, in the order needed
            to access the view
        :return: Sorted tuple of indices, to access original data
        """
        assert len(indices) == self.ndim
        sorted_indices = tuple(idx for (_, idx) in
                               sorted(zip(self.transposition, indices)))
        return sorted_indices

    def __getitem__(self, item):
        """Handle fancy indexing with regards to the dimension order as
        specified in :attr:`transposition`

        The supported fancy-indexing syntax is explained at
        http://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing.

        Additional restrictions exist if the data has been transposed:

            - numpy boolean array indexing is not supported
            - ellipsis objects are not supported

        :param item: Index, possibly fancy index (must be supported by h5py)
        :return:
        """
        # no transposition, let the original dataset handle indexing
        if self.transposition == list(range(self.ndim)):
            return self.dataset[item]

        # 1-D slicing -> n-D slicing (n=1)
        if not hasattr(item, "__len__"):
            # first dimension index is given
            item = [item]
            # following dimensions are indexed with : (all elements)
            item += [slice(0, sys.maxint, 1) for _i in range(self.ndim - 1)]

        # n-dimensional slicing
        if len(item) != self.ndim:
            raise IndexError(
                "N-dim slicing requires a tuple of N indices/slices. " +
                "Needed dimensions: %d" % self.ndim)

        # get list of indices sorted in the original dataset order
        sorted_indices = self.__sort_indices(item)

        output_data_not_transposed = self.dataset[sorted_indices]

        # now we must transpose the output data
        output_dimensions = []
        frozen_dimensions = []
        for i, idx in enumerate(item):
            # slices and sequences
            if not isinstance(idx, int):
                output_dimensions.append(self.transposition[i])
            # regular integer index
            else:
                # whenever a dimension is fixed (indexed by an integer)
                # the number of output dimension is reduced
                frozen_dimensions.append(self.transposition[i])

        # decrement output dimensions that are above frozen dimensions
        for frozen_dim in reversed(sorted(frozen_dimensions)):
            for i, out_dim in enumerate(output_dimensions):
                if out_dim > frozen_dim:
                    output_dimensions[i] -= 1

        assert (len(output_dimensions) + len(frozen_dimensions)) == self.ndim
        assert set(output_dimensions) == set(range(len(output_dimensions)))

        return numpy.transpose(output_data_not_transposed,
                               axes=output_dimensions)

    def __array__(self, dtype=None):
        """Cast the dataset into a numpy array, and return it.

        If a transposition has been done on this dataset, return
        a transposed view of a numpy array."""
        return numpy.transpose(numpy.array(self.dataset, dtype=dtype),
                               self.transposition)

    def transpose(self, transposition=None):
        """Return a re-ordered (dimensions permutated)
        :class:`TransposedDatasetView`.

        The returned object refers to
        the same dataset but with a different :attr:`transposition`.

        :param list[int] transposition: List of dimension numbers in the wanted order
        :return: Transposed TransposedDatasetView
        """
        # by default, reverse the dimensions
        if transposition is None:
            transposition = list(reversed(self.transposition))

        return TransposedDatasetView(self.dataset,
                                     transposition)

