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
"""Functions and classes for array-like objects, implementing common numpy
array features for datasets or nested sequences, while trying to avoid copying
data.

Classes:

    - :class:`DatasetView`: Similar to a numpy view, to access
      a h5py dataset as if it was transposed, without casting it into a
      numpy array (this lets h5py handle reading the data from the
      file into memory, as needed).
    - :class:`ListOfImages`: Similar to a numpy view, to access
      a list of 2D numpy arrays as if it was a 3D array (possibly transposed),
      without casting it into a numpy array.

Functions:

    - :func:`is_array`
    - :func:`is_list_of_arrays`
    - :func:`is_nested_sequence`
    - :func:`get_shape`
    - :func:`get_dtype`
    - :func:`get_concatenated_dtype`

"""

from __future__ import absolute_import, print_function, division

import sys

import numpy
import six
import numbers

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "26/04/2017"


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


def is_list_of_arrays(obj):
    """Return True if object is a sequence of numpy arrays,
    e.g. a list of images as 2D arrays.

    :param obj: list of arrays
    :return: boolean"""
    # object must not be a numpy array
    if is_array(obj):
        return False

    # object must have a __len__ method
    if not hasattr(obj, "__len__"):
        return False

    # all elements in sequence must be arrays
    for arr in obj:
        if not is_array(arr):
            return False

    return True


def is_nested_sequence(obj):
    """Return True if object is a nested sequence.

    A simple 1D sequence is considered to be a nested sequence.

    Numpy arrays and h5py datasets are not considered to be nested sequences.

    To test if an object is a nested sequence in a more general sense,
    including arrays and datasets, use::

        is_nested_sequence(obj) or is_array(obj)

    :param obj: nested sequence (numpy array, h5py dataset...)
    :return: boolean"""
    # object must not be a numpy array
    if is_array(obj):
        return False

    if not hasattr(obj, "__len__"):
        return False

    # obj must not be a list of (lists of) numpy arrays
    subsequence = obj
    while hasattr(subsequence, "__len__"):
        if is_array(subsequence):
            return False
        # strings cause infinite loops
        if isinstance(subsequence, six.string_types + (six.binary_type, )):
            return True
        subsequence = subsequence[0]

    # object has __len__ and is not an array
    return True


def get_shape(array_like):
    """Return shape of an array like object.

    In case the object is a nested sequence but not an array or dataset
    (list of lists, tuples...), the size of each dimension is assumed to be
    uniform, and is deduced from the length of the first sequence.

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
        # strings cause infinite loops
        if isinstance(subsequence, six.string_types + (six.binary_type, )):
            break
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

    subsequence = array_like
    while hasattr(subsequence, "__len__"):
        # strings cause infinite loops
        if isinstance(subsequence, six.string_types + (six.binary_type, )):
            break
        subsequence = subsequence[0]

    return numpy.dtype(type(subsequence))


def get_concatenated_dtype(arrays):
    """Return dtype of array resulting of concatenation
    of a list of arrays (without actually concatenating
    them).

    :param arrays: list of numpy arrays
    :return: resulting dtype after concatenating arrays
    """
    dtypes = {a.dtype for a in arrays}
    dummy = []
    for dt in dtypes:
        dummy.append(numpy.zeros((1, 1), dtype=dt))
    return numpy.array(dummy).dtype


class ListOfImages(object):
    """This class provides a way to access values and slices in a stack of
    images stored as a list of 2D numpy arrays, without creating a 3D numpy
    array first.

    A transposition can be specified, as a 3-tuple of dimensions in the wanted
    order. For example, to transpose from ``xyz`` ``(0, 1, 2)`` into ``yzx``,
    the transposition tuple is ``(1, 2, 0)``

    All the 2D arrays in the list must have the same shape.

    The global dtype of the stack of images is the one that would be obtained
    by casting the list of 2D arrays into a 3D numpy array.

    :param images: list of 2D numpy arrays, or :class:`ListOfImages` object
    :param transposition: Tuple of dimension numbers in the wanted order
    """
    def __init__(self, images, transposition=None):
        """

        """
        super(ListOfImages, self).__init__()

        # if images is a ListOfImages instance, get the underlying data
        # as a list of 2D arrays
        if isinstance(images, ListOfImages):
            images = images.images

        # test stack of images is as expected
        assert is_list_of_arrays(images), \
            "Image stack must be a list of arrays"
        image0_shape = images[0].shape
        for image in images:
            assert image.ndim == 2, \
                "Images must be 2D numpy arrays"
            assert image.shape == image0_shape, \
                "All images must have the same shape"

        self.images = images
        """List of images"""

        self.shape = (len(images), ) + image0_shape
        """Tuple of array dimensions"""
        self.dtype = get_concatenated_dtype(images)
        """Data-type of the global array"""
        self.ndim = 3
        """Number of array dimensions"""

        self.size = len(images) * image0_shape[0] * image0_shape[1]
        """Number of elements in the array."""

        self.transposition = list(range(self.ndim))
        """List of dimension indices, in an order depending on the
        specified transposition. By default this is simply
        [0, ..., self.ndim], but it can be changed by specifying a different
        ``transposition`` parameter at initialization.

        Use :meth:`transpose`, to create a new :class:`ListOfImages`
        with a different :attr:`transposition`.
        """

        if transposition is not None:
            assert len(transposition) == self.ndim
            assert set(transposition) == set(list(range(self.ndim))), \
                "Transposition must be a sequence containing all dimensions"
            self.transposition = transposition
            self.__sort_shape()

    def __sort_shape(self):
        """Sort shape in the order defined in :attr:`transposition`
        """
        new_shape = tuple(self.shape[dim] for dim in self.transposition)
        self.shape = new_shape

    def __sort_indices(self, indices):
        """Return array indices sorted in the order needed
        to access data in the original non-transposed images.

        :param indices: Tuple of ndim indices, in the order needed
            to access the transposed view
        :return: Sorted tuple of indices, to access original data
        """
        assert len(indices) == self.ndim
        sorted_indices = tuple(idx for (_, idx) in
                               sorted(zip(self.transposition, indices)))
        return sorted_indices

    def __array__(self, dtype=None):
        """Cast the images into a numpy array, and return it.

        If a transposition has been done on this images, return
        a transposed view of a numpy array."""
        return numpy.transpose(numpy.array(self.images, dtype=dtype),
                               self.transposition)

    def __len__(self):
        return self.shape[0]

    def transpose(self, transposition=None):
        """Return a re-ordered (dimensions permutated)
        :class:`ListOfImages`.

        The returned object refers to
        the same images but with a different :attr:`transposition`.

        :param List[int] transposition: List/tuple of dimension numbers in the
            wanted order.
            If ``None`` (default), reverse the dimensions.
        :return: new :class:`ListOfImages` object
        """
        # by default, reverse the dimensions
        if transposition is None:
            transposition = list(reversed(self.transposition))

        # If this ListOfImages is already transposed, sort new transposition
        # relative to old transposition
        elif list(self.transposition) != list(range(self.ndim)):
            transposition = [self.transposition[i] for i in transposition]

        return ListOfImages(self.images,
                            transposition)

    @property
    def T(self):
        """
        Same as self.transpose()

        :return: DatasetView with dimensions reversed."""
        return self.transpose()

    def __getitem__(self, item):
        """Handle a subset of numpy indexing with regards to the dimension
        order as specified in :attr:`transposition`

        Following features are **not supported**:

            - fancy indexing using numpy arrays
            - using ellipsis objects

        :param item: Index
        :return: value or slice as a numpy array
        """
        # 1-D slicing -> n-D slicing (n=1)
        if not hasattr(item, "__len__"):
            # first dimension index is given
            item = [item]
            # following dimensions are indexed with : (all elements)
            item += [slice(None) for _i in range(self.ndim - 1)]

        # n-dimensional slicing
        if len(item) != self.ndim:
            raise IndexError(
                "N-dim slicing requires a tuple of N indices/slices. " +
                "Needed dimensions: %d" % self.ndim)

        # get list of indices sorted in the original images order
        sorted_indices = self.__sort_indices(item)
        list_idx, array_idx = sorted_indices[0], sorted_indices[1:]

        images_selection = self.images[list_idx]

        # now we must transpose the output data
        output_dimensions = []
        frozen_dimensions = []
        for i, idx in enumerate(item):
            # slices and sequences
            if not isinstance(idx, numbers.Integral):
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

        # single list elements selected
        if isinstance(images_selection, numpy.ndarray):
            return numpy.transpose(images_selection[array_idx],
                                   axes=output_dimensions)
        # muliple list elements selected
        else:
            # apply selection first
            output_stack = []
            for img in images_selection:
                output_stack.append(img[array_idx])
            # then cast into a numpy array, and transpose
            return numpy.transpose(numpy.array(output_stack),
                                   axes=output_dimensions)

    def min(self):
        """
        :return: Global minimum value
        """
        min_value = self.images[0].min()
        if len(self.images) > 1:
            for img in self.images[1:]:
                min_value = min(min_value, img.min())
        return min_value

    def max(self):
        """
        :return: Global maximum value
        """
        max_value = self.images[0].max()
        if len(self.images) > 1:
            for img in self.images[1:]:
                max_value = max(max_value, img.max())
        return max_value


class DatasetView(object):
    """This class provides a way to transpose a dataset without
    casting it into a numpy array. This way, the dataset in a file need not
    necessarily be integrally read into memory to view it in a different
    transposition.

    .. note::
        The performances depend a lot on the way the dataset was written
        to file. Depending on the chunking strategy, reading a complete 2D slice
        in an unfavorable direction may still require the entire dataset to
        be read from disk.

    :param dataset: h5py dataset
    :param transposition: List of dimensions sorted in the order of
        transposition (relative to the original h5py dataset)
    """
    def __init__(self, dataset, transposition=None):
        """

        """
        super(DatasetView, self).__init__()
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

        Use :meth:`transpose`, to create a new :class:`DatasetView`
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
        :return: Sliced numpy array or numpy scalar
        """
        # no transposition, let the original dataset handle indexing
        if self.transposition == list(range(self.ndim)):
            return self.dataset[item]

        # 1-D slicing: create a list of indices to switch to n-D slicing
        if not hasattr(item, "__len__"):
            # first dimension index (list index) is given
            item = [item]
            # following dimensions are indexed with slices representing all elements
            item += [slice(None) for _i in range(self.ndim - 1)]

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

    def __len__(self):
        return self.shape[0]

    def transpose(self, transposition=None):
        """Return a re-ordered (dimensions permutated)
        :class:`DatasetView`.

        The returned object refers to
        the same dataset but with a different :attr:`transposition`.

        :param List[int] transposition: List of dimension numbers in the wanted order.
            If ``None`` (default), reverse the dimensions.
        :return: Transposed DatasetView
        """
        # by default, reverse the dimensions
        if transposition is None:
            transposition = list(reversed(self.transposition))

        # If this DatasetView is already transposed, sort new transposition
        # relative to old transposition
        elif list(self.transposition) != list(range(self.ndim)):
            transposition = [self.transposition[i] for i in transposition]

        return DatasetView(self.dataset,
                           transposition)

    @property
    def T(self):
        """
        Same as self.transpose()

        :return: DatasetView with dimensions reversed."""
        return self.transpose()
