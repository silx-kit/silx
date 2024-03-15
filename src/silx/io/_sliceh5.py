# /*##########################################################################
# Copyright (C) 2022-2023 European Synchrotron Radiation Facility
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
"""Provides a wrapper to expose a dataset slice as a `commonh5.Dataset`."""

from __future__ import annotations

from typing import Tuple, Union

import h5py
import numpy

from . import commonh5
from . import utils


IndexType = Union[int, slice, type(Ellipsis)]
IndicesType = Union[IndexType, Tuple[IndexType, ...]]
NormalisedIndicesType = Tuple[Union[int, slice], ...]


def _expand_indices(
    ndim: int,
    indices: IndicesType,
) -> NormalisedIndicesType:
    """Replace Ellipsis and complete indices to match ndim"""
    if not isinstance(indices, tuple):
        indices = (indices,)

    nb_ellipsis = indices.count(Ellipsis)
    if nb_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if nb_ellipsis == 1:
        ellipsis_index = indices.index(Ellipsis)
        return (
            indices[:ellipsis_index]
            + (slice(None),) * max(0, (ndim - len(indices) + 1))
            + indices[ellipsis_index + 1 :]
        )

    if len(indices) > ndim:
        raise IndexError(
            f"too many indices ({len(indices)}) for the number of dimensions ({ndim})"
        )
    return indices + (slice(None),) * (ndim - len(indices))


def _get_selection_shape(
    shape: tuple[int, ...],
    indices: NormalisedIndicesType,
) -> tuple[int, ...]:
    """Returns the shape of the selection of indices in a dataset of the given shape"""
    assert len(shape) == len(indices)

    selected_indices = (
        index.indices(length)
        for length, index in zip(shape, indices)
        if isinstance(index, slice)
    )
    return tuple(
        int(max(0, numpy.ceil((stop - start) / stride)))
        for start, stop, stride in selected_indices
    )


def _combine_indices(
    outer_shape: tuple[int, ...],
    outer_indices: NormalisedIndicesType,
    indices: IndicesType,
) -> NormalisedIndicesType:
    """Returns the combination of outer_indices and indices"""
    inner_shape = _get_selection_shape(outer_shape, outer_indices)
    inner_indices = _expand_indices(len(inner_shape), indices)
    inner_iter = zip(range(len(inner_shape)), inner_shape, inner_indices)

    combined_indices = []
    for outer_length, outer_index in zip(outer_shape, outer_indices):
        if isinstance(outer_index, int):
            combined_indices.append(outer_index)
            continue

        outer_start, outer_stop, outer_stride = outer_index.indices(outer_length)
        inner_axis, inner_length, inner_index = next(inner_iter)

        if isinstance(inner_index, int):
            if inner_index < -inner_length or inner_index >= inner_length:
                raise IndexError(
                    f"index {inner_index} is out of bounds for axis {inner_axis} with size {inner_length}"
                )
            index = outer_start + outer_stride * inner_index
            if inner_index < 0:
                index += outer_stride * inner_length
            combined_indices.append(index)
            continue

        inner_start, inner_stop, inner_stride = inner_index.indices(inner_length)
        combined_indices.append(
            slice(
                outer_start + outer_stride * inner_start,
                outer_start + outer_stride * inner_stop,
                outer_stride * inner_stride,
            )
        )

    return tuple(combined_indices)


class DatasetSlice(commonh5.Dataset):
    """Wrapper a dataset indexed selection as a commonh5.Dataset.
    :param h5file: h5py-like file containing the dataset
    :param dataset: h5py-like dataset from which to access a slice
    :param indices: The indexing to select
    :param attrs: dataset attributes
    """

    def __init__(
        self,
        dataset: Union[h5py.Dataset, commonh5.Dataset],
        indices: IndicesType,
        attrs: dict,
    ):
        if not utils.is_dataset(dataset):
            raise ValueError(f"Unsupported dataset '{dataset}'")

        self.__dataset = dataset
        self.__file = dataset.file  # Keep a ref on file to fix issue recovering it
        self.__indices = indices
        self.__expanded_indices = _expand_indices(len(self.__dataset.shape), indices)
        self.__shape = _get_selection_shape(
            self.__dataset.shape, self.__expanded_indices
        )
        super().__init__(
            self.__dataset.name, data=None, parent=self.__file, attrs=attrs
        )

    def _get_data(self) -> Union[h5py.Dataset, commonh5.Dataset]:
        # Give access to the underlying (h5py) dataset, not the selected data
        # All commonh5.Dataset methods using _get_data must be overridden
        return self.__dataset

    @property
    def dtype(self) -> numpy.dtype:
        return self.__dataset.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__shape

    @property
    def size(self) -> int:
        return numpy.prod(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, item):
        if item is Ellipsis:
            return numpy.asarray(self.__dataset[self.__expanded_indices])
        if item == ():
            return self.__dataset[self.__expanded_indices]

        if not self.__shape:
            raise IndexError("invalid index to scalar variable.")

        return self.__dataset[
            _combine_indices(
                self.__dataset.shape,
                self.__expanded_indices,
                item,
            )
        ]

    @property
    def value(self):
        return self[()]

    def __iter__(self):
        return self[()].__iter__()

    @property
    def file(self) -> Union[h5py.File, commonh5.File]:
        if isinstance(self.__file, h5py.File) and not self.__file.id:
            return None
        return self.__file

    @property
    def name(self) -> str:
        return self.basename

    @property
    def indices(self) -> IndicesType:
        return self.__indices

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the file"""
        self.__file.close()
