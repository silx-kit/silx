from __future__ import annotations

import logging
import math
import numbers
from typing import Generator, Literal

import numpy

from . import commonh5

from blissdata.h5api import abstract as _abstract
from blissdata.h5api import redis_hdf5 as _redis_hdf5
from blissdata.streams import BaseStream as _BaseStream

_logger = logging.getLogger(__name__)


class BlissDataH5(commonh5.File):
    def __init__(
        self,
        name: str,
        mode: Literal["r"] | None = None,
        attrs: dict | None = None,
    ) -> None:
        if mode not in ("r", None):
            raise ValueError(f"Unsupported mode: {mode}")

        if attrs is None:
            attrs = {}

        self.__file = _redis_hdf5.File(name)

        super().__init__(name, mode, attrs={**self.__file.attrs, **attrs})

        for child in _children(self.__file):
            self.add_node(child)

    def close(self) -> None:
        super().close()
        self.__file.close()
        self.__file = None


class BlissDataGroup(commonh5.LazyLoadableGroup):
    def __init__(
        self,
        name: str,
        group: _abstract.Group,
        parent: BlissDataH5 | "BlissDataGroup" | None = None,
        attrs: dict | None = None,
    ) -> None:
        super().__init__(name, parent, attrs)
        self.__group = group

    def _create_child(self) -> None:
        for child in _children(self.__group):
            self.add_node(child)


class BlissDataDataset(commonh5.Dataset):
    @property
    def shape(self) -> tuple[int, ...]:
        return self._get_data().shape

    @property
    def size(self) -> int:
        return self._get_data().size

    def __len__(self) -> int:
        return len(self._get_data())

    def __getitem__(self, item):
        # This method aims at avoiding accessing data that is not yet available
        # in order to avoid waiting for it
        data = self._get_data()
        if data.ndim == 0:
            if (isinstance(item, tuple) and item == ()) or item is Ellipsis:
                return data[()]
            return data[()][item]

        if not isinstance(item, tuple):
            item = (item,)
        if len(item) == 0:
            return self.__getFirstAxis(slice(None))

        first, others = item[0], item[1:]
        if first is Ellipsis:
            nExplicitIndices = sum(
                1 for index in others if index is not None and index is not Ellipsis
            )
            if nExplicitIndices >= data.ndim:
                # Ellipsis does not cover any axis
                return self[others]
            # Ellipsis covers the first axis: Keep it to apply on other axes
            first, others = slice(None), item

        if isinstance(first, slice) or (
            isinstance(first, numbers.Integral) and not isinstance(first, bool)
        ):
            result = self.__getFirstAxis(first)
            if not others:
                return result
            if isinstance(first, slice):
                # First axis is kept in the result
                others = (slice(None),) + others
            return numpy.asarray(result)[others]

        # Unsupported first index (array, list, None...): Read all available data
        return numpy.asarray(self.__getFirstAxis(slice(None)))[item]

    def __getFirstAxis(self, index: int | slice):
        """Index the first axis of the data, only with indices of available data.

        Indices are converted to positive int and slices with explicit
        positive bounds to avoid blissdata waiting for new data or the end of
        the scan (e.g., with negative indices or slices without stop).
        For integer index, the first axis is removed from the result.
        """
        data = self._get_data()
        length = len(self)

        if isinstance(index, slice):
            indices = range(*index.indices(length))
            if len(indices) == 0:
                return numpy.empty((0,) + tuple(self.shape[1:]), dtype=data.dtype)
            if indices.step < 0:
                # Not supported by blissdata streams: read in increasing order and reverse
                reverse = slice(indices[-1], indices[0] + 1, -indices.step)
                return self.__getFirstAxis(reverse)[::-1]
            return numpy.asarray(data[indices[0] : indices[-1] + 1 : indices.step])

        position = int(index)
        if position < 0:
            position += length
        if not 0 <= position < length:
            raise IndexError(
                f"Index ({index}) out of range for axis 0 of size {length}"
            )
        return data[position]

    _ITER_CHUNK_NBYTES = 16 * 1024**2
    """Maximum size of data read at once when iterating over a dataset

    This aims at avoiding loading too many data at once.
    """

    def __iter__(self):
        """Iterate over the first axis, on data available when iteration starts.

        Contrary to blissdata datasets, it does not wait for new data.
        """
        data = self._get_data()
        if data.ndim == 0:
            raise TypeError("Can't iterate over a scalar dataset")

        length = len(self)
        itemNBytes = data.dtype.itemsize * math.prod(data.shape[1:])
        chunkLength = max(1, self._ITER_CHUNK_NBYTES // max(1, itemNBytes))
        for start in range(0, length, chunkLength):
            stop = min(start + chunkLength, length)
            yield from self.__getFirstAxis(slice(start, stop))

    @property
    def value(self) -> numpy.ndarray:
        return self[()]


class BlissDataLima1Dataset(BlissDataDataset):
    """Dataset wrapping a lima1 stream.

    During the scan, lima1 images may not be immediately available,
    so they are not exposed until the stream is sealed.
    """

    def __init__(
        self,
        name: str,
        data: _abstract.Dataset,
        stream: _BaseStream,
        parent: BlissDataH5 | BlissDataGroup | None = None,
        attrs: dict | None = None,
    ) -> None:
        super().__init__(name, data, parent, attrs)
        self.__stream = stream
        self.__sealed = False

    def __isSealed(self) -> bool:
        if not self.__sealed:  # Once sealed, no need to query it again
            self.__sealed = self.__stream.is_sealed()
        return self.__sealed

    def __len__(self) -> int:
        return super().__len__() if self.__isSealed() else 0

    @property
    def shape(self) -> tuple[int, ...]:
        if self.__isSealed():
            return super().shape
        return (0,) + tuple(self.__stream.shape)

    @property
    def size(self) -> int:
        return super().size if self.__isSealed() else 0


def _blissdata_stream(dataset: _abstract.Dataset) -> _BaseStream | None:
    """Returns the stream wrapped by a blissdata Dataset if any, else None

    See https://gitlab.esrf.fr/bliss/blissdata/-/work_items/32
    """
    if isinstance(dataset, _redis_hdf5.Dataset) and isinstance(
        dataset._array, _redis_hdf5.StreamAsArray
    ):
        return dataset._array._stream
    else:
        return None


def _children(group: _abstract.Group) -> Generator[BlissDataDataset | BlissDataGroup]:
    for name in group.keys():
        item = group[name]
        if isinstance(item, _abstract.Group):
            yield BlissDataGroup(name, item, parent=group, attrs=item.attrs)
        elif isinstance(item, _abstract.Dataset):
            stream = _blissdata_stream(item)
            if stream is not None and stream.plugin == "lima":
                # Special case for Lima1 to prevent accessing frames during the scan
                # since this may hang until the end of the scan
                yield BlissDataLima1Dataset(
                    name, item, stream, parent=group, attrs=item.attrs
                )
            else:  # Default route for other streams and static datasets with live access
                yield BlissDataDataset(name, item, parent=group, attrs=item.attrs)
        else:
            _logger.warning(f"Cannot map child {name}: Ignored")
