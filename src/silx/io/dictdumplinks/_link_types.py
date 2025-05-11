from abc import abstractmethod
from typing import Any, TypeAlias
from collections.abc import Sequence

import h5py
from numpy.typing import DTypeLike

from ._vds_types import VdsSource


class LinkInterface:
    @abstractmethod
    def create(self, parent: h5py.Group, name: str) -> Any:
        pass


class InternalLink(LinkInterface):
    def __init__(self, data_path: str):
        self._data_path = data_path
        super().__init__()

    @property
    def data_path(self) -> str:
        return self._data_path

    def __eq__(self, other: Any):
        if isinstance(other, InternalLink):
            return self.data_path == other.data_path
        return super().__eq__(other)

    def create(self, parent: h5py.Group, name: str) -> None:
        parent[name] = h5py.SoftLink(self._data_path)


class ExternalLink(LinkInterface):
    def __init__(self, file_path: str, data_path: str):
        self._file_path = file_path
        self._data_path = data_path
        super().__init__()

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def data_path(self) -> str:
        return self._data_path

    def __eq__(self, other: Any):
        if isinstance(other, ExternalLink):
            return (
                self.file_path == other.file_path and self.data_path == other.data_path
            )
        return super().__eq__(other)

    def create(self, parent: h5py.Group, name: str) -> None:
        parent[name] = h5py.ExternalLink(self._file_path, self._data_path)


class ExternalBinaryLink(LinkInterface):
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        sources: Sequence[tuple[str, int, int]],
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self._sources = sources

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def sources(self) -> Sequence[tuple[str, int, int]]:
        return self._sources

    def __eq__(self, other: Any):
        if isinstance(other, ExternalBinaryLink):
            return (
                self.shape == other.shape
                and self.dtype == other.dtype
                and self.sources == other.sources
            )
        return super().__eq__(other)

    def create(self, parent: h5py.Group, name: str) -> h5py.Dataset:
        return parent.create_dataset(
            name=name, shape=self._shape, dtype=self._dtype, external=self._sources
        )


class VDSLink(LinkInterface):

    def __init__(
        self, shape: tuple[int, ...], dtype: DTypeLike, sources: Sequence[VdsSource]
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self._sources = sources

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def sources(self) -> Sequence[VdsSource]:
        return self._sources

    def __eq__(self, other: Any):
        if isinstance(other, VDSLink):
            return (
                self.shape == other.shape
                and self.dtype == other.dtype
                and self.sources == other.sources
            )
        return super().__eq__(other)

    def create(self, parent: h5py.Group, name: str) -> None:
        parent.create_virtual_dataset(name, self._get_layout())

    def _get_layout(self) -> h5py.VirtualLayout:
        layout = h5py.VirtualLayout(shape=self._shape, dtype=self._dtype)

        for source in self._sources:
            vsource = h5py.VirtualSource(
                source.file_path,
                name=source.data_path,
                shape=source.shape,
                dtype=source.dtype,
            )
            source_index = source.source_index
            target_index = source.target_index
            if source_index:
                layout[target_index] = vsource[source_index]
            else:
                layout[target_index] = vsource
        return layout


Hdf5LinkType: TypeAlias = ExternalLink | InternalLink | VDSLink | ExternalBinaryLink
