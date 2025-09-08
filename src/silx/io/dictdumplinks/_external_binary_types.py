from collections.abc import Sequence
from typing import TypedDict, NamedTuple, Literal, Any

import h5py
from numpy.typing import DTypeLike


class ExtSchemaV1(TypedDict):
    dictdump_schema: Literal["external_binary_link_v1"]
    shape: Sequence[int]
    dtype: str | DTypeLike
    sources: Sequence[tuple[str, int, int]]


class Ext(NamedTuple):
    """Attention: relative file names in external HDF5 datasets are relative
    with respect to the current working directory, not relative to the parent file.
    """

    shape: tuple[int, ...]
    dtype: str | DTypeLike
    sources: list[tuple[str, int, int]]  # (file_path, offset, bytecount)


def deserialize_ext_schema_v1(target: ExtSchemaV1) -> Ext:
    shape = tuple(target["shape"])
    dtype = target["dtype"]
    sources = [
        (file_path, offset, bytecount)
        for file_path, offset, bytecount in target["sources"]
    ]
    return Ext(shape, dtype, sources)


class ExternalBinaryLink:
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

    def serialize(self) -> ExtSchemaV1:
        return {
            "dictdump_schema": "external_binary_link_v1",
            "shape": self._shape,
            "dtype": self._dtype,
            "sources": self._sources,
        }
