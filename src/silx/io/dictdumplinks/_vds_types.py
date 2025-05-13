from collections.abc import Sequence
from typing import TypedDict, Any, Literal, NamedTuple, cast

import h5py
from numpy.typing import DTypeLike

from ._base_types import DsetIndex
from ._base_types import RawDsetIndex
from ._base_types import DsetIndexItem
from ._base_types import RawDsetIndexItem
from ._base_types import LinkInterface


class VdsSourceSchemaV1(TypedDict):
    file_path: str
    data_path: str
    shape: Sequence[int]
    dtype: str | DTypeLike
    source_index: DsetIndex
    target_index: DsetIndex


class VdsSchemaV1(TypedDict):
    dictdump_schema: Literal["virtual_dataset_v1"]
    shape: Sequence[int]
    dtype: str | DTypeLike
    sources: Sequence[VdsSourceSchemaV1]


class VdsSource(NamedTuple):
    file_path: str
    data_path: str
    shape: tuple[int, ...]
    dtype: str | DTypeLike
    source_index: RawDsetIndex
    target_index: RawDsetIndex


class Vds(NamedTuple):
    shape: tuple[int, ...]
    dtype: str | DTypeLike
    sources: list[VdsSource]


def deserialize_vds_schema_v1(target: VdsSchemaV1) -> Vds:
    shape = tuple(target["shape"])
    dtype = target["dtype"]
    sources = list()
    for source in target["sources"]:
        source_index = _as_raw_dset_index(source.get("source_index", tuple()))
        target_index = _as_raw_dset_index(source.get("target_index", tuple()))
        sources.append(
            VdsSource(
                file_path=source["file_path"],
                data_path=source["data_path"],
                shape=tuple(source["shape"]),
                dtype=dtype,
                source_index=source_index,
                target_index=target_index,
            )
        )
    return Vds(shape, dtype, sources)


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

    def serialize(self) -> VdsSchemaV1:
        return {
            "dictdump_schema": "virtual_dataset_v1",
            "shape": self._shape,
            "dtype": self._dtype,
            "sources": [
                {
                    "file_path": source.file_path,
                    "data_path": source.data_path,
                    "shape": source.shape,
                    "dtype": source.dtype,
                    "source_index": source.source_index,
                    "target_index": source.target_index,
                }
                for source in self._sources
            ],
        }


def _as_raw_dset_index(idx: DsetIndex) -> RawDsetIndex:
    if _is_raw_dset_index_item(idx):
        return _as_raw_dset_index_item(cast(DsetIndexItem, idx))
    if not isinstance(idx, Sequence):
        raise TypeError(f"Unsupported index type {type(idx)}: {idx}")
    return tuple(_as_raw_dset_index_item(idx_item) for idx_item in idx)


def _as_raw_dset_index_item(idx_item: DsetIndexItem) -> RawDsetIndexItem:
    if _is_raw_dset_index_item(idx_item):
        return cast(RawDsetIndexItem, idx_item)
    if _is_slice_arguments(idx_item):
        return slice(*cast(Sequence, idx_item))
    raise TypeError(f"Unsupported index item type {type(idx_item)}: {idx_item}")


def _is_raw_dset_index_item(idx_item: Any) -> bool:
    return isinstance(idx_item, (int, slice)) or idx_item in (None, Ellipsis)


def _is_slice_arguments(idx_item: Any) -> bool:
    return isinstance(idx_item, Sequence) and (2 <= len(idx_item) <= 3)
