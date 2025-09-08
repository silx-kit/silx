from collections.abc import Sequence
from typing import TypedDict, Any, Literal, NamedTuple, cast

import h5py
from numpy.typing import DTypeLike

from ._base_types import DsetIndex
from ._base_types import RawDsetIndex
from ._base_types import DsetIndexItem
from ._base_types import RawDsetIndexItem


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


def deserialize_vds_schema_v1(target: VdsSchemaV1) -> h5py.VirtualLayout:
    shape = tuple(target["shape"])
    dtype = target["dtype"]
    vds_layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

    for source in target["sources"]:
        source_index = _as_raw_dset_index(source.get("source_index", tuple()))
        target_index = _as_raw_dset_index(source.get("target_index", tuple()))

        vsource = h5py.VirtualSource(
            source["file_path"],
            name=source["data_path"],
            shape=tuple(source["shape"]),
            dtype=source["dtype"],
        )
        if source_index == tuple():
            vds_layout[target_index] = vsource
        else:
            vds_layout[target_index] = vsource[source_index]

    return vds_layout


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
