from collections.abc import Sequence
from typing import TypedDict, Any, Literal, NamedTuple

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
    dictdump_schema: Literal["external_virtual_link_v1"]
    shape: Sequence[int]
    dtype: str | DTypeLike
    sources: Sequence[VdsSourceSchemaV1]


class VdsSource(NamedTuple):
    file_path: str
    data_path: str
    shape: tuple[int]
    dtype: str | DTypeLike
    source_index: RawDsetIndex
    target_index: RawDsetIndex


class Vds(NamedTuple):
    shape: tuple[int]
    dtype: str | DTypeLike
    sources: list[VdsSource]


def parse_vds_schema_v1(target: VdsSchemaV1) -> Vds:
    shape = tuple(target["shape"])
    dtype = target["dtype"]
    sources = list()
    for source in target["sources"]:
        source_index = _as_raw_dset_index(source.get("source_index") or tuple())
        target_index = _as_raw_dset_index(source.get("target_index") or tuple())
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


def _as_raw_dset_index(idx: DsetIndex) -> tuple[RawDsetIndexItem]:
    if _is_raw_dset_index_item(idx):
        return _as_raw_dset_index_item(idx)
    if not isinstance(idx, Sequence):
        raise TypeError(f"Unsupported index type {type(idx)}: {idx}")
    return tuple(_as_raw_dset_index_item(idx_item) for idx_item in idx)


def _as_raw_dset_index_item(idx_item: DsetIndexItem) -> RawDsetIndexItem:
    if _is_raw_dset_index_item(idx_item):
        return idx_item
    if _is_slice_arguments(idx_item):
        return slice(*idx_item)
    raise TypeError(f"Unsupported index item type {type(idx_item)}: {idx_item}")


def _is_raw_dset_index_item(idx_item: Any) -> bool:
    return isinstance(idx_item, (int, slice)) or idx_item in (None, Ellipsis)


def _is_slice_arguments(idx_item: Any) -> bool:
    return isinstance(idx_item, Sequence) and (2 <= len(idx_item) <= 3)
