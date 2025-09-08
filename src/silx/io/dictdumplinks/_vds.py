from collections.abc import Sequence
from typing import Any
from typing import Literal
from typing import Optional
from typing import cast

import h5py
from pydantic import BaseModel
from pydantic import field_validator

from ._base_types import DsetIndex
from ._base_types import DsetIndexItem
from ._base_types import RawDsetIndex
from ._base_types import RawDsetIndexItem


class VdsSourceV1(BaseModel, arbitrary_types_allowed=True):
    file_path: str
    data_path: str
    shape: tuple[int, ...]
    dtype: Any  # DTypeLike gives pydantic.errors.PydanticUserError on Python < 3.12.
    source_index: Optional[RawDsetIndex] = tuple()
    target_index: Optional[RawDsetIndex] = tuple()

    @field_validator("source_index", "target_index", mode="before")
    @classmethod
    def as_raw_dset_index(cls, idx: DsetIndex) -> RawDsetIndex:
        if _is_raw_dset_index_item(idx):
            return _as_raw_dset_index_item(cast(DsetIndexItem, idx))
        if not isinstance(idx, Sequence):
            raise TypeError(f"Unsupported index type {type(idx)}: {idx}")
        return tuple(_as_raw_dset_index_item(idx_item) for idx_item in idx)


class VdsModelV1(BaseModel):
    dictdump_schema: Literal["virtual_dataset_v1"]
    shape: tuple[int, ...]
    dtype: Any  # DTypeLike gives pydantic.errors.PydanticUserError on Python < 3.12.
    sources: list[VdsSourceV1]


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


def deserialize_vds(model: VdsModelV1) -> h5py.VirtualLayout:
    vds_layout = h5py.VirtualLayout(shape=model.shape, dtype=model.dtype)
    for source in model.sources:
        vsource = h5py.VirtualSource(
            source.file_path,
            name=source.data_path,
            shape=source.shape,
            dtype=source.dtype,
        )
        if source.source_index == tuple():
            vds_layout[source.target_index] = vsource
        else:
            vds_layout[source.target_index] = vsource[source.source_index]
    return vds_layout
