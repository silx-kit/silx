from collections.abc import Sequence
from typing import Any
from typing import Literal
from typing import cast

import h5py
from pydantic import BaseModel
from pydantic import field_validator

from ..url import DataUrl
from ._base_types import DsetIndex
from ._base_types import DsetIndexItem
from ._base_types import RawDsetIndex
from ._base_types import RawDsetIndexItem
from ._link_types import Hdf5LinkModel
from ._parse_hdf5_utils import hdf5_url_to_vds_schema
from ._parse_hdf5_utils import hdf5_urls_to_vds_schema
from ._utils import normalize_vds_source_url


class VdsSourceV1(BaseModel, arbitrary_types_allowed=True):
    file_path: str = "."
    data_path: str
    shape: tuple[int, ...]
    dtype: Any  # DTypeLike gives pydantic.errors.PydanticUserError on Python < 3.12.
    source_index: RawDsetIndex = None
    target_index: RawDsetIndex = None

    @field_validator("source_index", "target_index", mode="before")
    @classmethod
    def as_raw_dset_index(cls, idx: DsetIndex) -> RawDsetIndex:
        if _is_raw_dset_index_item(idx):
            return _as_raw_dset_index_item(cast(DsetIndexItem, idx))
        if not isinstance(idx, Sequence):
            raise TypeError(f"Unsupported index type {type(idx)}: {idx}")
        return tuple(_as_raw_dset_index_item(idx_item) for idx_item in idx)


class VdsModelV1(Hdf5LinkModel):
    dictdump_schema: Literal["vds_v1"]
    shape: tuple[int, ...]
    dtype: Any  # DTypeLike gives pydantic.errors.PydanticUserError on Python < 3.12.
    sources: list[VdsSourceV1]

    def tolink(self, source: DataUrl) -> h5py.VirtualLayout:
        vds_layout = h5py.VirtualLayout(shape=self.shape, dtype=self.dtype)
        for vsource in self.sources:
            file_path, data_path = normalize_vds_source_url(
                vsource.file_path, vsource.data_path, source
            )
            vs = h5py.VirtualSource(
                file_path,
                name=data_path,
                shape=vsource.shape,
                dtype=vsource.dtype,
            )
            source_index = vsource.source_index
            target_index = vsource.target_index
            if source_index is None:
                source_index = tuple()
            if target_index is None:
                target_index = tuple()
            if isinstance(source_index, tuple):
                source_index = tuple(
                    slice(None) if idx is None else idx for idx in source_index
                )
            if isinstance(target_index, tuple):
                target_index = tuple(
                    slice(None) if idx is None else idx for idx in target_index
                )
            if source_index == tuple():
                vds_layout[target_index] = vs
            else:
                vds_layout[target_index] = vs[source_index]
        return vds_layout


class VdsUrlsModelV1(Hdf5LinkModel, arbitrary_types_allowed=True):
    dictdump_schema: Literal["vds_urls_v1"]
    source_shape: tuple[int, ...]
    source_dtype: (
        Any  # DTypeLike gives pydantic.errors.PydanticUserError on Python < 3.12.
    )
    sources: DataUrl | list[DataUrl]

    @field_validator("sources", mode="before")
    @classmethod
    def as_dataurl(
        cls, value: DataUrl | Sequence[DataUrl] | str | Sequence[str]
    ) -> RawDsetIndex:
        if isinstance(value, str):
            return DataUrl(value)
        if isinstance(value, Sequence) and isinstance(value[0], str):
            return [DataUrl(s) for s in value]
        return value

    def tolink(self, source: DataUrl) -> h5py.VirtualLayout:
        if isinstance(self.sources, DataUrl):
            target_schema = hdf5_url_to_vds_schema(
                source, self.sources, self.source_shape, self.source_dtype
            )
        else:
            target_schema = hdf5_urls_to_vds_schema(
                source,
                targets=self.sources,
                target_shape=self.source_shape,
                target_dtype=self.source_dtype,
            )
        return VdsModelV1(**target_schema).tolink(source)


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
