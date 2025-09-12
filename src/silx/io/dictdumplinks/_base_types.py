from collections.abc import Sequence
from types import EllipsisType
from typing import TypeAlias

import h5py

RawDsetIndexItem: TypeAlias = int | slice | EllipsisType | None
RawDsetIndex: TypeAlias = tuple[RawDsetIndexItem, ...] | RawDsetIndexItem

SliceArgs: TypeAlias = (
    tuple[int | None]  # slice(None, stop, None)
    | tuple[int | None, int | None]  # slice(start, stop)
    | tuple[int | None, int | None, int | None]  # slice(start, stop, step)
)

DsetIndexItem: TypeAlias = RawDsetIndexItem | SliceArgs
DsetIndex: TypeAlias = Sequence[DsetIndexItem] | DsetIndexItem

NativeHdf5LinkType: TypeAlias = h5py.SoftLink | h5py.ExternalLink | h5py.VirtualLayout
