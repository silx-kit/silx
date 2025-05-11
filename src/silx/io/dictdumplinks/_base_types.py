from typing import TypeAlias
from types import EllipsisType
from collections.abc import Sequence

RawDsetIndexItem: TypeAlias = int | slice | EllipsisType | None
RawDsetIndex: TypeAlias = tuple[RawDsetIndexItem, ...] | RawDsetIndexItem

SliceArgs: TypeAlias = (
    tuple[int | None]  # slice(None, stop, None)
    | tuple[int | None, int | None]  # slice(start, stop)
    | tuple[int | None, int | None, int | None]  # slice(start, stop, step)
)

DsetIndexItem: TypeAlias = RawDsetIndexItem | SliceArgs
DsetIndex: TypeAlias = Sequence[DsetIndexItem] | DsetIndexItem
