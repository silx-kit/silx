from typing import NewType
from collections.abc import Sequence

RawDsetIndexItem = NewType("RawDsetIndexItem", int | slice | type(Ellipsis) | None)
RawDsetIndex = NewType("RawDsetIndex", tuple[RawDsetIndexItem] | RawDsetIndexItem)

SliceArgs = NewType(
    "SliceArgs",
    Sequence[int | None]  # slice(None, stop, None)
    | Sequence[int | None, int | None]  # slice(start, stop)
    | Sequence[int | None, int | None, int | None],  # slice(start, stop, step)
)
DsetIndexItem = NewType("DsetIndexItem", RawDsetIndexItem | SliceArgs)
DsetIndex = NewType("DsetIndex", Sequence[DsetIndexItem] | DsetIndexItem)
