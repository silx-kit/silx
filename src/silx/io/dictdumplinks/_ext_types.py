from collections.abc import Sequence
from typing import TypedDict, NamedTuple, Literal

from numpy.typing import DTypeLike


class ExtSchemaV1(TypedDict):
    dictdump_schema: Literal["external_binary_link_v1"]
    shape: Sequence[int]
    dtype: str | DTypeLike
    sources: Sequence[Sequence[str, int, int]]


class Ext(NamedTuple):
    """Attention: relative file names in external HDF5 datasets are relative
    with respect to the current working directory, not relative to the parent file.
    """

    shape: tuple[int]
    dtype: str | DTypeLike
    sources: list[tuple[str, int, int]]  # (file_path, offset, bytecount)


def parse_ext_schema_v1(target: ExtSchemaV1) -> Ext:
    shape = tuple(target["shape"])
    dtype = target["dtype"]
    sources = [
        (file_path, offset, bytecount)
        for file_path, offset, bytecount in target["sources"]
    ]
    return Ext(shape, dtype, sources)
