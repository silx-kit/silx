from typing import TypeAlias
from collections.abc import Sequence

from ..url import DataUrl

from ._base_types import NativeHdf5LinkType
from ._vds_types import VdsSchemaV1
from ._external_binary_types import ExternalBinaryLink
from ._external_binary_types import ExtSchemaV1

Hdf5LinkType: TypeAlias = NativeHdf5LinkType | ExternalBinaryLink
SerializedHdf5LinkType: TypeAlias = (
    str | DataUrl | Sequence[str | DataUrl] | VdsSchemaV1 | ExtSchemaV1
)
