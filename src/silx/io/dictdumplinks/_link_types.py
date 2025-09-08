from collections.abc import Sequence
from typing import TypeAlias

from ..url import DataUrl
from ._base_types import NativeHdf5LinkType
from ._external_binary import ExternalBinaryLink

Hdf5LinkType: TypeAlias = NativeHdf5LinkType | ExternalBinaryLink
SerializedHdf5LinkType: TypeAlias = str | DataUrl | Sequence[str | DataUrl] | dict
