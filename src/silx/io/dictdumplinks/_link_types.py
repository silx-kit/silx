from abc import abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

from pydantic import BaseModel

from ..url import DataUrl
from ._base_types import NativeHdf5LinkType


class Hdf5Link:
    """Base class for implementing links that are not implemented
    in ``h5py`` like ``SoftLink``, ``ExternalLink`` and ``VirtualDataset``.
    """

    pass


Hdf5LinkType: TypeAlias = NativeHdf5LinkType | Hdf5Link
SerializedHdf5LinkType: TypeAlias = str | DataUrl | Sequence[str | DataUrl] | dict


class Hdf5LinkModel(BaseModel):
    """Base class for parsing and validating serialized link definitions."""

    dictdump_schema: str

    @abstractmethod
    def tolink(self, source: DataUrl) -> "Hdf5LinkModel":
        pass
