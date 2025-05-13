from typing import Any

import h5py

from ._base_types import LinkInterface


class InternalLink(LinkInterface):
    def __init__(self, data_path: str):
        self._data_path = data_path
        super().__init__()

    @property
    def data_path(self) -> str:
        return self._data_path

    def __eq__(self, other: Any):
        if isinstance(other, InternalLink):
            return self.data_path == other.data_path
        return super().__eq__(other)

    def create(self, parent: h5py.Group, name: str) -> None:
        parent[name] = h5py.SoftLink(self._data_path)

    def serialize(self) -> str:
        return self.data_path
