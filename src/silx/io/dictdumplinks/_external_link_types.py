from typing import Any

import h5py

from ._base_types import LinkInterface


class ExternalLink(LinkInterface):
    def __init__(self, file_path: str, data_path: str):
        self._file_path = file_path
        self._data_path = data_path
        super().__init__()

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def data_path(self) -> str:
        return self._data_path

    def __eq__(self, other: Any):
        if isinstance(other, ExternalLink):
            return (
                self.file_path == other.file_path and self.data_path == other.data_path
            )
        return super().__eq__(other)

    def create(self, parent: h5py.Group, name: str) -> None:
        parent[name] = h5py.ExternalLink(self._file_path, self._data_path)

    def serialize(self) -> str:
        return f"{self.file_path}::{self.data_path}"
