from typing import Any
from typing import Literal

import h5py
from numpy.typing import DTypeLike
from pydantic import BaseModel

from ..url import DataUrl
from ._utils import normalize_ext_source_path


class ExternalLinkModelV1(BaseModel):
    """Attention: relative file names in external HDF5 datasets are relative
    with respect to the current working directory, not relative to the parent file.
    """

    dictdump_schema: Literal["external_binary_link_v1"]
    shape: tuple[int, ...]
    dtype: Any  # DTypeLike gives pydantic.errors.PydanticUserError on Python < 3.12.
    sources: list[tuple[str, int, int]]  # file name, byte offset, byte size

    def tolink(self, source: DataUrl) -> "ExternalBinaryLink":
        model = self.model_copy(deep=True)
        model.sources = [
            (normalize_ext_source_path(file_path, source), offset, count)
            for file_path, offset, count in model.sources
        ]
        return ExternalBinaryLink(model)


class ExternalBinaryLink:
    def __init__(self, model: ExternalLinkModelV1) -> None:
        self._model = model

    @property
    def shape(self) -> tuple[int, ...]:
        return self._model.shape

    @property
    def dtype(self) -> DTypeLike:
        return self._model.dtype

    @property
    def sources(self) -> list[tuple[str, int, int]]:
        return self._model.sources

    def __eq__(self, other: Any):
        if isinstance(other, ExternalBinaryLink):
            return (
                self.shape == other.shape
                and self.dtype == other.dtype
                and self.sources == other.sources
            )
        return super().__eq__(other)

    def create(self, parent: h5py.Group, name: str) -> h5py.Dataset:
        return parent.create_dataset(
            name=name, shape=self.shape, dtype=self.dtype, external=self.sources
        )

    def serialize(self) -> dict:
        return self._model.model_dump()
