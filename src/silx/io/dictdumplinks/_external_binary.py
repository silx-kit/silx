from typing import Any
from typing import Literal

import h5py
from numpy.typing import DTypeLike
from pydantic import BaseModel
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from ..url import DataUrl
from ._link_types import Hdf5Link
from ._link_types import Hdf5LinkModel
from ._utils import normalize_ext_source_path


class ExternalLinkSourceV1(BaseModel):
    file_path: str
    offset: NonNegativeInt
    size: PositiveInt


class ExternalLinkModelV1(Hdf5LinkModel):
    """Attention: relative file names in external HDF5 datasets are relative
    with respect to the current working directory, not relative to the parent file.
    """

    dictdump_schema: Literal["external_binary_link_v1"]
    shape: tuple[int, ...]
    dtype: Any  # DTypeLike gives pydantic.errors.PydanticUserError on Python < 3.12.
    sources: list[ExternalLinkSourceV1]

    def tolink(self, source: DataUrl) -> "ExternalBinaryLink":
        model = self.model_copy(deep=True)
        model.sources = [
            (
                normalize_ext_source_path(ext_source.file_path, source),
                ext_source.offset,
                ext_source.size,
            )
            for ext_source in model.sources
        ]
        return ExternalBinaryLink(model)


class ExternalBinaryLink(Hdf5Link):
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
