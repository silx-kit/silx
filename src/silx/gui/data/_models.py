from dataclasses import dataclass

import h5py
import numpy

from ...io.nxdata._utils import AxisDataset
from ..plot.items import types


@dataclass(kw_only=True)
class _BaseNxField:
    errors: h5py.Dataset | numpy.ndarray | None
    name: str
    scale: types.AxisScaleType | None = None

    def __post_init__(self):
        self.name = "" if self.name is None else self.name

    def get_errors(self, selection) -> numpy.ndarray | None:
        if self.errors is None:
            return None

        return self.errors[selection]


@dataclass
class Signal(_BaseNxField):
    values: h5py.Dataset | numpy.ndarray


@dataclass
class Axis(_BaseNxField):
    values: AxisDataset
