from dataclasses import dataclass

import numpy

from ..plot.items.axis import AxisScaleType


@dataclass(kw_only=True)
class _BaseEntity:
    errors: numpy.ndarray | None
    name: str
    scale: AxisScaleType = "linear"

    def __post_init__(self):
        self.name = "" if self.name is None else self.name
        self.scale = "linear" if self.scale is None else self.scale

    def get_errors(self, selection):
        if self.errors is None:
            return None

        return self.errors[selection]


@dataclass
class Signal(_BaseEntity):
    values: numpy.ndarray


@dataclass
class Axis(_BaseEntity):
    values: numpy.ndarray | None
