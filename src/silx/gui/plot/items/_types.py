from typing import NamedTuple

from .types import AxisScaleType


class AxisInfo(NamedTuple):
    vmin: float
    vmax: float
    auto: bool
    scale: AxisScaleType

    def limits(self) -> tuple[float, float]:
        return self.vmin, self.vmax

    def has_limits(self) -> bool:
        return not self.isnan(self.vmin) and not self.isnan(self.vmax)

    @property
    def is_logarithmic(self) -> bool:
        return self.scale == "log"


class AxesInfo(NamedTuple):
    x: AxisInfo
    y: AxisInfo
