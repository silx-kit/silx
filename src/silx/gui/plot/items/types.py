from typing import NamedTuple

import numpy


class PlotDataRange(NamedTuple):
    """
    Object returned when requesting the data range.
    """

    x: tuple[float, float] | None
    y: tuple[float, float] | None
    yright: tuple[float, float] | None


class ItemBounds(NamedTuple):
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @classmethod
    def from_values(
        cls,
        xmin: float | None,
        xmax: float | None,
        ymin: float | None,
        ymax: float | None,
    ) -> "ItemBounds | None":
        """
        Create a :class:`ItemBounds` instance from optional values.

        ``None`` values are converted to ``NaN``.

        :param xmin: Minimum X bound or None.
        :param xmax: Maximum X bound or None.
        :param ymin: Minimum Y bound or None.
        :param ymax: Maximum Y bound or None.

        :returns:
            Returns a :class:`ItemBounds` instance. Returns ``None`` if
            all values are ``None`` or NaN, meaning the bounds are undefined.
        :rtype: :class:`ItemBounds` or ``None``
        """

        def none_to_nan(v):
            return float("nan") if v is None else v

        values = numpy.array(list(map(none_to_nan, (xmin, xmax, ymin, ymax))))

        if numpy.all(numpy.isnan(values)):
            return None

        return cls(*values)


class AxisInfo(NamedTuple):
    vmin: float
    vmax: float
    auto: bool
    log: bool

    def limits(self) -> tuple[float, float]:
        return self.vmin, self.vmax

    def has_limits(self) -> bool:
        return not self.isnan(self.vmin) and not self.isnan(self.vmax)


class AxesInfo(NamedTuple):
    x: AxisInfo
    y: AxisInfo
