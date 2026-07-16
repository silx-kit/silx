from typing import NamedTuple, Literal

import numpy

AxisScaleType = Literal["linear", "log", "asinh"]


class PlotDataRange(NamedTuple):
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
