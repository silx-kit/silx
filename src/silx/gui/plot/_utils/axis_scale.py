import numbers

import numpy
from numpy.typing import ArrayLike

from ..items.types import AxisScaleType


# Float 32 info ###############################################################
# Using min/max value below limits of float32
# so operation with such value (e.g., max - min) do not overflow

FLOAT32_SAFE_MIN = -1e37
FLOAT32_MINPOS = numpy.finfo(numpy.float32).tiny
FLOAT32_SAFE_MAX = 1e37


def isValid(axisScale: AxisScaleType, value: float) -> bool:
    if not numpy.isfinite(value):
        return False

    if axisScale == "linear":
        return True
    elif axisScale == "log":
        return value > 0.0
    elif axisScale == "asinh":
        return True
    else:
        raise ValueError(f"Unsupported axis scale: {axisScale}")


def apply(axisScale: AxisScaleType, value: float | ArrayLike) -> float | numpy.ndarray:
    if not isinstance(value, numbers.Real):
        value = numpy.asarray(value)

    if axisScale == "linear":
        return value
    elif axisScale == "log":
        if isinstance(value, numbers.Real):
            return numpy.log10(value) if value > 0.0 else float("nan")
        else:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                scaled = numpy.log10(value)
            scaled[~numpy.isfinite(scaled)] = numpy.nan
            return scaled
    elif axisScale == "asinh":
        return numpy.asinh(value)
    else:
        raise ValueError(f"Unsupported axis scale: {axisScale}")


def revert(axisScale: AxisScaleType, value: float | ArrayLike) -> float | numpy.ndarray:
    if not isinstance(value, numbers.Real):
        value = numpy.asarray(value)

    if axisScale == "linear":
        return value
    elif axisScale == "log":
        with numpy.errstate(over="ignore"):
            return numpy.pow(10.0, value)
    elif axisScale == "asinh":
        with numpy.errstate(over="ignore"):
            return numpy.sinh(value)
    else:
        raise ValueError(f"Unsupported axis scale: {axisScale}")


def safeRange(axisScale: AxisScaleType) -> tuple[float, float]:
    """Return axis range (min, max) below limits of float32 so that max - min does not overflow"""
    if axisScale == "linear":
        return FLOAT32_SAFE_MIN, FLOAT32_SAFE_MAX
    elif axisScale == "log":
        return FLOAT32_MINPOS, FLOAT32_SAFE_MAX
    elif axisScale == "asinh":
        return FLOAT32_SAFE_MIN, FLOAT32_SAFE_MAX
    else:
        raise ValueError(f"Unsupported axis scale: {axisScale}")


def inSafeRange(
    axisScale: AxisScaleType, value: float | ArrayLike
) -> bool | numpy.ndarray:
    if not isinstance(value, numbers.Real):
        value = numpy.asarray(value)

    min_, max_ = safeRange(axisScale)
    return (min_ <= value) & (value <= max_)


def clipToSafeRange(
    axisScale: AxisScaleType, value: float | ArrayLike
) -> float | numpy.ndarray:
    if not isinstance(value, numbers.Real):
        value = numpy.asarray(value)

    axisMin, axisMax = safeRange(axisScale)
    return numpy.clip(value, axisMin, axisMax)
