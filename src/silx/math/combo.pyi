import numpy
from typing import TypeVar, Generic


T = TypeVar('T')


class _MinMaxResult(object, Generic[T]):
    """Result from :func:`min_max`"""

    def __init__(self, minimum, min_pos, maximum, argmin, argmin_pos, argmax):
        ...

    @property
    def minimum(self) -> T:
        """Minimum value of the array"""
        ...

    @property
    def maximum(self) -> T:
        """Maximum value of the array"""
        ...

    @property
    def argmin(self) -> int:
        """Index of the first occurrence of the minimum value"""
        ...

    @property
    def argmax(self) -> int:
        """Index of the first occurrence of the maximum value"""
        ...

    @property
    def min_positive(self) -> T | None:
        """
        Strictly positive minimum value

        It is None if no value is strictly positive.
        """
        ...

    @property
    def argmin_positive(self) -> int | None:
        """
        Index of the strictly positive minimum value.

        It is None if no value is strictly positive.
        It is the index of the first occurrence.
        """
        ...

    def __getitem__(self, key: int):
        ...


def min_max(data: numpy.ndarray, min_positive: bint = False) -> _MinMaxResult[int] | _MinMaxResult[float]:
    """:func:`min_max` implementation for floats skipping infinite values

    See :func:`min_max` for documentation.
    """
    ...
