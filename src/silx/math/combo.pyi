import numpy


class _MinMaxResult(object):
    """Result from :func:`min_max`"""

    def __init__(self, minimum, min_pos, maximum, argmin, argmin_pos, argmax):
        ...

    @property
    def minimum(self) -> float:
        """Minimum value of the array"""
        ...

    @property
    def maximum(self) -> float:
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
    def min_positive(self) -> float | None:
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


def min_max(data: numpy.ndarray, min_positive: bint = False) -> _MinMaxResult:
    """:func:`min_max` implementation for floats skipping infinite values

    See :func:`min_max` for documentation.
    """
    ...
