import numpy
from typing import TypeVar, Generic

T = TypeVar("T")

class _MinMaxResult(object, Generic[T]):
    """Result from :func:`min_max`"""

    def __init__(self, minimum, min_pos, maximum, argmin, argmin_pos, argmax): ...
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

    def __getitem__(self, key: int): ...

def min_max(
    data: numpy.ndarray, min_positive: bint = False
) -> _MinMaxResult[int] | _MinMaxResult[float]:
    """:func:`min_max` implementation for floats skipping infinite values

    See :func:`min_max` for documentation.
    """
    ...


def mean_std(data: numpy.ndarray,
             ddof: float = 0.0,
             mask: numpy.ndarray|None = None,
             dummy: float = numpy.nan,
             delta_dummy: float = 0.0) -> Tuple[float, float]:
    """Computes mean and estimation of std in a single pass.

    Based on formula #12, #13 and #28 from :
    https://ds.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf

    All calculations are performed in double-precision (ieee754-64 bits)
    since single precision offers no advantage in speed and reduces
    significantly the quality of the variance.

    :param data: Array-like dataset,
    :param ddof:
       Means Delta Degrees of Freedom.
       The divisor used in calculations is (number_of_valid_points - ddof).
       Default: 0 (as in numpy.std).
    :param mask: array with 0 for valid values, same size as data
    :param dummy: dynamic mask for value=dummy. NaNs are always invalid
    :param delta_dummy: dynamic mask for abs(value-dummy)<=delta_dummy
    :returns: A tuple: (mean, std)
    :raises: ValueError if data is empty
             RuntimeError if the shape of the mask differs from data
    """
    ...