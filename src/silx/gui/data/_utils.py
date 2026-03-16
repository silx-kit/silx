import numbers

import h5py
import numpy

from ...io.commonh5 import Dataset
from ...io.nxdata.parse import NXdata
from ...math.calibration import (
    AbstractCalibration,
    ArrayCalibration,
    LinearCalibration,
    NoCalibration,
)
from ..hdf5 import H5Node
from ..plot.items import ImageBase

ImageAxis = h5py.Dataset | Dataset | None


def normalizeData(data):
    """Returns a normalized data.

    If the data embed a numpy data or a dataset it is returned.
    Else returns the input data."""
    if isinstance(data, H5Node):
        if data.is_broken:
            return None
        return data.h5py_object
    return data


def normalizeComplex(data):
    """Returns a normalized complex data.

    If the data is a numpy data with complex, returns the
    absolute value.
    Else returns the input data."""
    if hasattr(data, "dtype"):
        isComplex = numpy.issubdtype(data.dtype, numpy.complexfloating)
    else:
        isComplex = isinstance(data, numbers.Complex)
    if isComplex:
        data = numpy.absolute(data)
    return data


def isRgba(nxd: NXdata) -> bool:
    return (
        nxd.interpretation in ("rgb-image", "rgba-image")
        and nxd.signal_ndim >= 3
        and nxd.signal.shape[-1] in (3, 4)
    )


def isScatter(nxd: NXdata, naxes: int) -> bool:
    if nxd.signal.ndim != 1:
        return False

    # Check that all axes match the signal number of values
    if any(axis is None or axis.shape != nxd.signal.shape for axis in nxd.axes):
        return False
    return len(nxd.axes) == naxes


def _getAxisCalib(x_axis: ImageAxis, axis_length: int) -> AbstractCalibration:
    if x_axis is None:
        # no calibration
        return ArrayCalibration(numpy.arange(axis_length))
    if numpy.isscalar(x_axis) or len(x_axis) == 1:
        # constant axis
        return ArrayCalibration(x_axis * numpy.ones((axis_length,)))
    if len(x_axis) == 2:
        # linear calibration
        return LinearCalibration(slope=x_axis[0], y_intercept=x_axis[1])
    raise ValueError("Expected a scalar or two values. Got {x_axis}.")


def getAxesCalib(
    image_shape: tuple[int, int], x_axis: ImageAxis, y_axis: ImageAxis
) -> tuple[AbstractCalibration, AbstractCalibration]:
    if x_axis is None and y_axis is None:
        return NoCalibration(), NoCalibration()

    return _getAxisCalib(x_axis, image_shape[1]), _getAxisCalib(y_axis, image_shape[0])


def setImageCoords(
    item: ImageBase, xcalib: AbstractCalibration, ycalib: AbstractCalibration
):
    item.setOrigin((xcalib(0), ycalib(0)))
    item.setScale((xcalib.get_slope(), ycalib.get_slope()))
