import numbers

import numpy

from ...io.nxdata.parse import NXdata
from ..hdf5 import H5Node


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
