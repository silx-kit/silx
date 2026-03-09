import numbers
import numpy

import silx.io
from silx.io import nxdata
from silx.io.nxdata import get_attr_as_unicode

from ._utils import normalizeData


class DataInfo:
    """Store extracted information from a data"""

    def __init__(self, data):
        self.__priorities = {}
        data = self.normalizeData(data)
        self.isArray = False
        self.interpretation = None
        self.isNumeric = False
        self.isVoid = False
        self.isComplex = False
        self.isBoolean = False
        self.isRecord = False
        self.hasNXdata = False
        self.isInvalidNXdata = False
        self.countNumericColumns = 0
        self.shape = tuple()
        self.dim = 0
        self.size = 0

        if data is None:
            return

        if silx.io.is_group(data):
            nxd = nxdata.get_default(data)
            nx_class = get_attr_as_unicode(data, "NX_class")
            if nxd is not None:
                self.hasNXdata = True
                self.isInvalidNXdata = False
            elif nx_class == "NXdata":
                # group claiming to be NXdata could not be parsed
                self.isInvalidNXdata = True
            elif nx_class == "NXroot" or silx.io.is_file(data):
                # root claiming to have a default entry
                if "default" in data.attrs:
                    def_entry = data.attrs["default"]
                    if def_entry in data and "default" in data[def_entry].attrs:
                        # and entry claims to have default NXdata
                        self.isInvalidNXdata = True
            elif "default" in data.attrs:
                # group claiming to have a default NXdata could not be parsed
                self.isInvalidNXdata = True

        if isinstance(data, numpy.ndarray):
            self.isArray = True
        elif silx.io.is_dataset(data) and data.shape != tuple():
            self.isArray = True
        else:
            self.isArray = False

        if silx.io.is_dataset(data):
            if "interpretation" in data.attrs:
                self.interpretation = get_attr_as_unicode(data, "interpretation")
            else:
                self.interpretation = None
        elif self.hasNXdata:
            self.interpretation = nxd.interpretation
        else:
            self.interpretation = None

        if hasattr(data, "dtype"):
            if numpy.issubdtype(data.dtype, numpy.void):
                # That's a real opaque type, else it is a structured type
                self.isVoid = data.dtype.fields is None
            self.isNumeric = numpy.issubdtype(data.dtype, numpy.number)
            self.isRecord = data.dtype.fields is not None
            self.isComplex = numpy.issubdtype(data.dtype, numpy.complexfloating)
            self.isBoolean = numpy.issubdtype(data.dtype, numpy.bool_)
        elif self.hasNXdata:
            self.isNumeric = numpy.issubdtype(nxd.signal.dtype, numpy.number)
            self.isComplex = numpy.issubdtype(nxd.signal.dtype, numpy.complexfloating)
            self.isBoolean = numpy.issubdtype(nxd.signal.dtype, numpy.bool_)
        else:
            self.isNumeric = isinstance(data, numbers.Number)
            self.isComplex = isinstance(data, numbers.Complex)
            self.isBoolean = isinstance(data, bool)
            self.isRecord = False

        if hasattr(data, "shape"):
            self.shape = data.shape
        elif self.hasNXdata:
            self.shape = nxd.signal.shape
        else:
            self.shape = tuple()
        if self.shape is not None:
            self.dim = len(self.shape)

        if hasattr(data, "shape") and data.shape is None:
            # This test is expected to avoid to fall done on the h5py issue
            # https://github.com/h5py/h5py/issues/1044
            self.size = 0
        elif hasattr(data, "size"):
            self.size = int(data.size)
        else:
            self.size = 1

        if hasattr(data, "dtype"):
            if data.dtype.fields is not None:
                for field in data.dtype.fields:
                    if numpy.issubdtype(data.dtype[field], numpy.number):
                        self.countNumericColumns += 1

    def normalizeData(self, data):
        """Returns a normalized data if the embed a numpy or a dataset.
        Else returns the data."""
        return normalizeData(data)

    def cachePriority(self, view, priority):
        self.__priorities[view] = priority

    def getPriority(self, view):
        return self.__priorities[view]
