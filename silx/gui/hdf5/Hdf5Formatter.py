# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""This package provides a class sharred by widgets to format HDF5 data as
text."""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "06/06/2018"

import numpy
from silx.third_party import six
from silx.gui import qt
from silx.gui.data.TextFormatter import TextFormatter

try:
    import h5py
except ImportError:
    h5py = None


class Hdf5Formatter(qt.QObject):
    """Formatter to convert HDF5 data to string.
    """

    formatChanged = qt.Signal()
    """Emitted when properties of the formatter change."""

    def __init__(self, parent=None, textFormatter=None):
        """
        Constructor

        :param qt.QObject parent: Owner of the object
        :param TextFormatter formatter: Text formatter
        """
        qt.QObject.__init__(self, parent)
        if textFormatter is not None:
            self.__formatter = textFormatter
        else:
            self.__formatter = TextFormatter(self)
        self.__formatter.formatChanged.connect(self.__formatChanged)

    def textFormatter(self):
        """Returns the used text formatter

        :rtype: TextFormatter
        """
        return self.__formatter

    def setTextFormatter(self, textFormatter):
        """Set the text formatter to be used

        :param TextFormatter textFormatter: The text formatter to use
        """
        if textFormatter is None:
            raise ValueError("Formatter expected but None found")
        if self.__formatter is textFormatter:
            return
        self.__formatter.formatChanged.disconnect(self.__formatChanged)
        self.__formatter = textFormatter
        self.__formatter.formatChanged.connect(self.__formatChanged)
        self.__formatChanged()

    def __formatChanged(self):
        self.formatChanged.emit()

    def humanReadableShape(self, dataset):
        if dataset.shape is None:
            return "none"
        if dataset.shape == tuple():
            return "scalar"
        shape = [str(i) for i in dataset.shape]
        text = u" \u00D7 ".join(shape)
        return text

    def humanReadableValue(self, dataset):
        if dataset.shape is None:
            return "No data"

        dtype = dataset.dtype
        if dataset.dtype.type == numpy.void:
            if dtype.fields is None:
                return "Raw data"

        if dataset.shape == tuple():
            numpy_object = dataset[()]
            text = self.__formatter.toString(numpy_object, dtype=dataset.dtype)
        else:
            if dataset.size < 5 and dataset.compression is None:
                numpy_object = dataset[0:5]
                text = self.__formatter.toString(numpy_object, dtype=dataset.dtype)
            else:
                dimension = len(dataset.shape)
                if dataset.compression is not None:
                    text = "Compressed %dD data" % dimension
                else:
                    text = "%dD data" % dimension
        return text

    def humanReadableType(self, dataset, full=False):
        if hasattr(dataset, "dtype"):
            dtype = dataset.dtype
        else:
            # Fallback...
            dtype = type(dataset)
        return self.humanReadableDType(dtype, full)

    def humanReadableDType(self, dtype, full=False):
        if dtype == six.binary_type or numpy.issubdtype(dtype, numpy.string_):
            text = "string"
            if full:
                text = "ASCII " + text
            return text
        elif dtype == six.text_type or numpy.issubdtype(dtype, numpy.unicode_):
            text = "string"
            if full:
                text = "UTF-8 " + text
            return text
        elif dtype.type == numpy.object_:
            ref = h5py.check_dtype(ref=dtype)
            if ref is not None:
                return "reference"
            vlen = h5py.check_dtype(vlen=dtype)
            if vlen is not None:
                text = self.humanReadableDType(vlen, full=full)
                if full:
                    text = "variable-length " + text
                return text
            return "object"
        elif dtype.type == numpy.bool_:
            return "bool"
        elif dtype.type == numpy.void:
            if dtype.fields is None:
                return "opaque"
            else:
                if not full:
                    return "compound"
                else:
                    fields = sorted(dtype.fields.items(), key=lambda e: e[1][1])
                    compound = [d[1][0] for d in fields]
                    compound = [self.humanReadableDType(d) for d in compound]
                    return "compound(%s)" % ", ".join(compound)
        elif numpy.issubdtype(dtype, numpy.integer):
            if h5py is not None:
                enumType = h5py.check_dtype(enum=dtype)
                if enumType is not None:
                    return "enum"

        text = str(dtype.newbyteorder('N'))
        if numpy.issubdtype(dtype, numpy.floating):
            if hasattr(numpy, "float128") and dtype == numpy.float128:
                text = "float80"
                if full:
                    text += " (padding 128bits)"
            elif hasattr(numpy, "float96") and dtype == numpy.float96:
                text = "float80"
                if full:
                    text += " (padding 96bits)"

        if full:
            if dtype.byteorder == "<":
                text = "Little-endian " + text
            elif dtype.byteorder == ">":
                text = "Big-endian " + text
            elif dtype.byteorder == "=":
                text = "Native " + text

        dtype = dtype.newbyteorder('N')
        return text

    def humanReadableHdf5Type(self, dataset):
        """Format the internal HDF5 type as a string"""
        t = dataset.id.get_type()
        class_ = t.get_class()
        if class_ == h5py.h5t.NO_CLASS:
            return "NO_CLASS"
        elif class_ == h5py.h5t.INTEGER:
            return "INTEGER"
        elif class_ == h5py.h5t.FLOAT:
            return "FLOAT"
        elif class_ == h5py.h5t.TIME:
            return "TIME"
        elif class_ == h5py.h5t.STRING:
            charset = t.get_cset()
            strpad = t.get_strpad()
            text = ""

            if strpad == h5py.h5t.STR_NULLTERM:
                text += "NULLTERM"
            elif strpad == h5py.h5t.STR_NULLPAD:
                text += "NULLPAD"
            elif strpad == h5py.h5t.STR_SPACEPAD:
                text += "SPACEPAD"
            else:
                text += "UNKNOWN_STRPAD"

            if t.is_variable_str():
                text += " VARIABLE"

            if charset == h5py.h5t.CSET_ASCII:
                text += " ASCII"
            elif charset == h5py.h5t.CSET_UTF8:
                text += " UTF8"
            else:
                text += " UNKNOWN_CSET"

            return text + " STRING"
        elif class_ == h5py.h5t.BITFIELD:
            return "BITFIELD"
        elif class_ == h5py.h5t.OPAQUE:
            return "OPAQUE"
        elif class_ == h5py.h5t.COMPOUND:
            return "COMPOUND"
        elif class_ == h5py.h5t.REFERENCE:
            return "REFERENCE"
        elif class_ == h5py.h5t.ENUM:
            return "ENUM"
        elif class_ == h5py.h5t.VLEN:
            return "VLEN"
        elif class_ == h5py.h5t.ARRAY:
            return "ARRAY"
        else:
            return "UNKNOWN_CLASS"
