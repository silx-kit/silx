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
"""This package provides a class sharred by widget from the
data module to format data as text in the same way."""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "24/07/2018"

import numpy
import numbers
from silx.third_party import six
from silx.gui import qt
import logging

try:
    import h5py
except ImportError:
    h5py = None


_logger = logging.getLogger(__name__)


class TextFormatter(qt.QObject):
    """Formatter to convert data to string.

    The method :meth:`toString` returns a formatted string from an input data
    using parameters set to this object.

    It support most python and numpy data, expecting dictionary. Unsupported
    data are displayed using the string representation of the object (`str`).

    It provides a set of parameters to custom the formatting of integer and
    float values (:meth:`setIntegerFormat`, :meth:`setFloatFormat`).

    It also allows to custom the use of quotes to display text data
    (:meth:`setUseQuoteForText`), and custom unit used to display imaginary
    numbers (:meth:`setImaginaryUnit`).

    The object emit an event `formatChanged` every time a parametter is
    changed.
    """

    formatChanged = qt.Signal()
    """Emitted when properties of the formatter change."""

    def __init__(self, parent=None, formatter=None):
        """
        Constructor

        :param qt.QObject parent: Owner of the object
        :param TextFormatter formatter: Instantiate this object from the
            formatter
        """
        qt.QObject.__init__(self, parent)
        if formatter is not None:
            self.__integerFormat = formatter.integerFormat()
            self.__floatFormat = formatter.floatFormat()
            self.__useQuoteForText = formatter.useQuoteForText()
            self.__imaginaryUnit = formatter.imaginaryUnit()
            self.__enumFormat = formatter.enumFormat()
        else:
            self.__integerFormat = "%d"
            self.__floatFormat = "%g"
            self.__useQuoteForText = True
            self.__imaginaryUnit = u"j"
            self.__enumFormat = u"%(name)s(%(value)d)"

    def integerFormat(self):
        """Returns the format string controlling how the integer data
        are formated by this object.

        This is the C-style format string used by python when formatting
        strings with the modulus operator.

        :rtype: str
        """
        return self.__integerFormat

    def setIntegerFormat(self, value):
        """Set format string controlling how the integer data are
        formated by this object.

        :param str value: Format string (e.g. "%d", "%i", "%08i").
            This is the C-style format string used by python when formatting
            strings with the modulus operator.
        """
        if self.__integerFormat == value:
            return
        self.__integerFormat = value
        self.formatChanged.emit()

    def floatFormat(self):
        """Returns the format string controlling how the floating-point data
        are formated by this object.

        This is the C-style format string used by python when formatting
        strings with the modulus operator.

        :rtype: str
        """
        return self.__floatFormat

    def setFloatFormat(self, value):
        """Set format string controlling how the floating-point data are
        formated by this object.

        :param str value: Format string (e.g. "%.3f", "%d", "%-10.2f",
            "%10.3e").
            This is the C-style format string used by python when formatting
            strings with the modulus operator.
        """
        if self.__floatFormat == value:
            return
        self.__floatFormat = value
        self.formatChanged.emit()

    def useQuoteForText(self):
        """Returns true if the string data are formatted using double quotes.

        Else, no quotes are used.
        """
        return self.__integerFormat

    def setUseQuoteForText(self, useQuote):
        """Set the use of quotes to delimit string data.

        :param bool useQuote: True to use quotes.
        """
        if self.__useQuoteForText == useQuote:
            return
        self.__useQuoteForText = useQuote
        self.formatChanged.emit()

    def imaginaryUnit(self):
        """Returns the unit display for imaginary numbers.

        :rtype: str
        """
        return self.__imaginaryUnit

    def setImaginaryUnit(self, imaginaryUnit):
        """Set the unit display for imaginary numbers.

        :param str imaginaryUnit: Unit displayed after imaginary numbers
        """
        if self.__imaginaryUnit == imaginaryUnit:
            return
        self.__imaginaryUnit = imaginaryUnit
        self.formatChanged.emit()

    def setEnumFormat(self, value):
        """Set format string controlling how the enum data are
        formated by this object.

        :param str value: Format string (e.g. "%(name)s(%(value)d)").
            This is the C-style format string used by python when formatting
            strings with the modulus operator.
        """
        if self.__enumFormat == value:
            return
        self.__enumFormat = value
        self.formatChanged.emit()

    def enumFormat(self):
        """Returns the format string controlling how the enum data
        are formated by this object.

        This is the C-style format string used by python when formatting
        strings with the modulus operator.

        :rtype: str
        """
        return self.__enumFormat

    def __formatText(self, text):
        if self.__useQuoteForText:
            text = "\"%s\"" % text.replace("\\", "\\\\").replace("\"", "\\\"")
        return text

    def __formatBinary(self, data):
        if isinstance(data, numpy.void):
            if six.PY2:
                data = [ord(d) for d in data.data]
            else:
                data = data.item()
                if isinstance(data, numpy.ndarray):
                    # Before numpy 1.15.0 the item API was returning a numpy array
                    data = data.astype(numpy.uint8)
                else:
                    # Now it is supposed to be a bytes type
                    pass
        elif six.PY2:
            data = [ord(d) for d in data]
            # In python3 data is already a bytes array
        data = ["\\x%02X" % d for d in data]
        if self.__useQuoteForText:
            return "b\"%s\"" % "".join(data)
        else:
            return "".join(data)

    def __formatSafeAscii(self, data):
        if six.PY2:
            data = [ord(d) for d in data]
        data = [chr(d) if (d > 0x20 and d < 0x7F) else "\\x%02X" % d for d in data]
        if self.__useQuoteForText:
            data = [c if c != '"' else "\\" + c for c in data]
            return "b\"%s\"" % "".join(data)
        else:
            return "".join(data)

    def __formatCharString(self, data):
        """Format text of char.

        From the specifications we expect to have ASCII, but we also allow
        CP1252 in some ceases as fallback.

        If no encoding fits, it will display a readable ASCII chars, with
        escaped chars (using the python syntax) for non decoded characters.

        :param data: A binary string of char expected in ASCII
        :rtype: str
        """
        try:
            text = "%s" % data.decode("ascii")
            return self.__formatText(text)
        except UnicodeDecodeError:
            # Here we can spam errors, this is definitly a badly
            # generated file
            _logger.error("Invalid ASCII string %s.", data)
            if data == b"\xB0":
                _logger.error("Fallback using cp1252 encoding")
                return self.__formatText(u"\u00B0")
        return self.__formatSafeAscii(data)

    def __formatH5pyObject(self, data, dtype):
        # That's an HDF5 object
        ref = h5py.check_dtype(ref=dtype)
        if ref is not None:
            if bool(data):
                return "REF"
            else:
                return "NULL_REF"
        vlen = h5py.check_dtype(vlen=dtype)
        if vlen is not None:
            if vlen == six.text_type:
                # HDF5 UTF8
                return self.__formatText(data)
            elif vlen == six.binary_type:
                # HDF5 ASCII
                return self.__formatCharString(data)
            elif isinstance(vlen, numpy.dtype):
                return self.toString(data, vlen)
        return None

    def toString(self, data, dtype=None):
        """Format a data into a string using formatter options

        :param object data: Data to render
        :param dtype: enforce a dtype (mostly used to remember the h5py dtype,
             special h5py dtypes are not propagated from array to items)
        :rtype: str
        """
        if isinstance(data, tuple):
            text = [self.toString(d) for d in data]
            return "(" + " ".join(text) + ")"
        elif isinstance(data, list):
            text = [self.toString(d) for d in data]
            return "[" + " ".join(text) + "]"
        elif isinstance(data, (numpy.ndarray)):
            if dtype is None:
                dtype = data.dtype
            if data.shape == ():
                # it is a scaler
                return self.toString(data[()], dtype)
            else:
                text = [self.toString(d, dtype) for d in data]
                return "[" + " ".join(text) + "]"
        if dtype is not None and dtype.kind == 'O':
            text = self.__formatH5pyObject(data, dtype)
            if text is not None:
                return text
        elif isinstance(data, numpy.void):
            if dtype is None:
                dtype = data.dtype
            if dtype.fields is not None:
                text = []
                for index, field in enumerate(dtype.fields.items()):
                    text.append(field[0] + ":" + self.toString(data[index], field[1][0]))
                return "(" + " ".join(text) + ")"
            return self.__formatBinary(data)
        elif isinstance(data, (numpy.unicode_, six.text_type)):
            return self.__formatText(data)
        elif isinstance(data, (numpy.string_, six.binary_type)):
            if dtype is None and hasattr(data, "dtype"):
                dtype = data.dtype
            if dtype is not None:
                # Maybe a sub item from HDF5
                if dtype.kind == 'S':
                    return self.__formatCharString(data)
                elif dtype.kind == 'O':
                    if h5py is not None:
                        text = self.__formatH5pyObject(data, dtype)
                        if text is not None:
                            return text
            try:
                # Try ascii/utf-8
                text = "%s" % data.decode("utf-8")
                return self.__formatText(text)
            except UnicodeDecodeError:
                pass
            return self.__formatBinary(data)
        elif isinstance(data, six.string_types):
            text = "%s" % data
            return self.__formatText(text)
        elif isinstance(data, (numpy.integer)):
            if dtype is None:
                dtype = data.dtype
            if h5py is not None:
                enumType = h5py.check_dtype(enum=dtype)
                if enumType is not None:
                    for key, value in enumType.items():
                        if value == data:
                            result = {}
                            result["name"] = key
                            result["value"] = data
                            return self.__enumFormat % result
            return self.__integerFormat % data
        elif isinstance(data, (numbers.Integral)):
            return self.__integerFormat % data
        elif isinstance(data, (numbers.Real, numpy.floating)):
            # It have to be done before complex checking
            return self.__floatFormat % data
        elif isinstance(data, (numpy.complexfloating, numbers.Complex)):
            text = ""
            if data.real != 0:
                text += self.__floatFormat % data.real
            if data.real != 0 and data.imag != 0:
                if data.imag < 0:
                    template = self.__floatFormat + " - " + self.__floatFormat + self.__imaginaryUnit
                    params = (data.real, -data.imag)
                else:
                    template = self.__floatFormat + " + " + self.__floatFormat + self.__imaginaryUnit
                    params = (data.real, data.imag)
            else:
                if data.imag != 0:
                    template = self.__floatFormat + self.__imaginaryUnit
                    params = (data.imag)
                else:
                    template = self.__floatFormat
                    params = (data.real)
            return template % params
        elif h5py is not None and isinstance(data, h5py.h5r.Reference):
            dtype = h5py.special_dtype(ref=h5py.Reference)
            text = self.__formatH5pyObject(data, dtype)
            return text
        elif h5py is not None and isinstance(data, h5py.h5r.RegionReference):
            dtype = h5py.special_dtype(ref=h5py.RegionReference)
            text = self.__formatH5pyObject(data, dtype)
            return text
        elif isinstance(data, numpy.object_) or dtype is not None:
            if dtype is None:
                dtype = data.dtype
            if h5py is not None:
                text = self.__formatH5pyObject(data, dtype)
                if text is not None:
                    return text
            # That's a numpy object
            return str(data)
        return str(data)
