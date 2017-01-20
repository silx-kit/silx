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
__date__ = "20/01/2017"

import numpy
import numbers
import binascii
try:
    from silx.third_party import six
except ImportError:
    import six


class TextFormatter():
    """Formatter to convert data to string"""

    def __init__(self):
        self.__integerFormat = "%d"
        self.__floatFormat = "%g"
        self.__useQuoteForText = True
        self.__imaginaryUnit = u"j"

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
        self.__integerFormat = value

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
        self.__floatFormat = value

    def useQuoteForText(self):
        """Returns true if the string data are formatted using double quotes.

        Else, no quotes are used.
        """
        return self.__integerFormat

    def setUseQuoteForText(self, useQuote):
        """Set the use of quotes to delimit string data.

        :param bool useQuote: True to use quotes.
        """
        self.__useQuoteForText = useQuote

    def imaginaryUnit(self):
        """Returns the unit display for imaginary numbers.

        :rtype: str
        """
        return self.__imaginaryUnit

    def setImaginaryUnit(self, imaginaryUnit):
        """Set the unit display for imaginary numbers.

        :param str imaginaryUnit: Unit displayed after imaginary numbers
        """
        self.__imaginaryUnit = imaginaryUnit

    def toString(self, data):
        """Format a data into a string using formatter options

        :param data: Data to render
        :rtype: str
        """
        if isinstance(data, tuple):
            text = [self.toString(d) for d in data]
            return "(" + " ".join(text) + ")"
        elif isinstance(data, (list, numpy.ndarray)):
            text = [self.toString(d) for d in data]
            return "[" + " ".join(text) + "]"
        elif isinstance(data, numpy.void):
            return "0x" + binascii.hexlify(data).decode("ascii")
        elif isinstance(data, (numpy.string_, numpy.object_, bytes)):
            # This have to be done before checking python string inheritance
            try:
                text = "%s" % data.decode("utf-8")
                if self.__useQuoteForText:
                    text = "\"%s\"" % text.replace("\"", "\\\"")
                return text
            except UnicodeDecodeError:
                pass
            return "0x" + binascii.hexlify(data).decode("ascii")
        elif isinstance(data, six.string_types):
            text = "%s" % data
            if self.__useQuoteForText:
                text = "\"%s\"" % text.replace("\"", "\\\"")
            return text
        elif isinstance(data, (numpy.integer, numbers.Integral)):
            return self.__integerFormat % data
        elif isinstance(data, (numbers.Real, numpy.floating)):
            # It have to be done before complex checking
            return self.__floatFormat % data
        elif isinstance(data, (numpy.complex_, numbers.Complex)):
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
        return str(data)
