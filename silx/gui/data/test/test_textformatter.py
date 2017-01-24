# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "24/01/2017"

import unittest

from silx.gui.test.utils import TestCaseQt
from silx.gui.test.utils import SignalListener
from ..TextFormatter import TextFormatter


class TestTextFormatter(TestCaseQt):

    def test_copy(self):
        formatter = TextFormatter()
        copy = TextFormatter(formatter=formatter)
        self.assertIsNot(formatter, copy)
        copy.setFloatFormat("%.3f")
        self.assertEquals(formatter.integerFormat(), copy.integerFormat())
        self.assertNotEquals(formatter.floatFormat(), copy.floatFormat())
        self.assertEquals(formatter.useQuoteForText(), copy.useQuoteForText())
        self.assertEquals(formatter.imaginaryUnit(), copy.imaginaryUnit())

    def test_event(self):
        listener = SignalListener()
        formatter = TextFormatter()
        formatter.formatChanged.connect(listener)
        formatter.setFloatFormat("%.3f")
        formatter.setIntegerFormat("%03i")
        formatter.setUseQuoteForText(False)
        formatter.setImaginaryUnit("z")
        self.assertEquals(listener.callCount(), 4)

    def test_int(self):
        formatter = TextFormatter()
        formatter.setIntegerFormat("%05i")
        result = formatter.toString(512)
        self.assertEquals(result, "00512")

    def test_float(self):
        formatter = TextFormatter()
        formatter.setFloatFormat("%.3f")
        result = formatter.toString(1.3)
        self.assertEquals(result, "1.300")

    def test_complex(self):
        formatter = TextFormatter()
        formatter.setFloatFormat("%.1f")
        formatter.setImaginaryUnit("i")
        result = formatter.toString(1.0 + 5j)
        result = result.replace(" ", "")
        self.assertEquals(result, "1.0+5.0i")

    def test_string(self):
        formatter = TextFormatter()
        formatter.setIntegerFormat("%.1f")
        formatter.setImaginaryUnit("z")
        result = formatter.toString("toto")
        self.assertEquals(result, '"toto"')


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestTextFormatter))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
