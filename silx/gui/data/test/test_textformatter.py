# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
__date__ = "12/12/2017"

import unittest
import shutil
import tempfile

import numpy
import six

from silx.gui.utils.testutils import TestCaseQt
from silx.gui.utils.testutils import SignalListener
from ..TextFormatter import TextFormatter

import h5py


class TestTextFormatter(TestCaseQt):

    def test_copy(self):
        formatter = TextFormatter()
        copy = TextFormatter(formatter=formatter)
        self.assertIsNot(formatter, copy)
        copy.setFloatFormat("%.3f")
        self.assertEqual(formatter.integerFormat(), copy.integerFormat())
        self.assertNotEqual(formatter.floatFormat(), copy.floatFormat())
        self.assertEqual(formatter.useQuoteForText(), copy.useQuoteForText())
        self.assertEqual(formatter.imaginaryUnit(), copy.imaginaryUnit())

    def test_event(self):
        listener = SignalListener()
        formatter = TextFormatter()
        formatter.formatChanged.connect(listener)
        formatter.setFloatFormat("%.3f")
        formatter.setIntegerFormat("%03i")
        formatter.setUseQuoteForText(False)
        formatter.setImaginaryUnit("z")
        self.assertEqual(listener.callCount(), 4)

    def test_int(self):
        formatter = TextFormatter()
        formatter.setIntegerFormat("%05i")
        result = formatter.toString(512)
        self.assertEqual(result, "00512")

    def test_float(self):
        formatter = TextFormatter()
        formatter.setFloatFormat("%.3f")
        result = formatter.toString(1.3)
        self.assertEqual(result, "1.300")

    def test_complex(self):
        formatter = TextFormatter()
        formatter.setFloatFormat("%.1f")
        formatter.setImaginaryUnit("i")
        result = formatter.toString(1.0 + 5j)
        result = result.replace(" ", "")
        self.assertEqual(result, "1.0+5.0i")

    def test_string(self):
        formatter = TextFormatter()
        formatter.setIntegerFormat("%.1f")
        formatter.setImaginaryUnit("z")
        result = formatter.toString("toto")
        self.assertEqual(result, '"toto"')

    def test_numpy_void(self):
        formatter = TextFormatter()
        result = formatter.toString(numpy.void(b"\xFF"))
        self.assertEqual(result, 'b"\\xFF"')

    def test_char_cp1252(self):
        # degree character in cp1252
        formatter = TextFormatter()
        result = formatter.toString(numpy.bytes_(b"\xB0"))
        self.assertEqual(result, u'"\u00B0"')


class TestTextFormatterWithH5py(TestCaseQt):

    @classmethod
    def setUpClass(cls):
        super(TestTextFormatterWithH5py, cls).setUpClass()

        cls.tmpDirectory = tempfile.mkdtemp()
        cls.h5File = h5py.File("%s/formatter.h5" % cls.tmpDirectory, mode="w")
        cls.formatter = TextFormatter()

    @classmethod
    def tearDownClass(cls):
        super(TestTextFormatterWithH5py, cls).tearDownClass()
        cls.h5File.close()
        cls.h5File = None
        shutil.rmtree(cls.tmpDirectory)

    def create_dataset(self, data, dtype=None):
        testName = "%s" % self.id()
        dataset = self.h5File.create_dataset(testName, data=data, dtype=dtype)
        return dataset

    def testAscii(self):
        d = self.create_dataset(data=b"abc")
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, '"abc"')

    def testUnicode(self):
        d = self.create_dataset(data=u"i\u2661cookies")
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(len(result), 11)
        self.assertEqual(result, u'"i\u2661cookies"')

    def testBadAscii(self):
        d = self.create_dataset(data=b"\xF0\x9F\x92\x94")
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, 'b"\\xF0\\x9F\\x92\\x94"')

    def testVoid(self):
        d = self.create_dataset(data=numpy.void(b"abc\xF0"))
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, 'b"\\x61\\x62\\x63\\xF0"')

    def testEnum(self):
        dtype = h5py.special_dtype(enum=('i', {"RED": 0, "GREEN": 1, "BLUE": 42}))
        d = numpy.array(42, dtype=dtype)
        d = self.create_dataset(data=d)
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, 'BLUE(42)')

    def testRef(self):
        dtype = h5py.special_dtype(ref=h5py.Reference)
        d = numpy.array(self.h5File.ref, dtype=dtype)
        d = self.create_dataset(data=d)
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, 'REF')

    def testArrayAscii(self):
        d = self.create_dataset(data=[b"abc"])
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, '["abc"]')

    def testArrayUnicode(self):
        dtype = h5py.special_dtype(vlen=six.text_type)
        d = numpy.array([u"i\u2661cookies"], dtype=dtype)
        d = self.create_dataset(data=d)
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(len(result), 13)
        self.assertEqual(result, u'["i\u2661cookies"]')

    def testArrayBadAscii(self):
        d = self.create_dataset(data=[b"\xF0\x9F\x92\x94"])
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, '[b"\\xF0\\x9F\\x92\\x94"]')

    def testArrayVoid(self):
        d = self.create_dataset(data=numpy.void([b"abc\xF0"]))
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, '[b"\\x61\\x62\\x63\\xF0"]')

    def testArrayEnum(self):
        dtype = h5py.special_dtype(enum=('i', {"RED": 0, "GREEN": 1, "BLUE": 42}))
        d = numpy.array([42, 1, 100], dtype=dtype)
        d = self.create_dataset(data=d)
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, '[BLUE(42) GREEN(1) 100]')

    def testArrayRef(self):
        dtype = h5py.special_dtype(ref=h5py.Reference)
        d = numpy.array([self.h5File.ref, None], dtype=dtype)
        d = self.create_dataset(data=d)
        result = self.formatter.toString(d[()], dtype=d.dtype)
        self.assertEqual(result, '[REF NULL_REF]')


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestTextFormatter))
    test_suite.addTest(loadTests(TestTextFormatterWithH5py))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
