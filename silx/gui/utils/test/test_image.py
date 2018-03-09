# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""Test of utils module."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "16/01/2017"

import numpy
import unittest

from silx.gui import qt
from silx.gui.test.utils import TestCaseQt
from silx.gui.utils import _image


class TestQImageConversion(TestCaseQt):
    """Tests conversion of QImage to/from numpy array."""

    def testConvertArrayToQImage(self):
        """Test conversion of numpy array to QImage"""
        image = numpy.ones((3, 3, 3), dtype=numpy.uint8)
        qimage = _image.convertArrayToQImage(image)

        self.assertEqual(qimage.height(), image.shape[0])
        self.assertEqual(qimage.width(), image.shape[1])
        self.assertEqual(qimage.format(), qt.QImage.Format_RGB888)

        color = qt.QColor(1, 1, 1).rgb()
        self.assertEqual(qimage.pixel(1, 1), color)

    def testConvertQImageToArray(self):
        """Test conversion of QImage to numpy array"""
        qimage = qt.QImage(3, 3, qt.QImage.Format_RGB888)
        qimage.fill(0x010101)
        image = _image.convertQImageToArray(qimage)

        self.assertEqual(qimage.height(), image.shape[0])
        self.assertEqual(qimage.width(), image.shape[1])
        self.assertEqual(image.shape[2], 3)
        self.assertTrue(numpy.all(numpy.equal(image, 1)))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestQImageConversion))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
