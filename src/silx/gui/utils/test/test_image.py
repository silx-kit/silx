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
from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.utils.image import convertArrayToQImage, convertQImageToArray


class TestQImageConversion(TestCaseQt, ParametricTestCase):
    """Tests conversion of QImage to/from numpy array."""

    def testConvertArrayToQImage(self):
        """Test conversion of numpy array to QImage"""
        for format_, channels in [('Format_RGB888', 3),
                                  ('Format_ARGB32', 4)]:
            with self.subTest(format_):
                image = numpy.arange(
                    3*3*channels, dtype=numpy.uint8).reshape(3, 3, channels)
                qimage = convertArrayToQImage(image)

                self.assertEqual(qimage.height(), image.shape[0])
                self.assertEqual(qimage.width(), image.shape[1])
                self.assertEqual(qimage.format(), getattr(qt.QImage, format_))

                for row in range(3):
                    for col in range(3):
                        # Qrgb has no alpha channel, not compared
                        # Qt uses x,y while array is row,col...
                        self.assertEqual(qt.QColor(qimage.pixel(col, row)),
                                         qt.QColor(*image[row, col, :3]))


    def testConvertQImageToArray(self):
        """Test conversion of QImage to numpy array"""
        for format_, channels in [
                ('Format_RGB888', 3),  # Native support
                ('Format_ARGB32', 4),  # Native support
                ('Format_RGB32', 3)]:  # Conversion to RGB
            with self.subTest(format_):
                color = numpy.arange(channels)  # RGB(A) values
                qimage = qt.QImage(3, 3, getattr(qt.QImage, format_))
                qimage.fill(qt.QColor(*color))
                image = convertQImageToArray(qimage)

                self.assertEqual(qimage.height(), image.shape[0])
                self.assertEqual(qimage.width(), image.shape[1])
                self.assertEqual(image.shape[2], len(color))
                self.assertTrue(numpy.all(numpy.equal(image, color)))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(
        TestQImageConversion))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
