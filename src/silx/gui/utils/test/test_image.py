# /*##########################################################################
#
# Copyright (c) 2017-2023 European Synchrotron Radiation Facility
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
import pytest

from silx.gui import qt
from silx.gui.utils.image import convertArrayToQImage, convertQImageToArray


@pytest.mark.parametrize(
    "format_, channels",
    [
        (qt.QImage.Format_RGB888, 3),  # Native support
        (qt.QImage.Format_ARGB32, 4),  # Native support
    ],
)
def testConvertArrayToQImage(format_, channels):
    """Test conversion of numpy array to QImage"""
    image = numpy.arange(3 * 3 * channels, dtype=numpy.uint8).reshape(3, 3, channels)
    qimage = convertArrayToQImage(image)

    assert (qimage.height(), qimage.width()) == image.shape[:2]
    assert qimage.format() == format_

    for row in range(3):
        for col in range(3):
            # Qrgb has no alpha channel, not compared
            # Qt uses x,y while array is row,col...
            assert qt.QColor(qimage.pixel(col, row)) == qt.QColor(*image[row, col, :3])


@pytest.mark.parametrize(
    "format_, channels",
    [
        (qt.QImage.Format_RGB888, 3),  # Native support
        (qt.QImage.Format_ARGB32, 4),  # Native support
        (qt.QImage.Format_RGB32, 3),  # Conversion to RGB
    ],
)
def testConvertQImageToArray(format_, channels):
    """Test conversion of QImage to numpy array"""
    color = numpy.arange(channels)  # RGB(A) values
    qimage = qt.QImage(3, 3, format_)
    qimage.fill(qt.QColor(*color))
    image = convertQImageToArray(qimage)

    assert (qimage.height(), qimage.width(), len(color)) == image.shape
    assert numpy.all(numpy.equal(image, color))


def testConvertQImageToArrayGrayscale():
    """Test conversion of grayscale QImage to numpy array"""
    qimage = qt.QImage(3, 3, qt.QImage.Format_Grayscale8)
    qimage.fill(1)
    image = convertQImageToArray(qimage)

    assert (qimage.height(), qimage.width()) == image.shape
    assert numpy.all(numpy.equal(image, 1))
