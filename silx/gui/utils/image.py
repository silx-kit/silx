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
"""This module provides conversions between numpy.ndarray and QImage

- :func:`convertArrayToQImage`
- :func:`convertQImageToArray`
"""

from __future__ import division


__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "04/09/2018"


import sys
import numpy
from numpy.lib.stride_tricks import as_strided as _as_strided

from .. import qt


def convertArrayToQImage(image):
    """Convert an array-like RGB888 image to a QImage.

    The created QImage is using a copy of the array data.

    Limitation: Only supports RGB888 format.

    :param image: Array-like image data
    :type image: numpy.ndarray of uint8 of dimension HxWx3
    :return: Corresponding Qt image
    :rtype: QImage
    """
    # Possible extension: add a format argument to support more formats

    image = numpy.array(image, copy=False, order='C', dtype=numpy.uint8)

    height, width, depth = image.shape
    assert depth == 3

    qimage = qt.QImage(
        image.data,
        width,
        height,
        image.strides[0],  # bytesPerLine
        qt.QImage.Format_RGB888)

    return qimage.copy()  # Making a copy of the image and its data


def convertQImageToArray(image):
    """Convert a QImage to a numpy array.

    If QImage has not 8 bits per channel, it is converted to either
    ARG8888 or RGB888 depending whether it has a alpha channel or not.
    For format with padding bits (e.g. Format_RGB32), padding bits are preserved.

    The created numpy array is using a copy of the QImage data.

    :param QImage image: The QImage to convert.
    :return: The image array
    :rtype: numpy.ndarray of uint8 of shape (height, width, channels (3 or 4))
    """

    if image.format() == qt.QImage.Format_RGB888:
        channels = 3
    elif image.format() in (
            qt.QImage.Format_RGB32,
            qt.QImage.Format_ARGB32,
            qt.QImage.Format_ARGB32_Premultiplied,
            # Qt5 formats
            getattr(qt.QImage, 'Format_RGBX8888', 'none'),
            getattr(qt.QImage, 'Format_RGBA8888', 'none'),
            getattr(qt.QImage, 'Format_RGBA8888_Premultiplied', 'none')):
        channels = 4

    else:  # For other format: Convert to RGB888 or ARGB8888
        if image.hasAlphaChannel():
            image = image.convertToFormat(qt.QImage.Format_ARGB32)
            channels = 4
        else:
            image = image.convertToFormat(qt.QImage.Format_RGB888)
            channels = 3

    ptr = image.bits()
    if qt.BINDING not in ('PySide', 'PySide2'):
        ptr.setsize(image.byteCount())
        if qt.BINDING == 'PyQt4' and sys.version_info[0] == 2:
            ptr = ptr.asstring()
    elif sys.version_info[0] == 3:  # PySide with Python3
        ptr = ptr.tobytes()

    # Create an array view on QImage internal data
    view = _as_strided(
        numpy.frombuffer(ptr, dtype=numpy.uint8),
        shape=(image.height(), image.width(), channels),
        strides=(image.bytesPerLine(), channels, 1))

    return numpy.array(view, copy=True)
