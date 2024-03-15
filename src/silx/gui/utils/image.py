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
"""This module provides conversions between numpy.ndarray and QImage

- :func:`convertArrayToQImage`
- :func:`convertQImageToArray`
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "04/09/2018"


import sys
import numpy
from numpy.lib.stride_tricks import as_strided as _as_strided

from .. import qt


def convertArrayToQImage(array: numpy.ndarray) -> qt.QImage:
    """Convert an array-like image to a QImage.

    The created QImage is using a copy of the array data.

    Limitation: Only RGB or RGBA images with 8 bits per channel are supported.

    :param array: Array-like image data of shape (height, width, channels)
       Channels are expected to be either RGB or RGBA.
    :type array: numpy.ndarray of uint8
    :return: Corresponding Qt image with RGB888 or ARGB32 format.
    """
    array = numpy.asarray(array, order="C", dtype=numpy.uint8)

    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Image must be a 3D array with 3 or 4 channels per pixel")

    if array.shape[2] == 4:
        format_ = qt.QImage.Format_ARGB32
        # RGBA -> ARGB + take care of endianness
        if sys.byteorder == "little":  # RGBA -> BGRA
            array = array[:, :, (2, 1, 0, 3)]
        else:  # big endian: RGBA -> ARGB
            array = array[:, :, (3, 0, 1, 2)]

        array = numpy.array(array, order="C")  # Make a contiguous array

    else:  # array.shape[2] == 3
        format_ = qt.QImage.Format_RGB888

    height, width, depth = array.shape
    qimage = qt.QImage(
        array.data, width, height, array.strides[0], format_  # bytesPerLine
    )

    return qimage.copy()  # Making a copy of the image and its data


def convertQImageToArray(image: qt.QImage) -> numpy.ndarray:
    """Convert a QImage to a numpy array.

    If QImage format is not one of:

        - Format_Grayscale8
        - Format_RGB888
        - Format_RGBA8888
        - Format_ARGB32,

    it is first converted to one of this format.

    The created numpy array is using a copy of the QImage data.

    :param QImage image: The QImage to convert.
    :return: Image array of uint8 of shape:

        - (height, width) for grayscale images
        - (height, width, channels (3 or 4)) for RGB and RGBA images
    """
    supportedFormats = (
        qt.QImage.Format_Grayscale8,
        qt.QImage.Format_ARGB32,
        qt.QImage.Format_RGB888,
        qt.QImage.Format_RGBA8888,
    )

    # Convert to supported format if needed
    if image.format() not in supportedFormats:
        if image.hasAlphaChannel():
            image = image.convertToFormat(qt.QImage.Format_RGBA8888)
        else:
            image = image.convertToFormat(qt.QImage.Format_RGB888)

    format_ = image.format()
    if format_ == qt.QImage.Format_Grayscale8:
        channels = 1
    elif format_ == qt.QImage.Format_RGB888:
        channels = 3
    else:
        channels = 4

    ptr = image.bits()
    if qt.BINDING == "PyQt5":
        ptr.setsize(image.byteCount())
    elif qt.BINDING == "PyQt6":
        ptr.setsize(image.sizeInBytes())
    elif qt.BINDING == "PySide6":
        ptr = ptr.tobytes()
    else:
        raise RuntimeError("Unsupported Qt binding: %s" % qt.BINDING)

    # Create an array view on QImage internal data
    view = _as_strided(
        numpy.frombuffer(ptr, dtype=numpy.uint8),
        shape=(image.height(), image.width(), channels),
        strides=(image.bytesPerLine(), channels, 1),
    )

    if format_ == qt.QImage.Format_ARGB32:
        # Convert from ARGB to RGBA
        # Not a byte-ordered format: do care about endianness
        if sys.byteorder == "little":  # BGRA -> RGBA
            view = view[:, :, (2, 1, 0, 3)]
        else:  # big endian: ARGB -> RGBA
            view = view[:, :, (1, 2, 3, 0)]

    if channels == 1:  # Remove channel dimension
        view = view[:, :, 0]

    # Format_RGB888 and Format_RGBA8888 do not need reshuffling channels:
    # They are byte-ordered and already in the right order

    return numpy.array(view, copy=True, order="C")
