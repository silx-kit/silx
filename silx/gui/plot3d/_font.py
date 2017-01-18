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
"""Text rasterisation feature leveraging Qt font and text layout support."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "13/10/2016"


import logging
import sys
import numpy
from silx.gui import qt
from silx.gui._utils import convertQImageToArray


_logger = logging.getLogger(__name__)


def getDefaultFontFamily():
    """Returns the default font family of the application"""
    return qt.QApplication.instance().font().family()


# Font weights
ULTRA_LIGHT = 0
"""Lightest characters: Minimum font weight"""

LIGHT = 25
"""Light characters"""

NORMAL = 50
"""Normal characters"""

SEMI_BOLD = 63
"""Between normal and bold characters"""

BOLD = 74
"""Thicker characters"""

BLACK = 87
"""Really thick characters"""

ULTRA_BLACK = 99
"""Thickest characters: Maximum font weight"""


def rasterText(text, font, size=-1, weight=-1, italic=False):
    """Raster text using Qt.

    It supports multiple lines.

    :param str text: The text to raster
    :param font: Font name or QFont to use
    :type font: str or :class:`QFont`
    :param int size:
        Font size in points
        Used only if font is given as name.
    :param int weight:
        Font weight in [0, 99], see QFont.Weight.
        Used only if font is given as name.
    :param bool italic:
        True for italic font (default: False).
        Used only if font is given as name.
    :return: Corresponding image in gray scale and baseline offset from top
    :rtype: (HxW numpy.ndarray of uint8, int)
    """
    if not text:
        _logger.info("Trying to raster empty text, replaced by white space")
        text = ' '  # Replace empty text by white space to produce an image

    if not isinstance(font, qt.QFont):
        font = qt.QFont(font, size, weight, italic)

    metrics = qt.QFontMetrics(font)
    size = metrics.size(qt.Qt.TextExpandTabs, text)
    bounds = metrics.boundingRect(
        qt.QRect(0, 0, size.width(), size.height()),
        qt.Qt.TextExpandTabs,
        text)

    width = bounds.width() + 2  # Add extra border
    # align line size to 32 bits to ease conversion to numpy array
    width = 4 * ((width + 3) // 4)
    image = qt.QImage(width,
                      bounds.height(),
                      qt.QImage.Format_RGB888)
    # TODO if Qt5 use Format_Grayscale8 instead
    image.fill(0)

    # Raster text
    painter = qt.QPainter()
    painter.begin(image)
    painter.setPen(qt.Qt.white)
    painter.setFont(font)
    painter.drawText(bounds, qt.Qt.TextExpandTabs, text)
    painter.end()

    array = convertQImageToArray(image)

    # RGB to R
    array = numpy.ascontiguousarray(array[:, :, 0])

    return array, metrics.ascent()
