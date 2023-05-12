# /*##########################################################################
#
# Copyright (c) 2016-2022 European Synchrotron Radiation Facility
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
import numpy

from .. import qt
from ..utils.image import convertQImageToArray

try:
    from ..utils.matplotlib import rasterMathText
except ImportError:
    rasterMathText = None

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


def rasterTextQt(text, font, size=-1, weight=-1, italic=False, devicePixelRatio=1.0):
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
    :param float devicePixelRatio:
        The current ratio between device and device-independent pixel
        (default: 1.0)
    :return: Corresponding image in gray scale and baseline offset from top
    :rtype: (HxW numpy.ndarray of uint8, int)
    """
    if not text:
        _logger.info("Trying to raster empty text, replaced by white space")
        text = " "  # Replace empty text by white space to produce an image

    if not isinstance(font, qt.QFont):
        font = qt.QFont(font, size, weight, italic)

    # get text size
    image = qt.QImage(1, 1, qt.QImage.Format_RGB888)
    painter = qt.QPainter()
    painter.begin(image)
    painter.setPen(qt.Qt.white)
    painter.setFont(font)
    bounds = painter.boundingRect(
        qt.QRect(0, 0, 4096, 4096), qt.Qt.TextExpandTabs, text
    )
    painter.end()

    metrics = qt.QFontMetrics(font)

    # This does not provide the correct text bbox on macOS
    # size = metrics.size(qt.Qt.TextExpandTabs, text)
    # bounds = metrics.boundingRect(
    #     qt.QRect(0, 0, size.width(), size.height()),
    #     qt.Qt.TextExpandTabs,
    #     text)

    # Add extra border and handle devicePixelRatio
    width = bounds.width() * devicePixelRatio + 2
    # align line size to 32 bits to ease conversion to numpy array
    width = 4 * ((width + 3) // 4)
    image = qt.QImage(
        int(width), int(bounds.height() * devicePixelRatio + 2), qt.QImage.Format_RGB888
    )
    image.setDevicePixelRatio(devicePixelRatio)

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

    # Remove leading and trailing empty columns/rows but one on each side
    filled_rows = numpy.nonzero(numpy.sum(array, axis=1))[0]
    filled_columns = numpy.nonzero(numpy.sum(array, axis=0))[0]
    if len(filled_rows) == 0 or len(filled_columns) == 0:
        return array, metrics.ascent()

    min_row = max(0, filled_rows[0] - 1)
    array = array[
        min_row : filled_rows[-1] + 2,
        max(0, filled_columns[0] - 1) : filled_columns[-1] + 2,
    ]

    return array, metrics.ascent() - min_row


def rasterText(text, font, size=-1, weight=-1, italic=False, devicePixelRatio=1.0):
    """Raster text using Qt or matplotlib if there may be math syntax.

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
    :param float devicePixelRatio:
        The current ratio between device and device-independent pixel
        (default: 1.0)
    :return: Corresponding image in gray scale and baseline offset from top
    :rtype: (HxW numpy.ndarray of uint8, int)
    """
    if rasterMathText is not None and text.count("$") >= 2:
        return rasterMathText(text, font, size, weight, italic, devicePixelRatio)
    else:
        return rasterTextQt(text, font, size, weight, italic, devicePixelRatio)
