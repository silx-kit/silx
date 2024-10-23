# /*##########################################################################
#
# Copyright (c) 2016-2024 European Synchrotron Radiation Facility
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
from __future__ import annotations
"""Text rasterisation feature leveraging Qt font and text layout support."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "13/10/2016"


import logging
import numpy

from .. import qt
from ..utils.image import convertQImageToArray
from ..utils.matplotlib import rasterMathText


_logger = logging.getLogger(__name__)


def getDefaultFontFamily() -> str:
    """Returns the default font family of the application"""
    return qt.QApplication.instance().font().family()


def rasterTextQt(
    text: str,
    font: qt.QFont,
    dotsPerInch: float = 96.0,
) -> tuple[numpy.ndarray, float]:
    """Raster text using Qt.

    It supports multiple lines.

    :param text: The text to raster
    :param font: Font to use
    :param dotsPerInch: The DPI resolution of the created image
    :return: Corresponding image in gray scale and baseline offset from top
    """
    if not text:
        _logger.info("Trying to raster empty text, replaced by white space")
        text = " "  # Replace empty text by white space to produce an image

    dotsPerMeter = int(dotsPerInch * 100 / 2.54)

    # get text size
    image = qt.QImage(1, 1, qt.QImage.Format_Grayscale8)
    image.setDotsPerMeterX(dotsPerMeter)
    image.setDotsPerMeterY(dotsPerMeter)

    painter = qt.QPainter()
    painter.begin(image)
    painter.setPen(qt.Qt.white)
    painter.setFont(font)
    bounds = painter.boundingRect(
        qt.QRect(0, 0, 4096, 4096), qt.Qt.TextExpandTabs, text
    )
    painter.end()

    metrics = qt.QFontMetrics(font)
    offset = metrics.ascent() / 72.0 * dotsPerInch

    # This does not provide the correct text bbox on macOS
    # size = metrics.size(qt.Qt.TextExpandTabs, text)
    # bounds = metrics.boundingRect(
    #     qt.QRect(0, 0, size.width(), size.height()),
    #     qt.Qt.TextExpandTabs,
    #     text)

    # Add extra border
    width = bounds.width() + 2
    # align line size to 32 bits to ease conversion to numpy array
    width = 4 * ((width + 3) // 4)
    image = qt.QImage(
        int(width),
        int(bounds.height() + 2),
        qt.QImage.Format_Grayscale8,
    )
    image.setDotsPerMeterX(dotsPerMeter)
    image.setDotsPerMeterY(dotsPerMeter)
    image.fill(0)

    # Raster text
    painter = qt.QPainter()
    painter.begin(image)
    painter.setPen(qt.Qt.white)
    painter.setFont(font)
    painter.drawText(bounds, qt.Qt.TextExpandTabs, text)
    painter.end()

    array = convertQImageToArray(image)

    # Remove leading and trailing empty columns/rows but one on each side
    filled_rows = numpy.nonzero(numpy.sum(array, axis=1))[0]
    filled_columns = numpy.nonzero(numpy.sum(array, axis=0))[0]

    if len(filled_rows) == 0 or len(filled_columns) == 0:
        return array, offset
    return (
        numpy.ascontiguousarray(
            array[
                0 : filled_rows[-1] + 2,
                max(0, filled_columns[0] - 1) : filled_columns[-1] + 2,
            ]
        ),
        offset,
    )


def rasterText(
    text: str,
    font: qt.QFont,
    dotsPerInch: float = 96.0,
) -> tuple[numpy.ndarray, float]:
    """Raster text using Qt or matplotlib if there may be math syntax.

    It supports multiple lines.

    :param text: The text to raster
    :param font: Font name or QFont to use
    :param dotsPerInch: Created image resolution
    :return: Corresponding image in gray scale and baseline offset from top
    """

    if text.count("$") >= 2:
        return rasterMathText(text, font, dotsPerInch)
    return rasterTextQt(text, font, dotsPerInch)
