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

"""This module initializes matplotlib and sets-up the backend to use.

It MUST be imported prior to any other import of matplotlib.

It provides the matplotlib :class:`FigureCanvasQTAgg` class corresponding
to the used backend.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/05/2018"


import io
from pkg_resources import parse_version
import matplotlib
import numpy

from .. import qt


def rasterMathText(text, font, size=-1, weight=-1, italic=False, devicePixelRatio=1.0):
    """Raster text using matplotlib supporting latex-like math syntax.

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
    # Implementation adapted from:
    # https://github.com/matplotlib/matplotlib/blob/d624571a19aec7c7d4a24123643288fc27db17e7/lib/matplotlib/mathtext.py#L264

    # Lazy import to avoid imports before setting matplotlib's rcParams
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    from matplotlib import figure

    dpi = 96  # default
    qapp = qt.QApplication.instance()
    if qapp:
        screen = qapp.primaryScreen()
        if screen:
            dpi = screen.logicalDotsPerInchY()

    # Make sure dpi is even, it causes issues with array reshape otherwise
    dpi = ((dpi * devicePixelRatio) // 2) * 2

    stripped_text = text.strip("\n")

    parser = MathTextParser("path")
    width, height, depth, _, _ = parser.parse(stripped_text, dpi=dpi)
    width *= 2
    height *= 2 * (stripped_text.count("\n") + 1)

    if not isinstance(font, qt.QFont):
        font = qt.QFont(font, size, weight, italic)
    prop = FontProperties(
        family=font.family(),
        style="italic" if font.italic() else "normal",
        weight=10 * font.weight(),
        size=font.pointSize(),
    )

    fig = figure.Figure(figsize=(width / dpi, height / dpi))
    fig.text(0, depth / height, stripped_text, fontproperties=prop)
    with io.BytesIO() as buffer:
        fig.savefig(buffer, dpi=dpi, format="raw")
        buffer.seek(0)
        image = numpy.frombuffer(buffer.read(), dtype=numpy.uint8).reshape(
            int(height), int(width), 4
        )

    # RGB to inverted R channel
    array = 255 - image[:, :, 0]

    # Remove leading and trailing empty columns/rows but one on each side
    filled_rows = numpy.nonzero(numpy.sum(array, axis=1))[0]
    filled_columns = numpy.nonzero(numpy.sum(array, axis=0))[0]
    if len(filled_rows) == 0 or len(filled_columns) == 0:
        return array, image.shape[0] - 1

    clipped_array = numpy.ascontiguousarray(
        array[
            max(0, filled_rows[0] - 1) : filled_rows[-1] + 2,
            max(0, filled_columns[0] - 1) : filled_columns[-1] + 2,
        ]
    )

    return clipped_array, image.shape[0] - 1  # baseline not available


def _matplotlib_use(backend, force):
    """Wrapper of `matplotlib.use` to set-up backend.

    It adds extra initialization for PySide2 with matplotlib < 2.2.
    """
    # This is kept for compatibility with matplotlib < 2.2
    if (
        parse_version(matplotlib.__version__) < parse_version("2.2")
        and qt.BINDING == "PySide2"
    ):
        matplotlib.rcParams["backend.qt5"] = "PySide2"

    matplotlib.use(backend, force=force)


if qt.BINDING in ("PySide6", "PyQt6", "PyQt5", "PySide2"):
    _matplotlib_use("Qt5Agg", force=False)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa

else:
    raise ImportError("Unsupported Qt binding: %s" % qt.BINDING)
