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

"""This module initializes matplotlib and sets-up the backend to use.

It MUST be imported prior to any other import of matplotlib.

It provides the matplotlib :class:`FigureCanvasQTAgg` class corresponding
to the used backend.
"""
from __future__ import annotations


__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/05/2018"


import io
import matplotlib
import numpy

from .. import qt

# This must be performed before any import from matplotlib
if qt.BINDING in ("PySide6", "PyQt6", "PyQt5"):
    matplotlib.use("Qt5Agg", force=False)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa

else:
    raise ImportError("Unsupported Qt binding: %s" % qt.BINDING)


from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from matplotlib.ticker import ScalarFormatter as _ScalarFormatter
from matplotlib import figure, font_manager
from packaging.version import Version

_MATPLOTLIB_VERSION = Version(matplotlib.__version__)


class DefaultTickFormatter(_ScalarFormatter):
    """Tick label formatter"""

    def __init__(self):
        super().__init__(useOffset=True, useMathText=True)
        self.set_scientific(True)
        self.create_dummy_axis()

    if _MATPLOTLIB_VERSION < Version("3.1.0"):

        def format_ticks(self, values):
            self.set_locs(values)
            return [self(value, i) for i, value in enumerate(values)]


_FONT_STYLES = {
    qt.QFont.StyleNormal: "normal",
    qt.QFont.StyleItalic: "italic",
    qt.QFont.StyleOblique: "oblique",
}


def qFontToFontProperties(font: qt.QFont):
    """Convert a QFont to a matplotlib FontProperties"""
    weightFactor = 10 if qt.BINDING == "PyQt5" else 1
    families = [font.family(), font.defaultFamily()]
    if _MATPLOTLIB_VERSION >= Version("3.6.0"):
        # Prevent 'Font family not found' warnings
        availableNames = font_manager.get_font_names()
        families = [f for f in families if f in availableNames]
        families.append(font_manager.fontManager.defaultFamily["ttf"])

    if "Sans" in font.family():
        families.insert(0, "sans-serif")

    return FontProperties(
        family=families,
        style=_FONT_STYLES[font.style()],
        weight=weightFactor * font.weight(),
        size=font.pointSizeF(),
    )


def rasterMathText(
    text: str,
    font: qt.QFont,
    dotsPerInch: float = 96.0,
) -> tuple[numpy.ndarray, float]:
    """Raster text using matplotlib supporting latex-like math syntax.

    It supports multiple lines.

    :param text: The text to raster
    :param font: Font to use
    :param dotsPerInch: The DPI resolution of the created image
    :return: Corresponding image in gray scale and baseline offset from top
    """
    # Implementation adapted from:
    # https://github.com/matplotlib/matplotlib/blob/d624571a19aec7c7d4a24123643288fc27db17e7/lib/matplotlib/mathtext.py#L264

    stripped_text = text.strip("\n")
    font_prop = qFontToFontProperties(font)

    parser = MathTextParser("path")
    lines_info = [
        parser.parse(line, prop=font_prop, dpi=dotsPerInch)
        for line in stripped_text.split("\n")
    ]
    max_line_width = max(info[0] for info in lines_info)
    # Use lp string as minimum height/ascent
    ref_info = parser.parse("lp", prop=font_prop, dpi=dotsPerInch)
    line_height = max(
        ref_info[1],
        *(info[1] for info in lines_info),
    )
    first_line_ascent = max(
        ref_info[1] - ref_info[2], lines_info[0][1] - lines_info[0][2]
    )

    linespacing = 1.2

    figure_height = numpy.ceil(line_height * len(lines_info) * linespacing) + 2
    fig = figure.Figure(
        figsize=(
            (max_line_width + 1) / dotsPerInch,
            figure_height / dotsPerInch,
        )
    )
    fig.set_dpi(dotsPerInch)
    text = fig.text(
        0,
        1,
        stripped_text,
        fontproperties=font_prop,
        verticalalignment="top",
    )
    text.set_linespacing(linespacing)
    with io.BytesIO() as buffer:
        fig.savefig(buffer, dpi=dotsPerInch, format="raw")
        canvas_width, canvas_height = fig.get_window_extent().max
        buffer.seek(0)
        image = numpy.frombuffer(buffer.read(), dtype=numpy.uint8).reshape(
            int(canvas_height), int(canvas_width), 4
        )

    # RGB to inverted R channel
    array = 255 - image[:, :, 0]

    # Remove leading/trailing empty columns and trailing rows but one on each side
    filled_rows = numpy.nonzero(numpy.sum(array, axis=1))[0]
    filled_columns = numpy.nonzero(numpy.sum(array, axis=0))[0]
    if len(filled_rows) == 0 or len(filled_columns) == 0:
        return array, first_line_ascent
    return (
        numpy.ascontiguousarray(
            array[
                0 : filled_rows[-1] + 2,
                max(0, filled_columns[0] - 1) : filled_columns[-1] + 2,
            ]
        ),
        first_line_ascent,
    )
