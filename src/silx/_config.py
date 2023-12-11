#!/usr/bin/env python
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
"""This module contains library wide configuration.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "09/11/2018"


class Config(object):
    """
    Class containing shared global configuration for the silx library.

    .. versionadded:: 0.8
    """

    DEFAULT_PLOT_BACKEND = "matplotlib", "opengl"
    """Default plot backend.

    It will be used as default backend for all the next created PlotWidget.

    This attribute can be set with:

    - 'matplotlib' (default) or 'mpl'
    - 'opengl', 'gl'
    - 'none'
    - A :class:`silx.gui.plot.backend.BackendBase.BackendBase` class
    - A callable returning backend class or binding name

    If multiple backends are provided, the first available one is used.

    .. versionadded:: 0.8
    """

    DEFAULT_COLORMAP_NAME = "gray"
    """Default LUT for the plot widgets.

    The available list of names are available in the module
    :module:`silx.gui.colors`.

    .. versionadded:: 0.8
    """

    DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION = "upward"
    """Default Y-axis orientation for plot widget displaying images.

    This attribute can be set with:

    - 'upward' (default), which set the origin to the bottom with an upward
        orientation.
    - 'downward', which set the origin to the top with a backward orientation.

    It will have an influence on:

    - :class:`silx.gui.plot.StackWidget`
    - :class:`silx.gui.plot.ComplexImageView`
    - :class:`silx.gui.plot.Plot2D`
    - :class:`silx.gui.plot.ImageView`

    .. versionadded:: 0.8
    """

    DEFAULT_PLOT_CURVE_COLORS = [
        "#1f77b4",  # tab:blue
        "#ff7f0e",  # tab:orange
        "#2ca02c",  # tab:green
        "#d62728",  # tab:red
        "#9467bd",  # tab:purple
        "#8c564b",  # tab:brown
        "#e377c2",  # tab:pink
        "#7f7f7f",  # tab:gray
        "#bcbd22",  # tab:olive
        "#17becf",  # tab:cyan
    ]
    """Default list of colors for plot widget displaying curves.

    It is based on the color cycle of matplotlib 2.0.
    See https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_color_data.py#L14

    It will have an influence on:

    - :class:`silx.gui.plot.PlotWidget`

    .. versionadded:: 0.9
    """

    DEFAULT_PLOT_CURVE_SYMBOL_MODE = False
    """Whether to display curves with markers or not by default in PlotWidget.

    It will have an influence on PlotWidget curve items.

    .. versionadded:: 0.10
    """

    DEFAULT_PLOT_SYMBOL = "o"
    """Default marker of the item.

    It will have an influence on PlotWidget items

    Supported symbols:

        - 'o', 'Circle'
        - 'd', 'Diamond'
        - 's', 'Square'
        - '+', 'Plus'
        - 'x', 'Cross'
        - '.', 'Point'
        - ',', 'Pixel'
        - '',  'None'

    .. versionadded:: 0.10
    """

    DEFAULT_PLOT_SYMBOL_SIZE = 6.0
    """Default marker size of the item.

    It will have an influence on PlotWidget items

    .. versionadded:: 0.10
    """

    DEFAULT_PLOT_ACTIVE_CURVE_COLOR = None
    """Default color for the active curve.

    It will have an influence on PlotWidget curve items

    .. versionadded:: 2.0
    """

    DEFAULT_PLOT_ACTIVE_CURVE_LINEWIDTH = 2
    """Default line width for the active curve.

    It will have an influence on PlotWidget curve items

    .. versionadded:: 2.0
    """

    DEFAULT_PLOT_MARKER_TEXT_FONT_SIZE = None
    """Default font size for marker text.

    It will have an influence on PlotWidget marker items

    .. versionadded:: 2.0
    """

    _MPL_TIGHT_LAYOUT = False
    """If true the matplotlib backend will use the
    experimental tight layout.

    .. versionadded:: 2.0
    """
