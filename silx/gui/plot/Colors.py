# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""Color conversion function and dictionary."""

__authors__ = ["V.A. Sole", "T. VINCENT"]
__license__ = "MIT"
__date__ = "15/09/2016"


import numpy


COLORDICT = {}
"""Dictionary of common colors."""

COLORDICT['b'] = COLORDICT['blue'] = '#0000ff'
COLORDICT['r'] = COLORDICT['red'] = '#ff0000'
COLORDICT['g'] = COLORDICT['green'] = '#00ff00'
COLORDICT['k'] = COLORDICT['black'] = '#000000'
COLORDICT['w'] = COLORDICT['white'] = '#ffffff'
COLORDICT['pink'] = '#ff66ff'
COLORDICT['brown'] = '#a52a2a'
COLORDICT['orange'] = '#ff9900'
COLORDICT['violet'] = '#6600ff'
COLORDICT['gray'] = COLORDICT['grey'] = '#a0a0a4'
# COLORDICT['darkGray'] = COLORDICT['darkGrey'] = '#808080'
# COLORDICT['lightGray'] = COLORDICT['lightGrey'] = '#c0c0c0'
COLORDICT['y'] = COLORDICT['yellow'] = '#ffff00'
COLORDICT['m'] = COLORDICT['magenta'] = '#ff00ff'
COLORDICT['c'] = COLORDICT['cyan'] = '#00ffff'
COLORDICT['darkBlue'] = '#000080'
COLORDICT['darkRed'] = '#800000'
COLORDICT['darkGreen'] = '#008000'
COLORDICT['darkBrown'] = '#660000'
COLORDICT['darkCyan'] = '#008080'
COLORDICT['darkYellow'] = '#808000'
COLORDICT['darkMagenta'] = '#800080'


def rgba(color, colorDict=None):
    """Convert color code '#RRGGBB' and '#RRGGBBAA' to (R, G, B, A)

    It also convert RGB(A) values from uint8 to float in [0, 1]

    :param str code: The color code to conver
    :param dict colorDict: A dictionary of color name conversion to color code
    :returns: RGBA colors as floats in [0., 1.]
    :rtype: tuple
    """
    if colorDict is None:
        colorDict = COLORDICT

    values = numpy.asarray(color).ravel()

    if values.dtype.kind in 'iuf':  # integer or float
        # Color is an array
        assert len(values) in (3, 4)

        # Convert from integers in [0, 255] to float in [0, 1]
        if values.dtype.kind in 'iu':
            values = values / 255.

        # Clip to [0, 1]
        values[values < 0.] = 0.
        values[values > 1.] = 1.

        if len(values) == 3:
            return values[0], values[1], values[2], 1.
        else:
            return tuple(values)

    # We assume color is a string
    if not color.startswith('#'):
        color = colorDict[color]

    assert len(color) in (7, 9) and color[0] == '#'
    r = int(color[1:3], 16) / 255.
    g = int(color[3:5], 16) / 255.
    b = int(color[5:7], 16) / 255.
    a = int(color[7:9], 16) / 255. if len(color) == 9 else 1.
    return r, g, b, a


_COLORMAP_CURSOR_COLORS = {
    'gray': 'pink',
    'reversed gray': 'pink',
    'temperature': 'black',
    'red': 'white',
    'green': 'white',
    'blue': 'white',
    'jet': 'white',
    'viridis': 'white',
    'magma': 'white',
    'inferno': 'white',
    'plasma': 'white',
}


def cursorColorForColormap(colormapName):
    """Get a color suitable for overlay over a colormap.

    :param str colormapName: The name of the colormap.
    :return: Name of the color.
    :rtype: str
    """
    return _COLORMAP_CURSOR_COLORS.get(colormapName, 'black')
