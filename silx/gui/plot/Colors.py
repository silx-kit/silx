# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
"""Color conversion function, color dictionary and colormap tools."""

from __future__ import absolute_import

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "15/05/2017"


import logging
import numpy

# First of all init matplotlib and set its backend
from .matplotlib import matplotlib  #noqa
from .matplotlib import Colormap as MPLColormap

try:
    from matplotlib import cm as matplotlib_cm
except ImportError:
    matplotlib_cm = None


_logger = logging.getLogger(__name__)


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

    It also convert RGB(A) values from uint8 to float in [0, 1] and
    accept a QColor as color argument.

    :param str color: The color to convert
    :param dict colorDict: A dictionary of color name conversion to color code
    :returns: RGBA colors as floats in [0., 1.]
    :rtype: tuple
    """
    if colorDict is None:
        colorDict = COLORDICT

    if hasattr(color, 'getRgbF'):  # QColor support
        color = color.getRgbF()

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
    'temperature': 'pink',
    'red': 'green',
    'green': 'pink',
    'blue': 'yellow',
    'jet': 'pink',
    'viridis': 'pink',
    'magma': 'green',
    'inferno': 'green',
    'plasma': 'green',
}


def cursorColorForColormap(colormapName):
    """Get a color suitable for overlay over a colormap.

    :param str colormapName: The name of the colormap.
    :return: Name of the color.
    :rtype: str
    """
    return _COLORMAP_CURSOR_COLORS.get(colormapName, 'black')


def applyColormapToData(data,
                        name='gray',
                        normalization='linear',
                        autoscale=True,
                        vmin=0.,
                        vmax=1.,
                        colors=None):
    """Apply a colormap to the data and returns the RGBA image

    This supports data of any dimensions (not only of dimension 2).
    The returned array will have one more dimension (with 4 entries)
    than the input data to store the RGBA channels
    corresponding to each bin in the array.

    :param numpy.ndarray data: The data to convert.
    :param str name: Name of the colormap (default: 'gray').
    :param str normalization: Colormap mapping: 'linear' or 'log'.
    :param bool autoscale: Whether to use data min/max (True, default)
                           or [vmin, vmax] range (False).
    :param float vmin: The minimum value of the range to use if
                       'autoscale' is False.
    :param float vmax: The maximum value of the range to use if
                       'autoscale' is False.
    :param numpy.ndarray colors: Only used if name is None.
        Custom colormap colors as Nx3 or Nx4 RGB or RGBA arrays
    :return: The computed RGBA image
    :rtype: numpy.ndarray of uint8
    """
    # Debian 7 specific support
    # No transparent colormap with matplotlib < 1.2.0
    # Add support for transparent colormap for uint8 data with
    # colormap with 256 colors, linear norm, [0, 255] range
    if matplotlib.__version__ < '1.2.0':
        if name is None and colors is not None:
            colors = numpy.array(colors, copy=False)
            if (colors.shape[-1] == 4 and
                    not numpy.all(numpy.equal(colors[3], 255))):
                # This is a transparent colormap
                if (colors.shape == (256, 4) and
                        normalization == 'linear' and
                        not autoscale and
                        vmin == 0 and vmax == 255 and
                        data.dtype == numpy.uint8):
                    # Supported case, convert data to RGBA
                    return colors[data.reshape(-1)].reshape(
                        data.shape + (4,))
                else:
                    _logger.warning(
                        'matplotlib %s does not support transparent '
                        'colormap.', matplotlib.__version__)

    colormap = dict(name=name,
                    normalization=normalization,
                    autoscale=autoscale,
                    vmin=vmin,
                    vmax=vmax,
                    colors=colors)
    scalarMappable = MPLColormap.getScalarMappable(colormap, data)
    rgbaImage = scalarMappable.to_rgba(data, bytes=True)

    return rgbaImage


def getSupportedColormaps():
    """Get the supported colormap names as a tuple of str.

    The list should at least contain and start by:
    ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue')
    """
    default = ('gray', 'reversed gray',
               'temperature',
               'red', 'green', 'blue')
    if matplotlib_cm is None:
        return default
    else:
        maps = [m for m in matplotlib_cm.datad]
        maps.sort()
        return default + tuple(maps)
