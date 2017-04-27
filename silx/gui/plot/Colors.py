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

__authors__ = ["V.A. Sole", "T. VINCENT"]
__license__ = "MIT"
__date__ = "16/01/2017"


import logging

import numpy

import matplotlib
import matplotlib.colors
import matplotlib.cm

from . import MPLColormap


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


_CMAPS = {}  # Store additional colormaps


def getMPLColormap(name):
    """Returns matplotlib colormap corresponding to given name

    :param str name: The name of the colormap
    :return: The corresponding colormap
    :rtype: matplolib.colors.Colormap
    """
    if not _CMAPS:  # Lazy initialization of own colormaps
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        _CMAPS['red'] = matplotlib.colors.LinearSegmentedColormap(
            'red', cdict, 256)

        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        _CMAPS['green'] = matplotlib.colors.LinearSegmentedColormap(
            'green', cdict, 256)

        cdict = {'red': ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 1.0, 1.0))}
        _CMAPS['blue'] = matplotlib.colors.LinearSegmentedColormap(
            'blue', cdict, 256)

        # Temperature as defined in spslut
        cdict = {'red': ((0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (0.75, 1.0, 1.0),
                         (1.0, 1.0, 1.0)),
                 'green': ((0.0, 0.0, 0.0),
                           (0.25, 1.0, 1.0),
                           (0.75, 1.0, 1.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 1.0, 1.0),
                          (0.25, 1.0, 1.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}
        # but limited to 256 colors for a faster display (of the colorbar)
        _CMAPS['temperature'] = \
            matplotlib.colors.LinearSegmentedColormap(
                'temperature', cdict, 256)

        # reversed gray
        cdict = {'red': ((0.0, 1.0, 1.0),
                         (1.0, 0.0, 0.0)),
                 'green': ((0.0, 1.0, 1.0),
                           (1.0, 0.0, 0.0)),
                 'blue': ((0.0, 1.0, 1.0),
                          (1.0, 0.0, 0.0))}

        _CMAPS['reversed gray'] = \
            matplotlib.colors.LinearSegmentedColormap(
                'yerg', cdict, 256)

    if name in _CMAPS:
        return _CMAPS[name]
    elif hasattr(MPLColormap, name):  # viridis and sister colormaps
        return getattr(MPLColormap, name)
    else:
        # matplotlib built-in
        return matplotlib.cm.get_cmap(name)


def getMPLScalarMappable(colormap, data=None):
    """Returns matplotlib ScalarMappable corresponding to colormap

    :param dict colormap: The colormap to convert
    :param numpy.ndarray data:
        The data on which the colormap is applied.
        If provided, it is used to compute autoscale.
    :return: matplotlib object corresponding to colormap
    :rtype: matplotlib.cm.ScalarMappable
    """
    assert colormap is not None

    if colormap['name'] is not None:
        cmap = getMPLColormap(colormap['name'])

    else:  # No name, use custom colors
        if 'colors' not in colormap:
            raise ValueError(
                'addImage: colormap no name nor list of colors.')
        colors = numpy.array(colormap['colors'], copy=True)
        assert len(colors.shape) == 2
        assert colors.shape[-1] in (3, 4)
        if colors.dtype == numpy.uint8:
            # Convert to float in [0., 1.]
            colors = colors.astype(numpy.float32) / 255.
        cmap = matplotlib.colors.ListedColormap(colors)

    if colormap['normalization'].startswith('log'):
        vmin, vmax = None, None
        if not colormap['autoscale']:
            if colormap['vmin'] > 0.:
                vmin = colormap['vmin']
            if colormap['vmax'] > 0.:
                vmax = colormap['vmax']

            if vmin is None or vmax is None:
                _logger.warning('Log colormap with negative bounds, ' +
                                'changing bounds to positive ones.')
            elif vmin > vmax:
                _logger.warning('Colormap bounds are inverted.')
                vmin, vmax = vmax, vmin

        # Set unset/negative bounds to positive bounds
        if (vmin is None or vmax is None) and data is not None:
            finiteData = data[numpy.isfinite(data)]
            posData = finiteData[finiteData > 0]
            if vmax is None:
                # 1. as an ultimate fallback
                vmax = posData.max() if posData.size > 0 else 1.
            if vmin is None:
                vmin = posData.min() if posData.size > 0 else vmax
            if vmin > vmax:
                vmin = vmax

        norm = matplotlib.colors.LogNorm(vmin, vmax)

    else:  # Linear normalization
        if colormap['autoscale']:
            if data is None:
                vmin, vmax = None, None
            else:
                finiteData = data[numpy.isfinite(data)]
                vmin = finiteData.min()
                vmax = finiteData.max()
        else:
            vmin = colormap['vmin']
            vmax = colormap['vmax']
            if vmin > vmax:
                _logger.warning('Colormap bounds are inverted.')
                vmin, vmax = vmax, vmin

        norm = matplotlib.colors.Normalize(vmin, vmax)

    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


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
    scalarMappable = getMPLScalarMappable(colormap, data)
    rgbaImage = scalarMappable.to_rgba(data, bytes=True)

    return rgbaImage
