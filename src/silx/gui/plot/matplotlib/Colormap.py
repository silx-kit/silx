# /*##########################################################################
# Copyright (C) 2017-2020 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Matplotlib's new colormaps"""

import numpy
import logging
from matplotlib.colors import ListedColormap
import matplotlib.colors
import matplotlib.cm
import silx.resources
from silx.utils.deprecation import deprecated, deprecated_warning


deprecated_warning(type_='module',
                   name=__file__,
                   replacement='silx.gui.colors.Colormap',
                   since_version='0.10.0')


_logger = logging.getLogger(__name__)

_AVAILABLE_AS_RESOURCE = ('magma', 'inferno', 'plasma', 'viridis')
"""List available colormap name as resources"""

_AVAILABLE_AS_BUILTINS = ('gray', 'reversed gray',
                          'temperature', 'red', 'green', 'blue')
"""List of colormaps available through built-in declarations"""

_CMAPS = {}
"""Cache colormaps"""


@property
@deprecated(since_version='0.10.0')
def magma():
    return getColormap('magma')


@property
@deprecated(since_version='0.10.0')
def inferno():
    return getColormap('inferno')


@property
@deprecated(since_version='0.10.0')
def plasma():
    return getColormap('plasma')


@property
@deprecated(since_version='0.10.0')
def viridis():
    return getColormap('viridis')


@deprecated(since_version='0.10.0')
def getColormap(name):
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
    elif name in _AVAILABLE_AS_RESOURCE:
        filename = silx.resources.resource_filename("gui/colormaps/%s.npy" % name)
        data = numpy.load(filename)
        lut = ListedColormap(data, name=name)
        _CMAPS[name] = lut
        return lut
    else:
        # matplotlib built-in
        return matplotlib.cm.get_cmap(name)


@deprecated(since_version='0.10.0')
def getScalarMappable(colormap, data=None):
    """Returns matplotlib ScalarMappable corresponding to colormap

    :param :class:`.Colormap` colormap: The colormap to convert
    :param numpy.ndarray data:
        The data on which the colormap is applied.
        If provided, it is used to compute autoscale.
    :return: matplotlib object corresponding to colormap
    :rtype: matplotlib.cm.ScalarMappable
    """
    assert colormap is not None

    if colormap.getName() is not None:
        cmap = getColormap(colormap.getName())

    else:  # No name, use custom colors
        if colormap.getColormapLUT() is None:
            raise ValueError(
                'addImage: colormap no name nor list of colors.')
        colors = colormap.getColormapLUT()
        assert len(colors.shape) == 2
        assert colors.shape[-1] in (3, 4)
        if colors.dtype == numpy.uint8:
            # Convert to float in [0., 1.]
            colors = colors.astype(numpy.float32) / 255.
        cmap = matplotlib.colors.ListedColormap(colors)

    vmin, vmax = colormap.getColormapRange(data)
    normalization = colormap.getNormalization()
    if normalization == colormap.LOGARITHM:
        norm = matplotlib.colors.LogNorm(vmin, vmax)
    elif normalization == colormap.LINEAR:
        norm = matplotlib.colors.Normalize(vmin, vmax)
    else:
        raise RuntimeError("Unsupported normalization: %s" % normalization)

    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


@deprecated(replacement='silx.colors.Colormap.applyToData',
            since_version='0.8.0')
def applyColormapToData(data, colormap):
    """Apply a colormap to the data and returns the RGBA image

    This supports data of any dimensions (not only of dimension 2).
    The returned array will have one more dimension (with 4 entries)
    than the input data to store the RGBA channels
    corresponding to each bin in the array.

    :param numpy.ndarray data: The data to convert.
    :param :class:`.Colormap`: The colormap to apply
    """
    # Debian 7 specific support
    # No transparent colormap with matplotlib < 1.2.0
    # Add support for transparent colormap for uint8 data with
    # colormap with 256 colors, linear norm, [0, 255] range
    if matplotlib.__version__ < '1.2.0':
        if (colormap.getName() is None and
                colormap.getColormapLUT() is not None):
            colors = colormap.getColormapLUT()
            if (colors.shape[-1] == 4 and
                    not numpy.all(numpy.equal(colors[3], 255))):
                # This is a transparent colormap
                if (colors.shape == (256, 4) and
                        colormap.getNormalization() == 'linear' and
                        not colormap.isAutoscale() and
                        colormap.getVMin() == 0 and
                        colormap.getVMax() == 255 and
                        data.dtype == numpy.uint8):
                    # Supported case, convert data to RGBA
                    return colors[data.reshape(-1)].reshape(
                        data.shape + (4,))
                else:
                    _logger.warning(
                        'matplotlib %s does not support transparent '
                        'colormap.', matplotlib.__version__)

    scalarMappable = getScalarMappable(colormap, data)
    rgbaImage = scalarMappable.to_rgba(data, bytes=True)

    return rgbaImage


@deprecated(replacement='silx.colors.Colormap.getSupportedColormaps',
            since_version='0.10.0')
def getSupportedColormaps():
    """Get the supported colormap names as a tuple of str.
    """
    colormaps = set(matplotlib.cm.datad.keys())
    colormaps.update(_AVAILABLE_AS_BUILTINS)
    colormaps.update(_AVAILABLE_AS_RESOURCE)
    return tuple(sorted(colormaps))
