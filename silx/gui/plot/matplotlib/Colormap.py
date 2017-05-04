# New matplotlib colormaps by Nathaniel J. Smith, Stefan van der Walt,
# and (in the case of viridis) Eric Firing.
#
# This file and the colormaps in it are released under the CC0 license /
# public domain dedication. We would appreciate credit if you use or
# redistribute these colormaps, but do not impose any legal restrictions.
#
# To the extent possible under law, the persons who associated CC0 with
# mpl-colormaps have waived all copyright and related or neighboring rights
# to mpl-colormaps.
#
# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
"""Matplotlib's new colormaps"""

import numpy
import logging
from matplotlib.colors import ListedColormap
import matplotlib.colors
import matplotlib.cm
import silx.resources

_logger = logging.getLogger(__name__)

_AVAILABLE_AS_RESOURCE = ('magma', 'inferno', 'plasma', 'viridis')
"""List available colormap name as resources"""

_CMAPS = {}
"""Cache colormaps"""


@property
def magma():
    return getColormap('magma')


@property
def inferno():
    return getColormap('inferno')


@property
def plasma():
    return getColormap('plasma')


@property
def viridis():
    return getColormap('viridis')


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


def getScalarMappable(colormap, data=None):
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
        cmap = getColormap(colormap['name'])

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
