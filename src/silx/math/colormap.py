# /*##########################################################################
#
# Copyright (c) 2018-2021 European Synchrotron Radiation Facility
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
"""This module provides helper functions for applying colormaps to datasets"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/08/2021"


import collections
import warnings
import numpy

from ..resources import resource_filename as _resource_filename
from .combo import min_max as _min_max
from . import _colormap
from ._colormap import cmap  # noqa


__all__ = ["apply_colormap", "cmap"]


_LUT_DESCRIPTION = collections.namedtuple("_LUT_DESCRIPTION", ["source", "cursor_color"])
"""Description of a LUT for internal purpose."""


_AVAILABLE_LUTS = collections.OrderedDict([
    ('gray', _LUT_DESCRIPTION('builtin', '#ff66ff')),
    ('reversed gray', _LUT_DESCRIPTION('builtin', '#ff66ff')),
    ('red', _LUT_DESCRIPTION('builtin', '#00ff00')),
    ('green', _LUT_DESCRIPTION('builtin', '#ff66ff')),
    ('blue', _LUT_DESCRIPTION('builtin', '#ffff00')),
    ('viridis', _LUT_DESCRIPTION('resource', '#ff66ff')),
    ('cividis', _LUT_DESCRIPTION('resource', '#ff66ff')),
    ('magma', _LUT_DESCRIPTION('resource', '#00ff00')),
    ('inferno', _LUT_DESCRIPTION('resource', '#00ff00')),
    ('plasma', _LUT_DESCRIPTION('resource', '#00ff00')),
    ('temperature', _LUT_DESCRIPTION('builtin', '#ff66ff')),
])
"""Description for internal porpose of all the default LUT provided by the library."""


# Colormap loader

_COLORMAP_CACHE = {}
"""Cache already used colormaps as name: color LUT"""


def array_to_rgba8888(colors):
    """Convert colors from a numpy array using float (0..1) int or uint
    (0..255) to uint8 RGBA.

    :param numpy.ndarray colors: Array of float int or uint  colors to convert
    :return: colors as uint8
    :rtype: numpy.ndarray
    """
    assert len(colors.shape) == 2
    assert colors.shape[1] in (3, 4)

    if colors.dtype == numpy.uint8:
        pass
    elif colors.dtype.kind == 'f':
        # Each bin is [N, N+1[ except the last one: [255, 256]
        colors = numpy.clip(colors.astype(numpy.float64) * 256, 0., 255.)
        colors = colors.astype(numpy.uint8)
    elif colors.dtype.kind in 'iu':
        colors = numpy.clip(colors, 0, 255)
        colors = colors.astype(numpy.uint8)

    if colors.shape[1] == 3:
        tmp = numpy.empty((len(colors), 4), dtype=numpy.uint8)
        tmp[:, 0:3] = colors
        tmp[:, 3] = 255
        colors = tmp

    return colors


def _create_colormap_lut(name):
    """Returns the color LUT corresponding to a colormap name

    :param str name: Name of the colormap to load
    :returns: Corresponding table of colors
    :rtype: numpy.ndarray
    :raise ValueError: If no colormap corresponds to name
    """
    description = _AVAILABLE_LUTS.get(name)
    if description is not None:
        if description.source == "builtin":
            # Build colormap LUT
            lut = numpy.zeros((256, 4), dtype=numpy.uint8)
            lut[:, 3] = 255

            if name == 'gray':
                lut[:, :3] = numpy.arange(256, dtype=numpy.uint8).reshape(-1, 1)
            elif name == 'reversed gray':
                lut[:, :3] = numpy.arange(255, -1, -1, dtype=numpy.uint8).reshape(-1, 1)
            elif name == 'red':
                lut[:, 0] = numpy.arange(256, dtype=numpy.uint8)
            elif name == 'green':
                lut[:, 1] = numpy.arange(256, dtype=numpy.uint8)
            elif name == 'blue':
                lut[:, 2] = numpy.arange(256, dtype=numpy.uint8)
            elif name == 'temperature':
                # Red
                lut[128:192, 0] = numpy.arange(2, 255, 4, dtype=numpy.uint8)
                lut[192:, 0] = 255
                # Green
                lut[:64, 1] = numpy.arange(0, 255, 4, dtype=numpy.uint8)
                lut[64:192, 1] = 255
                lut[192:, 1] = numpy.arange(252, -1, -4, dtype=numpy.uint8)
                # Blue
                lut[:64, 2] = 255
                lut[64:128, 2] = numpy.arange(254, 0, -4, dtype=numpy.uint8)
            else:
                raise RuntimeError("Built-in colormap not implemented")
            return lut

        elif description.source == "resource":
            # Load colormap LUT
            colors = numpy.load(_resource_filename("gui/colormaps/%s.npy" % name))
            # Convert to uint8 and add alpha channel
            lut = array_to_rgba8888(colors)
            return lut

        else:
            raise RuntimeError("Internal LUT source '%s' unsupported" % description.source)

    raise ValueError("Unknown colormap '%s'" % name)


def register_colormap(name, lut, cursor_color='#000000'):
    """Register a custom colormap LUT

    It can override existing LUT names.

    :param str name: Name of the LUT as defined to configure colormaps
    :param numpy.ndarray lut: The custom LUT to register.
            Nx3 or Nx4 numpy array of RGB(A) colors,
            either uint8 or float in [0, 1].
    :param str cursor_color: Color used to display overlay over images using
        colormap with this LUT.
    """
    description = _LUT_DESCRIPTION('user', cursor_color)
    colors = array_to_rgba8888(lut)
    _AVAILABLE_LUTS[name] = description

    # Register the cache as the LUT was already loaded
    _COLORMAP_CACHE[name] = colors


def get_registered_colormaps():
    """Returns currently registered colormap names"""
    return tuple(_AVAILABLE_LUTS.keys())


def get_colormap_cursor_color(name):
    """Get a color suitable for overlay over a colormap.

    :param str name: The name of the colormap.
    :return: Name of the color.
    :rtype: str
    """
    description = _AVAILABLE_LUTS.get(name, None)
    if description is not None:
        color = description.cursor_color
        if color is not None:
            return color
    return 'black'


def get_colormap_lut(name):
    """Returns the color LUT corresponding to a colormap name

    :param str name: Name of the colormap to load
    :returns: Corresponding table of colors
    :rtype: numpy.ndarray
    :raise ValueError: If no colormap corresponds to name
    """
    name = str(name)
    if name not in _COLORMAP_CACHE:
        lut = _create_colormap_lut(name)
        _COLORMAP_CACHE[name] = lut
    return _COLORMAP_CACHE[name]


# Normalizations

class _NormalizationMixIn:
    """Colormap normalization mix-in class"""

    DEFAULT_RANGE = 0, 1
    """Fallback for (vmin, vmax)"""

    def is_valid(self, value):
        """Check if a value is in the valid range for this normalization.

        Override in subclass.

        :param Union[float,numpy.ndarray] value:
        :rtype: Union[bool,numpy.ndarray]
        """
        if isinstance(value, collections.abc.Iterable):
            return numpy.ones_like(value, dtype=numpy.bool_)
        else:
            return True

    def autoscale(self, data, mode):
        """Returns range for given data and autoscale mode.

        :param Union[None,numpy.ndarray] data:
        :param str mode: Autoscale mode: 'minmax' or 'stddev3'
        :returns: Range as (min, max)
        :rtype: Tuple[float,float]
        """
        data = None if data is None else numpy.array(data, copy=False)
        if data is None or data.size == 0:
            return self.DEFAULT_RANGE

        if mode == "minmax":
            vmin, vmax = self.autoscale_minmax(data)
        elif mode == "stddev3":
            dmin, dmax = self.autoscale_minmax(data)
            stdmin, stdmax = self.autoscale_mean3std(data)
            if dmin is None:
                vmin = stdmin
            elif stdmin is None:
                vmin = dmin
            else:
                vmin = max(dmin, stdmin)

            if dmax is None:
                vmax = stdmax
            elif stdmax is None:
                vmax = dmax
            else:
                vmax = min(dmax, stdmax)

        else:
            raise ValueError('Unsupported mode: %s' % mode)

        # Check returned range and handle fallbacks
        if vmin is None or not numpy.isfinite(vmin):
            vmin = self.DEFAULT_RANGE[0]
        if vmax is None or not numpy.isfinite(vmax):
            vmax = self.DEFAULT_RANGE[1]
        if vmax < vmin:
            vmax = vmin
        return float(vmin), float(vmax)

    def autoscale_minmax(self, data):
        """Autoscale using min/max

        :param numpy.ndarray data:
        :returns: (vmin, vmax)
        :rtype: Tuple[float,float]
        """
        data = data[self.is_valid(data)]
        if data.size == 0:
            return None, None
        result = _min_max(data, min_positive=False, finite=True)
        return result.minimum, result.maximum

    def autoscale_mean3std(self, data):
        """Autoscale using mean+/-3std

        This implementation only works for normalization that do NOT
        use the data range.
        Override this method for normalization using the range.

        :param numpy.ndarray data:
        :returns: (vmin, vmax)
        :rtype: Tuple[float,float]
        """
        # Use [0, 1] as data range for normalization not using range
        normdata = self.apply(data, 0., 1.)
        if normdata.dtype.kind == 'f':  # Replaces inf by NaN
            normdata[numpy.isfinite(normdata) == False] = numpy.nan
        if normdata.size == 0:  # Fallback
            return None, None

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Ignore nanmean "Mean of empty slice" warning and
            # nanstd "Degrees of freedom <= 0 for slice" warning
            mean, std = numpy.nanmean(normdata), numpy.nanstd(normdata)

        return self.revert(mean - 3 * std, 0., 1.), self.revert(mean + 3 * std, 0., 1.)


class _LinearNormalizationMixIn(_NormalizationMixIn):
    """Colormap normalization mix-in class specific to autoscale taken from initial range"""

    def autoscale_mean3std(self, data):
        """Autoscale using mean+/-3std

        Do the autoscale on the data itself, not the normalized data.

        :param numpy.ndarray data:
        :returns: (vmin, vmax)
        :rtype: Tuple[float,float]
        """
        if data.dtype.kind == 'f':  # Replaces inf by NaN
            data = numpy.array(data, copy=True)  # Work on a copy
            data[numpy.isfinite(data) == False] = numpy.nan
        if data.size == 0:  # Fallback
            return None, None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Ignore nanmean "Mean of empty slice" warning and
            # nanstd "Degrees of freedom <= 0 for slice" warning
            mean, std = numpy.nanmean(data), numpy.nanstd(data)
        return mean - 3 * std, mean + 3 * std


class LinearNormalization(_colormap.LinearNormalization, _LinearNormalizationMixIn):
    """Linear normalization"""
    def __init__(self):
        _colormap.LinearNormalization.__init__(self)
        _LinearNormalizationMixIn.__init__(self)


class LogarithmicNormalization(_colormap.LogarithmicNormalization, _NormalizationMixIn):
    """Logarithm normalization"""

    DEFAULT_RANGE = 1, 10

    def __init__(self):
        _colormap.LogarithmicNormalization.__init__(self)
        _NormalizationMixIn.__init__(self)

    def is_valid(self, value):
        return value > 0.

    def autoscale_minmax(self, data):
        result = _min_max(data, min_positive=True, finite=True)
        return result.min_positive, result.maximum


class SqrtNormalization(_colormap.SqrtNormalization, _NormalizationMixIn):
    """Square root normalization"""

    DEFAULT_RANGE = 0, 1

    def __init__(self):
        _colormap.SqrtNormalization.__init__(self)
        _NormalizationMixIn.__init__(self)

    def is_valid(self, value):
        return value >= 0.


class GammaNormalization(_colormap.PowerNormalization, _LinearNormalizationMixIn):
    """Gamma correction normalization:

    Linear normalization to [0, 1] followed by power normalization.

    :param gamma: Gamma correction factor
    """
    def __init__(self, gamma):
        _colormap.PowerNormalization.__init__(self, gamma)
        _LinearNormalizationMixIn.__init__(self)


# Backward compatibility
PowerNormalization = GammaNormalization


class ArcsinhNormalization(_colormap.ArcsinhNormalization, _NormalizationMixIn):
    """Inverse hyperbolic sine normalization"""

    def __init__(self):
        _colormap.ArcsinhNormalization.__init__(self)
        _NormalizationMixIn.__init__(self)


# Colormap function

_BASIC_NORMALIZATIONS = {
    "linear": LinearNormalization(),
    "log": LogarithmicNormalization(),
    "sqrt": SqrtNormalization(),
    "arcsinh": ArcsinhNormalization(),
}

_DEFAULT_NAN_COLOR = 255, 255, 255, 0

def apply_colormap(data,
                   colormap: str,
                   norm: str="linear",
                   autoscale: str="minmax",
                   vmin=None,
                   vmax=None,
                   gamma=1.0):
    """Apply colormap to data with given normalization and autoscale.

    :param numpy.ndarray data: Data on which to apply the colormap
    :param str colormap: Name of the colormap to use
    :param str norm: Normalization to use
    :param str autoscale: Autoscale mode: "minmax" (default) or "stddev3"
    :param vmin: Lower bound, None (default) to autoscale
    :param vmax: Upper bound, None (default) to autoscale
    :param float gamma:
        Gamma correction parameter (used only for "gamma" normalization)
    :returns: Array of colors
    """
    colors = get_colormap_lut(colormap)

    if norm == "gamma":
        normalizer = GammaNormalization(gamma)
    else:
        normalizer = _BASIC_NORMALIZATIONS[norm]

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = normalizer.autoscale(data, autoscale)
        if vmin is None:  # Set vmin respecting provided vmax
            vmin = auto_vmin if vmax is None else min(auto_vmin, vmax)
        if vmax is None:
            vmax = max(auto_vmax, vmin)  # Handle max_ <= 0 for log scale

    return _colormap.cmap(
        data,
        colors,
        vmin,
        vmax,
        normalizer,
        _DEFAULT_NAN_COLOR,
    )
