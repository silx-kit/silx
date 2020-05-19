# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2020 European Synchrotron Radiation Facility
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
"""This module provides API to manage colors.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent", "H.Payno"]
__license__ = "MIT"
__date__ = "29/01/2019"

import numpy
import logging
import collections
from silx.gui import qt
from silx.math.combo import min_max
from silx.math import colormap as _colormap
from silx.utils.exceptions import NotEditableError
from silx.utils import deprecation
from silx.resources import resource_filename as _resource_filename


_logger = logging.getLogger(__file__)

try:
    from matplotlib import cm as _matplotlib_cm
except ImportError:
    _logger.info("matplotlib not available, only embedded colormaps available")
    _matplotlib_cm = None


_COLORDICT = {}
"""Dictionary of common colors."""

_COLORDICT['b'] = _COLORDICT['blue'] = '#0000ff'
_COLORDICT['r'] = _COLORDICT['red'] = '#ff0000'
_COLORDICT['g'] = _COLORDICT['green'] = '#00ff00'
_COLORDICT['k'] = _COLORDICT['black'] = '#000000'
_COLORDICT['w'] = _COLORDICT['white'] = '#ffffff'
_COLORDICT['pink'] = '#ff66ff'
_COLORDICT['brown'] = '#a52a2a'
_COLORDICT['orange'] = '#ff9900'
_COLORDICT['violet'] = '#6600ff'
_COLORDICT['gray'] = _COLORDICT['grey'] = '#a0a0a4'
# _COLORDICT['darkGray'] = _COLORDICT['darkGrey'] = '#808080'
# _COLORDICT['lightGray'] = _COLORDICT['lightGrey'] = '#c0c0c0'
_COLORDICT['y'] = _COLORDICT['yellow'] = '#ffff00'
_COLORDICT['m'] = _COLORDICT['magenta'] = '#ff00ff'
_COLORDICT['c'] = _COLORDICT['cyan'] = '#00ffff'
_COLORDICT['darkBlue'] = '#000080'
_COLORDICT['darkRed'] = '#800000'
_COLORDICT['darkGreen'] = '#008000'
_COLORDICT['darkBrown'] = '#660000'
_COLORDICT['darkCyan'] = '#008080'
_COLORDICT['darkYellow'] = '#808000'
_COLORDICT['darkMagenta'] = '#800080'
_COLORDICT['transparent'] = '#00000000'


# FIXME: It could be nice to expose a functional API instead of that attribute
COLORDICT = _COLORDICT


_LUT_DESCRIPTION = collections.namedtuple("_LUT_DESCRIPTION", ["source", "cursor_color", "preferred"])
"""Description of a LUT for internal purpose."""


_AVAILABLE_LUTS = collections.OrderedDict([
    ('gray', _LUT_DESCRIPTION('builtin', 'pink', True)),
    ('reversed gray', _LUT_DESCRIPTION('builtin', 'pink', True)),
    ('temperature', _LUT_DESCRIPTION('builtin', 'pink', True)),
    ('red', _LUT_DESCRIPTION('builtin', 'green', True)),
    ('green', _LUT_DESCRIPTION('builtin', 'pink', True)),
    ('blue', _LUT_DESCRIPTION('builtin', 'yellow', True)),
    ('jet', _LUT_DESCRIPTION('matplotlib', 'pink', True)),
    ('viridis', _LUT_DESCRIPTION('resource', 'pink', True)),
    ('cividis', _LUT_DESCRIPTION('resource', 'pink', True)),
    ('magma', _LUT_DESCRIPTION('resource', 'green', True)),
    ('inferno', _LUT_DESCRIPTION('resource', 'green', True)),
    ('plasma', _LUT_DESCRIPTION('resource', 'green', True)),
    ('hsv', _LUT_DESCRIPTION('matplotlib', 'black', True)),
])
"""Description for internal porpose of all the default LUT provided by the library."""


DEFAULT_MIN_LIN = 0
"""Default min value if in linear normalization"""
DEFAULT_MAX_LIN = 1
"""Default max value if in linear normalization"""


def rgba(color, colorDict=None):
    """Convert color code '#RRGGBB' and '#RRGGBBAA' to a tuple (R, G, B, A)
    of floats.

    It also supports RGB(A) from uint8 in [0, 255], float in [0, 1], and
    QColor as color argument.

    :param str color: The color to convert
    :param dict colorDict: A dictionary of color name conversion to color code
    :returns: RGBA colors as floats in [0., 1.]
    :rtype: tuple
    """
    if colorDict is None:
        colorDict = _COLORDICT

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


def greyed(color, colorDict=None):
    """Convert color code '#RRGGBB' and '#RRGGBBAA' to a grey color
    (R, G, B, A).

    It also supports RGB(A) from uint8 in [0, 255], float in [0, 1], and
    QColor as color argument.

    :param str color: The color to convert
    :param dict colorDict: A dictionary of color name conversion to color code
    :returns: RGBA colors as floats in [0., 1.]
    :rtype: tuple
    """
    r, g, b, a = rgba(color=color, colorDict=colorDict)
    g = 0.21 * r + 0.72 * g + 0.07 * b
    return g, g, g, a


def asQColor(color):
    """Convert color code '#RRGGBB' and '#RRGGBBAA' to a `qt.QColor`.

    It also supports RGB(A) from uint8 in [0, 255], float in [0, 1], and
    QColor as color argument.

    :param str color: The color to convert
    :rtype: qt.QColor
    """
    color = rgba(color)
    return qt.QColor.fromRgbF(*color)


def cursorColorForColormap(colormapName):
    """Get a color suitable for overlay over a colormap.

    :param str colormapName: The name of the colormap.
    :return: Name of the color.
    :rtype: str
    """
    description = _AVAILABLE_LUTS.get(colormapName, None)
    if description is not None:
        color = description.cursor_color
        if color is not None:
            return color
    return 'black'


# Colormap loader

_COLORMAP_CACHE = {}
"""Cache already used colormaps as name: color LUT"""


def _arrayToRgba8888(colors):
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


def _createColormapLut(name):
    """Returns the color LUT corresponding to a colormap name

    :param str name: Name of the colormap to load
    :returns: Corresponding table of colors
    :rtype: numpy.ndarray
    :raise ValueError: If no colormap corresponds to name
    """
    description = _AVAILABLE_LUTS.get(name)
    use_mpl = False
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
            lut = _arrayToRgba8888(colors)
            return lut

        elif description.source == "matplotlib":
            use_mpl = True

        else:
            raise RuntimeError("Internal LUT source '%s' unsupported" % description.source)

    # Here it expect a matplotlib LUTs

    if use_mpl:
        # matplotlib is mandatory
        if _matplotlib_cm is None:
            raise ValueError("The colormap '%s' expect matplotlib, but matplotlib is not installed" % name)

    if _matplotlib_cm is not None:  # Try to load with matplotlib
        colormap = _matplotlib_cm.get_cmap(name)
        lut = colormap(numpy.linspace(0, 1, colormap.N, endpoint=True))
        lut = _arrayToRgba8888(lut)
        return lut

    raise ValueError("Unknown colormap '%s'" % name)


def _getColormap(name):
    """Returns the color LUT corresponding to a colormap name

    :param str name: Name of the colormap to load
    :returns: Corresponding table of colors
    :rtype: numpy.ndarray
    :raise ValueError: If no colormap corresponds to name
    """
    name = str(name)
    if name not in _COLORMAP_CACHE:
        lut = _createColormapLut(name)
        _COLORMAP_CACHE[name] = lut
    return _COLORMAP_CACHE[name]


# Normalizations

class _NormalizationMixIn:
    """Colormap normalization mix-in class"""

    DEFAULT_RANGE = 0, 1
    """Fallback for (vmin, vmax)"""

    def isValid(self, value):
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
        :param str mode: Autoscale mode, see :class:`Colormap`
        :returns: Range as (min, max)
        :rtype: Tuple[float,float]
        """
        data = None if data is None else numpy.array(data, copy=False)
        if data is None or data.size == 0:
            return self.DEFAULT_RANGE

        if mode == Colormap.MINMAX:
            vmin, vmax = self.autoscaleMinMax(data)
        elif mode == Colormap.STDDEV3:
            vmin, vmax = self.autoscaleMean3Std(data)
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

    def autoscaleMinMax(self, data):
        """Autoscale using min/max

        :param numpy.ndarray data:
        :returns: (vmin, vmax)
        :rtype: Tuple[float,float]
        """
        data = data[self.isValid(data)]
        if data.size == 0:
            return None, None
        result = min_max(data, min_positive=False, finite=True)
        return result.minimum, result.maximum

    def autoscaleMean3Std(self, data):
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
        mean, std = numpy.nanmean(normdata), numpy.nanstd(normdata)
        return self.revert(mean - 3 * std, 0., 1.), self.revert(mean + 3 * std, 0., 1.)


class _LinearNormalizationMixIn(_NormalizationMixIn):
    """Colormap normalization mix-in class specific to autoscale taken from initial range"""

    def autoscaleMean3Std(self, data):
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
        mean, std = numpy.nanmean(data), numpy.nanstd(data)
        return mean - 3 * std, mean + 3 * std


class _LinearNormalization(_colormap.LinearNormalization, _LinearNormalizationMixIn):
    """Linear normalization"""
    def __init__(self):
        _colormap.LinearNormalization.__init__(self)
        _LinearNormalizationMixIn.__init__(self)


class _LogarithmicNormalization(_colormap.LogarithmicNormalization, _NormalizationMixIn):
    """Logarithm normalization"""

    DEFAULT_RANGE = 1, 10

    def __init__(self):
        _colormap.LogarithmicNormalization.__init__(self)
        _NormalizationMixIn.__init__(self)

    def isValid(self, value):
        return value > 0.

    def autoscaleMinMax(self, data):
        result = min_max(data, min_positive=True, finite=True)
        return result.min_positive, result.maximum


class _SqrtNormalization(_colormap.SqrtNormalization, _NormalizationMixIn):
    """Square root normalization"""

    DEFAULT_RANGE = 0, 1

    def __init__(self):
        _colormap.SqrtNormalization.__init__(self)
        _NormalizationMixIn.__init__(self)

    def isValid(self, value):
        return value >= 0.


class _GammaNormalization(_colormap.PowerNormalization, _LinearNormalizationMixIn):
    """Gamma correction normalization:

    Linear normalization to [0, 1] followed by power normalization.

    :param gamma: Gamma correction factor
    """
    def __init__(self, gamma):
        _colormap.PowerNormalization.__init__(self, gamma)
        _LinearNormalizationMixIn.__init__(self)


class _ArcsinhNormalization(_colormap.ArcsinhNormalization, _NormalizationMixIn):
    """Inverse hyperbolic sine normalization"""

    def __init__(self):
        _colormap.ArcsinhNormalization.__init__(self)
        _NormalizationMixIn.__init__(self)


class Colormap(qt.QObject):
    """Description of a colormap

    If no `name` nor `colors` are provided, a default gray LUT is used.

    :param str name: Name of the colormap
    :param tuple colors: optional, custom colormap.
            Nx3 or Nx4 numpy array of RGB(A) colors,
            either uint8 or float in [0, 1].
            If 'name' is None, then this array is used as the colormap.
    :param str normalization: Normalization: 'linear' (default) or 'log'
    :param vmin: Lower bound of the colormap or None for autoscale (default)
    :type vmin: Union[None, float]
    :param vmax: Upper bounds of the colormap or None for autoscale (default)
    :type vmax: Union[None, float]
    """

    LINEAR = 'linear'
    """constant for linear normalization"""

    LOGARITHM = 'log'
    """constant for logarithmic normalization"""

    SQRT = 'sqrt'
    """constant for square root normalization"""

    GAMMA = 'gamma'
    """Constant for gamma correction normalization"""

    ARCSINH = 'arcsinh'
    """constant for inverse hyperbolic sine normalization"""

    _BASIC_NORMALIZATIONS = {
        LINEAR: _LinearNormalization(),
        LOGARITHM: _LogarithmicNormalization(),
        SQRT: _SqrtNormalization(),
        ARCSINH: _ArcsinhNormalization(),
        }
    """Normalizations without parameters"""

    NORMALIZATIONS = LINEAR, LOGARITHM, SQRT, GAMMA, ARCSINH
    """Tuple of managed normalizations"""

    MINMAX = 'minmax'
    """constant for autoscale using min/max data range"""

    STDDEV3 = 'stddev3'
    """constant for autoscale using mean +/- 3*std(data)"""

    AUTOSCALE_MODES = (MINMAX, STDDEV3)
    """Tuple of managed auto scale algorithms"""

    sigChanged = qt.Signal()
    """Signal emitted when the colormap has changed."""

    def __init__(self, name=None, colors=None, normalization=LINEAR, vmin=None, vmax=None, autoscaleMode=MINMAX):
        qt.QObject.__init__(self)
        self._editable = True
        self.__gamma = 2.0

        assert normalization in Colormap.NORMALIZATIONS
        assert autoscaleMode in Colormap.AUTOSCALE_MODES

        if normalization is Colormap.LOGARITHM:
            if (vmin is not None and vmin < 0) or (vmax is not None and vmax < 0):
                m = "Unsuported vmin (%s) and/or vmax (%s) given for a log scale."
                m += ' Autoscale will be performed.'
                m = m % (vmin, vmax)
                _logger.warning(m)
                vmin = None
                vmax = None

        self._name = None
        self._colors = None

        if colors is not None and name is not None:
            deprecation.deprecated_warning("Argument",
                                           name="silx.gui.plot.Colors",
                                           reason="name and colors can't be used at the same time",
                                           since_version="0.10.0",
                                           skip_backtrace_count=1)

            colors = None

        if name is not None:
            self.setName(name)  # And resets colormap LUT
        elif colors is not None:
            self.setColormapLUT(colors)
        else:
            # Default colormap is grey
            self.setName("gray")

        self._normalization = str(normalization)
        self._autoscaleMode = str(autoscaleMode)
        self._vmin = float(vmin) if vmin is not None else None
        self._vmax = float(vmax) if vmax is not None else None

    def setFromColormap(self, other):
        """Set this colormap using information from the `other` colormap.

        :param ~silx.gui.colors.Colormap other: Colormap to use as reference.
        """
        if not self.isEditable():
            raise NotEditableError('Colormap is not editable')
        if self == other:
            return
        old = self.blockSignals(True)
        name = other.getName()
        if name is not None:
            self.setName(name)
        else:
            self.setColormapLUT(other.getColormapLUT())
        self.setNormalization(other.getNormalization())
        self.setVRange(other.getVMin(), other.getVMax())
        self.blockSignals(old)
        self.sigChanged.emit()

    def getNColors(self, nbColors=None):
        """Returns N colors computed by sampling the colormap regularly.

        :param nbColors:
            The number of colors in the returned array or None for the default value.
            The default value is the size of the colormap LUT.
        :type nbColors: int or None
        :return: 2D array of uint8 of shape (nbColors, 4)
        :rtype: numpy.ndarray
        """
        # Handle default value for nbColors
        if nbColors is None:
            return numpy.array(self._colors, copy=True)
        else:
            nbColors = int(nbColors)
            colormap = self.copy()
            colormap.setNormalization(Colormap.LINEAR)
            colormap.setVRange(vmin=0, vmax=nbColors - 1)
            colors = colormap.applyToData(
                numpy.arange(nbColors, dtype=numpy.int))
            return colors

    def getName(self):
        """Return the name of the colormap
        :rtype: str
        """
        return self._name

    def setName(self, name):
        """Set the name of the colormap to use.

        :param str name: The name of the colormap.
            At least the following names are supported: 'gray',
            'reversed gray', 'temperature', 'red', 'green', 'blue', 'jet',
            'viridis', 'magma', 'inferno', 'plasma'.
        """
        name = str(name)
        if self._name == name:
            return
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        if name not in self.getSupportedColormaps():
            raise ValueError("Colormap name '%s' is not supported" % name)
        self._name = name
        self._colors = _getColormap(self._name)
        self.sigChanged.emit()

    def getColormapLUT(self, copy=True):
        """Return the list of colors for the colormap or None if not set.

        This returns None if the colormap was set with :meth:`setName`.
        Use :meth:`getNColors` to get the colormap LUT for any colormap.

        :param bool copy: If true a copy of the numpy array is provided
        :return: the list of colors for the colormap or None if not set
        :rtype: numpy.ndarray or None
        """
        if self._name is None:
            return numpy.array(self._colors, copy=copy)
        else:
            return None

    def setColormapLUT(self, colors):
        """Set the colors of the colormap.

        :param numpy.ndarray colors: the colors of the LUT.
           If float, it is converted from [0, 1] to uint8 range.
           Otherwise it is casted to uint8.

        .. warning: this will set the value of name to None
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        assert colors is not None

        colors = numpy.array(colors, copy=False)
        if colors.shape == ():
            raise TypeError("An array is expected for 'colors' argument. '%s' was found." % type(colors))
        assert len(colors) != 0
        assert colors.ndim >= 2
        colors.shape = -1, colors.shape[-1]
        self._colors = _arrayToRgba8888(colors)
        self._name = None
        self.sigChanged.emit()

    def getNormalization(self):
        """Return the normalization of the colormap.

        See :meth:`setNormalization` for returned values.

        :return: the normalization of the colormap
        :rtype: str
        """
        return self._normalization

    def setNormalization(self, norm):
        """Set the colormap normalization.

        Accepted normalizations: 'log', 'linear', 'sqrt'

        :param str norm: the norm to set
        """
        assert norm in self.NORMALIZATIONS
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        self._normalization = str(norm)
        self.sigChanged.emit()

    def setGammaNormalizationParameter(self, gamma: float) -> None:
        """Set the gamma correction parameter.

        Only used for gamma correction normalization.

        :param float gamma:
        :raise ValueError: If gamma is not valid
        """
        if gamma < 0. or not numpy.isfinite(gamma):
            raise ValueError("Gamma value not supported")
        if gamma != self.__gamma:
            self.__gamma = gamma
            self.sigChanged.emit()

    def getGammaNormalizationParameter(self) -> float:
        """Returns the gamma correction parameter value.

        :rtype: float
        """
        return self.__gamma

    def getAutoscaleMode(self):
        """Return the autoscale mode of the colormap ('minmax' or 'stddev3')

        :rtype: str
        """
        return self._autoscaleMode

    def setAutoscaleMode(self, mode):
        """Set the autoscale mode: either 'minmax' or 'stddev3'

        :param str mode: the mode to set
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        assert mode in self.AUTOSCALE_MODES
        if mode != self._autoscaleMode:
            self._autoscaleMode = mode
            self.sigChanged.emit()

    def isAutoscale(self):
        """Return True if both min and max are in autoscale mode"""
        return self._vmin is None and self._vmax is None

    def getVMin(self):
        """Return the lower bound of the colormap

         :return: the lower bound of the colormap
         :rtype: float or None
         """
        return self._vmin

    def setVMin(self, vmin):
        """Set the minimal value of the colormap

        :param float vmin: Lower bound of the colormap or None for autoscale
            (default)
            value)
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        if vmin is not None:
            if self._vmax is not None and vmin > self._vmax:
                err = "Can't set vmin because vmin >= vmax. " \
                      "vmin = %s, vmax = %s" % (vmin, self._vmax)
                raise ValueError(err)

        self._vmin = vmin
        self.sigChanged.emit()

    def getVMax(self):
        """Return the upper bounds of the colormap or None

        :return: the upper bounds of the colormap or None
        :rtype: float or None
        """
        return self._vmax

    def setVMax(self, vmax):
        """Set the maximal value of the colormap

        :param float vmax: Upper bounds of the colormap or None for autoscale
            (default)
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        if vmax is not None:
            if self._vmin is not None and vmax < self._vmin:
                err = "Can't set vmax because vmax <= vmin. " \
                      "vmin = %s, vmax = %s" % (self._vmin, vmax)
                raise ValueError(err)

        self._vmax = vmax
        self.sigChanged.emit()

    def isEditable(self):
        """ Return if the colormap is editable or not

        :return: editable state of the colormap
         :rtype: bool
        """
        return self._editable

    def setEditable(self, editable):
        """
        Set the editable state of the colormap

        :param bool editable: is the colormap editable
        """
        assert type(editable) is bool
        self._editable = editable
        self.sigChanged.emit()

    def _getNormalizer(self):
        """Returns normalizer object"""
        normalization = self.getNormalization()
        if normalization == self.GAMMA:
            return _GammaNormalization(self.getGammaNormalizationParameter())
        else:
            return self._BASIC_NORMALIZATIONS[normalization]

    def _computeAutoscaleRange(self, data):
        """Compute the data range which will be used in autoscale mode.

        :param numpy.ndarray data: The data for which to compute the range
        :return: (vmin, vmax) range
        """
        return self._getNormalizer().autoscale(
            data, mode=self.getAutoscaleMode())

    def getColormapRange(self, data=None):
        """Return (vmin, vmax) the range of the colormap for the given data or item.

        :param Union[numpy.ndarray,~silx.gui.plot.items.ColormapMixIn] data:
            The data or item to use for autoscale bounds.
        :return: (vmin, vmax) corresponding to the colormap applied to data if provided.
        :rtype: tuple
        """
        vmin = self._vmin
        vmax = self._vmax
        assert vmin is None or vmax is None or vmin <= vmax  # TODO handle this in setters

        normalizer = self._getNormalizer()

        # Handle invalid bounds as autoscale
        if vmin is not None and not normalizer.isValid(vmin):
            _logger.info(
                'Invalid vmin, switching to autoscale for lower bound')
            vmin = None
        if vmax is not None and not normalizer.isValid(vmax):
            _logger.info(
                'Invalid vmax, switching to autoscale for upper bound')
            vmax = None

        if vmin is None or vmax is None:  # Handle autoscale
            from .plot.items.core import ColormapMixIn  # avoid cyclic import
            if isinstance(data, ColormapMixIn):
                min_, max_ = data._getColormapAutoscaleRange(self)
                # Make sure min_, max_ are not None
                min_ = normalizer.DEFAULT_RANGE[0] if min_ is None else min_
                max_ = normalizer.DEFAULT_RANGE[1] if max_ is None else max_
            else:
                min_, max_ = normalizer.autoscale(
                    data, mode=self.getAutoscaleMode())

            if vmin is None:  # Set vmin respecting provided vmax
                vmin = min_ if vmax is None else min(min_, vmax)

            if vmax is None:
                vmax = max(max_, vmin)  # Handle max_ <= 0 for log scale

        return vmin, vmax

    def getVRange(self):
        """Get the bounds of the colormap

        :rtype: Tuple(Union[float,None],Union[float,None])
        :returns: A tuple of 2 values for min and max. Or None instead of float
            for autoscale
        """
        return self.getVMin(), self.getVMax()

    def setVRange(self, vmin, vmax):
        """Set the bounds of the colormap

        :param vmin: Lower bound of the colormap or None for autoscale
            (default)
        :param vmax: Upper bounds of the colormap or None for autoscale
            (default)
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        if vmin is not None and vmax is not None:
            if vmin > vmax:
                err = "Can't set vmin and vmax because vmin >= vmax " \
                      "vmin = %s, vmax = %s" % (vmin, vmax)
                raise ValueError(err)

        if self._vmin == vmin and self._vmax == vmax:
            return

        self._vmin = vmin
        self._vmax = vmax
        self.sigChanged.emit()

    def __getitem__(self, item):
        if item == 'autoscale':
            return self.isAutoscale()
        elif item == 'name':
            return self.getName()
        elif item == 'normalization':
            return self.getNormalization()
        elif item == 'vmin':
            return self.getVMin()
        elif item == 'vmax':
            return self.getVMax()
        elif item == 'colors':
            return self.getColormapLUT()
        elif item == 'autoscaleMode':
            return self.getAutoscaleMode()
        else:
            raise KeyError(item)

    def _toDict(self):
        """Return the equivalent colormap as a dictionary
        (old colormap representation)

        :return: the representation of the Colormap as a dictionary
        :rtype: dict
        """
        return {
            'name': self._name,
            'colors': self.getColormapLUT(),
            'vmin': self._vmin,
            'vmax': self._vmax,
            'autoscale': self.isAutoscale(),
            'normalization': self.getNormalization(),
            'autoscaleMode': self.getAutoscaleMode(),
            }

    def _setFromDict(self, dic):
        """Set values to the colormap from a dictionary

        :param dict dic: the colormap as a dictionary
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        name = dic['name'] if 'name' in dic else None
        colors = dic['colors'] if 'colors' in dic else None
        if name is not None and colors is not None:
            if isinstance(colors, int):
                # Filter out argument which was supported but never used
                _logger.info("Unused 'colors' from colormap dictionary filterer.")
                colors = None
        vmin = dic['vmin'] if 'vmin' in dic else None
        vmax = dic['vmax'] if 'vmax' in dic else None
        if 'normalization' in dic:
            normalization = dic['normalization']
        else:
            warn = 'Normalization not given in the dictionary, '
            warn += 'set by default to ' + Colormap.LINEAR
            _logger.warning(warn)
            normalization = Colormap.LINEAR

        if name is None and colors is None:
            err = 'The colormap should have a name defined or a tuple of colors'
            raise ValueError(err)
        if normalization not in Colormap.NORMALIZATIONS:
            err = 'Given normalization is not recognized (%s)' % normalization
            raise ValueError(err)

        autoscaleMode = dic.get('autoscaleMode', Colormap.MINMAX)
        if autoscaleMode not in Colormap.AUTOSCALE_MODES:
            err = 'Given autoscale mode is not recognized (%s)' % autoscaleMode
            raise ValueError(err)

        # If autoscale, then set boundaries to None
        if dic.get('autoscale', False):
            vmin, vmax = None, None

        if name is not None:
            self.setName(name)
        else:
            self.setColormapLUT(colors)
        self._vmin = vmin
        self._vmax = vmax
        self._autoscale = True if (vmin is None and vmax is None) else False
        self._normalization = normalization
        self._autoscaleMode = autoscaleMode

        self.sigChanged.emit()

    @staticmethod
    def _fromDict(dic):
        colormap = Colormap()
        colormap._setFromDict(dic)
        return colormap

    def copy(self):
        """Return a copy of the Colormap.

        :rtype: silx.gui.colors.Colormap
        """
        colormap = Colormap(name=self._name,
                        colors=self.getColormapLUT(),
                        vmin=self._vmin,
                        vmax=self._vmax,
                        normalization=self.getNormalization(),
                        autoscaleMode=self.getAutoscaleMode())
        colormap.setGammaNormalizationParameter(
            self.getGammaNormalizationParameter())
        return colormap

    def applyToData(self, data, reference=None):
        """Apply the colormap to the data

        :param Union[numpy.ndarray,~silx.gui.plot.item.ColormapMixIn] data:
            The data to convert or the item for which to apply the colormap.
        :param Union[numpy.ndarray,~silx.gui.plot.item.ColormapMixIn,None] reference:
            The data or item to use as reference to compute autoscale
        """
        if reference is None:
            reference = data
        vmin, vmax = self.getColormapRange(reference)

        if hasattr(data, "getColormappedData"):  # Use item's data
            data = data.getColormappedData()

        return _colormap.cmap(
            data, self._colors, vmin, vmax, self._getNormalizer())

    @staticmethod
    def getSupportedColormaps():
        """Get the supported colormap names as a tuple of str.

        The list should at least contain and start by:

         ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue',
         'viridis', 'magma', 'inferno', 'plasma')

        :rtype: tuple
        """
        colormaps = set()
        if _matplotlib_cm is not None:
            colormaps.update(_matplotlib_cm.cmap_d.keys())
        colormaps.update(_AVAILABLE_LUTS.keys())

        colormaps = tuple(cmap for cmap in sorted(colormaps)
                          if cmap not in _AVAILABLE_LUTS.keys())

        return tuple(_AVAILABLE_LUTS.keys()) + colormaps

    def __str__(self):
        return str(self._toDict())

    def __eq__(self, other):
        """Compare colormap values and not pointers"""
        if other is None:
            return False
        if not isinstance(other, Colormap):
            return False
        if self.getNormalization() != other.getNormalization():
            return False
        if self.getNormalization() == self.GAMMA:
            delta = self.getGammaNormalizationParameter() - other.getGammaNormalizationParameter()
            if abs(delta) > 0.001:
                return False
        return (self.getName() == other.getName() and
                self.getAutoscaleMode() == other.getAutoscaleMode() and
                self.getVMin() == other.getVMin() and
                self.getVMax() == other.getVMax() and
                numpy.array_equal(self.getColormapLUT(), other.getColormapLUT())
                )

    _SERIAL_VERSION = 2

    def restoreState(self, byteArray):
        """
        Read the colormap state from a QByteArray.

        :param qt.QByteArray byteArray: Stream containing the state
        :return: True if the restoration sussseed
        :rtype: bool
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        stream = qt.QDataStream(byteArray, qt.QIODevice.ReadOnly)

        className = stream.readQString()
        if className != self.__class__.__name__:
            _logger.warning("Classname mismatch. Found %s." % className)
            return False

        version = stream.readUInt32()
        if version not in (1, self._SERIAL_VERSION):
            _logger.warning("Serial version mismatch. Found %d." % version)
            return False

        name = stream.readQString()
        isNull = stream.readBool()
        if not isNull:
            vmin = stream.readQVariant()
        else:
            vmin = None
        isNull = stream.readBool()
        if not isNull:
            vmax = stream.readQVariant()
        else:
            vmax = None

        normalization = stream.readQString()
        if normalization == Colormap.GAMMA:
            gamma = stream.readFloat()
        else:
            gamma = None

        if version == 1:
            autoscaleMode = Colormap.MINMAX
        else:
            autoscaleMode = stream.readQString()

        # emit change event only once
        old = self.blockSignals(True)
        try:
            self.setName(name)
            self.setNormalization(normalization)
            self.setAutoscaleMode(autoscaleMode)
            self.setVRange(vmin, vmax)
            if gamma is not None:
                self.setGammaNormalizationParameter(gamma)
        finally:
            self.blockSignals(old)
        self.sigChanged.emit()
        return True

    def saveState(self):
        """
        Save state of the colomap into a QDataStream.

        :rtype: qt.QByteArray
        """
        data = qt.QByteArray()
        stream = qt.QDataStream(data, qt.QIODevice.WriteOnly)

        stream.writeQString(self.__class__.__name__)
        stream.writeUInt32(self._SERIAL_VERSION)
        stream.writeQString(self.getName())
        stream.writeBool(self.getVMin() is None)
        if self.getVMin() is not None:
            stream.writeQVariant(self.getVMin())
        stream.writeBool(self.getVMax() is None)
        if self.getVMax() is not None:
            stream.writeQVariant(self.getVMax())
        stream.writeQString(self.getNormalization())
        if self.getNormalization() == Colormap.GAMMA:
            stream.writeFloat(self.getGammaNormalizationParameter())
        stream.writeQString(self.getAutoscaleMode())
        return data


_PREFERRED_COLORMAPS = None
"""
Tuple of preferred colormap names accessed with :meth:`preferredColormaps`.
"""


def preferredColormaps():
    """Returns the name of the preferred colormaps.

    This list is used by widgets allowing to change the colormap
    like the :class:`ColormapDialog` as a subset of colormap choices.

    :rtype: tuple of str
    """
    global _PREFERRED_COLORMAPS
    if _PREFERRED_COLORMAPS is None:
        # Initialize preferred colormaps
        default_preferred = []
        for name, info in _AVAILABLE_LUTS.items():
            if (info.preferred and
                    (info.source != 'matplotlib' or _matplotlib_cm is not None)):
                default_preferred.append(name)
        setPreferredColormaps(default_preferred)
    return tuple(_PREFERRED_COLORMAPS)


def setPreferredColormaps(colormaps):
    """Set the list of preferred colormap names.

    Warning: If a colormap name is not available
    it will be removed from the list.

    :param colormaps: Not empty list of colormap names
    :type colormaps: iterable of str
    :raise ValueError: if the list of available preferred colormaps is empty.
    """
    supportedColormaps = Colormap.getSupportedColormaps()
    colormaps = [cmap for cmap in colormaps if cmap in supportedColormaps]
    if len(colormaps) == 0:
        raise ValueError("Cannot set preferred colormaps to an empty list")

    global _PREFERRED_COLORMAPS
    _PREFERRED_COLORMAPS = colormaps


def registerLUT(name, colors, cursor_color='black', preferred=True):
    """Register a custom LUT to be used with `Colormap` objects.

    It can override existing LUT names.

    :param str name: Name of the LUT as defined to configure colormaps
    :param numpy.ndarray colors: The custom LUT to register.
            Nx3 or Nx4 numpy array of RGB(A) colors,
            either uint8 or float in [0, 1].
    :param bool preferred: If true, this LUT will be displayed as part of the
        preferred colormaps in dialogs.
    :param str cursor_color: Color used to display overlay over images using
        colormap with this LUT.
    """
    description = _LUT_DESCRIPTION('user', cursor_color, preferred=preferred)
    colors = _arrayToRgba8888(colors)
    _AVAILABLE_LUTS[name] = description

    if preferred:
        # Invalidate the preferred cache
        global _PREFERRED_COLORMAPS
        if _PREFERRED_COLORMAPS is not None:
            if name not in _PREFERRED_COLORMAPS:
                _PREFERRED_COLORMAPS.append(name)
        else:
            # The cache is not yet loaded, it's fine
            pass

    # Register the cache as the LUT was already loaded
    _COLORMAP_CACHE[name] = colors
