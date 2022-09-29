# /*##########################################################################
#
# Copyright (c) 2015-2021 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent", "H.Payno"]
__license__ = "MIT"
__date__ = "29/01/2019"

import numpy
import logging

from silx.gui import qt
from silx.gui.utils import blockSignals
from silx.math import colormap as _colormap
from silx.utils.exceptions import NotEditableError
from silx.utils import deprecation


_logger = logging.getLogger(__name__)

try:
    import silx.gui.utils.matplotlib  # noqa  Initalize matplotlib
    from matplotlib import cm as _matplotlib_cm
    from matplotlib.pyplot import colormaps as _matplotlib_colormaps
except ImportError:
    _logger.info("matplotlib not available, only embedded colormaps available")
    _matplotlib_cm = None
    _matplotlib_colormaps = None


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

    if hasattr(color, 'getRgb'):  # QColor support
        color = color.getRgb()

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
    return _colormap.get_colormap_cursor_color(colormapName)


# Colormap loader

def _registerColormapFromMatplotlib(name, cursor_color='black', preferred=False):
    colormap = _matplotlib_cm.get_cmap(name)
    lut = colormap(numpy.linspace(0, 1, colormap.N, endpoint=True))
    colors = _colormap.array_to_rgba8888(lut)
    registerLUT(name, colors, cursor_color, preferred)


def _getColormap(name):
    """Returns the color LUT corresponding to a colormap name
    :param str name: Name of the colormap to load
    :returns: Corresponding table of colors
    :rtype: numpy.ndarray
    :raise ValueError: If no colormap corresponds to name
    """
    name = str(name)
    try:
        return _colormap.get_colormap_lut(name)
    except ValueError:
        # Colormap is not available, try to load it from matplotlib
        _registerColormapFromMatplotlib(name, 'black', False)
    return _colormap.get_colormap_lut(name)


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
        LINEAR: _colormap.LinearNormalization(),
        LOGARITHM: _colormap.LogarithmicNormalization(),
        SQRT: _colormap.SqrtNormalization(),
        ARCSINH: _colormap.ArcsinhNormalization(),
        }
    """Normalizations without parameters"""

    NORMALIZATIONS = LINEAR, LOGARITHM, SQRT, GAMMA, ARCSINH
    """Tuple of managed normalizations"""

    MINMAX = 'minmax'
    """constant for autoscale using min/max data range"""

    STDDEV3 = 'stddev3'
    """constant for autoscale using mean +/- 3*std(data)
    with a clamp on min/max of the data"""

    AUTOSCALE_MODES = (MINMAX, STDDEV3)
    """Tuple of managed auto scale algorithms"""

    sigChanged = qt.Signal()
    """Signal emitted when the colormap has changed."""

    _DEFAULT_NAN_COLOR = 255, 255, 255, 0

    def __init__(self, name=None, colors=None, normalization=LINEAR, vmin=None, vmax=None, autoscaleMode=MINMAX):
        qt.QObject.__init__(self)
        self._editable = True
        self.__gamma = 2.0
        # Default NaN color: fully transparent white
        self.__nanColor = numpy.array(self._DEFAULT_NAN_COLOR, dtype=numpy.uint8)

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
        self.__warnBadVmin = True
        self.__warnBadVmax = True

    def setFromColormap(self, other):
        """Set this colormap using information from the `other` colormap.

        :param ~silx.gui.colors.Colormap other: Colormap to use as reference.
        """
        if not self.isEditable():
            raise NotEditableError('Colormap is not editable')
        if self == other:
            return
        with blockSignals(self):
            name = other.getName()
            if name is not None:
                self.setName(name)
            else:
                self.setColormapLUT(other.getColormapLUT())
            self.setNaNColor(other.getNaNColor())
            self.setNormalization(other.getNormalization())
            self.setGammaNormalizationParameter(
                other.getGammaNormalizationParameter())
            self.setAutoscaleMode(other.getAutoscaleMode())
            self.setVRange(*other.getVRange())
            self.setEditable(other.isEditable())
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
                numpy.arange(nbColors, dtype=numpy.int32))
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
        self._colors = _colormap.array_to_rgba8888(colors)
        self._name = None
        self.sigChanged.emit()

    def getNaNColor(self):
        """Returns the color to use for Not-A-Number floating point value.

        :rtype: QColor
        """
        return qt.QColor(*self.__nanColor)

    def setNaNColor(self, color):
        """Set the color to use for Not-A-Number floating point value.

        :param color: RGB(A) color to use for NaN values
        :type color: QColor, str, tuple of uint8 or float in [0., 1.]
        """
        color = (numpy.array(rgba(color)) * 255).astype(numpy.uint8)
        if not numpy.array_equal(self.__nanColor, color):
            self.__nanColor = color
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
        norm = str(norm)
        if norm != self._normalization:
            self._normalization = norm
            self.__warnBadVmin = True
            self.__warnBadVmax = True
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

        if vmin != self._vmin:
            self._vmin = vmin
            self.__warnBadVmin = True
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

        if vmax != self._vmax:
            self._vmax = vmax
            self.__warnBadVmax = True
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
            return _colormap.GammaNormalization(self.getGammaNormalizationParameter())
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
        if vmin is not None and not normalizer.is_valid(vmin):
            if self.__warnBadVmin:
                self.__warnBadVmin = False
                _logger.info(
                    'Invalid vmin, switching to autoscale for lower bound')
            vmin = None
        if vmax is not None and not normalizer.is_valid(vmax):
            if self.__warnBadVmax:
                self.__warnBadVmax = False
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

        if vmin != self._vmin:
            self.__warnBadVmin = True
        self._vmin = vmin
        if vmax != self._vmax:
            self.__warnBadVmax = True
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

        self.__warnBadVmin = True
        self.__warnBadVmax = True
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
        colormap.setNaNColor(self.getNaNColor())
        colormap.setGammaNormalizationParameter(
            self.getGammaNormalizationParameter())
        colormap.setEditable(self.isEditable())
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
            data = data.getColormappedData(copy=False)

        return _colormap.cmap(
            data,
            self._colors,
            vmin,
            vmax,
            self._getNormalizer(),
            self.__nanColor)

    @staticmethod
    def getSupportedColormaps():
        """Get the supported colormap names as a tuple of str.

        The list should at least contain and start by:

         ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue',
         'viridis', 'magma', 'inferno', 'plasma')

        :rtype: tuple
        """
        registered_colormaps = _colormap.get_registered_colormaps()
        colormaps = set(registered_colormaps)
        if _matplotlib_colormaps is not None:
            colormaps.update(_matplotlib_colormaps())

        # Put registered_colormaps first
        colormaps = tuple(cmap for cmap in sorted(colormaps)
                          if cmap not in registered_colormaps)
        return registered_colormaps + colormaps

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

    _SERIAL_VERSION = 3

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
        if version not in numpy.arange(1, self._SERIAL_VERSION+1):
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

        if version <= 2:
            nanColor = self._DEFAULT_NAN_COLOR
        else:
            nanColor = stream.readInt32(), stream.readInt32(), stream.readInt32(), stream.readInt32()

        # emit change event only once
        old = self.blockSignals(True)
        try:
            self.setName(name)
            self.setNormalization(normalization)
            self.setAutoscaleMode(autoscaleMode)
            self.setVRange(vmin, vmax)
            if gamma is not None:
                self.setGammaNormalizationParameter(gamma)
            self.setNaNColor(nanColor)
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
        nanColor = self.getNaNColor()
        stream.writeInt32(nanColor.red())
        stream.writeInt32(nanColor.green())
        stream.writeInt32(nanColor.blue())
        stream.writeInt32(nanColor.alpha())

        return data


_PREFERRED_COLORMAPS = None
"""
Tuple of preferred colormap names accessed with :meth:`preferredColormaps`.
"""

_DEFAULT_PREFERRED_COLORMAPS = (
    'gray', 'reversed gray', 'red', 'green', 'blue',
    'viridis', 'cividis', 'magma', 'inferno', 'plasma',
    'temperature',
    'jet', 'hsv'
)


def preferredColormaps():
    """Returns the name of the preferred colormaps.

    This list is used by widgets allowing to change the colormap
    like the :class:`ColormapDialog` as a subset of colormap choices.

    :rtype: tuple of str
    """
    global _PREFERRED_COLORMAPS
    if _PREFERRED_COLORMAPS is None:
        # Initialize preferred colormaps
        setPreferredColormaps(_DEFAULT_PREFERRED_COLORMAPS)
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
    _colormap.register_colormap(name, colors, cursor_color)

    if preferred:
        # Invalidate the preferred cache
        global _PREFERRED_COLORMAPS
        if _PREFERRED_COLORMAPS is not None:
            if name not in _PREFERRED_COLORMAPS:
                _PREFERRED_COLORMAPS.append(name)
        else:
            # The cache is not yet loaded, it's fine
            pass


# Load some colormaps from matplotlib by default
if _matplotlib_cm is not None:
    _registerColormapFromMatplotlib('jet', cursor_color='pink', preferred=True)
    _registerColormapFromMatplotlib('hsv', cursor_color='black', preferred=True)
