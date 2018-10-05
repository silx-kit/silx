# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
__date__ = "05/10/2018"

from silx.gui import qt
import copy as copy_mdl
import numpy
import logging
from silx.math.combo import min_max
from silx.math.colormap import cmap as _cmap
from silx.utils.exceptions import NotEditableError

_logger = logging.getLogger(__file__)


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


# FIXME: It could be nice to expose a functional API instead of that attribute
COLORDICT = _COLORDICT


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


DEFAULT_COLORMAPS = (
    'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue')
"""Tuple of supported colormap names."""

DEFAULT_MIN_LIN = 0
"""Default min value if in linear normalization"""
DEFAULT_MAX_LIN = 1
"""Default max value if in linear normalization"""
DEFAULT_MIN_LOG = 1
"""Default min value if in log normalization"""
DEFAULT_MAX_LOG = 10
"""Default max value if in log normalization"""


class Colormap(qt.QObject):
    """Description of a colormap

    :param str name: Name of the colormap
    :param tuple colors: optional, custom colormap.
            Nx3 or Nx4 numpy array of RGB(A) colors,
            either uint8 or float in [0, 1].
            If 'name' is None, then this array is used as the colormap.
    :param str normalization: Normalization: 'linear' (default) or 'log'
    :param float vmin:
        Lower bound of the colormap or None for autoscale (default)
    :param float vmax:
        Upper bounds of the colormap or None for autoscale (default)
    """

    LINEAR = 'linear'
    """constant for linear normalization"""

    LOGARITHM = 'log'
    """constant for logarithmic normalization"""

    NORMALIZATIONS = (LINEAR, LOGARITHM)
    """Tuple of managed normalizations"""

    sigChanged = qt.Signal()
    """Signal emitted when the colormap has changed."""

    def __init__(self, name='gray', colors=None, normalization=LINEAR, vmin=None, vmax=None):
        qt.QObject.__init__(self)
        assert normalization in Colormap.NORMALIZATIONS
        assert not (name is None and colors is None)
        if normalization is Colormap.LOGARITHM:
            if (vmin is not None and vmin < 0) or (vmax is not None and vmax < 0):
                m = "Unsuported vmin (%s) and/or vmax (%s) given for a log scale."
                m += ' Autoscale will be performed.'
                m = m % (vmin, vmax)
                _logger.warning(m)
                vmin = None
                vmax = None

        self._name = str(name) if name is not None else None
        self._setColors(colors)
        self._normalization = str(normalization)
        self._vmin = float(vmin) if vmin is not None else None
        self._vmax = float(vmax) if vmax is not None else None
        self._editable = True

    def isAutoscale(self):
        """Return True if both min and max are in autoscale mode"""
        return self._vmin is None and self._vmax is None

    def getName(self):
        """Return the name of the colormap
        :rtype: str
        """
        return self._name

    @staticmethod
    def _convertColorsFromFloatToUint8(colors):
        """Convert colors from float in [0, 1] to uint8

        :param numpy.ndarray colors: Array of float colors to convert
        :return: colors as uint8
        :rtype: numpy.ndarray
        """
        # Each bin is [N, N+1[ except the last one: [255, 256]
        return numpy.clip(
            colors.astype(numpy.float64) * 256, 0., 255.).astype(numpy.uint8)

    def _setColors(self, colors):
        if colors is None:
            self._colors = None
        else:
            colors = numpy.array(colors, copy=False)
            if colors.shape == ():
                raise TypeError("An array is expected for 'colors' argument. '%s' was found." % type(colors))
            colors.shape = -1, colors.shape[-1]
            if colors.dtype.kind == 'f':
                colors = self._convertColorsFromFloatToUint8(colors)

            # Makes sure it is RGBA8888
            self._colors = numpy.zeros((len(colors), 4), dtype=numpy.uint8)
            self._colors[:, 3] = 255  # Alpha channel
            self._colors[:, :colors.shape[1]] = colors  # Copy colors

    def getNColors(self, nbColors=None):
        """Returns N colors computed by sampling the colormap regularly.

        :param nbColors:
            The number of colors in the returned array or None for the default value.
            The default value is 256 for colormap with a name (see :meth:`setName`) and
            it is the size of the LUT for colormap defined with :meth:`setColormapLUT`.
        :type nbColors: int or None
        :return: 2D array of uint8 of shape (nbColors, 4)
        :rtype: numpy.ndarray
        """
        # Handle default value for nbColors
        if nbColors is None:
            lut = self.getColormapLUT()
            if lut is not None:  # In this case uses LUT length
                nbColors = len(lut)
            else:  # Default to 256
                nbColors = 256

        nbColors = int(nbColors)

        colormap = self.copy()
        colormap.setNormalization(Colormap.LINEAR)
        colormap.setVRange(vmin=None, vmax=None)
        colors = colormap.applyToData(
            numpy.arange(nbColors, dtype=numpy.int))
        return colors

    def setName(self, name):
        """Set the name of the colormap to use.

        :param str name: The name of the colormap.
            At least the following names are supported: 'gray',
            'reversed gray', 'temperature', 'red', 'green', 'blue', 'jet',
            'viridis', 'magma', 'inferno', 'plasma'.
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        assert name in self.getSupportedColormaps()
        self._name = str(name)
        self._colors = None
        self.sigChanged.emit()

    def getColormapLUT(self):
        """Return the list of colors for the colormap or None if not set

        :return: the list of colors for the colormap or None if not set
        :rtype: numpy.ndarray or None
        """
        if self._colors is None:
            return None
        else:
            return numpy.array(self._colors, copy=True)

    def setColormapLUT(self, colors):
        """Set the colors of the colormap.

        :param numpy.ndarray colors: the colors of the LUT.
           If float, it is converted from [0, 1] to uint8 range.
           Otherwise it is casted to uint8.

        .. warning: this will set the value of name to None
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        self._setColors(colors)
        if len(colors) is 0:
            self._colors = None

        self._name = None
        self.sigChanged.emit()

    def getNormalization(self):
        """Return the normalization of the colormap ('log' or 'linear')

        :return: the normalization of the colormap
        :rtype: str
        """
        return self._normalization

    def setNormalization(self, norm):
        """Set the norm ('log', 'linear')

        :param str norm: the norm to set
        """
        if self.isEditable() is False:
            raise NotEditableError('Colormap is not editable')
        self._normalization = str(norm)
        self.sigChanged.emit()

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

    def getColormapRange(self, data=None):
        """Return (vmin, vmax)

        :return: the tuple vmin, vmax fitting vmin, vmax, normalization and
            data if any given
        :rtype: tuple
        """
        vmin = self._vmin
        vmax = self._vmax
        assert vmin is None or vmax is None or vmin <= vmax  # TODO handle this in setters

        if self.getNormalization() == self.LOGARITHM:
            # Handle negative bounds as autoscale
            if vmin is not None and (vmin is not None and vmin <= 0.):
                mess = 'negative vmin, moving to autoscale for lower bound'
                _logger.warning(mess)
                vmin = None
            if vmax is not None and (vmax is not None and vmax <= 0.):
                mess = 'negative vmax, moving to autoscale for upper bound'
                _logger.warning(mess)
                vmax = None

        if vmin is None or vmax is None:  # Handle autoscale
            # Get min/max from data
            if data is not None:
                data = numpy.array(data, copy=False)
                if data.size == 0:  # Fallback an array but no data
                    min_, max_ = self._getDefaultMin(), self._getDefaultMax()
                else:
                    if self.getNormalization() == self.LOGARITHM:
                        result = min_max(data, min_positive=True, finite=True)
                        min_ = result.min_positive  # >0 or None
                        max_ = result.maximum  # can be <= 0
                    else:
                        min_, max_ = min_max(data, min_positive=False, finite=True)

                    # Handle fallback
                    if min_ is None or not numpy.isfinite(min_):
                        min_ = self._getDefaultMin()
                    if max_ is None or not numpy.isfinite(max_):
                        max_ = self._getDefaultMax()
            else:  # Fallback if no data is provided
                min_, max_ = self._getDefaultMin(), self._getDefaultMax()

            if vmin is None:  # Set vmin respecting provided vmax
                vmin = min_ if vmax is None else min(min_, vmax)

            if vmax is None:
                vmax = max(max_, vmin)  # Handle max_ <= 0 for log scale

        return vmin, vmax

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
            'colors': copy_mdl.copy(self._colors),
            'vmin': self._vmin,
            'vmax': self._vmax,
            'autoscale': self.isAutoscale(),
            'normalization': self._normalization
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
            err = 'Given normalization is not recoginized (%s)' % normalization
            raise ValueError(err)

        # If autoscale, then set boundaries to None
        if dic.get('autoscale', False):
            vmin, vmax = None, None

        self._name = name
        self._colors = colors
        self._vmin = vmin
        self._vmax = vmax
        self._autoscale = True if (vmin is None and vmax is None) else False
        self._normalization = normalization

        self.sigChanged.emit()

    @staticmethod
    def _fromDict(dic):
        colormap = Colormap(name="")
        colormap._setFromDict(dic)
        return colormap

    def copy(self):
        """Return a copy of the Colormap.

        :rtype: silx.gui.colors.Colormap
        """
        return Colormap(name=self._name,
                        colors=copy_mdl.copy(self._colors),
                        vmin=self._vmin,
                        vmax=self._vmax,
                        normalization=self._normalization)

    def applyToData(self, data):
        """Apply the colormap to the data

        :param numpy.ndarray data: The data to convert.
        """
        name = self.getName()
        if name is not None:  # Get colormap definition from matplotlib
            # FIXME: If possible remove dependency to the plot
            from .plot.matplotlib import Colormap as MPLColormap
            mplColormap = MPLColormap.getColormap(name)
            colors = mplColormap(numpy.linspace(0, 1, 256, endpoint=True))
            colors = self._convertColorsFromFloatToUint8(colors)

        else:  # Use user defined LUT
            colors = self.getColormapLUT()

        vmin, vmax = self.getColormapRange(data)
        normalization = self.getNormalization()

        return _cmap(data, colors, vmin, vmax, normalization)

    @staticmethod
    def getSupportedColormaps():
        """Get the supported colormap names as a tuple of str.

        The list should at least contain and start by:
        ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue')
        :rtype: tuple
        """
        # FIXME: If possible remove dependency to the plot
        from .plot.matplotlib import Colormap as MPLColormap
        maps = MPLColormap.getSupportedColormaps()
        return DEFAULT_COLORMAPS + maps

    def __str__(self):
        return str(self._toDict())

    def _getDefaultMin(self):
        return DEFAULT_MIN_LIN if self._normalization == Colormap.LINEAR else DEFAULT_MIN_LOG

    def _getDefaultMax(self):
        return DEFAULT_MAX_LIN if self._normalization == Colormap.LINEAR else DEFAULT_MAX_LOG

    def __eq__(self, other):
        """Compare colormap values and not pointers"""
        return (self.getName() == other.getName() and
                self.getNormalization() == other.getNormalization() and
                self.getVMin() == other.getVMin() and
                self.getVMax() == other.getVMax() and
                numpy.array_equal(self.getColormapLUT(), other.getColormapLUT())
                )

    _SERIAL_VERSION = 1

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
        if version != self._SERIAL_VERSION:
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

        # emit change event only once
        old = self.blockSignals(True)
        try:
            self.setName(name)
            self.setNormalization(normalization)
            self.setVRange(vmin, vmax)
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
        _PREFERRED_COLORMAPS = DEFAULT_COLORMAPS
        # Initialize preferred colormaps
        setPreferredColormaps(('gray', 'reversed gray',
                               'temperature', 'red', 'green', 'blue', 'jet',
                               'viridis', 'magma', 'inferno', 'plasma',
                               'hsv'))
    return _PREFERRED_COLORMAPS


def setPreferredColormaps(colormaps):
    """Set the list of preferred colormap names.

    Warning: If a colormap name is not available
    it will be removed from the list.

    :param colormaps: Not empty list of colormap names
    :type colormaps: iterable of str
    :raise ValueError: if the list of available preferred colormaps is empty.
    """
    supportedColormaps = Colormap.getSupportedColormaps()
    colormaps = tuple(
        cmap for cmap in colormaps if cmap in supportedColormaps)
    if len(colormaps) == 0:
        raise ValueError("Cannot set preferred colormaps to an empty list")

    global _PREFERRED_COLORMAPS
    _PREFERRED_COLORMAPS = colormaps
