# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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
"""This module provides mix-in classes for :class:`Item3D`.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import collections
import numpy

from silx.math.combo import min_max

from .....utils.enum import Enum as _Enum
from ...plot.items.core import ItemMixInBase
from ...plot.items.core import ColormapMixIn as _ColormapMixIn
from ...plot.items.core import SymbolMixIn as _SymbolMixIn
from ...colors import rgba

from ..scene import primitives
from .core import Item3DChangedType, ItemChangedType


class InterpolationMixIn(ItemMixInBase):
    """Mix-in class for image interpolation mode

    :param str mode: 'linear' (default) or 'nearest'
    :param primitive:
        scene object for which to sync interpolation mode.
        This object MUST have an interpolation property that is updated.
    """

    NEAREST_INTERPOLATION = 'nearest'
    """Nearest interpolation mode (see :meth:`setInterpolation`)"""

    LINEAR_INTERPOLATION = 'linear'
    """Linear interpolation mode (see :meth:`setInterpolation`)"""

    INTERPOLATION_MODES = NEAREST_INTERPOLATION, LINEAR_INTERPOLATION
    """Supported interpolation modes for :meth:`setInterpolation`"""

    def __init__(self, mode=NEAREST_INTERPOLATION, primitive=None):
        self.__primitive = primitive
        self._syncPrimitiveInterpolation()

        self.__interpolationMode = None
        self.setInterpolation(mode)

    def _setPrimitive(self, primitive):

        """Set the scene object for which to sync interpolation"""
        self.__primitive = primitive
        self._syncPrimitiveInterpolation()

    def _syncPrimitiveInterpolation(self):
        """Synchronize scene object's interpolation"""
        if self.__primitive is not None:
            self.__primitive.interpolation = self.getInterpolation()

    def setInterpolation(self, mode):
        """Set image interpolation mode

        :param str mode: 'nearest' or 'linear'
        """
        mode = str(mode)
        assert mode in self.INTERPOLATION_MODES
        if mode != self.__interpolationMode:
            self.__interpolationMode = mode
            self._syncPrimitiveInterpolation()
            self._updated(Item3DChangedType.INTERPOLATION)

    def getInterpolation(self):
        """Returns the interpolation mode set by :meth:`setInterpolation`

        :rtype: str
        """
        return self.__interpolationMode


class ColormapMixIn(_ColormapMixIn):
    """Mix-in class for Item3D object with a colormap

    :param sceneColormap:
        The plot3d scene colormap to sync with Colormap object.
    """

    def __init__(self, sceneColormap=None):
        super(ColormapMixIn, self).__init__()

        self._dataRange = None
        self.__sceneColormap = sceneColormap
        self._syncSceneColormap()

    def _colormapChanged(self):
        """Handle colormap updates"""
        self._syncSceneColormap()
        super(ColormapMixIn, self)._colormapChanged()

    def _setRangeFromData(self, data=None):
        """Compute the data range the colormap should use from provided data.

        :param data: Data set from which to compute the range or None
        """
        if data is None or data.size == 0:
            dataRange = None
        else:
            dataRange = min_max(data, min_positive=True, finite=True)
            if dataRange.minimum is None:  # Only non-finite data
                dataRange = None

            if dataRange is not None:
                min_positive = dataRange.min_positive
                if min_positive is None:
                    min_positive = float('nan')
                dataRange = dataRange.minimum, min_positive, dataRange.maximum

        self._dataRange = dataRange

        colormap = self.getColormap()
        if None in (colormap.getVMin(), colormap.getVMax()):
            self._colormapChanged()

    def _getDataRange(self):
        """Returns the data range as used in the scene for colormap

        :rtype: Union[List[float],None]
        """
        return self._dataRange

    def _setSceneColormap(self, sceneColormap):
        """Set the scene colormap to sync with Colormap object.

        :param sceneColormap:
            The plot3d scene colormap to sync with Colormap object.
        """
        self.__sceneColormap = sceneColormap
        self._syncSceneColormap()

    def _getSceneColormap(self):
        """Returns scene colormap that is sync"""
        return self.__sceneColormap

    def _syncSceneColormap(self):
        """Synchronizes scene's colormap with Colormap object"""
        if self.__sceneColormap is not None:
            colormap = self.getColormap()

            self.__sceneColormap.colormap = colormap.getNColors()
            self.__sceneColormap.norm = colormap.getNormalization()
            range_ = colormap.getColormapRange(data=self._dataRange)
            self.__sceneColormap.range_ = range_


class ComplexMixIn(ItemMixInBase):
    """Mix-in class for converting complex data to scalar value"""

    class Mode(_Enum):
        """Identify available display mode for complex"""
        ABSOLUTE = 'amplitude'
        PHASE = 'phase'
        REAL = 'real'
        IMAGINARY = 'imaginary'
        AMPLITUDE_PHASE = 'amplitude_phase'
        LOG10_AMPLITUDE_PHASE = 'log10_amplitude_phase'
        SQUARE_AMPLITUDE = 'square_amplitude'

    def __init__(self):
        self._mode = self.Mode.ABSOLUTE

    def getComplexMode(self):
        """Returns the current complex visualization mode.

        :rtype: Mode
        """
        return self._mode

    def setComplexMode(self, mode):
        """Set the complex visualization mode.

        :param Mode mode: The visualization mode in:
            'real', 'imaginary', 'phase', 'amplitude'
        """
        mode = self.Mode.asmember(str(mode))
        assert mode in self.supportedComplexModes()

        if mode != self._mode:
            self._mode = mode
            self._updated(ItemChangedType.VISUALIZATION_MODE)

    def _convertComplexData(self, data, mode=None):
        """Convert complex data to the specific mode.

        :param Union[Mode,None] mode:
            The kind of value to compute.
            If None (the default), the current complex mode is used.
        :return: The converted dataset
        :rtype: Union[numpy.ndarray[float],None]
        """
        if data is None:
            return None

        if mode is None:
            mode = self.getComplexMode()

        if mode is self.Mode.REAL:
            return numpy.real(data)
        elif mode is self.Mode.IMAGINARY:
            return numpy.imag(data)
        elif mode is self.Mode.ABSOLUTE:
            return numpy.absolute(data)
        elif mode is self.Mode.PHASE:
            return numpy.angle(data)
        elif mode is self.Mode.SQUARE_AMPLITUDE:
            return numpy.absolute(data) ** 2
        else:
            raise ValueError('Unsupported conversion mode: %s', str(mode))

    @classmethod
    def supportedComplexModes(cls):
        """Returns the list of supported complex visualization modes.

        See :meth:`setComplexMode`.

        :rtype: List[Mode]
        """
        return (cls.Mode.REAL,
                cls.Mode.IMAGINARY,
                cls.Mode.ABSOLUTE,
                cls.Mode.PHASE,
                cls.Mode.SQUARE_AMPLITUDE)


class SymbolMixIn(_SymbolMixIn):
    """Mix-in class for symbol and symbolSize properties for Item3D"""

    _SUPPORTED_SYMBOLS = collections.OrderedDict((
        ('o', 'Circle'),
        ('d', 'Diamond'),
        ('s', 'Square'),
        ('+', 'Plus'),
        ('x', 'Cross'),
        ('*', 'Star'),
        ('|', 'Vertical Line'),
        ('_', 'Horizontal Line'),
        ('.', 'Point'),
        (',', 'Pixel')))

    def _getSceneSymbol(self):
        """Returns a symbol name and size suitable for scene primitives.

        :return: (symbol, size)
        """
        symbol = self.getSymbol()
        size = self.getSymbolSize()
        if symbol == ',':  # pixel
            return 's', 1.
        elif symbol == '.':  # point
            # Size as in plot OpenGL backend, mimic matplotlib
            return 'o', numpy.ceil(0.5 * size) + 1.
        else:
            return symbol, size


class PlaneMixIn(ItemMixInBase):
    """Mix-in class for plane items (based on PlaneInGroup primitive)"""

    def __init__(self, plane):
        assert isinstance(plane, primitives.PlaneInGroup)
        self.__plane = plane
        self.__plane.alpha = 1.
        self.__plane.addListener(self._planeChanged)
        self.__plane.plane.addListener(self._planePositionChanged)

    def _getPlane(self):
        """Returns plane primitive

        :rtype: primitives.PlaneInGroup
        """
        return self.__plane

    def _planeChanged(self, source, *args, **kwargs):
        """Handle events from the plane primitive"""
        # Sync visibility
        if source.visible != self.isVisible():
            self.setVisible(source.visible)

    def _planePositionChanged(self, source, *args, **kwargs):
        """Handle update of cut plane position and normal"""
        if self.__plane.visible:  # TODO send even if hidden? or send also when showing if moved while hidden
            self._updated(ItemChangedType.POSITION)

    # Plane position

    def moveToCenter(self):
        """Move cut plane to center of data set"""
        self.__plane.moveToCenter()

    def isValid(self):
        """Returns whether the cut plane is defined or not (bool)"""
        return self.__plane.isValid

    def getNormal(self):
        """Returns the normal of the plane (as a unit vector)

        :return: Normal (nx, ny, nz), vector is 0 if no plane is defined
        :rtype: numpy.ndarray
        """
        return self.__plane.plane.normal

    def setNormal(self, normal):
        """Set the normal of the plane

        :param normal: 3-tuple of float: nx, ny, nz
        """
        self.__plane.plane.normal = normal

    def getPoint(self):
        """Returns a point on the plane

        :return: (x, y, z)
        :rtype: numpy.ndarray
        """
        return self.__plane.plane.point

    def setPoint(self, point):
        """Set a point contained in the plane.

        Warning: The plane might not intersect the bounding box of the data.

        :param point: (x, y, z) position
        :type point: 3-tuple of float
        """
        self.__plane.plane.point = point  # TODO rework according to PR #1303

    def getParameters(self):
        """Returns the plane equation parameters: a*x + b*y + c*z + d = 0

        :return: Plane equation parameters: (a, b, c, d)
        :rtype: numpy.ndarray
        """
        return self.__plane.plane.parameters

    def setParameters(self, parameters):
        """Set the plane equation parameters: a*x + b*y + c*z + d = 0

        Warning: The plane might not intersect the bounding box of the data.
        The given parameters will be normalized.

        :param parameters: (a, b, c, d) equation parameters
        """
        self.__plane.plane.parameters = parameters

    # Border stroke

    def _setForegroundColor(self, color):
        """Set the color of the plane border.

        :param color: RGBA color as 4 floats in [0, 1]
        """
        self.__plane.color = rgba(color)
        if hasattr(super(PlaneMixIn, self), '_setForegroundColor'):
            super(PlaneMixIn, self)._setForegroundColor(color)
