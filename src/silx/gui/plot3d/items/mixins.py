# /*##########################################################################
#
# Copyright (c) 2017-2020 European Synchrotron Radiation Facility
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

from ...plot.items.core import ItemMixInBase
from ...plot.items.core import ColormapMixIn as _ColormapMixIn
from ...plot.items.core import SymbolMixIn as _SymbolMixIn
from ...plot.items.core import ComplexMixIn as _ComplexMixIn
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

        self.__sceneColormap = sceneColormap
        self._syncSceneColormap()

    def _colormapChanged(self):
        """Handle colormap updates"""
        self._syncSceneColormap()
        super(ColormapMixIn, self)._colormapChanged()

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
            self.__sceneColormap.gamma = colormap.getGammaNormalizationParameter()
            self.__sceneColormap.range_ = colormap.getColormapRange(self)
            self.__sceneColormap.nancolor = rgba(colormap.getNaNColor())


class ComplexMixIn(_ComplexMixIn):
    __doc__ = _ComplexMixIn.__doc__  # Reuse docstring

    _SUPPORTED_COMPLEX_MODES = (
        _ComplexMixIn.ComplexMode.REAL,
        _ComplexMixIn.ComplexMode.IMAGINARY,
        _ComplexMixIn.ComplexMode.ABSOLUTE,
        _ComplexMixIn.ComplexMode.PHASE,
        _ComplexMixIn.ComplexMode.SQUARE_AMPLITUDE)
    """Overrides supported ComplexMode"""


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
