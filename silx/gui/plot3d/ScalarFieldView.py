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
"""This module provides a window to view a 3D scalar field.

It supports iso-surfaces, a cutting plane and the definition of
a region of interest.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "14/06/2018"

import re
import logging
import time
from collections import deque

import numpy

from silx.gui import qt, icons
from silx.gui.colors import rgba
from silx.gui.colors import Colormap

from silx.math.marchingcubes import MarchingCubes
from silx.math.combo import min_max

from .scene import axes, cutplane, interaction, primitives, transform
from . import scene
from .Plot3DWindow import Plot3DWindow
from .tools import InteractiveModeToolBar

_logger = logging.getLogger(__name__)


class Isosurface(qt.QObject):
    """Class representing an iso-surface

    :param parent: The View widget this iso-surface belongs to
    """

    sigLevelChanged = qt.Signal(float)
    """Signal emitted when the iso-surface level has changed.

    This signal provides the new level value (might be nan).
    """

    sigColorChanged = qt.Signal()
    """Signal emitted when the iso-surface color has changed"""

    sigVisibilityChanged = qt.Signal(bool)
    """Signal emitted when the iso-surface visibility has changed.

    This signal provides the new visibility status.
    """

    def __init__(self, parent):
        super(Isosurface, self).__init__(parent=parent)
        self._level = float('nan')
        self._autoLevelFunction = None
        self._color = rgba('#FFD700FF')
        self._data = None
        self._group = scene.Group()

    def _setData(self, data, copy=True):
        """Set the data set from which to build the iso-surface.

        :param numpy.ndarray data: The 3D dataset or None
        :param bool copy: True to make a copy, False to use as is if possible
        """
        if data is None:
            self._data = None
        else:
            self._data = numpy.array(data, copy=copy, order='C')

        self._update()

    def _get3DPrimitive(self):
        """Return the group containing the mesh of the iso-surface if any"""
        return self._group

    def isVisible(self):
        """Returns True if iso-surface is visible, else False"""
        return self._group.visible

    def setVisible(self, visible):
        """Set the visibility of the iso-surface in the view.

        :param bool visible: True to show the iso-surface, False to hide
        """
        visible = bool(visible)
        if visible != self._group.visible:
            self._group.visible = visible
            self.sigVisibilityChanged.emit(visible)

    def getLevel(self):
        """Return the level of this iso-surface (float)"""
        return self._level

    def setLevel(self, level):
        """Set the value at which to build the iso-surface.

        Setting this value reset auto-level function

        :param float level: The value at which to build the iso-surface
        """
        self._autoLevelFunction = None
        level = float(level)
        if level != self._level:
            self._level = level
            self._update()
            self.sigLevelChanged.emit(level)

    def isAutoLevel(self):
        """True if iso-level is rebuild for each data set."""
        return self.getAutoLevelFunction() is not None

    def getAutoLevelFunction(self):
        """Return the function computing the iso-level (callable or None)"""
        return self._autoLevelFunction

    def setAutoLevelFunction(self, autoLevel):
        """Set the function used to compute the iso-level.

        WARNING: The function might get called in a thread.

        :param callable autoLevel:
            A function taking a 3D numpy.ndarray of float32 and returning
            a float used as iso-level.
            Example: numpy.mean(data) + numpy.std(data)
        """
        assert callable(autoLevel)
        self._autoLevelFunction = autoLevel
        self._update()

    def getColor(self):
        """Return the color of this iso-surface (QColor)"""
        return qt.QColor.fromRgbF(*self._color)

    def setColor(self, color):
        """Set the color of the iso-surface

        :param color: RGBA color of the isosurface
        :type color: QColor, str or array-like of 4 float in [0., 1.]
        """
        color = rgba(color)
        if color != self._color:
            self._color = color
            if len(self._group.children) != 0:
                self._group.children[0].setAttribute('color', self._color)
            self.sigColorChanged.emit()

    def _update(self):
        """Update underlying mesh"""
        self._group.children = []

        if self._data is None:
            if self.isAutoLevel():
                self._level = float('nan')

        else:
            if self.isAutoLevel():
                st = time.time()
                try:
                    level = float(self.getAutoLevelFunction()(self._data))

                except Exception:
                    module = self.getAutoLevelFunction().__module__
                    name = self.getAutoLevelFunction().__name__
                    _logger.error(
                        "Error while executing iso level function %s.%s",
                        module,
                        name,
                        exc_info=True)
                    level = float('nan')

                else:
                    _logger.info(
                        'Computed iso-level in %f s.', time.time() - st)

                if level != self._level:
                    self._level = level
                    self.sigLevelChanged.emit(level)

            if not numpy.isfinite(self._level):
                return

            st = time.time()
            vertices, normals, indices = MarchingCubes(
                self._data,
                isolevel=self._level)
            _logger.info('Computed iso-surface in %f s.', time.time() - st)

            if len(vertices) == 0:
                return
            else:
                mesh = primitives.Mesh3D(vertices,
                                         colors=self._color,
                                         normals=normals,
                                         mode='triangles',
                                         indices=indices)
                self._group.children = [mesh]


class SelectedRegion(object):
    """Selection of a 3D region aligned with the axis.

    :param arrayRange: Range of the selection in the array
        ((zmin, zmax), (ymin, ymax), (xmin, xmax))
    :param dataBBox: Bounding box of the selection in data coordinates
        ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    :param translation: Offset from array to data coordinates (ox, oy, oz)
    :param scale: Scale from array to data coordinates (sx, sy, sz)
    """

    def __init__(self, arrayRange, dataBBox,
                 translation=(0., 0., 0.),
                 scale=(1., 1., 1.)):
        self._arrayRange = numpy.array(arrayRange, copy=True, dtype=numpy.int)
        assert self._arrayRange.shape == (3, 2)
        assert numpy.all(self._arrayRange[:, 1] >= self._arrayRange[:, 0])

        self._dataRange = dataBBox

        self._translation = numpy.array(translation, dtype=numpy.float32)
        assert self._translation.shape == (3,)
        self._scale = numpy.array(scale, dtype=numpy.float32)
        assert self._scale.shape == (3,)

    def getArrayRange(self):
        """Returns array ranges of the selection: 3x2 array of int

        :return: A numpy array with ((zmin, zmax), (ymin, ymax), (xmin, xmax))
        :rtype: numpy.ndarray
        """
        return self._arrayRange.copy()

    def getArraySlices(self):
        """Slices corresponding to the selected range in the array

        :return: A numpy array with (zslice, yslice, zslice)
        :rtype: numpy.ndarray
        """
        return (slice(*self._arrayRange[0]),
                slice(*self._arrayRange[1]),
                slice(*self._arrayRange[2]))

    def getDataRange(self):
        """Range in the data coordinates of the selection: 3x2 array of float

        When the transform matrix is not the identity matrix
        (e.g., rotation, skew) the returned range is the one of the selected
        region bounding box in data coordinates.

        :return: A numpy array with ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        :rtype: numpy.ndarray
        """
        return self._dataRange.copy()

    def getDataScale(self):
        """Scale from array to data coordinates: (sx, sy, sz)

        :return: A numpy array with (sx, sy, sz)
        :rtype: numpy.ndarray
        """
        return self._scale.copy()

    def getDataTranslation(self):
        """Offset from array to data coordinates: (ox, oy, oz)

        :return: A numpy array with (ox, oy, oz)
        :rtype: numpy.ndarray
        """
        return self._translation.copy()


class CutPlane(qt.QObject):
    """Class representing a cutting plane

    :param ~silx.gui.plot3d.ScalarFieldView.ScalarFieldView sfView:
        Widget in which the cut plane is applied.
    """

    sigVisibilityChanged = qt.Signal(bool)
    """Signal emitted when the cut visibility has changed.

    This signal provides the new visibility status.
    """

    sigDataChanged = qt.Signal()
    """Signal emitted when the data this plane is cutting has changed."""

    sigPlaneChanged = qt.Signal()
    """Signal emitted when the cut plane has moved"""

    sigColormapChanged = qt.Signal(Colormap)
    """Signal emitted when the colormap has changed

    This signal provides the new colormap.
    """

    sigTransparencyChanged = qt.Signal()
    """Signal emitted when the transparency of the plane has changed.

    This signal is emitted when calling :meth:`setDisplayValuesBelowMin`.
    """

    sigInterpolationChanged = qt.Signal(str)
    """Signal emitted when the cut plane interpolation has changed

    This signal provides the new interpolation mode.
    """

    def __init__(self, sfView):
        super(CutPlane, self).__init__(parent=sfView)

        self._dataRange = None
        self._visible = False

        self.__syncPlane = True

        # Plane stroke on the outer bounding box
        self._planeStroke = primitives.PlaneInGroup(normal=(0, 1, 0))
        self._planeStroke.visible = self._visible
        self._planeStroke.addListener(self._planeChanged)
        self._planeStroke.plane.addListener(self._planePositionChanged)

        # Plane with texture on the data bounding box
        self._dataPlane = cutplane.CutPlane(normal=(0, 1, 0))
        self._dataPlane.strokeVisible = False
        self._dataPlane.alpha = 1.
        self._dataPlane.visible = self._visible
        self._dataPlane.plane.addListener(self._planePositionChanged)

        self._colormap = Colormap(
            name='gray', normalization='linear', vmin=None, vmax=None)
        self.getColormap().sigChanged.connect(self._colormapChanged)
        self._updateSceneColormap()

        sfView.sigDataChanged.connect(self._sfViewDataChanged)
        sfView.sigTransformChanged.connect(self._sfViewTransformChanged)

    def _get3DPrimitives(self):
        """Return the cut plane scene node."""
        return self._planeStroke, self._dataPlane

    def _keepPlaneInBBox(self):
        """Makes sure the plane intersect its parent bounding box if any"""
        bounds = self._planeStroke.parent.bounds(dataBounds=True)
        if bounds is not None:
            self._planeStroke.plane.point = numpy.clip(
                self._planeStroke.plane.point,
                a_min=bounds[0], a_max=bounds[1])

    @staticmethod
    def _syncPlanes(master, slave):
        """Move slave PlaneInGroup so that it is coplanar with master.

        :param PlaneInGroup master: Reference PlaneInGroup
        :param PlaneInGroup slave: PlaneInGroup to align
        """
        masterToSlave = transform.StaticTransformList([
            slave.objectToSceneTransform.inverse(),
            master.objectToSceneTransform])

        point = masterToSlave.transformPoint(
            master.plane.point)
        normal = masterToSlave.transformNormal(
            master.plane.normal)
        slave.plane.setPlane(point, normal)

    def _sfViewDataChanged(self):
        """Handle data change in the ScalarFieldView this plane belongs to"""
        self._dataPlane.setData(self.sender().getData(), copy=False)

        # Store data range info as 3-tuple of values
        self._dataRange = self.sender().getDataRange()

        self.sigDataChanged.emit()

        # Update colormap range when autoscale
        if self.getColormap().isAutoscale():
            self._updateSceneColormap()

        self._keepPlaneInBBox()

    def _sfViewTransformChanged(self):
        """Handle transform changed in the ScalarFieldView"""
        self._keepPlaneInBBox()
        self._syncPlanes(master=self._planeStroke,
                         slave=self._dataPlane)
        self.sigPlaneChanged.emit()

    def _planeChanged(self, source, *args, **kwargs):
        """Handle events from the plane primitive"""
        # Using _visible for now, until scene as more info in events
        if source.visible != self._visible:
            self._visible = source.visible
            self.sigVisibilityChanged.emit(source.visible)

    def _planePositionChanged(self, source, *args, **kwargs):
        """Handle update of cut plane position and normal"""
        if self.__syncPlane:
            self.__syncPlane = False
            if source is self._planeStroke.plane:
                self._syncPlanes(master=self._planeStroke,
                                 slave=self._dataPlane)
            elif source is self._dataPlane.plane:
                self._syncPlanes(master=self._dataPlane,
                                 slave=self._planeStroke)
            else:
                _logger.error('Received an unknown object %s',
                              str(source))

            if self._planeStroke.visible or self._dataPlane.visible:
                self.sigPlaneChanged.emit()

            self.__syncPlane = True

    # Plane position

    def moveToCenter(self):
        """Move cut plane to center of data set"""
        self._planeStroke.moveToCenter()

    def isValid(self):
        """Returns whether the cut plane is defined or not (bool)"""
        return self._planeStroke.isValid

    def _plane(self, coordinates='array'):
        """Returns the scene plane to set.

        :param str coordinates: The coordinate system to use:
            Either 'scene' or 'array' (default)
        :rtype: Plane
        :raise ValueError: If coordinates is not correct
        """
        if coordinates == 'scene':
            return self._planeStroke.plane
        elif coordinates == 'array':
            return self._dataPlane.plane
        else:
             raise ValueError(
                'Unsupported coordinates: %s' % str(coordinates))

    def getNormal(self, coordinates='array'):
        """Returns the normal of the plane (as a unit vector)

        :param str coordinates: The coordinate system to use:
            Either 'scene' or 'array' (default)
        :return: Normal (nx, ny, nz), vector is 0 if no plane is defined
        :rtype: numpy.ndarray
        :raise ValueError: If coordinates is not correct
        """
        return self._plane(coordinates).normal

    def setNormal(self, normal, coordinates='array'):
        """Set the normal of the plane.

        :param normal: 3-tuple of float: nx, ny, nz
        :param str coordinates: The coordinate system to use:
            Either 'scene' or 'array' (default)
        :raise ValueError: If coordinates is not correct
        """
        self._plane(coordinates).normal = normal

    def getPoint(self, coordinates='array'):
        """Returns a point on the plane.

        :param str coordinates: The coordinate system to use:
            Either 'scene' or 'array' (default)
        :return: (x, y, z)
        :rtype: numpy.ndarray
        :raise ValueError: If coordinates is not correct
        """
        return self._plane(coordinates).point

    def setPoint(self, point, constraint=True, coordinates='array'):
        """Set a point contained in the plane.

        Warning: The plane might not intersect the bounding box of the data.

        :param point: (x, y, z) position
        :type point: 3-tuple of float
        :param bool constraint:
            True (default) to make sure the plane intersect data bounding box,
            False to set the plane without any constraint.
        :raise ValueError: If coordinates is not correc
        """
        self._plane(coordinates).point = point
        if constraint:
            self._keepPlaneInBBox()

    def getParameters(self, coordinates='array'):
        """Returns the plane equation parameters: a*x + b*y + c*z + d = 0

        :param str coordinates: The coordinate system to use:
            Either 'scene' or 'array' (default)
        :return: Plane equation parameters: (a, b, c, d)
        :rtype: numpy.ndarray
        :raise ValueError: If coordinates is not correct
        """
        return self._plane(coordinates).parameters

    def setParameters(self, parameters, constraint=True, coordinates='array'):
        """Set the plane equation parameters: a*x + b*y + c*z + d = 0

        Warning: The plane might not intersect the bounding box of the data.

        :param parameters: (a, b, c, d) plane equation parameters.
        :type parameters: 4-tuple of float
        :param bool constraint:
            True (default) to make sure the plane intersect data bounding box,
            False to set the plane without any constraint.
        :raise ValueError: If coordinates is not correc
        """
        self._plane(coordinates).parameters = parameters
        if constraint:
            self._keepPlaneInBBox()

    # Visibility

    def isVisible(self):
        """Returns True if the plane is visible, False otherwise"""
        return self._planeStroke.visible

    def setVisible(self, visible):
        """Set the visibility of the plane

        :param bool visible: True to make plane visible
        """
        visible = bool(visible)
        self._planeStroke.visible = visible
        self._dataPlane.visible = visible

    # Border stroke

    def getStrokeColor(self):
        """Returns the color of the plane border (QColor)"""
        return qt.QColor.fromRgbF(*self._planeStroke.color)

    def setStrokeColor(self, color):
        """Set the color of the plane border.

        :param color: RGB color: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        self._planeStroke.color = color
        self._dataPlane.color = color

    # Data

    def getImageData(self):
        """Returns the data and information corresponding to the cut plane.

        The returned data is not interpolated,
        it is a slice of the 3D scalar field.

        Image data axes are so that plane normal is towards the point of view.

        :return: An object containing the 2D data slice and information
        """
        return _CutPlaneImage(self)

    # Interpolation

    def getInterpolation(self):
        """Returns the interpolation used to display to cut plane.

        :return: 'nearest' or 'linear'
        :rtype: str
        """
        return self._dataPlane.interpolation

    def setInterpolation(self, interpolation):
        """Set the interpolation used to display to cut plane

        The default interpolation is 'linear'

        :param str interpolation: 'nearest' or 'linear'
        """
        if interpolation != self.getInterpolation():
            self._dataPlane.interpolation = interpolation
            self.sigInterpolationChanged.emit(interpolation)

    # Colormap

    # def getAlpha(self):
    #     """Returns the transparency of the plane as a float in [0., 1.]"""
    #     return self._plane.alpha

    # def setAlpha(self, alpha):
    #     """Set the plane transparency.
    #
    #     :param float alpha: Transparency in [0., 1]
    #     """
    #     self._plane.alpha = alpha

    def getDisplayValuesBelowMin(self):
        """Return whether values <= colormap min are displayed or not.

        :rtype: bool
        """
        return self._dataPlane.colormap.displayValuesBelowMin

    def setDisplayValuesBelowMin(self, display):
        """Set whether to display values <= colormap min.

        :param bool display: True to show values below min,
                             False to discard them
        """
        display = bool(display)
        if display != self.getDisplayValuesBelowMin():
            self._dataPlane.colormap.displayValuesBelowMin = display
            self.sigTransparencyChanged.emit()

    def getColormap(self):
        """Returns the colormap set by :meth:`setColormap`.

        :return: The colormap
        :rtype: ~silx.gui.colors.Colormap
        """
        return self._colormap

    def setColormap(self,
                    name='gray',
                    norm=None,
                    vmin=None,
                    vmax=None):
        """Set the colormap to use.

        By either providing a :class:`Colormap` object or
        its name, normalization and range.

        :param name: Name of the colormap in
            'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue'.
            Or Colormap object.
        :type name: str or ~silx.gui.colors.Colormap
        :param str norm: Colormap mapping: 'linear' or 'log'.
        :param float vmin: The minimum value of the range or None for autoscale
        :param float vmax: The maximum value of the range or None for autoscale
        """
        _logger.debug('setColormap %s %s (%s, %s)',
                      name, str(norm), str(vmin), str(vmax))

        self._colormap.sigChanged.disconnect(self._colormapChanged)

        if isinstance(name, Colormap):  # Use it as it is
            assert (norm, vmin, vmax) == (None, None, None)
            self._colormap = name
        else:
            if norm is None:
                norm = 'linear'
            self._colormap = Colormap(
                name=name, normalization=norm, vmin=vmin, vmax=vmax)

        self._colormap.sigChanged.connect(self._colormapChanged)
        self._colormapChanged()

    def getColormapEffectiveRange(self):
        """Returns the currently used range of the colormap.

        This range is computed from the data set if colormap is in autoscale.
        Range is clipped to positive values when using log scale.

        :return: 2-tuple of float
        """
        return self._dataPlane.colormap.range_

    def _updateSceneColormap(self):
        """Synchronizes scene's colormap with Colormap object"""
        colormap = self.getColormap()
        sceneCMap = self._dataPlane.colormap

        sceneCMap.colormap = colormap.getNColors()

        sceneCMap.norm = colormap.getNormalization()
        range_ = colormap.getColormapRange(data=self._dataRange)
        sceneCMap.range_ = range_

    def _colormapChanged(self):
        """Handle update of Colormap object"""
        self._updateSceneColormap()
        # Forward colormap changed event
        self.sigColormapChanged.emit(self.getColormap())


class _CutPlaneImage(object):
    """Object representing the data sliced by a cut plane

    :param CutPlane cutPlane: The CutPlane from which to generate image info
    """

    def __init__(self, cutPlane):
        # Init attributes with default values
        self._isValid = False
        self._data = numpy.zeros((0, 0), dtype=numpy.float32)
        self._index = 0
        self._xLabel = ''
        self._yLabel = ''
        self._normalLabel = ''
        self._scale = float('nan'), float('nan')
        self._translation = float('nan'), float('nan')
        self._position = float('nan')

        sfView = cutPlane.parent()
        if not sfView or not cutPlane.isValid():
            _logger.info("No plane available")
            return

        data = sfView.getData(copy=False)
        if data is None:
            _logger.info("No data available")
            return

        normal = cutPlane.getNormal(coordinates='array')
        point = cutPlane.getPoint(coordinates='array')

        if numpy.linalg.norm(numpy.cross(normal, (1., 0., 0.))) < 0.0017:
            if not 0 <= point[0] <= data.shape[2]:
                _logger.info("Plane outside dataset")
                return
            index = max(0, min(int(point[0]), data.shape[2] - 1))
            slice_ = data[:, :, index]
            xAxisIndex, yAxisIndex, normalAxisIndex = 1, 2, 0  # y, z, x

        elif numpy.linalg.norm(numpy.cross(normal, (0., 1., 0.))) < 0.0017:
            if not 0 <= point[1] <= data.shape[1]:
                _logger.info("Plane outside dataset")
                return
            index = max(0, min(int(point[1]), data.shape[1] - 1))
            slice_ = numpy.transpose(data[:, index, :])
            xAxisIndex, yAxisIndex, normalAxisIndex = 2, 0, 1  # z, x, y

        elif numpy.linalg.norm(numpy.cross(normal, (0., 0., 1.))) < 0.0017:
            if not 0 <= point[2] <= data.shape[0]:
                _logger.info("Plane outside dataset")
                return
            index = max(0, min(int(point[2]), data.shape[0] - 1))
            slice_ = data[index, :, :]
            xAxisIndex, yAxisIndex, normalAxisIndex = 0, 1, 2  # x, y, z
        else:
            _logger.warning('Unsupported normal: (%f, %f, %f)',
                            normal[0], normal[1], normal[2])
            return

        # Store cut plane image info

        self._isValid = True
        self._data = numpy.array(slice_, copy=True)
        self._index = index

        # Only store extra information when no transform matrix is set
        # Otherwise this information can be meaningless
        if numpy.all(numpy.equal(sfView.getTransformMatrix(),
                                 numpy.identity(3, dtype=numpy.float32))):
            labels = sfView.getAxesLabels()
            self._xLabel = labels[xAxisIndex]
            self._yLabel = labels[yAxisIndex]
            self._normalLabel = labels[normalAxisIndex]

            scale = sfView.getScale()
            self._scale = scale[xAxisIndex], scale[yAxisIndex]

            translation = sfView.getTranslation()
            self._translation = translation[xAxisIndex], translation[yAxisIndex]

            self._position = float(index * scale[normalAxisIndex] +
                                   translation[normalAxisIndex])

    def isValid(self):
        """Returns True if the cut plane image is defined (bool)"""
        return self._isValid

    def getData(self, copy=True):
        """Returns the image data sliced by the cut plane.

        :param bool copy: True to get a copy, False otherwise
        :return: The 2D image data corresponding to the cut plane
        :rtype: numpy.ndarray
        """
        return numpy.array(self._data, copy=copy)

    def getXLabel(self):
        """Returns the label associated to the X axis of the image (str)"""
        return self._xLabel

    def getYLabel(self):
        """Returns the label associated to the Y axis of the image (str)"""
        return self._yLabel

    def getNormalLabel(self):
        """Returns the label of the 3D axis of the plane normal (str)"""
        return self._normalLabel

    def getScale(self):
        """Returns the scales of the data as a 2-tuple of float (sx, sy)"""
        return self._scale

    def getTranslation(self):
        """Returns the offset of the data as a 2-tuple of float (ox, oy)"""
        return self._translation

    def getIndex(self):
        """Returns the index in the data array of the cut plane (int)"""
        return self._index

    def getPosition(self):
        """Returns the cut plane position along the normal axis (flaot)"""
        return self._position


class ScalarFieldView(Plot3DWindow):
    """Widget computing and displaying an iso-surface from a 3D scalar dataset.

    Limitation: Currently, iso-surfaces are generated with higher values
    than the iso-level 'inside' the surface.

    :param parent: See :class:`QMainWindow`
    """

    sigDataChanged = qt.Signal()
    """Signal emitted when the scalar data field has changed."""

    sigTransformChanged = qt.Signal()
    """Signal emitted when the transformation has changed.

    It is emitted by :meth:`setTranslation`, :meth:`setTransformMatrix`,
    :meth:`setScale`.
    """

    sigSelectedRegionChanged = qt.Signal(object)
    """Signal emitted when the selected region has changed.

    This signal provides the new selected region.
    """

    def __init__(self, parent=None):
        super(ScalarFieldView, self).__init__(parent)
        self._colormap = Colormap(
            name='gray', normalization='linear', vmin=None, vmax=None)
        self._selectedRange = None

        # Store iso-surfaces
        self._isosurfaces = []

        # Transformations
        self._dataScale = transform.Scale()
        self._dataTranslate = transform.Translate()
        self._dataTransform = transform.Matrix()   # default to identity

        self._foregroundColor = 1., 1., 1., 1.
        self._highlightColor = 0.7, 0.7, 0., 1.

        self._data = None
        self._dataRange = None

        self._group = primitives.BoundedGroup()
        self._group.transforms = [
            self._dataTranslate, self._dataTransform, self._dataScale]

        self._bbox = axes.LabelledAxes()
        self._bbox.children = [self._group]
        self._outerScale = transform.Scale(1., 1., 1.)
        self._bbox.transforms = [self._outerScale]
        self.getPlot3DWidget().viewport.scene.children.append(self._bbox)

        self._selectionBox = primitives.Box()
        self._selectionBox.strokeSmooth = False
        self._selectionBox.strokeWidth = 1.
        # self._selectionBox.fillColor = 1., 1., 1., 0.3
        # self._selectionBox.fillCulling = 'back'
        self._selectionBox.visible = False
        self._group.children.append(self._selectionBox)

        self._cutPlane = CutPlane(sfView=self)
        self._cutPlane.sigVisibilityChanged.connect(
            self._planeVisibilityChanged)
        planeStroke, dataPlane = self._cutPlane._get3DPrimitives()
        self._bbox.children.append(planeStroke)
        self._group.children.append(dataPlane)

        self._isogroup = primitives.GroupDepthOffset()
        self._isogroup.transforms = [
            # Convert from z, y, x from marching cubes to x, y, z
            transform.Matrix((
                (0., 0., 1., 0.),
                (0., 1., 0., 0.),
                (1., 0., 0., 0.),
                (0., 0., 0., 1.))),
            # Offset to match cutting plane coords
            transform.Translate(0.5, 0.5, 0.5)
        ]
        self._group.children.append(self._isogroup)

        self._initPanPlaneAction()

        self._updateColors()

        self.getPlot3DWidget().viewport.light.shininess = 32

    def saveConfig(self, ioDevice):
        """
        Saves this view state. Only isosurfaces at the moment. Does not save
        the isosurface's function.

        :param qt.QIODevice ioDevice: A `qt.QIODevice`.
        """

        stream = qt.QDataStream(ioDevice)

        stream.writeString('<ScalarFieldView>')

        isoSurfaces = self.getIsosurfaces()

        nIsoSurfaces = len(isoSurfaces)

        # TODO : delegate the serialization to the serialized items
        # isosurfaces
        if nIsoSurfaces:
            tagIn = '<IsoSurfaces nIso={0}>'.format(nIsoSurfaces)
            stream.writeString(tagIn)

            for surface in isoSurfaces:
                color = surface.getColor()
                level = surface.getLevel()
                visible = surface.isVisible()
                stream << color
                stream.writeDouble(level)
                stream.writeBool(visible)

            stream.writeString('</IsoSurfaces>')

        stream.writeString('<Style>')
        background = self.getBackgroundColor()
        foreground = self.getForegroundColor()
        highlight = self.getHighlightColor()
        stream << background << foreground << highlight
        stream.writeString('</Style>')

        stream.writeString('</ScalarFieldView>')

    def loadConfig(self, ioDevice):
        """
        Loads this view state.
        See ScalarFieldView.saveView to know what is supported at the moment.

        :param qt.QIODevice ioDevice: A `qt.QIODevice`.
        """

        tagStack = deque()

        tagInRegex = re.compile('<(?P<itemId>[^ /]*) *'
                                '(?P<args>.*)>')

        tagOutRegex = re.compile('</(?P<itemId>[^ ]*)>')

        tagRootInRegex = re.compile('<ScalarFieldView>')

        isoSurfaceArgsRegex = re.compile('nIso=(?P<nIso>[0-9]*)')

        stream = qt.QDataStream(ioDevice)

        tag = stream.readString()
        tagMatch = tagRootInRegex.match(tag)

        if tagMatch is None:
            # TODO : explicit error
            raise ValueError('Unknown data.')

        itemId = 'ScalarFieldView'

        tagStack.append(itemId)

        while True:

            tag = stream.readString()

            tagMatch = tagOutRegex.match(tag)
            if tagMatch:
                closeId = tagMatch.groupdict()['itemId']
                if closeId != itemId:
                    # TODO : explicit error
                    raise ValueError('Unexpected closing tag {0} '
                                     '(expected {1})'
                                     ''.format(closeId, itemId))

                if itemId == 'ScalarFieldView':
                    # reached end
                    break
                else:
                    itemId = tagStack.pop()
                    # fetching next tag
                    continue

            tagMatch = tagInRegex.match(tag)

            if tagMatch is None:
                # TODO : explicit error
                raise ValueError('Unknown data.')

            tagStack.append(itemId)

            matchDict = tagMatch.groupdict()

            itemId = matchDict['itemId']

            # TODO : delegate the deserialization to the serialized items
            if itemId == 'IsoSurfaces':
                argsMatch = isoSurfaceArgsRegex.match(matchDict['args'])
                if not argsMatch:
                    # TODO : explicit error
                    raise ValueError('Failed to parse args "{0}".'
                                     ''.format(matchDict['args']))
                argsDict = argsMatch.groupdict()
                nIso = int(argsDict['nIso'])
                if nIso:
                    for surface in self.getIsosurfaces():
                        self.removeIsosurface(surface)
                    for isoIdx in range(nIso):
                        color = qt.QColor()
                        stream >> color
                        level = stream.readDouble()
                        visible = stream.readBool()
                        surface = self.addIsosurface(level, color=color)
                        surface.setVisible(visible)
            elif itemId == 'Style':
                background = qt.QColor()
                foreground = qt.QColor()
                highlight = qt.QColor()
                stream >> background >> foreground >> highlight
                self.setBackgroundColor(background)
                self.setForegroundColor(foreground)
                self.setHighlightColor(highlight)
            else:
                raise ValueError('Unknown entry tag {0}.'
                                 ''.format(itemId))

    def _initPanPlaneAction(self):
        """Creates and init the pan plane action"""
        self._panPlaneAction = qt.QAction(self)
        self._panPlaneAction.setIcon(icons.getQIcon('3d-plane-pan'))
        self._panPlaneAction.setText('Pan plane')
        self._panPlaneAction.setCheckable(True)
        self._panPlaneAction.setToolTip(
            'Pan the cutting plane. Press <b>Ctrl</b> to rotate the scene.')
        self._panPlaneAction.setEnabled(False)

        self._panPlaneAction.triggered[bool].connect(self._planeActionTriggered)
        self.getPlot3DWidget().sigInteractiveModeChanged.connect(
            self._interactiveModeChanged)

        toolbar = self.findChild(InteractiveModeToolBar)
        if toolbar is not None:
            toolbar.addAction(self._panPlaneAction)

    def _planeActionTriggered(self, checked=False):
        self._panPlaneAction.setChecked(True)
        self.setInteractiveMode('plane')

    def _interactiveModeChanged(self):
        self._panPlaneAction.setChecked(self.getInteractiveMode() == 'plane')
        self._updateColors()

    def _planeVisibilityChanged(self, visible):
        """Handle visibility events from the plane"""
        if visible != self._panPlaneAction.isEnabled():
            self._panPlaneAction.setEnabled(visible)
            if visible:
                self.setInteractiveMode('plane')
            elif self._panPlaneAction.isChecked():
                self.setInteractiveMode('rotate')

    def setInteractiveMode(self, mode):
        """Choose the current interaction.

        :param str mode: Either rotate, pan or plane
        """
        if mode == self.getInteractiveMode():
            return

        sceneScale = self.getPlot3DWidget().viewport.scene.transforms[0]
        if mode == 'plane':
            mode = interaction.PanPlaneZoomOnWheelControl(
                self.getPlot3DWidget().viewport,
                self._cutPlane._get3DPrimitives()[0],
                mode='position',
                orbitAroundCenter=False,
                scaleTransform=sceneScale)

        self.getPlot3DWidget().setInteractiveMode(mode)
        self._updateColors()

    def getInteractiveMode(self):
        """Returns the current interaction mode, see :meth:`setInteractiveMode`
        """
        if isinstance(self.getPlot3DWidget().eventHandler,
                      interaction.PanPlaneZoomOnWheelControl):
            return 'plane'
        else:
            return self.getPlot3DWidget().getInteractiveMode()

    # Handle scalar field

    def setData(self, data, copy=True):
        """Set the 3D scalar data set to use for building the iso-surface.

        Dataset order is zyx (i.e., first dimension is z).

        :param data: scalar field from which to extract the iso-surface
        :type data: 3D numpy.ndarray of float32 with shape at least (2, 2, 2)
        :param bool copy:
            True (default) to make a copy,
            False to avoid copy (DO NOT MODIFY data afterwards)
        """
        if data is None:
            self._data = None
            self._dataRange = None
            self.setSelectedRegion(zrange=None, yrange=None, xrange_=None)
            self._group.shape = None
            self.centerScene()

        else:
            data = numpy.array(data, copy=copy, dtype=numpy.float32, order='C')
            assert data.ndim == 3
            assert min(data.shape) >= 2

            wasData = self._data is not None
            previousSelectedRegion = self.getSelectedRegion()

            self._data = data

            # Store data range info
            dataRange = min_max(self._data, min_positive=True, finite=True)
            if dataRange.minimum is None:  # Only non-finite data
                dataRange = None

            if dataRange is not None:
                min_positive = dataRange.min_positive
                if min_positive is None:
                    min_positive = float('nan')
                dataRange = dataRange.minimum, min_positive, dataRange.maximum
            self._dataRange = dataRange

            if previousSelectedRegion is not None:
                # Update selected region to ensure it is clipped to array range
                self.setSelectedRegion(*previousSelectedRegion.getArrayRange())

            self._group.shape = self._data.shape

            if not wasData:
                self.centerScene()  # Reset viewpoint the first time only

        # Update iso-surfaces
        for isosurface in self.getIsosurfaces():
            isosurface._setData(self._data, copy=False)

        self.sigDataChanged.emit()

    def getData(self, copy=True):
        """Get the 3D scalar data currently used to build the iso-surface.

        :param bool copy:
           True (default) to get a copy,
           False to get the internal data (DO NOT modify!)
        :return: The data set (or None if not set)
        """
        if self._data is None:
            return None
        else:
            return numpy.array(self._data, copy=copy)

    def getDataRange(self):
        """Return the range of the data as a 3-tuple of values.

        positive min is NaN if no data is positive.

        :return: (min, positive min, max) or None.
        """
        return self._dataRange

    # Transformations

    def setOuterScale(self, sx=1., sy=1., sz=1.):
        """Set the scale to apply to the whole scene including the axes.

        This is useful when axis lengths in data space are really different.

        :param float sx: Scale factor along the X axis
        :param float sy: Scale factor along the Y axis
        :param float sz: Scale factor along the Z axis
        """
        self._outerScale.setScale(sx, sy, sz)
        self.centerScene()

    def getOuterScale(self):
        """Returns the scales provided by :meth:`setOuterScale`.

        :rtype: numpy.ndarray
        """
        return self._outerScale.scale

    def setScale(self, sx=1., sy=1., sz=1.):
        """Set the scale of the 3D scalar field (i.e., size of a voxel).

        :param float sx: Scale factor along the X axis
        :param float sy: Scale factor along the Y axis
        :param float sz: Scale factor along the Z axis
        """
        scale = numpy.array((sx, sy, sz), dtype=numpy.float32)
        if not numpy.all(numpy.equal(scale, self.getScale())):
            self._dataScale.scale = scale
            self.sigTransformChanged.emit()
            self.centerScene()  # Reset viewpoint

    def getScale(self):
        """Returns the scales provided by :meth:`setScale` as a numpy.ndarray.
        """
        return self._dataScale.scale

    def setTranslation(self, x=0., y=0., z=0.):
        """Set the translation of the origin of the data array in data coordinates.

        :param float x: Offset of the data origin on the X axis
        :param float y: Offset of the data origin on the Y axis
        :param float z: Offset of the data origin on the Z axis
        """
        translation = numpy.array((x, y, z), dtype=numpy.float32)
        if not numpy.all(numpy.equal(translation, self.getTranslation())):
            self._dataTranslate.translation = translation
            self.sigTransformChanged.emit()
            self.centerScene()  # Reset viewpoint

    def getTranslation(self):
        """Returns the offset set by :meth:`setTranslation` as a numpy.ndarray.
        """
        return self._dataTranslate.translation

    def setTransformMatrix(self, matrix3x3):
        """Set the transform matrix applied to the data.

        :param numpy.ndarray matrix: 3x3 transform matrix
        """
        matrix3x3 = numpy.array(matrix3x3, copy=True, dtype=numpy.float32)
        if not numpy.all(numpy.equal(matrix3x3, self.getTransformMatrix())):
            matrix = numpy.identity(4, dtype=numpy.float32)
            matrix[:3, :3] = matrix3x3
            self._dataTransform.setMatrix(matrix)
            self.sigTransformChanged.emit()
            self.centerScene()  # Reset viewpoint

    def getTransformMatrix(self):
        """Returns the transform matrix applied to the data.

        See :meth:`setTransformMatrix`.

        :rtype: numpy.ndarray
        """
        return self._dataTransform.getMatrix()[:3, :3]

    # Axes labels

    def isBoundingBoxVisible(self):
        """Returns axes labels, grid and bounding box visibility.

        :rtype: bool
        """
        return self._bbox.boxVisible

    def setBoundingBoxVisible(self, visible):
        """Set axes labels, grid and bounding box visibility.

        :param bool visible: True to show axes, False to hide
        """
        visible = bool(visible)
        self._bbox.boxVisible = visible

    def setAxesLabels(self, xlabel=None, ylabel=None, zlabel=None):
        """Set the text labels of the axes.

        :param str xlabel: Label of the X axis, None to leave unchanged.
        :param str ylabel: Label of the Y axis, None to leave unchanged.
        :param str zlabel: Label of the Z axis, None to leave unchanged.
        """
        if xlabel is not None:
            self._bbox.xlabel = xlabel

        if ylabel is not None:
            self._bbox.ylabel = ylabel

        if zlabel is not None:
            self._bbox.zlabel = zlabel

    class _Labels(tuple):
        """Return type of :meth:`getAxesLabels`"""

        def getXLabel(self):
            """Label of the X axis (str)"""
            return self[0]

        def getYLabel(self):
            """Label of the Y axis (str)"""
            return self[1]

        def getZLabel(self):
            """Label of the Z axis (str)"""
            return self[2]

    def getAxesLabels(self):
        """Returns the text labels of the axes

        >>> widget = ScalarFieldView()
        >>> widget.setAxesLabels(xlabel='X')

        You can get the labels either as a 3-tuple:

        >>> xlabel, ylabel, zlabel = widget.getAxesLabels()

        Or as an object with methods getXLabel, getYLabel and getZLabel:

        >>> labels = widget.getAxesLabels()
        >>> labels.getXLabel()
        ... 'X'

        :return: object describing the labels
        """
        return self._Labels((self._bbox.xlabel,
                             self._bbox.ylabel,
                             self._bbox.zlabel))

    # Colors

    def _updateColors(self):
        """Update item depending on foreground/highlight color"""
        self._bbox.tickColor = self._foregroundColor
        self._selectionBox.strokeColor = self._foregroundColor
        if self.getInteractiveMode() == 'plane':
            self._cutPlane.setStrokeColor(self._highlightColor)
            self._bbox.color = self._foregroundColor
        else:
            self._cutPlane.setStrokeColor(self._foregroundColor)
            self._bbox.color = self._highlightColor

    def getForegroundColor(self):
        """Return color used for text and bounding box (QColor)"""
        return qt.QColor.fromRgbF(*self._foregroundColor)

    def setForegroundColor(self, color):
        """Set the foreground color.

        :param color: RGB color: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        if color != self._foregroundColor:
            self._foregroundColor = color
            self._updateColors()

    def getHighlightColor(self):
        """Return color used for highlighted item bounding box (QColor)"""
        return qt.QColor.fromRgbF(*self._highlightColor)

    def setHighlightColor(self, color):
        """Set hightlighted item color.

        :param color: RGB color: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        if color != self._highlightColor:
            self._highlightColor = color
            self._updateColors()

    # Cut Plane

    def getCutPlanes(self):
        """Return an iterable of all cut planes of the view.

        This includes hidden cut planes.

        For now, there is always one cut plane.
        """
        return (self._cutPlane,)

    # Selection

    def setSelectedRegion(self, zrange=None, yrange=None, xrange_=None):
        """Set the 3D selected region aligned with the axes.

        Provided range are array indices range.
        The provided ranges are clipped to the data.
        If a range is None, the range of the array on this dimension is used.

        :param zrange: (zmin, zmax) range of the selection
        :param yrange: (ymin, ymax) range of the selection
        :param xrange_: (xmin, xmax) range of the selection
        """
        # No range given: unset selection
        if zrange is None and yrange is None and xrange_ is None:
            selectedRange = None

        else:
            # Handle default ranges
            if self._data is not None:
                if zrange is None:
                    zrange = 0, self._data.shape[0]
                if yrange is None:
                    yrange = 0, self._data.shape[1]
                if xrange_ is None:
                    xrange_ = 0, self._data.shape[2]

            elif None in (xrange_, yrange, zrange):
                # One of the range is None and no data available
                raise RuntimeError(
                    'Data is not set, cannot get default range from it.')

            # Clip selected region to data shape and make sure min <= max
            selectedRange = numpy.array((
                (max(0, min(*zrange)),
                 min(self._data.shape[0], max(*zrange))),
                (max(0, min(*yrange)),
                 min(self._data.shape[1], max(*yrange))),
                (max(0, min(*xrange_)),
                 min(self._data.shape[2], max(*xrange_))),
                ), dtype=numpy.int)

        # numpy.equal supports None
        if not numpy.all(numpy.equal(selectedRange, self._selectedRange)):
            self._selectedRange = selectedRange

            # Update scene accordingly
            if self._selectedRange is None:
                self._selectionBox.visible = False
            else:
                self._selectionBox.visible = True
                scales = self._selectedRange[:, 1] - self._selectedRange[:, 0]
                self._selectionBox.size = scales[::-1]
                self._selectionBox.transforms = [
                    transform.Translate(*self._selectedRange[::-1, 0])]

            self.sigSelectedRegionChanged.emit(self.getSelectedRegion())

    def getSelectedRegion(self):
        """Returns the currently selected region or None."""
        if self._selectedRange is None:
            return None
        else:
            dataBBox = self._group.transforms.transformBounds(
                self._selectedRange[::-1].T).T
            return SelectedRegion(self._selectedRange, dataBBox,
                                  translation=self.getTranslation(),
                                  scale=self.getScale())

    # Handle iso-surfaces

    sigIsosurfaceAdded = qt.Signal(object)
    """Signal emitted when a new iso-surface is added to the view.

    The newly added iso-surface is provided by this signal
    """

    sigIsosurfaceRemoved = qt.Signal(object)
    """Signal emitted when an iso-surface is removed from the view

    The removed iso-surface is provided by this signal.
    """

    def addIsosurface(self, level, color):
        """Add an iso-surface to the view.

        :param level:
            The value at which to build the iso-surface or a callable
            (e.g., a function) taking a 3D numpy.ndarray as input and
            returning a float.
            Example: numpy.mean(data) + numpy.std(data)
        :type level: float or callable
        :param color: RGBA color of the isosurface
        :type color: str or array-like of 4 float in [0., 1.]
        :return: Isosurface object describing this isosurface
        """
        isosurface = Isosurface(parent=self)
        isosurface.setColor(color)
        if callable(level):
            isosurface.setAutoLevelFunction(level)
        else:
            isosurface.setLevel(level)
        isosurface._setData(self._data, copy=False)
        isosurface.sigLevelChanged.connect(self._updateIsosurfaces)

        self._isosurfaces.append(isosurface)

        self._updateIsosurfaces()

        self.sigIsosurfaceAdded.emit(isosurface)
        return isosurface

    def getIsosurfaces(self):
        """Return an iterable of all iso-surfaces of the view"""
        return tuple(self._isosurfaces)

    def removeIsosurface(self, isosurface):
        """Remove an iso-surface from the view.

        :param isosurface: The isosurface object to remove"""
        if isosurface not in self.getIsosurfaces():
            _logger.warning(
                "Try to remove isosurface that is not in the list: %s",
                str(isosurface))
        else:
            isosurface.sigLevelChanged.disconnect(self._updateIsosurfaces)
            self._isosurfaces.remove(isosurface)
            self._updateIsosurfaces()
            self.sigIsosurfaceRemoved.emit(isosurface)

    def clearIsosurfaces(self):
        """Remove all iso-surfaces from the view."""
        for isosurface in self.getIsosurfaces():
            self.removeIsosurface(isosurface)

    def _updateIsosurfaces(self, level=None):
        """Handle updates of iso-surfaces level and add/remove"""
        # Sorting using minus, this supposes data 'object' to be max values
        sortedIso = sorted(self.getIsosurfaces(),
                           key=lambda iso: - iso.getLevel())
        self._isogroup.children = [iso._get3DPrimitive() for iso in sortedIso]
