# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module provides 3D array item class and its sub-items.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"

import logging
import time
import numpy

from silx.math.combo import min_max
from silx.math.marchingcubes import MarchingCubes

from ... import qt
from ...colors import rgba

from ..scene import cutplane, primitives, transform, utils

from .core import BaseNodeItem, Item3D, ItemChangedType, Item3DChangedType
from .mixins import ColormapMixIn, InterpolationMixIn, PlaneMixIn
from ._pick import PickingResult


_logger = logging.getLogger(__name__)


class CutPlane(Item3D, ColormapMixIn, InterpolationMixIn, PlaneMixIn):
    """Class representing a cutting plane in a :class:`ScalarField3D` item.

    :param parent: 3D Data set in which the cut plane is applied.
    """

    def __init__(self, parent):
        plane = cutplane.CutPlane(normal=(0, 1, 0))

        Item3D.__init__(self, parent=parent)
        ColormapMixIn.__init__(self)
        InterpolationMixIn.__init__(self)
        PlaneMixIn.__init__(self, plane=plane)

        self._dataRange = None

        self._getScenePrimitive().children = [plane]

        # Connect scene primitive to mix-in class
        ColormapMixIn._setSceneColormap(self, plane.colormap)
        InterpolationMixIn._setPrimitive(self, plane)

        parent.sigItemChanged.connect(self._parentChanged)

    def _parentChanged(self, event):
        """Handle data change in the parent this plane belongs to"""
        if event == ItemChangedType.DATA:
            self._getPlane().setData(self.sender().getData(copy=False),
                                     copy=False)

            # Store data range info as 3-tuple of values
            self._dataRange = self.sender().getDataRange()

            self.sigItemChanged.emit(ItemChangedType.DATA)

    # Colormap

    def getDisplayValuesBelowMin(self):
        """Return whether values <= colormap min are displayed or not.

        :rtype: bool
        """
        return self._getPlane().colormap.displayValuesBelowMin

    def setDisplayValuesBelowMin(self, display):
        """Set whether to display values <= colormap min.

        :param bool display: True to show values below min,
                             False to discard them
        """
        display = bool(display)
        if display != self.getDisplayValuesBelowMin():
            self._getPlane().colormap.displayValuesBelowMin = display
            self.sigItemChanged.emit(ItemChangedType.ALPHA)

    def getDataRange(self):
        """Return the range of the data as a 3-tuple of values.

        positive min is NaN if no data is positive.

        :return: (min, positive min, max) or None.
        """
        return self._dataRange

    def getData(self, copy=True):
        """Return 3D dataset.

        :param bool copy:
           True (default) to get a copy,
           False to get the internal data (DO NOT modify!)
        :return: The data set (or None if not set)
        """
        parent = self.parent()
        return None if parent is None else parent.getData(copy=copy)

    def _pickFull(self, context):
        """Perform picking in this item at given widget position.

        :param PickContext context: Current picking context
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        rayObject = context.getPickingSegment(frame=self._getScenePrimitive())
        if rayObject is None:
            return None

        points = utils.segmentPlaneIntersect(
            rayObject[0, :3],
            rayObject[1, :3],
            planeNorm=self.getNormal(),
            planePt=self.getPoint())

        if len(points) == 1:  # Single intersection
            if numpy.any(points[0] < 0.):
                return None  # Outside volume
            z, y, x = int(points[0][2]), int(points[0][1]), int(points[0][0])

            data = self.getData(copy=False)
            if data is None:
                return None  # No dataset

            depth, height, width = data.shape
            if z < depth and y < height and x < width:
                return PickingResult(self,
                                     positions=[points[0]],
                                     indices=([z], [y], [x]))
            else:
                return None  # Outside image
        else:  # Either no intersection or segment and image are coplanar
            return None


class Isosurface(Item3D):
    """Class representing an iso-surface in a :class:`ScalarField3D` item.

    :param parent: The DataItem3D this iso-surface belongs to
    """

    def __init__(self, parent):
        Item3D.__init__(self, parent=parent)
        assert isinstance(parent, ScalarField3D)
        parent.sigItemChanged.connect(self._scalarField3DChanged)
        self._level = float('nan')
        self._autoLevelFunction = None
        self._color = rgba('#FFD700FF')
        self._updateScenePrimitive()

    def _scalarField3DChanged(self, event):
        """Handle parent's ScalarField3D sigItemChanged"""
        if event == ItemChangedType.DATA:
            self._updateScenePrimitive()

    def getData(self, copy=True):
        """Return 3D dataset.

        :param bool copy:
           True (default) to get a copy,
           False to get the internal data (DO NOT modify!)
        :return: The data set (or None if not set)
        """
        parent = self.parent()
        return None if parent is None else parent.getData(copy=copy)

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
            self._updateScenePrimitive()
            self._updated(Item3DChangedType.ISO_LEVEL)

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
        self._updateScenePrimitive()

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
            primitive = self._getScenePrimitive()
            if len(primitive.children) != 0:
                primitive.children[0].setAttribute('color', self._color)
            self._updated(ItemChangedType.COLOR)

    def _updateScenePrimitive(self):
        """Update underlying mesh"""
        self._getScenePrimitive().children = []

        data = self.getData(copy=False)

        if data is None:
            if self.isAutoLevel():
                self._level = float('nan')

        else:
            if self.isAutoLevel():
                st = time.time()
                try:
                    level = float(self.getAutoLevelFunction()(data))

                except Exception:
                    module_ = self.getAutoLevelFunction().__module__
                    name = self.getAutoLevelFunction().__name__
                    _logger.error(
                        "Error while executing iso level function %s.%s",
                        module_,
                        name,
                        exc_info=True)
                    level = float('nan')

                else:
                    _logger.info(
                        'Computed iso-level in %f s.', time.time() - st)

                if level != self._level:
                    self._level = level
                    self._updated(Item3DChangedType.ISO_LEVEL)

            if not numpy.isfinite(self._level):
                return

            st = time.time()
            vertices, normals, indices = MarchingCubes(
                data,
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
                self._getScenePrimitive().children = [mesh]

    def _pickFull(self, context):
        """Perform picking in this item at given widget position.

        :param PickContext context: Current picking context
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        rayObject = context.getPickingSegment(frame=self._getScenePrimitive())
        if rayObject is None:
            return None
        rayObject = rayObject[:, :3]

        data = self.getData(copy=False)
        bins = utils.segmentVolumeIntersect(
            rayObject, numpy.array(data.shape) - 1)
        if bins is None:
            return None

        # gather bin data
        offsets = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        indices = bins[:, numpy.newaxis, :] + offsets
        binsData = data[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]
        # binsData.shape = nbins, 8
        # TODO up-to this point everything can be done once for all isosurfaces

        # check bin candidates
        level = self.getLevel()
        mask = numpy.logical_and(numpy.nanmin(binsData, axis=1) <= level,
                                 level <= numpy.nanmax(binsData, axis=1))
        bins = bins[mask]
        binsData = binsData[mask]

        if len(bins) == 0:
            return None  # No bin candidate

        # do picking on candidates
        intersections = []
        depths = []
        for currentBin, data in zip(bins, binsData):
            mc = MarchingCubes(data.reshape(2, 2, 2), isolevel=level)
            points = mc.get_vertices() + currentBin
            triangles = points[mc.get_indices()]
            t = utils.segmentTrianglesIntersection(rayObject, triangles)[1]
            t = numpy.unique(t)  # Duplicates happen on triangle edges
            if len(t) != 0:
                # Compute intersection points and get closest data point
                points = t.reshape(-1, 1) * (rayObject[1] - rayObject[0]) + rayObject[0]
                # Get closest data points by rounding to int
                intersections.extend(points)
                depths.extend(t)

        if len(intersections) == 0:
            return None  # No intersected triangles

        intersections = numpy.array(intersections)[numpy.argsort(depths)]
        indices = numpy.transpose(numpy.round(intersections).astype(numpy.int))
        return PickingResult(self, positions=intersections, indices=indices)


class ScalarField3D(BaseNodeItem):
    """3D scalar field on a regular grid.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        BaseNodeItem.__init__(self, parent=parent)

        # Gives this item the shape of the data, no matter
        # of the isosurface/cut plane size
        self._boundedGroup = primitives.BoundedGroup()

        # Store iso-surfaces
        self._isosurfaces = []

        self._data = None
        self._dataRange = None

        self._cutPlane = CutPlane(parent=self)
        self._cutPlane.setVisible(False)

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

        self._getScenePrimitive().children = [
            self._boundedGroup,
            self._cutPlane._getScenePrimitive(),
            self._isogroup]

    def setData(self, data, copy=True):
        """Set the 3D scalar data represented by this item.

        Dataset order is zyx (i.e., first dimension is z).

        :param data: 3D array
        :type data: 3D numpy.ndarray of float32 with shape at least (2, 2, 2)
        :param bool copy:
            True (default) to make a copy,
            False to avoid copy (DO NOT MODIFY data afterwards)
        """
        if data is None:
            self._data = None
            self._dataRange = None
            self._boundedGroup.shape = None

        else:
            data = numpy.array(data, copy=copy, dtype=numpy.float32, order='C')
            assert data.ndim == 3
            assert min(data.shape) >= 2

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

            self._boundedGroup.shape = self._data.shape

        self._updated(ItemChangedType.DATA)

    def getData(self, copy=True):
        """Return 3D dataset.

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

    # Cut Plane

    def getCutPlanes(self):
        """Return an iterable of all :class:`CutPlane` of this item.

        This includes hidden cut planes.

        For now, there is always one cut plane.
        """
        return (self._cutPlane,)

    # Handle iso-surfaces

    # TODO rename to sigItemAdded|Removed?
    sigIsosurfaceAdded = qt.Signal(object)
    """Signal emitted when a new iso-surface is added to the view.

    The newly added iso-surface is provided by this signal
    """

    sigIsosurfaceRemoved = qt.Signal(object)
    """Signal emitted when an iso-surface is removed from the view

    The removed iso-surface is provided by this signal.
    """

    def addIsosurface(self, level, color):
        """Add an isosurface to this item.

        :param level:
            The value at which to build the iso-surface or a callable
            (e.g., a function) taking a 3D numpy.ndarray as input and
            returning a float.
            Example: numpy.mean(data) + numpy.std(data)
        :type level: float or callable
        :param color: RGBA color of the isosurface
        :type color: str or array-like of 4 float in [0., 1.]
        :return: isosurface object
        :rtype: ~silx.gui.plot3d.items.volume.Isosurface
        """
        isosurface = Isosurface(parent=self)
        isosurface.setColor(color)
        if callable(level):
            isosurface.setAutoLevelFunction(level)
        else:
            isosurface.setLevel(level)
        isosurface.sigItemChanged.connect(self._isosurfaceItemChanged)

        self._isosurfaces.append(isosurface)

        self._updateIsosurfaces()

        self.sigIsosurfaceAdded.emit(isosurface)
        return isosurface

    def getIsosurfaces(self):
        """Return an iterable of all :class:`.Isosurface` instance of this item"""
        return tuple(self._isosurfaces)

    def removeIsosurface(self, isosurface):
        """Remove an iso-surface from this item.

        :param ~silx.gui.plot3d.Plot3DWidget.Isosurface isosurface:
            The isosurface object to remove
        """
        if isosurface not in self.getIsosurfaces():
            _logger.warning(
                "Try to remove isosurface that is not in the list: %s",
                str(isosurface))
        else:
            isosurface.sigItemChanged.disconnect(self._isosurfaceItemChanged)
            self._isosurfaces.remove(isosurface)
            self._updateIsosurfaces()
            self.sigIsosurfaceRemoved.emit(isosurface)

    def clearIsosurfaces(self):
        """Remove all :class:`.Isosurface` instances from this item."""
        for isosurface in self.getIsosurfaces():
            self.removeIsosurface(isosurface)

    def _isosurfaceItemChanged(self, event):
        """Handle update of isosurfaces upon level changed"""
        if event == Item3DChangedType.ISO_LEVEL:
            self._updateIsosurfaces()

    def _updateIsosurfaces(self):
        """Handle updates of iso-surfaces level and add/remove"""
        # Sorting using minus, this supposes data 'object' to be max values
        sortedIso = sorted(self.getIsosurfaces(),
                           key=lambda isosurface: - isosurface.getLevel())
        self._isogroup.children = [iso._getScenePrimitive() for iso in sortedIso]

    # BaseNodeItem

    def getItems(self):
        """Returns the list of items currently present in the ScalarField3D.

        :rtype: tuple
        """
        return self.getCutPlanes() + self.getIsosurfaces()
