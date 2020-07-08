# coding: utf-8
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
"""This module provides 2D and 3D scatter data item class.
"""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/11/2017"

try:
    from collections import abc
except ImportError:  # Python2 support
    import collections as abc
import logging
import numpy

from ....utils.deprecation import deprecated
from ... import _glutils as glu
from ...plot._utils.delaunay import delaunay
from ..scene import function, primitives, utils

from ...plot.items import ScatterVisualizationMixIn
from .core import DataItem3D, Item3DChangedType, ItemChangedType
from .mixins import ColormapMixIn, SymbolMixIn
from ._pick import PickingResult


_logger = logging.getLogger(__name__)


class Scatter3D(DataItem3D, ColormapMixIn, SymbolMixIn):
    """Description of a 3D scatter plot.

    :param parent: The View widget this item belongs to.
    """

    # TODO supports different size for each point

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        ColormapMixIn.__init__(self)
        SymbolMixIn.__init__(self)

        noData = numpy.zeros((0, 1), dtype=numpy.float32)
        symbol, size = self._getSceneSymbol()
        self._scatter = primitives.Points(
            x=noData, y=noData, z=noData, value=noData, size=size)
        self._scatter.marker = symbol
        self._getScenePrimitive().children.append(self._scatter)

        # Connect scene primitive to mix-in class
        ColormapMixIn._setSceneColormap(self, self._scatter.colormap)

    def _updated(self, event=None):
        """Handle mix-in class updates"""
        if event in (ItemChangedType.SYMBOL, ItemChangedType.SYMBOL_SIZE):
            symbol, size = self._getSceneSymbol()
            self._scatter.marker = symbol
            self._scatter.setAttribute('size', size, copy=True)

        super(Scatter3D, self)._updated(event)

    def setData(self, x, y, z, value, copy=True):
        """Set the data of the scatter plot

        :param numpy.ndarray x: Array of X coordinates (single value not accepted)
        :param y: Points Y coordinate (array-like or single value)
        :param z: Points Z coordinate (array-like or single value)
        :param value: Points values (array-like or single value)
        :param bool copy:
            True (default) to copy the data,
            False to use provided data (do not modify!)
        """
        self._scatter.setAttribute('x', x, copy=copy)
        self._scatter.setAttribute('y', y, copy=copy)
        self._scatter.setAttribute('z', z, copy=copy)
        self._scatter.setAttribute('value', value, copy=copy)

        self._setColormappedData(self.getValueData(copy=False), copy=False)
        self._updated(ItemChangedType.DATA)

    def getData(self, copy=True):
        """Returns data as provided to :meth:`setData`.

        :param bool copy: True to get a copy,
                          False to return internal data (do not modify!)
        :return: (x, y, z, value)
        """
        return (self.getXData(copy),
                self.getYData(copy),
                self.getZData(copy),
                self.getValueData(copy))

    def getXData(self, copy=True):
        """Returns X data coordinates.

        :param bool copy: True to get a copy,
                          False to return internal array (do not modify!)
        :return: X coordinates
        :rtype: numpy.ndarray
        """
        return self._scatter.getAttribute('x', copy=copy).reshape(-1)

    def getYData(self, copy=True):
        """Returns Y data coordinates.

        :param bool copy: True to get a copy,
                          False to return internal array (do not modify!)
        :return: Y coordinates
        :rtype: numpy.ndarray
        """
        return self._scatter.getAttribute('y', copy=copy).reshape(-1)

    def getZData(self, copy=True):
        """Returns Z data coordinates.

        :param bool copy: True to get a copy,
                          False to return internal array (do not modify!)
        :return: Z coordinates
        :rtype: numpy.ndarray
        """
        return self._scatter.getAttribute('z', copy=copy).reshape(-1)

    def getValueData(self, copy=True):
        """Returns data values.

        :param bool copy: True to get a copy,
                          False to return internal array (do not modify!)
        :return: data values
        :rtype: numpy.ndarray
        """
        return self._scatter.getAttribute('value', copy=copy).reshape(-1)

    @deprecated(reason="Consistency with PlotWidget items",
                replacement="getValueData", since_version="0.10.0")
    def getValues(self, copy=True):
        return self.getValueData(copy)

    def _pickFull(self, context, threshold=0., sort='depth'):
        """Perform picking in this item at given widget position.

        :param PickContext context: Current picking context
        :param float threshold: Picking threshold in pixel.
            Perform picking in a square of size threshold x threshold.
        :param str sort: How returned indices are sorted:

            - 'index' (default): sort by the value of the indices
            - 'depth':  Sort by the depth of the points from the current
              camera point of view.
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        assert sort in ('index', 'depth')

        rayNdc = context.getPickingSegment(frame='ndc')
        if rayNdc is None:  # No picking outside viewport
            return None

        # Project data to NDC
        xData = self.getXData(copy=False)
        if len(xData) == 0:  # No data in the scatter
            return None

        primitive = self._getScenePrimitive()

        dataPoints = numpy.transpose((xData,
                                      self.getYData(copy=False),
                                      self.getZData(copy=False),
                                      numpy.ones_like(xData)))

        pointsNdc = primitive.objectToNDCTransform.transformPoints(
            dataPoints, perspectiveDivide=True)

        # Perform picking
        distancesNdc = numpy.abs(pointsNdc[:, :2] - rayNdc[0, :2])
        # TODO issue with symbol size: using pixel instead of points
        threshold += self.getSymbolSize()
        thresholdNdc = 2. * threshold / numpy.array(primitive.viewport.size)
        picked = numpy.where(numpy.logical_and(
                numpy.all(distancesNdc < thresholdNdc, axis=1),
                numpy.logical_and(rayNdc[0, 2] <= pointsNdc[:, 2],
                                  pointsNdc[:, 2] <= rayNdc[1, 2])))[0]

        if sort == 'depth':
            # Sort picked points from front to back
            picked = picked[numpy.argsort(pointsNdc[picked, 2])]

        if picked.size > 0:
            return PickingResult(self,
                                 positions=dataPoints[picked, :3],
                                 indices=picked,
                                 fetchdata=self.getValueData)
        else:
            return None


class Scatter2D(DataItem3D, ColormapMixIn, SymbolMixIn,
                ScatterVisualizationMixIn):
    """2D scatter data with settable visualization mode.

    :param parent: The View widget this item belongs to.
    """

    _VISUALIZATION_PROPERTIES = {
        ScatterVisualizationMixIn.Visualization.POINTS:
            ('symbol', 'symbolSize'),
        ScatterVisualizationMixIn.Visualization.LINES:
            ('lineWidth',),
        ScatterVisualizationMixIn.Visualization.SOLID: (),
    }
    """Dict {visualization mode: property names used in this mode}"""

    _SUPPORTED_SCATTER_VISUALIZATION = tuple(_VISUALIZATION_PROPERTIES.keys())
    """Overrides supported Visualizations"""

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        ColormapMixIn.__init__(self)
        SymbolMixIn.__init__(self)
        ScatterVisualizationMixIn.__init__(self)

        self._heightMap = False
        self._lineWidth = 1.

        self._x = numpy.zeros((0,), dtype=numpy.float32)
        self._y = numpy.zeros((0,), dtype=numpy.float32)
        self._value = numpy.zeros((0,), dtype=numpy.float32)

        self._cachedLinesIndices = None
        self._cachedTrianglesIndices = None

        # Connect scene primitive to mix-in class
        ColormapMixIn._setSceneColormap(self, function.Colormap())

    def _updated(self, event=None):
        """Handle mix-in class updates"""
        if event in (ItemChangedType.SYMBOL, ItemChangedType.SYMBOL_SIZE):
            symbol, size = self._getSceneSymbol()
            for child in self._getScenePrimitive().children:
                if isinstance(child, primitives.Points):
                    child.marker = symbol
                    child.setAttribute('size', size, copy=True)

        elif event is ItemChangedType.VISIBLE:
            # TODO smart update?, need dirty flags
            self._updateScene()

        elif event is ItemChangedType.VISUALIZATION_MODE:
            self._updateScene()

        super(Scatter2D, self)._updated(event)

    def isPropertyEnabled(self, name, visualization=None):
        """Returns true if the property is used with visualization mode.

        :param str name: The name of the property to check, in:
                         'lineWidth', 'symbol', 'symbolSize'
        :param str visualization:
            The visualization mode for which to get the info.
            By default, it is the current visualization mode.
        :return:
        """
        assert name in ('lineWidth', 'symbol', 'symbolSize')
        if visualization is None:
            visualization = self.getVisualization()
        assert visualization in self.supportedVisualizations()
        return name in self._VISUALIZATION_PROPERTIES[visualization]

    def setHeightMap(self, heightMap):
        """Set whether to display the data has a height map or not.

        When displayed as a height map, the data values are used as
        z coordinates.

        :param bool heightMap:
            True to display a height map,
            False to display as 2D data with z=0
        """
        heightMap = bool(heightMap)
        if heightMap != self.isHeightMap():
            self._heightMap = heightMap
            self._updateScene()
            self._updated(Item3DChangedType.HEIGHT_MAP)

    def isHeightMap(self):
        """Returns True if data is displayed as a height map.

        :rtype: bool
        """
        return self._heightMap

    def getLineWidth(self):
        """Return the curve line width in pixels (float)"""
        return self._lineWidth

    def setLineWidth(self, width):
        """Set the width in pixel of the curve line

        See :meth:`getLineWidth`.

        :param float width: Width in pixels
        """
        width = float(width)
        assert width >= 1.
        if width != self._lineWidth:
            self._lineWidth = width
            for child in self._getScenePrimitive().children:
                if hasattr(child, 'lineWidth'):
                    child.lineWidth = width
            self._updated(ItemChangedType.LINE_WIDTH)

    def setData(self, x, y, value, copy=True):
        """Set the data represented by this item.

        Provided arrays must have the same length.

        :param numpy.ndarray x: X coordinates (array-like)
        :param numpy.ndarray y: Y coordinates (array-like)
        :param value: Points value: array-like or single scalar
        :param bool copy:
            True (default) to make a copy of the data,
            False to avoid copy if possible (do not modify the arrays).
        """
        x = numpy.array(
            x, copy=copy, dtype=numpy.float32, order='C').reshape(-1)
        y = numpy.array(
            y, copy=copy, dtype=numpy.float32, order='C').reshape(-1)
        assert len(x) == len(y)

        if isinstance(value, abc.Iterable):
            value = numpy.array(
                value, copy=copy, dtype=numpy.float32, order='C').reshape(-1)
            assert len(value) == len(x)
        else:  # Single scalar
            value = numpy.array((float(value),), dtype=numpy.float32)

        self._x = x
        self._y = y
        self._value = value

        # Reset cache
        self._cachedLinesIndices = None
        self._cachedTrianglesIndices = None

        self._setColormappedData(self.getValueData(copy=False), copy=False)

        self._updateScene()

        self._updated(ItemChangedType.DATA)

    def getData(self, copy=True):
        """Returns data as provided to :meth:`setData`.

        :param bool copy: True to get a copy,
                          False to return internal data (do not modify!)
        :return: (x, y, value)
        """
        return (self.getXData(copy=copy),
                self.getYData(copy=copy),
                self.getValueData(copy=copy))

    def getXData(self, copy=True):
        """Returns X data coordinates.

        :param bool copy: True to get a copy,
                          False to return internal array (do not modify!)
        :return: X coordinates
        :rtype: numpy.ndarray
        """
        return numpy.array(self._x, copy=copy)

    def getYData(self, copy=True):
        """Returns Y data coordinates.

        :param bool copy: True to get a copy,
                          False to return internal array (do not modify!)
        :return: Y coordinates
        :rtype: numpy.ndarray
        """
        return numpy.array(self._y, copy=copy)

    def getValueData(self, copy=True):
        """Returns data values.

        :param bool copy: True to get a copy,
                          False to return internal array (do not modify!)
        :return: data values
        :rtype: numpy.ndarray
        """
        return numpy.array(self._value, copy=copy)

    @deprecated(reason="Consistency with PlotWidget items",
                replacement="getValueData", since_version="0.10.0")
    def getValues(self, copy=True):
        return self.getValueData(copy)

    def _pickPoints(self, context, points, threshold=1., sort='depth'):
        """Perform picking while in 'points' visualization mode

        :param PickContext context: Current picking context
        :param float threshold: Picking threshold in pixel.
            Perform picking in a square of size threshold x threshold.
        :param str sort: How returned indices are sorted:

            - 'index' (default): sort by the value of the indices
            - 'depth':  Sort by the depth of the points from the current
              camera point of view.
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        assert sort in ('index', 'depth')

        rayNdc = context.getPickingSegment(frame='ndc')
        if rayNdc is None:  # No picking outside viewport
            return None

        # Project data to NDC
        primitive = self._getScenePrimitive()
        pointsNdc = primitive.objectToNDCTransform.transformPoints(
            points, perspectiveDivide=True)

        # Perform picking
        distancesNdc = numpy.abs(pointsNdc[:, :2] - rayNdc[0, :2])
        thresholdNdc = threshold / numpy.array(primitive.viewport.size)
        picked = numpy.where(numpy.logical_and(
            numpy.all(distancesNdc < thresholdNdc, axis=1),
            numpy.logical_and(rayNdc[0, 2] <= pointsNdc[:, 2],
                              pointsNdc[:, 2] <= rayNdc[1, 2])))[0]

        if sort == 'depth':
            # Sort picked points from front to back
            picked = picked[numpy.argsort(pointsNdc[picked, 2])]

        if picked.size > 0:
            return PickingResult(self,
                                 positions=points[picked, :3],
                                 indices=picked,
                                 fetchdata=self.getValueData)
        else:
            return None

    def _pickSolid(self, context, points):
        """Perform picking while in 'solid' visualization mode

        :param PickContext context: Current picking context
        """
        if self._cachedTrianglesIndices is None:
            _logger.info("Picking on Scatter2D before rendering")
            return None

        rayObject = context.getPickingSegment(frame=self._getScenePrimitive())
        if rayObject is None:  # No picking outside viewport
            return None
        rayObject = rayObject[:, :3]

        trianglesIndices = self._cachedTrianglesIndices.reshape(-1, 3)
        triangles = points[trianglesIndices, :3]
        selectedIndices, t, barycentric = glu.segmentTrianglesIntersection(
            rayObject, triangles)
        closest = numpy.argmax(barycentric, axis=1)

        indices = trianglesIndices.reshape(-1, 3)[selectedIndices, closest]

        if len(indices) == 0:  # No point is picked
            return None

        # Compute intersection points and get closest data point
        positions = t.reshape(-1, 1) * (rayObject[1] - rayObject[0]) + rayObject[0]

        return PickingResult(self,
                             positions=positions,
                             indices=indices,
                             fetchdata=self.getValueData)

    def _pickFull(self, context):
        """Perform picking in this item at given widget position.

        :param PickContext context: Current picking context
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        xData = self.getXData(copy=False)
        if len(xData) == 0:  # No data in the scatter
            return None

        if self.isHeightMap():
            zData = self.getValueData(copy=False)
        else:
            zData = numpy.zeros_like(xData)

        points = numpy.transpose((xData,
                                  self.getYData(copy=False),
                                  zData,
                                  numpy.ones_like(xData)))

        mode = self.getVisualization()
        if mode is self.Visualization.POINTS:
            # TODO issue with symbol size: using pixel instead of points
            # Get "corrected" symbol size
            _, threshold = self._getSceneSymbol()
            return self._pickPoints(
                context, points, threshold=max(3., threshold))

        elif mode is self.Visualization.LINES:
            # Picking only at point
            return self._pickPoints(context, points, threshold=5.)

        else:  # mode == 'solid'
            return self._pickSolid(context, points)

    def _updateScene(self):
        self._getScenePrimitive().children = []  # Remove previous primitives

        if not self.isVisible():
            return  # Update when visible

        x, y, value = self.getData(copy=False)
        if len(x) == 0:
            return  # Nothing to display

        mode = self.getVisualization()
        heightMap = self.isHeightMap()

        if mode is self.Visualization.POINTS:
            z = value if heightMap else 0.
            symbol, size = self._getSceneSymbol()
            primitive = primitives.Points(
                x=x, y=y, z=z, value=value,
                size=size,
                colormap=self._getSceneColormap())
            primitive.marker = symbol

        else:
            # TODO run delaunay in a thread
            # Compute lines/triangles indices if not cached
            if self._cachedTrianglesIndices is None:
                triangulation = delaunay(x, y)
                if triangulation is None:
                    return None
                self._cachedTrianglesIndices = numpy.ravel(
                    triangulation.simplices.astype(numpy.uint32))

            if (mode is self.Visualization.LINES and
                    self._cachedLinesIndices is None):
                # Compute line indices
                self._cachedLinesIndices = utils.triangleToLineIndices(
                    self._cachedTrianglesIndices, unicity=True)

            if mode is self.Visualization.LINES:
                indices = self._cachedLinesIndices
                renderMode = 'lines'
            else:
                indices = self._cachedTrianglesIndices
                renderMode = 'triangles'

            # TODO supports x, y instead of copy
            if heightMap:
                if len(value) == 1:
                    value = numpy.ones_like(x) * value
                coordinates = numpy.array((x, y, value), dtype=numpy.float32).T
            else:
                coordinates = numpy.array((x, y), dtype=numpy.float32).T

            # TODO option to enable/disable light, cache normals
            # TODO smooth surface
            if mode is self.Visualization.SOLID:
                if heightMap:
                    coordinates = coordinates[indices]
                    if len(value) > 1:
                        value = value[indices]
                    triangleNormals = utils.trianglesNormal(coordinates)
                    normal = numpy.empty((len(triangleNormals) * 3, 3),
                                         dtype=numpy.float32)
                    normal[0::3, :] = triangleNormals
                    normal[1::3, :] = triangleNormals
                    normal[2::3, :] = triangleNormals
                    indices = None
                else:
                    normal = (0., 0., 1.)
            else:
                normal = None

            primitive = primitives.ColormapMesh3D(
                coordinates,
                value.reshape(-1, 1),  # Makes it a 2D array
                normal=normal,
                colormap=self._getSceneColormap(),
                indices=indices,
                mode=renderMode)
            primitive.lineWidth = self.getLineWidth()
            primitive.lineSmooth = False

        self._getScenePrimitive().children = [primitive]
