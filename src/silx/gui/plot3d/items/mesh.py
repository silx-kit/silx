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
"""This module provides regular mesh item class.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/07/2018"


import logging
import numpy

from ... import _glutils as glu
from ..scene import primitives, utils, function
from ..scene.transform import Rotate
from .core import DataItem3D, ItemChangedType
from .mixins import ColormapMixIn
from ._pick import PickingResult


_logger = logging.getLogger(__name__)


class _MeshBase(DataItem3D):
    """Base class for :class:`Mesh' and :class:`ColormapMesh`.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        self._mesh = None

    def _setMesh(self, mesh):
        """Set mesh primitive

        :param Union[None,Geometry] mesh: The scene primitive
        """
        self._getScenePrimitive().children = []  # Remove any previous mesh

        self._mesh = mesh
        if self._mesh is not None:
            self._getScenePrimitive().children.append(self._mesh)

        self._updated(ItemChangedType.DATA)

    def _getMesh(self):
        """Returns the underlying Mesh scene primitive"""
        return self._mesh

    def getPositionData(self, copy=True):
        """Get the mesh vertex positions.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The (x, y, z) positions as a (N, 3) array
        :rtype: numpy.ndarray
        """
        if self._getMesh() is None:
            return numpy.empty((0, 3), dtype=numpy.float32)
        else:
            return self._getMesh().getAttribute('position', copy=copy)

    def getNormalData(self, copy=True):
        """Get the mesh vertex normals.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The normals as a (N, 3) array, a single normal or None
        :rtype: Union[numpy.ndarray,None]
        """
        if self._getMesh() is None:
            return None
        else:
            return self._getMesh().getAttribute('normal', copy=copy)

    def getIndices(self, copy=True):
        """Get the vertex indices.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The vertex indices as an array or None.
        :rtype: Union[numpy.ndarray,None]
        """
        if self._getMesh() is None:
            return None
        else:
            return self._getMesh().getIndices(copy=copy)

    def getDrawMode(self):
        """Get mesh rendering mode.

        :return: The drawing mode of this primitive
        :rtype: str
        """
        return self._getMesh().drawMode

    def _pickFull(self, context):
        """Perform precise picking in this item at given widget position.

        :param PickContext context: Current picking context
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        rayObject = context.getPickingSegment(frame=self._getScenePrimitive())
        if rayObject is None:  # No picking outside viewport
            return None
        rayObject = rayObject[:, :3]

        positions = self.getPositionData(copy=False)
        if positions.size == 0:
            return None

        mode = self.getDrawMode()

        vertexIndices = self.getIndices(copy=False)
        if vertexIndices is not None:  # Expand indices
            positions = utils.unindexArrays(mode, vertexIndices, positions)[0]
            triangles = positions.reshape(-1, 3, 3)
        else:
            if mode == 'triangles':
                triangles = positions.reshape(-1, 3, 3)

            elif mode == 'triangle_strip':
                # Expand strip
                triangles = numpy.empty((len(positions) - 2, 3, 3),
                                        dtype=positions.dtype)
                triangles[:, 0] = positions[:-2]
                triangles[:, 1] = positions[1:-1]
                triangles[:, 2] = positions[2:]

            elif mode == 'fan':
                # Expand fan
                triangles = numpy.empty((len(positions) - 2, 3, 3),
                                        dtype=positions.dtype)
                triangles[:, 0] = positions[0]
                triangles[:, 1] = positions[1:-1]
                triangles[:, 2] = positions[2:]

            else:
                _logger.warning("Unsupported draw mode: %s" % mode)
                return None

        trianglesIndices, t, barycentric = glu.segmentTrianglesIntersection(
            rayObject, triangles)

        if len(trianglesIndices) == 0:
            return None

        points = t.reshape(-1, 1) * (rayObject[1] - rayObject[0]) + rayObject[0]

        # Get vertex index from triangle index and closest point in triangle
        closest = numpy.argmax(barycentric, axis=1)

        if mode == 'triangles':
            indices = trianglesIndices * 3 + closest

        elif mode == 'triangle_strip':
            indices = trianglesIndices + closest

        elif mode == 'fan':
            indices = trianglesIndices + closest  # For corners 1 and 2
            indices[closest == 0] = 0  # For first corner (common)

        if vertexIndices is not None:
            # Convert from indices in expanded triangles to input vertices
            indices = vertexIndices[indices]

        return PickingResult(self,
                             positions=points,
                             indices=indices,
                             fetchdata=self.getPositionData)


class Mesh(_MeshBase):
    """Description of mesh.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        _MeshBase.__init__(self, parent=parent)

    def setData(self,
                position,
                color,
                normal=None,
                mode='triangles',
                indices=None,
                copy=True):
        """Set mesh geometry data.

        Supported drawing modes are: 'triangles', 'triangle_strip', 'fan'

        :param numpy.ndarray position:
            Position (x, y, z) of each vertex as a (N, 3) array
        :param numpy.ndarray color: Colors for each point or a single color
        :param Union[numpy.ndarray,None] normal: Normals for each point or None (default)
        :param str mode: The drawing mode.
        :param Union[List[int],None] indices:
            Array of vertex indices or None to use arrays directly.
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """
        assert mode in ('triangles', 'triangle_strip', 'fan')
        if position is None or len(position) == 0:
            mesh = None
        else:
            mesh = primitives.Mesh3D(
                position, color, normal, mode=mode, indices=indices, copy=copy)
        self._setMesh(mesh)

    def getData(self, copy=True):
        """Get the mesh geometry.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The positions, colors, normals and mode
        :rtype: tuple of numpy.ndarray
        """
        return (self.getPositionData(copy=copy),
                self.getColorData(copy=copy),
                self.getNormalData(copy=copy),
                self.getDrawMode())

    def getColorData(self, copy=True):
        """Get the mesh vertex colors.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The RGBA colors as a (N, 4) array or a single color
        :rtype: numpy.ndarray
        """
        if self._getMesh() is None:
            return numpy.empty((0, 4), dtype=numpy.float32)
        else:
            return self._getMesh().getAttribute('color', copy=copy)


class ColormapMesh(_MeshBase, ColormapMixIn):
    """Description of mesh which color is defined by scalar and a colormap.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        _MeshBase.__init__(self, parent=parent)
        ColormapMixIn.__init__(self, function.Colormap())

    def setData(self,
                position,
                value,
                normal=None,
                mode='triangles',
                indices=None,
                copy=True):
        """Set mesh geometry data.

        Supported drawing modes are: 'triangles', 'triangle_strip', 'fan'

        :param numpy.ndarray position:
            Position (x, y, z) of each vertex as a (N, 3) array
        :param numpy.ndarray value: Data value for each vertex.
        :param Union[numpy.ndarray,None] normal: Normals for each point or None (default)
        :param str mode: The drawing mode.
        :param Union[List[int],None] indices:
            Array of vertex indices or None to use arrays directly.
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """
        assert mode in ('triangles', 'triangle_strip', 'fan')
        if position is None or len(position) == 0:
            mesh = None
        else:
            mesh = primitives.ColormapMesh3D(
                position=position,
                value=numpy.array(value, copy=False).reshape(-1, 1),  # Make it a 2D array
                colormap=self._getSceneColormap(),
                normal=normal,
                mode=mode,
                indices=indices,
                copy=copy)
        self._setMesh(mesh)

        self._setColormappedData(self.getValueData(copy=False), copy=False)

    def getData(self, copy=True):
        """Get the mesh geometry.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The positions, values, normals and mode
        :rtype: tuple of numpy.ndarray
        """
        return (self.getPositionData(copy=copy),
                self.getValueData(copy=copy),
                self.getNormalData(copy=copy),
                self.getDrawMode())

    def getValueData(self, copy=True):
        """Get the mesh vertex values.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: Array of data values
        :rtype: numpy.ndarray
        """
        if self._getMesh() is None:
            return numpy.empty((0,), dtype=numpy.float32)
        else:
            return self._getMesh().getAttribute('value', copy=copy)


class _CylindricalVolume(DataItem3D):
    """Class that represents a volume with a rotational symmetry along z

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        self._mesh = None
        self._nbFaces = 0

    def getPosition(self, copy=True):
        """Get primitive positions.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: Position of the primitives as a (N, 3) array.
        :rtype: numpy.ndarray
        """
        raise NotImplementedError("Must be implemented in subclass")

    def _setData(self, position, radius, height, angles, color, flatFaces,
                 rotation):
        """Set volume geometry data.

        :param numpy.ndarray position:
            Center position (x, y, z) of each volume as (N, 3) array.
        :param float radius: External radius ot the volume.
        :param float height: Height of the volume(s).
        :param numpy.ndarray angles: Angles of the edges.
        :param numpy.array color: RGB color of the volume(s).
        :param bool flatFaces:
            If the volume as flat faces or not. Used for normals calculation.
        """

        self._getScenePrimitive().children = []  # Remove any previous mesh

        if position is None or len(position) == 0:
            self._mesh = None
            self._nbFaces = 0
        else:
            self._nbFaces = len(angles) - 1

            volume = numpy.empty(shape=(len(angles) - 1, 12, 3),
                                 dtype=numpy.float32)
            normal = numpy.empty(shape=(len(angles) - 1, 12, 3),
                                 dtype=numpy.float32)

            for i in range(0, len(angles) - 1):
                #       c6
                #       /\
                #      /  \
                #     /    \
                #  c4|------|c5
                #    | \    |
                #    |  \   |
                #    |   \  |
                #    |    \ |
                #  c2|------|c3
                #     \    /
                #      \  /
                #       \/
                #       c1
                c1 = numpy.array([0, 0, -height/2])
                c1 = rotation.transformPoint(c1)
                c2 = numpy.array([radius * numpy.cos(angles[i]),
                                  radius * numpy.sin(angles[i]),
                                  -height/2])
                c2 = rotation.transformPoint(c2)
                c3 = numpy.array([radius * numpy.cos(angles[i+1]),
                                  radius * numpy.sin(angles[i+1]),
                                  -height/2])
                c3 = rotation.transformPoint(c3)
                c4 = numpy.array([radius * numpy.cos(angles[i]),
                                  radius * numpy.sin(angles[i]),
                                  height/2])
                c4 = rotation.transformPoint(c4)
                c5 = numpy.array([radius * numpy.cos(angles[i+1]),
                                  radius * numpy.sin(angles[i+1]),
                                  height/2])
                c5 = rotation.transformPoint(c5)
                c6 = numpy.array([0, 0, height/2])
                c6 = rotation.transformPoint(c6)

                volume[i] = numpy.array([c1, c3, c2,
                                         c2, c3, c4,
                                         c3, c5, c4,
                                         c4, c5, c6])
                if flatFaces:
                    normal[i] = numpy.array([numpy.cross(c3-c1, c2-c1),  # c1
                                             numpy.cross(c2-c3, c1-c3),  # c3
                                             numpy.cross(c1-c2, c3-c2),  # c2
                                             numpy.cross(c3-c2, c4-c2),  # c2
                                             numpy.cross(c4-c3, c2-c3),  # c3
                                             numpy.cross(c2-c4, c3-c4),  # c4
                                             numpy.cross(c5-c3, c4-c3),  # c3
                                             numpy.cross(c4-c5, c3-c5),  # c5
                                             numpy.cross(c3-c4, c5-c4),  # c4
                                             numpy.cross(c5-c4, c6-c4),  # c4
                                             numpy.cross(c6-c5, c5-c5),  # c5
                                             numpy.cross(c4-c6, c5-c6)])  # c6
                else:
                    normal[i] = numpy.array([numpy.cross(c3-c1, c2-c1),
                                             numpy.cross(c2-c3, c1-c3),
                                             numpy.cross(c1-c2, c3-c2),
                                             c2-c1, c3-c1, c4-c6,  # c2 c2 c4
                                             c3-c1, c5-c6, c4-c6,  # c3 c5 c4
                                             numpy.cross(c5-c4, c6-c4),
                                             numpy.cross(c6-c5, c5-c5),
                                             numpy.cross(c4-c6, c5-c6)])

            # Multiplication according to the number of positions
            vertices = numpy.tile(volume.reshape(-1, 3), (len(position), 1))\
                .reshape((-1, 3))
            normals = numpy.tile(normal.reshape(-1, 3), (len(position), 1))\
                .reshape((-1, 3))

            # Translations
            numpy.add(vertices, numpy.tile(position, (1, (len(angles)-1) * 12))
                      .reshape((-1, 3)), out=vertices)

            # Colors
            if numpy.ndim(color) == 2:
                color = numpy.tile(color, (1, 12 * (len(angles) - 1)))\
                    .reshape(-1, 3)

            self._mesh = primitives.Mesh3D(
                vertices, color, normals, mode='triangles', copy=False)
            self._getScenePrimitive().children.append(self._mesh)

        self._updated(ItemChangedType.DATA)

    def _pickFull(self, context):
        """Perform precise picking in this item at given widget position.

        :param PickContext context: Current picking context
        :return: Object holding the results or None
        :rtype: Union[None,PickingResult]
        """
        if self._mesh is None or self._nbFaces == 0:
            return None

        rayObject = context.getPickingSegment(frame=self._getScenePrimitive())
        if rayObject is None:  # No picking outside viewport
            return None
        rayObject = rayObject[:, :3]

        positions = self._mesh.getAttribute('position', copy=False)
        triangles = positions.reshape(-1, 3, 3)  # 'triangle' draw mode

        trianglesIndices, t = glu.segmentTrianglesIntersection(
            rayObject, triangles)[:2]

        if len(trianglesIndices) == 0:
            return None

        # Get object index from triangle index
        indices = trianglesIndices // (4 * self._nbFaces)

        # Select closest intersection point for each primitive
        indices, firstIndices = numpy.unique(indices, return_index=True)
        t = t[firstIndices]

        # Resort along t as result of numpy.unique is not sorted by t
        sortedIndices = numpy.argsort(t)
        t = t[sortedIndices]
        indices = indices[sortedIndices]

        points = t.reshape(-1, 1) * (rayObject[1] - rayObject[0]) + rayObject[0]

        return PickingResult(self,
                             positions=points,
                             indices=indices,
                             fetchdata=self.getPosition)


class Box(_CylindricalVolume):
    """Description of a box.

    Can be used to draw one box or many similar boxes.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        super(Box, self).__init__(parent)
        self.position = None
        self.size = None
        self.color = None
        self.rotation = None
        self.setData()

    def setData(self, size=(1, 1, 1), color=(1, 1, 1),
                position=(0, 0, 0), rotation=(0, (0, 0, 0))):
        """
        Set Box geometry data.

        :param numpy.array size: Size (dx, dy, dz) of the box(es).
        :param numpy.array color: RGB color of the box(es).
        :param numpy.ndarray position:
            Center position (x, y, z) of each box as a (N, 3) array.
        :param tuple(float, array) rotation:
                Angle (in degrees) and axis of rotation.
                If (0, (0, 0, 0)) (default), the hexagonal faces are on
                xy plane and a side face is aligned with x axis.
        """
        self.position = numpy.atleast_2d(numpy.array(position, copy=True))
        self.size = numpy.array(size, copy=True)
        self.color = numpy.array(color, copy=True)
        self.rotation = Rotate(rotation[0],
                               rotation[1][0], rotation[1][1], rotation[1][2])

        assert (numpy.ndim(self.color) == 1 or
                len(self.color) == len(self.position))

        diagonal = numpy.sqrt(self.size[0]**2 + self.size[1]**2)
        alpha = 2 * numpy.arcsin(self.size[1] / diagonal)
        beta = 2 * numpy.arcsin(self.size[0] / diagonal)
        angles = numpy.array([0,
                              alpha,
                              alpha + beta,
                              alpha + beta + alpha,
                              2 * numpy.pi])
        numpy.subtract(angles, 0.5 * alpha, out=angles)
        self._setData(self.position,
                      numpy.sqrt(self.size[0]**2 + self.size[1]**2)/2,
                      self.size[2],
                      angles,
                      self.color,
                      True,
                      self.rotation)

    def getPosition(self, copy=True):
        """Get box(es) position(s).

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: Position of the box(es) as a (N, 3) array.
        :rtype: numpy.ndarray
        """
        return numpy.array(self.position, copy=copy)

    def getSize(self):
        """Get box(es) size.

        :return: Size (dx, dy, dz) of the box(es).
        :rtype: numpy.ndarray
        """
        return numpy.array(self.size, copy=True)

    def getColor(self, copy=True):
        """Get box(es) color.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: RGB color of the box(es).
        :rtype: numpy.ndarray
        """
        return numpy.array(self.color, copy=copy)


class Cylinder(_CylindricalVolume):
    """Description of a cylinder.

    Can be used to draw one cylinder or many similar cylinders.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        super(Cylinder, self).__init__(parent)
        self.position = None
        self.radius = None
        self.height = None
        self.color = None
        self.nbFaces = 0
        self.rotation = None
        self.setData()

    def setData(self, radius=1, height=1, color=(1, 1, 1), nbFaces=20,
                position=(0, 0, 0), rotation=(0, (0, 0, 0))):
        """
        Set the cylinder geometry data

        :param float radius: Radius of the cylinder(s).
        :param float height: Height of the cylinder(s).
        :param numpy.array color: RGB color of the cylinder(s).
        :param int nbFaces:
            Number of faces for cylinder approximation (default 20).
        :param numpy.ndarray position:
            Center position (x, y, z) of each cylinder as a (N, 3) array.
        :param tuple(float, array) rotation:
                Angle (in degrees) and axis of rotation.
                If (0, (0, 0, 0)) (default), the hexagonal faces are on
                xy plane and a side face is aligned with x axis.
        """
        self.position = numpy.atleast_2d(numpy.array(position, copy=True))
        self.radius = float(radius)
        self.height = float(height)
        self.color = numpy.array(color, copy=True)
        self.nbFaces = int(nbFaces)
        self.rotation = Rotate(rotation[0],
                               rotation[1][0], rotation[1][1], rotation[1][2])

        assert (numpy.ndim(self.color) == 1 or
                len(self.color) == len(self.position))

        angles = numpy.linspace(0, 2*numpy.pi, self.nbFaces + 1)
        self._setData(self.position,
                      self.radius,
                      self.height,
                      angles,
                      self.color,
                      False,
                      self.rotation)

    def getPosition(self, copy=True):
        """Get cylinder(s) position(s).

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: Position(s) of the cylinder(s) as a (N, 3) array.
        :rtype: numpy.ndarray
        """
        return numpy.array(self.position, copy=copy)

    def getRadius(self):
        """Get cylinder(s) radius.

        :return: Radius of the cylinder(s).
        :rtype: float
        """
        return self.radius

    def getHeight(self):
        """Get cylinder(s) height.

        :return: Height of the cylinder(s).
        :rtype: float
        """
        return self.height

    def getColor(self, copy=True):
        """Get cylinder(s) color.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: RGB color of the cylinder(s).
        :rtype: numpy.ndarray
        """
        return numpy.array(self.color, copy=copy)


class Hexagon(_CylindricalVolume):
    """Description of a uniform hexagonal prism.

    Can be used to draw one hexagonal prim or many similar hexagonal
    prisms.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        super(Hexagon, self).__init__(parent)
        self.position = None
        self.radius = 0
        self.height = 0
        self.color = None
        self.rotation = None
        self.setData()

    def setData(self, radius=1, height=1, color=(1, 1, 1),
                position=(0, 0, 0), rotation=(0, (0, 0, 0))):
        """
        Set the uniform hexagonal prism geometry data

        :param float radius: External radius of the hexagonal prism
        :param float height: Height of the hexagonal prism
        :param numpy.array color: RGB color of the prism(s)
        :param numpy.ndarray position:
            Center position (x, y, z) of each prism as a (N, 3) array
        :param tuple(float, array) rotation:
                Angle (in degrees) and axis of rotation.
                If (0, (0, 0, 0)) (default), the hexagonal faces are on
                xy plane and a side face is aligned with x axis.
        """
        self.position = numpy.atleast_2d(numpy.array(position, copy=True))
        self.radius = float(radius)
        self.height = float(height)
        self.color = numpy.array(color, copy=True)
        self.rotation = Rotate(rotation[0], rotation[1][0], rotation[1][1],
                               rotation[1][2])

        assert (numpy.ndim(self.color) == 1 or
                len(self.color) == len(self.position))

        angles = numpy.linspace(0, 2*numpy.pi, 7)
        self._setData(self.position,
                      self.radius,
                      self.height,
                      angles,
                      self.color,
                      True,
                      self.rotation)

    def getPosition(self, copy=True):
        """Get hexagonal prim(s) position(s).

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
         :return: Position(s) of hexagonal prism(s) as a (N, 3) array.
         :rtype: numpy.ndarray
         """
        return numpy.array(self.position, copy=copy)

    def getRadius(self):
        """Get hexagonal prism(s) radius.

        :return: Radius of hexagon(s).
        :rtype: float
        """
        return self.radius

    def getHeight(self):
        """Get hexagonal prism(s) height.

        :return: Height of hexagonal prism(s).
        :rtype: float
        """
        return self.height

    def getColor(self, copy=True):
        """Get hexagonal prism(s) color.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: RGB color of the hexagonal prism(s).
        :rtype: numpy.ndarray
        """
        return numpy.array(self.color, copy=copy)
