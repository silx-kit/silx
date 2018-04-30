# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/11/2017"

import numpy

from ..scene import primitives
from .core import DataItem3D, ItemChangedType
from ..scene.utils import trianglesNormal


class Mesh(DataItem3D):
    """Description of mesh.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        self._mesh = None

    def setData(self,
                position,
                color,
                normal=None,
                mode='triangles',
                copy=True):
        """Set mesh geometry data.

        Supported drawing modes are:

        - For points: 'points'
        - For lines: 'lines', 'line_strip', 'loop'
        - For triangles: 'triangles', 'triangle_strip', 'fan'

        :param numpy.ndarray position:
            Position (x, y, z) of each vertex as a (N, 3) array
        :param numpy.ndarray color: Colors for each point or a single color
        :param numpy.ndarray normal: Normals for each point or None (default)
        :param str mode: The drawing mode.
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """
        self._getScenePrimitive().children = []  # Remove any previous mesh

        if position is None or len(position) == 0:
            self._mesh = 0
        else:
            self._mesh = primitives.Mesh3D(
                position, color, normal, mode=mode, copy=copy)
            self._getScenePrimitive().children.append(self._mesh)

        self.sigItemChanged.emit(ItemChangedType.DATA)

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

    def getPositionData(self, copy=True):
        """Get the mesh vertex positions.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The (x, y, z) positions as a (N, 3) array
        :rtype: numpy.ndarray
        """
        if self._mesh is None:
            return numpy.empty((0, 3), dtype=numpy.float32)
        else:
            return self._mesh.getAttribute('position', copy=copy)

    def getColorData(self, copy=True):
        """Get the mesh vertex colors.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The RGBA colors as a (N, 4) array or a single color
        :rtype: numpy.ndarray
        """
        if self._mesh is None:
            return numpy.empty((0, 4), dtype=numpy.float32)
        else:
            return self._mesh.getAttribute('color', copy=copy)

    def getNormalData(self, copy=True):
        """Get the mesh vertex normals.

        :param bool copy:
            True (default) to get a copy,
            False to get internal representation (do not modify!).
        :return: The normals as a (N, 3) array, a single normal or None
        :rtype: numpy.ndarray or None
        """
        if self._mesh is None:
            return None
        else:
            return self._mesh.getAttribute('normal', copy=copy)

    def getDrawMode(self):
        """Get mesh rendering mode.

        :return: The drawing mode of this primitive
        :rtype: str
        """
        return self._mesh.drawMode


class _CylindricalVolume(DataItem3D):
    """Class that represents a volume with a rotational symmetry along z

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        self._mesh = None

    def _setData(self, position, radius, height, color, nbFaces, flatFaces,
                 phase, copy=True):
        """Set volume geometry data.

        :param numpy.ndarray position:
            Center position (x, y, z) of each volume as (N, 3) array.
        :param float radius: External radius ot the volume.
        :param float height: Height of the volume(s).
        :param numpy.array color: RGB color of the volume(s).
        :param int nbFaces: Number of faces (symmetry order).
        :param bool flatFaces:
            If the volume as flat faces or not. Used for normals calculation.
        :param float phase:
            rotation angle (in degrees) of the volume along z axis.
            If 0, the first edge is on y = 0.
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """

        self._getScenePrimitive().children = []  # Remove any previous mesh
        self.N = nbFaces

        if position is None or len(position) == 0:
            self._mesh = 0
        else:
            phase = numpy.deg2rad(phase)
            alpha = (2 * numpy.pi / self.N) + phase
            """
                   c6
                   /\
                  /  \
                 /    \
              c4|------|c5
                | \    |
                |  \   |
                |   \  |
                |    \ |
              c2|------|c3
                 \    /
                  \  /
                   \/
                   c1     
            """
            c1 = numpy.array([0, 0, -height/2])
            c2 = numpy.array([radius * numpy.cos(phase),
                              radius * numpy.sin(phase), -height/2])
            c3 = numpy.array([radius * numpy.cos(alpha),
                              radius * numpy.sin(alpha), -height/2])
            c4 = numpy.array([radius * numpy.cos(phase),
                              radius * numpy.sin(phase), height/2])
            c5 = numpy.array([radius * numpy.cos(alpha),
                              radius * numpy.sin(alpha), height/2])
            c6 = numpy.array([0, 0, height/2])

            # One volume
            volume = numpy.ndarray(shape=(self.N, 12, 3), dtype=numpy.float32)
            normal = numpy.ndarray(shape=(self.N, 12, 3), dtype=numpy.float32)
            for i in range(0, self.N):
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
                alpha += 2 * numpy.pi / self.N
                c2 = c3
                c4 = c5
                c3 = numpy.array([radius * numpy.cos(alpha),
                                  radius * numpy.sin(alpha), -height/2])
                c5 = numpy.array([radius * numpy.cos(alpha),
                                  radius * numpy.sin(alpha), height/2])

            # Multiplication according to the number of positions
            vertices = numpy.tile(volume.reshape(-1, 3), (len(position), 1))\
                .reshape((-1, 3))
            normals = numpy.tile(normal.reshape(-1, 3), (len(position), 1))\
                .reshape((-1, 3))

            # Translations
            numpy.add(vertices, numpy.tile(position, (1, self.N * 12))
                      .reshape((-1, 3)), out=vertices)

            self._mesh = primitives.Mesh3D(
                vertices, color, normals, mode='triangles', copy=copy)
            self._getScenePrimitive().children.append(self._mesh)

        self.sigItemChanged.emit(ItemChangedType.DATA)


class Box(_CylindricalVolume):
    """Description of a box.
    Can be used to draw one box or an array of the same box.

    :param parent: The View widget this item belongs to.
    """

    def __init__(self, parent=None):
        super(Box, self).__init__(parent)

    def setData(self, position, size, color, phase=45, copy=True):
        """
        Set Box geometry data.

        :param numpy.ndarray position:
            Center position (x, y, z) of each box as a (N, 3) array.
        :param numpy.array size: Size (dx, dy, dz) of the box(es).
        :param numpy.array color: RGB color of the box(es).
        :param float phase:
            Rotation angle (in degrees) of the volume along z (default 45).
            If 0, the first edge is on y = 0.
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """
        self._setData(position, numpy.sqrt(size[0]**2 + size[1]**2)/2, size[2],
                      color, 4, True, phase, copy)


class Cylinder(_CylindricalVolume):
    """Description of a cylinder.
    Can be used to draw one cylinder or an array of the same cylinder.

    :param parent: The View widget this item belongs to.
    """
    def __init__(self, parent=None):
        super(Cylinder, self).__init__(parent)

    def setData(self, position, radius, height, color, nbFaces=20, copy=True):
        """
        Set the cylinder geometry data

        :param numpy.ndarray position:
            Center position (x, y, z) of each cylinder as a (N, 3) array.
        :param float radius: Radius of the cylinder(s).
        :param float height: Height of the cylinder(s).
        :param numpy.array color: RGB color of the cylinder(s).
        :param int nbFaces:
            Number of faces for cylinder approximation (default 20).
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """
        self._setData(position, radius, height, color, nbFaces, False, 0, copy)


class Hexagon(_CylindricalVolume):
    """Description of a uniform hexagonal prism.
    Can be used to draw one hexagonal prim or an array of the same hexagonal
    prism.

    :param parent: The View widget this item belongs to.
    """
    def __init__(self, parent=None):
        super(Hexagon, self).__init__(parent)

    def setData(self, position, radius, height, color, phase=0, copy=True):
        """
        Set the uniform hexagonal prism geometry data

        :param numpy.ndarray position:
            Center position (x, y, z) of each prism as a (N, 3) array
        :param float radius: External radius of the hexagonal prism
        :param float height: Height of the hexagonal prism
        :param numpy.array color: RGB color of the prism(s)
        :param float phase:
                Rotation angle (in degrees) of the prism(s).
                If 0 (default), the first edge is on y = 0.
        :param bool copy: True (default) to copy the data,
                          False to use as is (do not modify!).
        """
        self._setData(position, radius, height, color, 6, True, phase, copy)
