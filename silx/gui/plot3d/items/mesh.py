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

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        self._mesh = None

    def _setData(self, position, radius, height, color, nbFaces=0, normal=None, copy=True):

        self._getScenePrimitive().children = []  # Remove any previous mesh
        self.N = nbFaces

        if position is None or len(position) == 0:
            self._mesh = 0
        else:
            alpha = (2 * numpy.pi / self.N)
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
            c1 = numpy.array([0,                         0,                         -height/2])
            c2 = numpy.array([radius,                    0,                         -height/2])
            c3 = numpy.array([radius * numpy.cos(alpha), radius * numpy.sin(alpha), -height/2])
            c4 = numpy.array([radius,                    0,                         height/2])
            c5 = numpy.array([radius * numpy.cos(alpha), radius * numpy.sin(alpha), height/2])
            c6 = numpy.array([0,                         0,                         height/2])

            volume = numpy.ndarray(shape=(self.N, 12, 3), dtype=numpy.float32)
            for i in range(0, self.N):
                volume[i] = numpy.array([c1, c2, c3,
                                         c2, c4, c3,
                                         c3, c4, c5,
                                         c4, c6, c5])
                alpha += 2 * numpy.pi / self.N
                c2 = c3
                c4 = c5
                c3 = numpy.array([radius * numpy.cos(alpha), radius * numpy.sin(alpha), -height/2])
                c5 = numpy.array([radius * numpy.cos(alpha), radius * numpy.sin(alpha), height/2])

            # construct one volume at each position
            vertices = numpy.ndarray(shape=(len(position), self.N, 12, 3), dtype=numpy.float32)
            for i in range(0, len(position)):
                numpy.add(volume, position[i], out=vertices[i])
            vertices = numpy.reshape(vertices, (-1, 3))

            self._mesh = primitives.Mesh3D(
                vertices, color, normal, mode='triangles', copy=copy)
            self._getScenePrimitive().children.append(self._mesh)

        self.sigItemChanged.emit(ItemChangedType.DATA)


class Box(_CylindricalVolume):

    def __init__(self, parent=None):
        super(Box, self).__init__(parent)

    def setData(self, position, size, color, normal=None, copy=True):
        self._setData(position, size[0]/2, size[2], color, 4, normal, copy)


class Cylinder(_CylindricalVolume):

    def __init__(self, parent=None):
        super(Cylinder, self).__init__(parent)

    def setData(self, position, radius, height, color, nbFaces=20, normal=None, copy=True):
        self._setData(position, radius, height, color, nbFaces, normal, copy)


class Hexagon(_CylindricalVolume):

    def __init__(self, parent=None):
        super(Hexagon, self).__init__(parent)

    def setData(self, position, radius, height, color, normal=None, copy=True):
        self._setData(position, radius, height, color, 6, normal, copy)
