from __future__ import absolute_import

import numpy

from ..scene import primitives
from .core import DataItem3D, ItemChangedType


class Cylinder(DataItem3D):

    N = 20  # Number of faces for cylinder approximation

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        self._mesh = None

    def setData(self,
                position,
                radius,
                height,
                color,
                normal=None,
                copy=True):

        self._getScenePrimitive().children = []  # Remove any previous mesh

        if position is None or len(position) == 0:
            self._mesh = 0
        else:
            # Definition of vertices to draw 1 cylinder
            alpha = (2*numpy.pi/Cylinder.N)
            center = numpy.array([0, 0, -height/2])
            side1 = numpy.array([radius, 0, -height/2])
            side2 = numpy.array([radius * numpy.cos(alpha), radius * numpy.sin(alpha), -height/2])
            cylinder = numpy.ndarray(shape=(Cylinder.N, 12, 3), dtype=numpy.float32)
            angle = alpha
            for i in range(0, Cylinder.N):
                cylinder[i] = numpy.array([center, side1, side2,
                                           side1, side1, side2,
                                           side1, side2, side2,
                                           side1, center, side2])
                cylinder[i][4][2] = height/2
                cylinder[i][5][2] = height/2
                cylinder[i][8][2] = height/2
                cylinder[i][9][2] = height/2
                cylinder[i][11][2] = height/2
                angle = angle + 2 * numpy.pi / Cylinder.N
                side1 = side2
                side2 = numpy.array([radius * numpy.cos(angle), radius * numpy.sin(angle), -height/2])

            # add all the cylinders to vertices
            vertices = numpy.ndarray(shape=(len(position), Cylinder.N, 12, 3), dtype=numpy.float32)
            for i in range(0, len(position)):
                numpy.add(cylinder, position[i], out=vertices[i])
            vertices = numpy.reshape(vertices, (-1, 3))
            self._mesh = primitives.Mesh3D(
                vertices, color, normal, mode='triangles', copy=copy)
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
