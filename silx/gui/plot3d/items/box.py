from __future__ import absolute_import

import numpy

from ..scene import primitives
from .core import DataItem3D, ItemChangedType


class Box(DataItem3D):

    def __init__(self, parent=None):
        DataItem3D.__init__(self, parent=parent)
        self._mesh = None

    def setData(self,
                position,
                size,
                color,
                normal=None,
                copy=True):

        # Define the vertices and their order to draw 1 cube (triangle strip mode)
        drawingPath = numpy.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 1., 0.],
            [1., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 1., 1.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 0., 1.]
        ])

        # Scaling
        numpy.multiply(drawingPath, size, out=drawingPath)

        self._getScenePrimitive().children = []  # Remove any previous mesh

        if position is None or len(position) == 0:
            self._mesh = 0
        else:
            # Compute vertices for all the cubes
            vertices = numpy.ndarray(shape=(len(position), 16, 3), dtype=numpy.float32)
            for i in range(0, len(position)):
                numpy.add(drawingPath, position[i], out=vertices[i])
            vertices = numpy.reshape(vertices, (-1, 3))
            self._mesh = primitives.Mesh3D(
                vertices, color, normal, mode='triangle_strip', copy=copy)
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
