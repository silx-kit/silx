# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
"""This module provides a scene object rendering an isosurface."""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import logging
import time
import numpy

from silx.math.marchingcubes import MarchingCubes

from . import primitives
from . import core
from . import utils


_logger = logging.getLogger(__name__)


# IsoSurface ##################################################################

class IsoSurface(core.PrivateGroup):
    """An isosurface that can be rendered as points, lines or triangles.

    :param data: numpy.ndarray-like 3D dataset of float.
    :param float isolevel: The value of the isosurface to generate.
    :param str renderMode: Kind of rendering in: 'mesh', 'lines', 'points'.
    :param color: The color of the iso-surface.
    :type color: 4-tuple of float in [0., 1.].
    :param float size: The width of lines or the size on points in pixels.
    :param bool normalsInverted: Whether to invert normal orientation
                                 or not (the default).
    :param bool showNormals: Whether to show normals as lines
                             or not (the default).
    :param bool copy: Whether to make a copy of the data (the default)
                      or not.
    """

    _RENDER_MODES = 'mesh', 'lines', 'points'

    def __init__(self, data, isolevel,
                 renderMode='mesh', color=(1., 1., 1., 1.), size=1.,
                 normalsInverted=False, showNormals=False,
                 copy=True):
        assert renderMode in self._RENDER_MODES

        super(IsoSurface, self).__init__()
        self._data = numpy.array(data, order='C', copy=copy)
        self._isolevel = isolevel
        self._renderMode = renderMode
        self._color = color
        self._size = size
        self._normalsInverted = normalsInverted
        self._showNormals = showNormals

        self._update()  # TODO Move in prepare?, better in a thread...

    def getData(self, copy=True):
        """Data used to compute the isosurface.

        :param bool copy: True (the default) to get a copy of the array,
                          False to get the internal array, do not modify!
        :return: Data array as provided to constructor.
        """
        if copy:
            return self._data
        else:
            return self._data.copy()

    @property
    def isolevel(self):
        """The value of the iso-surface."""
        return self._isolevel

    @isolevel.setter
    def isolevel(self, isolevel):
        isolevel = float(isolevel)
        if isolevel != self._isolevel:
            self._isolevel = isolevel
            self._update()

    @property
    def renderMode(self):
        """The rendering of the isosurface: 'mesh', 'lines' or 'points'."""
        return self._renderMode

    @renderMode.setter
    def renderMode(self, mode):
        assert mode in ('mesh', 'lines', 'points')
        self._renderMode = mode
        self._update()  # TODO Quick and dirty support, far from optimum!

    @property
    def color(self):
        """The color of the iso-surface (4-tuple of float in [0., 1.])."""
        return self._color

    @color.setter
    def color(self, color):
        self._color = color
        self._update()  # TODO Quick and dirty support, far from optimum!

    @property
    def size(self):
        """Line width or point size in pixels."""
        return self._size

    @property
    def normalsInverted(self):
        return self._normalsInverted

    def _update(self):
        """Generate the iso-surface."""
        start = time.time()
        vertices, normals, indices = MarchingCubes(
            self.getData(copy=False), self.isolevel,
            invert_normals=self._normalsInverted)

        if self.renderMode == 'lines':
            indices = utils.triangleToLineIndices(indices, unicity=True)

        _logger.info('Marching Cubes duration %fs Nb vertices %d',
                     time.time() - start, len(vertices))

        self._children = []
        if len(vertices) != 0:
            if self.renderMode == 'mesh':
                mesh = primitives.Mesh3D(vertices,
                                         colors=self.color,
                                         normals=normals,
                                         mode='triangles',
                                         indices=indices)
                self._children.append(mesh)

            elif self.renderMode == 'lines':
                lines = primitives.Lines(vertices,
                                         indices=indices,
                                         normals=normals,
                                         mode='lines',
                                         width=self.size,
                                         colors=self.color)
                self._children.append(lines)

            elif self.renderMode == 'points':
                points = primitives.ColorPoints(vertices,
                                                colors=self.color,
                                                sizes=self.size)
                self._children.append(points)

            if self._showNormals:
                # Normals as Lines
                normallines = utils.verticesNormalsToLines(
                    vertices, normals)
                normalslines = primitives.Lines(normallines,
                                                mode='lines',
                                                width=1.,
                                                colors=(1., 1., 0., 1.))
                self._children.append(normalslines)

    def _bounds(self, dataBounds=False):
        if dataBounds:  # TODO check order of axes?
            depth, height, width = self._data.shape
            return numpy.array(((0., 0., 0.),
                                (depth - 1., height - 1., width - 1.)),
                               dtype=numpy.float32)
        else:
            super(IsoSurface, self)._bounds(dataBounds)
