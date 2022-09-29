# /*##########################################################################
#
# Copyright (c) 2019-2021 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""
This module provides a class to render a set of 2D triangles
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


import ctypes

import numpy

from .....math.combo import min_max
from .... import _glutils as glutils
from ...._glutils import gl
from .GLPlotItem import GLPlotItem


class GLPlotTriangles(GLPlotItem):
    """Handle rendering of a set of colored triangles"""

    _PROGRAM = glutils.Program(
        vertexShader="""
        #version 120

        uniform mat4 matrix;
        attribute float xPos;
        attribute float yPos;
        attribute vec4 color;

        varying vec4 vColor;

        void main(void) {
            gl_Position = matrix * vec4(xPos, yPos, 0.0, 1.0);
            vColor = color;
        }
        """,
        fragmentShader="""
        #version 120

        uniform float alpha;
        varying vec4 vColor;

        void main(void) {
            gl_FragColor = vColor;
            gl_FragColor.a *= alpha;
        }
        """,
        attrib0='xPos')

    def __init__(self, x, y, color, triangles, alpha=1.):
        """

        :param numpy.ndarray x: X coordinates of triangle corners
        :param numpy.ndarray y: Y coordinates of triangle corners
        :param numpy.ndarray color: color for each point
        :param numpy.ndarray triangles: (N, 3) array of indices of triangles
        :param float alpha: Opacity in [0, 1]
        """
        super().__init__()
        # Check and convert input data
        x = numpy.ravel(numpy.array(x, dtype=numpy.float32))
        y = numpy.ravel(numpy.array(y, dtype=numpy.float32))
        color = numpy.array(color, copy=False)
        # Cast to uint32
        triangles = numpy.array(triangles, copy=False, dtype=numpy.uint32)

        assert x.size == y.size
        assert x.size == len(color)
        assert color.ndim == 2 and color.shape[1] in (3, 4)
        if numpy.issubdtype(color.dtype, numpy.floating):
            color = numpy.array(color, dtype=numpy.float32, copy=False)
        elif numpy.issubdtype(color.dtype, numpy.integer):
            color = numpy.array(color, dtype=numpy.uint8, copy=False)
        else:
            raise ValueError('Unsupported color type')
        assert triangles.ndim == 2 and triangles.shape[1] == 3

        self.__x_y_color = x, y, color
        self.xMin, self.xMax = min_max(x, finite=True)
        self.yMin, self.yMax = min_max(y, finite=True)
        self.__triangles = triangles
        self.__alpha = numpy.clip(float(alpha), 0., 1.)
        self.__vbos = None
        self.__indicesVbo = None
        self.__picking_triangles = None

    def pick(self, x, y):
        """Perform picking

        :param float x: X coordinates in plot data frame
        :param float y: Y coordinates in plot data frame
        :return: List of picked data point indices
        :rtype: Union[List[int],None]
        """
        if (x < self.xMin or x > self.xMax or
                y < self.yMin or y > self.yMax):
            return None

        xPts, yPts = self.__x_y_color[:2]
        if self.__picking_triangles is None:
            self.__picking_triangles = numpy.zeros(
                self.__triangles.shape + (3,), dtype=numpy.float32)
            self.__picking_triangles[:, :, 0] = xPts[self.__triangles]
            self.__picking_triangles[:, :, 1] = yPts[self.__triangles]

        segment = numpy.array(((x, y, -1), (x, y, 1)), dtype=numpy.float32)
        # Picked triangle indices
        indices = glutils.segmentTrianglesIntersection(
            segment, self.__picking_triangles)[0]
        # Point indices
        indices = numpy.unique(numpy.ravel(self.__triangles[indices]))

        # Sorted from furthest to closest point
        dists = (xPts[indices] - x) ** 2 + (yPts[indices] - y) ** 2
        indices = indices[numpy.flip(numpy.argsort(dists), axis=0)]

        return tuple(indices) if len(indices) > 0 else None

    def discard(self):
        """Release resources on the GPU"""
        if self.isInitialized():
            self.__vbos[0].vbo.discard()
            self.__vbos = None
            self.__indicesVbo.discard()
            self.__indicesVbo = None

    def isInitialized(self):
        return self.__vbos is not None

    def prepare(self):
        """Allocate resources on the GPU"""
        if self.__vbos is None:
            self.__vbos = glutils.vertexBuffer(self.__x_y_color)
            # Normalization is need for color
            self.__vbos[-1].normalization = True

        if self.__indicesVbo is None:
            self.__indicesVbo = glutils.VertexBuffer(
                numpy.ravel(self.__triangles),
                usage=gl.GL_STATIC_DRAW,
                target=gl.GL_ELEMENT_ARRAY_BUFFER)

    def render(self, context):
        """Perform rendering

        :param RenderContext context: Rendering information
        """
        self.prepare()

        if self.__vbos is None or self.__indicesVbo is None:
            return  # Nothing to display

        self._PROGRAM.use()

        gl.glUniformMatrix4fv(self._PROGRAM.uniforms['matrix'],
                              1,
                              gl.GL_TRUE,
                              context.matrix.astype(numpy.float32))

        gl.glUniform1f(self._PROGRAM.uniforms['alpha'], self.__alpha)

        for index, name in enumerate(('xPos', 'yPos', 'color')):
            attr = self._PROGRAM.attributes[name]
            gl.glEnableVertexAttribArray(attr)
            self.__vbos[index].setVertexAttrib(attr)

        with self.__indicesVbo:
            gl.glDrawElements(gl.GL_TRIANGLES,
                              self.__triangles.size,
                              glutils.numpyToGLType(self.__triangles.dtype),
                              ctypes.c_void_p(0))
