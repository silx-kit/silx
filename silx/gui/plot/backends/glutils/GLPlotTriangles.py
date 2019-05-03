# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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


import numpy


from ...._glutils import gl, Program, vertexBuffer


class GLPlotTriangles(object):

    _PROGRAM = Program(
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

        varying vColor;

        void main(void) {
            gl_FragColor = vColor;
        }
        """,
        attrib0='xPos')

    def __init__(self, xData=None, yData=None, color=None):
        self.__x_y_color = xData, yData, color
        self.__vbos = None

    def discard(self):
        if self.__vbos is not None:
            self.__vbos[0].vbo.discard()
            self.__vbos = None

    def prepare(self):
        if self.__vbos is None and None not in self.__x_y_color:
            self.__vbos = vertexBuffer(self.__x_y_color)

    def render(self, matrix):
        """Perform rendering

        :param numpy.ndarray matrix: 4x4 transform matrix to use
        """
        self.prepare()

        if self.__vbos is None:
            return  # Nothing to display

        self._PROGRAM.use()

        gl.glUniformMatrix4fv(
            self._PROGRAM.uniforms['matrix'], 1, gl.GL_TRUE,
            matrix.astype(numpy.float32))

        for index, name in enumerate(('xPos', 'yPos', 'color')):
            attr = self._PROGRAM.attributes[name]
            gl.glEnableVertexAttribArray(attr)
            self.__vbos[index].setVertexAttrib(attr)

        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, self.__vbos[0].size)
