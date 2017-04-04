# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2017 European Synchrotron Radiation Facility
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
This module provides minimalistic text support for OpenGL.
It provides Latin-1 (ISO8859-1) characters for one monospace font at one size.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


import numpy
from ctypes import c_void_p, sizeof, c_float
from ...._glutils import gl, getGLContext, Program
from . import FontLatin1_12 as font
from .GLSupport import mat4Translate

# TODO: Font should be configurable by the main program


# Text2D ######################################################################

LEFT, CENTER, RIGHT = 'left', 'center', 'right'
TOP, BASELINE, BOTTOM = 'top', 'baseline', 'bottom'
ROTATE_90, ROTATE_180, ROTATE_270 = 90, 180, 270


class Text2D(object):

    _SHADERS = {
        'vertex': """
    #version 120

    attribute vec2 position;
    attribute vec2 texCoords;
    uniform mat4 matrix;

    varying vec2 vCoords;

    void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
        vCoords = texCoords;
    }
    """,
        'fragment': """
    #version 120

    uniform sampler2D texText;
    uniform vec4 color;
    uniform vec4 bgColor;

    varying vec2 vCoords;

    void main(void) {
        gl_FragColor = mix(bgColor, color, texture2D(texText, vCoords).r);
    }
    """
    }

    _program = Program(_SHADERS['vertex'],
                       _SHADERS['fragment'])

    _textures = {}

    def __init__(self, text, x=0, y=0,
                 color=(0., 0., 0., 1.),
                 bgColor=None,
                 align=LEFT, valign=BASELINE,
                 rotate=0):
        self._vertices = None
        self._text = text
        self.x = x
        self.y = y
        self.color = color
        self.bgColor = bgColor

        if align not in (LEFT, CENTER, RIGHT):
            raise RuntimeError(
                "Horizontal alignment not supported: {0}".format(align))
        self._align = align

        if valign not in (TOP, CENTER, BASELINE, BOTTOM):
            raise RuntimeError(
                "Vertical alignment not supported: {0}".format(valign))
        self._valign = valign

        self._rotate = numpy.radians(rotate)

    @classmethod
    def _getTexture(cls):
        # Loaded once for all Text2D instances per OpenGL context
        context = getGLContext()
        try:
            tex = cls._textures[context]
        except KeyError:
            cls._textures[context] = font.loadTexture()
            tex = cls._textures[context]
        return tex

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        if self._text != text:
            self._vertices = None
            self._text = text

    @property
    def size(self):
        return len(self._text) * font.cWidth, font.cHeight

    def getVertices(self):
        if self._vertices is None:
            self._vertices = numpy.empty((len(self.text), 4, 4),
                                         dtype='float32')

            if self._align == LEFT:
                xOrig = 0
            elif self._align == RIGHT:
                xOrig = - len(self._text) * font.cWidth
            else:  # CENTER
                xOrig = - (len(self._text) * font.cWidth) // 2

            if self._valign == BASELINE:
                yOrig = - font.bearingY
            elif self._valign == TOP:
                yOrig = 0
            elif self._valign == BOTTOM:
                yOrig = - font.cHeight
            else:  # CENTER
                yOrig = - font.cHeight // 2

            cos, sin = numpy.cos(self._rotate), numpy.sin(self._rotate)

            for index, char in enumerate(self.text):
                uMin, vMin, uMax, vMax = font.charTexCoords(char)
                vertices = ((xOrig + index * font.cWidth, yOrig + font.cHeight,
                             uMin, vMax),
                            (xOrig + index * font.cWidth, yOrig, uMin, vMin),
                            (xOrig + (index + 1) * font.cWidth,
                             yOrig + font.cHeight, uMax, vMax),
                            (xOrig + (index + 1) * font.cWidth, yOrig,
                             uMax, vMin))

                self._vertices[index] = [
                    (cos * x - sin * y, sin * x + cos * y, u, v)
                    for x, y, u, v in vertices]

        return self._vertices

    def getStride(self):
        vertices = self.getVertices()
        return vertices.shape[-1] * vertices.itemsize

    def render(self, matrix):
        if not self.text:
            return

        prog = self._program
        prog.use()

        texUnit = 0
        self._getTexture().bind(texUnit)

        gl.glUniform1i(prog.uniforms['texText'], texUnit)

        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE,
                              matrix * mat4Translate(self.x, self.y))

        gl.glUniform4f(prog.uniforms['color'], *self.color)
        if self.bgColor is not None:
            bgColor = self.bgColor
        else:
            bgColor = self.color[0], self.color[1], self.color[2], 0.
        gl.glUniform4f(prog.uniforms['bgColor'], *bgColor)

        stride, vertices = self.getStride(), self.getVertices()

        posAttrib = prog.attributes['position']
        gl.glEnableVertexAttribArray(posAttrib)
        gl.glVertexAttribPointer(posAttrib,
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 stride, vertices)

        texAttrib = prog.attributes['texCoords']
        gl.glEnableVertexAttribArray(texAttrib)
        gl.glVertexAttribPointer(texAttrib,
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 stride,
                                 c_void_p(vertices.ctypes.data +
                                          2 * sizeof(c_float))
                                 )

        nbChar, nbVert, _ = vertices.shape
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, nbChar * nbVert)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
