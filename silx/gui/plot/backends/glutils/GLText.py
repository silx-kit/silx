# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2018 European Synchrotron Radiation Facility
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

from ...._glutils import font, gl, getGLContext, Program, Texture
from .GLSupport import mat4Translate


# TODO: Font should be configurable by the main program: using mpl.rcParams?


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

    _TEX_COORDS = numpy.array(((0., 0.), (1., 0.), (0., 1.), (1., 1.)),
                              dtype=numpy.float32).ravel()

    _program = Program(_SHADERS['vertex'],
                       _SHADERS['fragment'],
                       attrib0='position')

    _textures = {}
    """Cache already created textures"""
    # TODO limit cache size and discard least recent used

    _sizes = {}
    """Cache already computed sizes"""

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
            raise ValueError(
                "Horizontal alignment not supported: {0}".format(align))
        self._align = align

        if valign not in (TOP, CENTER, BASELINE, BOTTOM):
            raise ValueError(
                "Vertical alignment not supported: {0}".format(valign))
        self._valign = valign

        self._rotate = numpy.radians(rotate)

    def _getTexture(self, text):
        key = getGLContext(), text

        if key not in self._textures:
            image, offset = font.rasterText(text,
                                            font.getDefaultFontFamily())
            if text not in self._sizes:
                self._sizes[text] = image.shape[1], image.shape[0]

            self._textures[key] = (
                Texture(gl.GL_RED,
                        data=image,
                        minFilter=gl.GL_NEAREST,
                        magFilter=gl.GL_NEAREST,
                        wrap=(gl.GL_CLAMP_TO_EDGE,
                              gl.GL_CLAMP_TO_EDGE)),
                offset)

        return self._textures[key]

    @property
    def text(self):
        return self._text

    @property
    def size(self):
        if self.text not in self._sizes:
            image, offset = font.rasterText(self.text,
                                            font.getDefaultFontFamily())
            self._sizes[self.text] = image.shape[1], image.shape[0]
        return self._sizes[self.text]

    def getVertices(self, offset, shape):
        height, width = shape

        if self._align == LEFT:
            xOrig = 0
        elif self._align == RIGHT:
            xOrig = - width
        else:  # CENTER
            xOrig = - width // 2

        if self._valign == BASELINE:
            yOrig = - offset
        elif self._valign == TOP:
            yOrig = 0
        elif self._valign == BOTTOM:
            yOrig = - height
        else:  # CENTER
            yOrig = - height // 2

        vertices = numpy.array((
            (xOrig, yOrig),
            (xOrig + width, yOrig),
            (xOrig, yOrig + height),
            (xOrig + width, yOrig + height)), dtype=numpy.float32)

        cos, sin = numpy.cos(self._rotate), numpy.sin(self._rotate)
        vertices = numpy.ascontiguousarray(numpy.transpose(numpy.array((
            cos * vertices[:, 0] - sin * vertices[:, 1],
            sin * vertices[:, 0] + cos * vertices[:, 1]),
            dtype=numpy.float32)))

        return vertices

    def render(self, matrix):
        if not self.text:
            return

        prog = self._program
        prog.use()

        texUnit = 0
        texture, offset = self._getTexture(self.text)

        gl.glUniform1i(prog.uniforms['texText'], texUnit)

        mat = numpy.dot(matrix, mat4Translate(int(self.x), int(self.y)))
        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE,
                              mat.astype(numpy.float32))

        gl.glUniform4f(prog.uniforms['color'], *self.color)
        if self.bgColor is not None:
            bgColor = self.bgColor
        else:
            bgColor = self.color[0], self.color[1], self.color[2], 0.
        gl.glUniform4f(prog.uniforms['bgColor'], *bgColor)

        vertices = self.getVertices(offset, texture.shape)

        posAttrib = prog.attributes['position']
        gl.glEnableVertexAttribArray(posAttrib)
        gl.glVertexAttribPointer(posAttrib,
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0,
                                 vertices)

        texAttrib = prog.attributes['texCoords']
        gl.glEnableVertexAttribArray(texAttrib)
        gl.glVertexAttribPointer(texAttrib,
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0,
                                 self._TEX_COORDS)

        with texture:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
