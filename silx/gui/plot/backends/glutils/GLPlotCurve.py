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
This module provides classes to render 2D lines and scatter plots
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


import math
import logging

import numpy

from silx.math.combo import min_max

from ...._glutils import gl
from ...._glutils import numpyToGLType, Program, vertexBuffer
from ..._utils import FLOAT32_MINPOS
from .GLSupport import buildFillMaskIndices


_logger = logging.getLogger(__name__)


_MPL_NONES = None, 'None', '', ' '


# fill ########################################################################

class _Fill2D(object):
    _LINEAR, _LOG10_X, _LOG10_Y, _LOG10_X_Y = 0, 1, 2, 3

    _SHADERS = {
        'vertexTransforms': {
            _LINEAR: """
        vec4 transformXY(float x, float y) {
            return vec4(x, y, 0.0, 1.0);
        }
        """,
            _LOG10_X: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(oneOverLog10 * log(x), y, 0.0, 1.0);
        }
        """,
            _LOG10_Y: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(x, oneOverLog10 * log(y), 0.0, 1.0);
        }
        """,
            _LOG10_X_Y: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(oneOverLog10 * log(x),
                        oneOverLog10 * log(y),
                        0.0, 1.0);
        }
        """
        },
        'vertex': """
        #version 120

        uniform mat4 matrix;
        attribute float xPos;
        attribute float yPos;

        %s

        void main(void) {
            gl_Position = matrix * transformXY(xPos, yPos);
        }
        """,
        'fragment': """
        #version 120

        uniform vec4 color;

        void main(void) {
            gl_FragColor = color;
        }
        """
    }

    _programs = {
        _LINEAR: Program(
            _SHADERS['vertex'] % _SHADERS['vertexTransforms'][_LINEAR],
            _SHADERS['fragment'], attrib0='xPos'),
        _LOG10_X: Program(
            _SHADERS['vertex'] % _SHADERS['vertexTransforms'][_LOG10_X],
            _SHADERS['fragment'], attrib0='xPos'),
        _LOG10_Y: Program(
            _SHADERS['vertex'] % _SHADERS['vertexTransforms'][_LOG10_Y],
            _SHADERS['fragment'], attrib0='xPos'),
        _LOG10_X_Y: Program(
            _SHADERS['vertex'] % _SHADERS['vertexTransforms'][_LOG10_X_Y],
            _SHADERS['fragment'], attrib0='xPos'),
    }

    def __init__(self, xFillVboData=None, yFillVboData=None,
                 xMin=None, yMin=None, xMax=None, yMax=None,
                 color=(0., 0., 0., 1.)):
        self.xFillVboData = xFillVboData
        self.yFillVboData = yFillVboData
        self.xMin, self.yMin = xMin, yMin
        self.xMax, self.yMax = xMax, yMax
        self.color = color

        self._bboxVertices = None
        self._indices = None
        self._indicesType = None

    def prepare(self):
        if self._indices is None:
            self._indices = buildFillMaskIndices(self.xFillVboData.size)
            self._indicesType = numpyToGLType(self._indices.dtype)

        if self._bboxVertices is None:
            yMin, yMax = min(self.yMin, 1e-32), max(self.yMax, 1e-32)
            self._bboxVertices = numpy.array(((self.xMin, self.xMin,
                                               self.xMax, self.xMax),
                                              (yMin, yMax, yMin, yMax)),
                                             dtype=numpy.float32)

    def render(self, matrix, isXLog, isYLog):
        self.prepare()

        if isXLog:
            transform = self._LOG10_X_Y if isYLog else self._LOG10_X
        else:
            transform = self._LOG10_Y if isYLog else self._LINEAR

        prog = self._programs[transform]
        prog.use()

        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE, matrix)

        gl.glUniform4f(prog.uniforms['color'], *self.color)

        xPosAttrib = prog.attributes['xPos']
        yPosAttrib = prog.attributes['yPos']

        gl.glEnableVertexAttribArray(xPosAttrib)
        self.xFillVboData.setVertexAttrib(xPosAttrib)

        gl.glEnableVertexAttribArray(yPosAttrib)
        self.yFillVboData.setVertexAttrib(yPosAttrib)

        # Prepare fill mask
        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glStencilMask(1)
        gl.glStencilFunc(gl.GL_ALWAYS, 1, 1)
        gl.glStencilOp(gl.GL_INVERT, gl.GL_INVERT, gl.GL_INVERT)
        gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
        gl.glDepthMask(gl.GL_FALSE)

        gl.glDrawElements(gl.GL_TRIANGLE_STRIP, self._indices.size,
                          self._indicesType, self._indices)

        gl.glStencilFunc(gl.GL_EQUAL, 1, 1)
        # Reset stencil while drawing
        gl.glStencilOp(gl.GL_ZERO, gl.GL_ZERO, gl.GL_ZERO)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
        gl.glDepthMask(gl.GL_TRUE)

        gl.glVertexAttribPointer(xPosAttrib, 1, gl.GL_FLOAT, gl.GL_FALSE, 0,
                                 self._bboxVertices[0])
        gl.glVertexAttribPointer(yPosAttrib, 1, gl.GL_FLOAT, gl.GL_FALSE, 0,
                                 self._bboxVertices[1])
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, self._bboxVertices[0].size)

        gl.glDisable(gl.GL_STENCIL_TEST)


# line ########################################################################

SOLID, DASHED, DASHDOT, DOTTED = '-', '--', '-.', ':'


class _Lines2D(object):
    STYLES = SOLID, DASHED, DASHDOT, DOTTED
    """Supported line styles"""

    _LINEAR, _LOG10_X, _LOG10_Y, _LOG10_X_Y = 0, 1, 2, 3

    _SHADERS = {
        'vertexTransforms': {
            _LINEAR: """
        vec4 transformXY(float x, float y) {
            return vec4(x, y, 0.0, 1.0);
        }
        """,
            _LOG10_X: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(oneOverLog10 * log(x), y, 0.0, 1.0);
        }
        """,
            _LOG10_Y: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(x, oneOverLog10 * log(y), 0.0, 1.0);
        }
        """,
            _LOG10_X_Y: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(oneOverLog10 * log(x),
                        oneOverLog10 * log(y),
                        0.0, 1.0);
        }
        """
        },
        'solid': {
            'vertex': """
        #version 120

        uniform mat4 matrix;
        attribute float xPos;
        attribute float yPos;
        attribute vec4 color;

        varying vec4 vColor;

        %s

        void main(void) {
            gl_Position = matrix * transformXY(xPos, yPos);
            vColor = color;
        }
        """,
            'fragment': """
        #version 120

        varying vec4 vColor;

        void main(void) {
            gl_FragColor = vColor;
        }
        """
        },


        # Limitation: Dash using an estimate of distance in screen coord
        # to avoid computing distance when viewport is resized
        # results in inequal dashes when viewport aspect ratio is far from 1
        'dashed': {
            'vertex': """
        #version 120

        uniform mat4 matrix;
        uniform vec2 halfViewportSize;
        attribute float xPos;
        attribute float yPos;
        attribute vec4 color;
        attribute float distance;

        varying float vDist;
        varying vec4 vColor;

        %s

        void main(void) {
            gl_Position = matrix * transformXY(xPos, yPos);
            //Estimate distance in pixels
            vec2 probe = vec2(matrix * vec4(1., 1., 0., 0.)) *
                         halfViewportSize;
            float pixelPerDataEstimate = length(probe)/sqrt(2.);
            vDist = distance * pixelPerDataEstimate;
            vColor = color;
        }
        """,
            'fragment': """
        #version 120

        /* Dashes: [0, x], [y, z]
           Dash period: w */
        uniform vec4 dash;

        varying float vDist;
        varying vec4 vColor;

        void main(void) {
            float dist = mod(vDist, dash.w);
            if ((dist > dash.x && dist < dash.y) || dist > dash.z) {
                discard;
            }
            gl_FragColor = vColor;
        }
        """
        }
    }

    _programs = {}

    def __init__(self, xVboData=None, yVboData=None,
                 colorVboData=None, distVboData=None,
                 style=SOLID, color=(0., 0., 0., 1.),
                 width=1, dashPeriod=20, drawMode=None):
        self.xVboData = xVboData
        self.yVboData = yVboData
        self.distVboData = distVboData
        self.colorVboData = colorVboData
        self.useColorVboData = colorVboData is not None

        self.color = color
        self._width = 1
        self.width = width
        self._style = None
        self.style = style
        self.dashPeriod = dashPeriod

        self._drawMode = drawMode if drawMode is not None else gl.GL_LINE_STRIP

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        if style in _MPL_NONES:
            self._style = None
            self.render = self._renderNone
        else:
            assert style in self.STYLES
            self._style = style
            if style == SOLID:
                self.render = self._renderSolid
            else:  # DASHED, DASHDOT, DOTTED
                self.render = self._renderDash

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        # try:
        #    widthRange = self._widthRange
        # except AttributeError:
        #    widthRange = gl.glGetFloatv(gl.GL_ALIASED_LINE_WIDTH_RANGE)
        #    # Shared among contexts, this should be enough..
        #    _Lines2D._widthRange = widthRange
        # assert width >= widthRange[0] and width <= widthRange[1]
        self._width = width

    @classmethod
    def _getProgram(cls, transform, style):
        try:
            prgm = cls._programs[(transform, style)]
        except KeyError:
            sources = cls._SHADERS[style]
            vertexShdr = sources['vertex'] % \
                cls._SHADERS['vertexTransforms'][transform]
            prgm = Program(vertexShdr, sources['fragment'], attrib0='xPos')
            cls._programs[(transform, style)] = prgm
        return prgm

    @classmethod
    def init(cls):
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

    def _renderNone(self, matrix, isXLog, isYLog):
        pass

    render = _renderNone  # Overridden in style setter

    def _renderSolid(self, matrix, isXLog, isYLog):
        if isXLog:
            transform = self._LOG10_X_Y if isYLog else self._LOG10_X
        else:
            transform = self._LOG10_Y if isYLog else self._LINEAR

        prog = self._getProgram(transform, 'solid')
        prog.use()

        gl.glEnable(gl.GL_LINE_SMOOTH)

        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE, matrix)

        colorAttrib = prog.attributes['color']
        if self.useColorVboData and self.colorVboData is not None:
            gl.glEnableVertexAttribArray(colorAttrib)
            self.colorVboData.setVertexAttrib(colorAttrib)
        else:
            gl.glDisableVertexAttribArray(colorAttrib)
            gl.glVertexAttrib4f(colorAttrib, *self.color)

        xPosAttrib = prog.attributes['xPos']
        gl.glEnableVertexAttribArray(xPosAttrib)
        self.xVboData.setVertexAttrib(xPosAttrib)

        yPosAttrib = prog.attributes['yPos']
        gl.glEnableVertexAttribArray(yPosAttrib)
        self.yVboData.setVertexAttrib(yPosAttrib)

        gl.glLineWidth(self.width)
        gl.glDrawArrays(self._drawMode, 0, self.xVboData.size)

        gl.glDisable(gl.GL_LINE_SMOOTH)

    def _renderDash(self, matrix, isXLog, isYLog):
        if isXLog:
            transform = self._LOG10_X_Y if isYLog else self._LOG10_X
        else:
            transform = self._LOG10_Y if isYLog else self._LINEAR

        prog = self._getProgram(transform, 'dashed')
        prog.use()

        gl.glEnable(gl.GL_LINE_SMOOTH)

        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE, matrix)
        x, y, viewWidth, viewHeight = gl.glGetFloatv(gl.GL_VIEWPORT)
        gl.glUniform2f(prog.uniforms['halfViewportSize'],
                       0.5 * viewWidth, 0.5 * viewHeight)

        if self.style == DOTTED:
            dash = (0.1 * self.dashPeriod,
                    0.6 * self.dashPeriod,
                    0.7 * self.dashPeriod,
                    self.dashPeriod)
        elif self.style == DASHDOT:
            dash = (0.3 * self.dashPeriod,
                    0.5 * self.dashPeriod,
                    0.6 * self.dashPeriod,
                    self.dashPeriod)
        else:
            dash = (0.5 * self.dashPeriod,
                    self.dashPeriod,
                    self.dashPeriod,
                    self.dashPeriod)

        gl.glUniform4f(prog.uniforms['dash'], *dash)

        colorAttrib = prog.attributes['color']
        if self.useColorVboData and self.colorVboData is not None:
            gl.glEnableVertexAttribArray(colorAttrib)
            self.colorVboData.setVertexAttrib(colorAttrib)
        else:
            gl.glDisableVertexAttribArray(colorAttrib)
            gl.glVertexAttrib4f(colorAttrib, *self.color)

        distAttrib = prog.attributes['distance']
        gl.glEnableVertexAttribArray(distAttrib)
        self.distVboData.setVertexAttrib(distAttrib)

        xPosAttrib = prog.attributes['xPos']
        gl.glEnableVertexAttribArray(xPosAttrib)
        self.xVboData.setVertexAttrib(xPosAttrib)

        yPosAttrib = prog.attributes['yPos']
        gl.glEnableVertexAttribArray(yPosAttrib)
        self.yVboData.setVertexAttrib(yPosAttrib)

        gl.glLineWidth(self.width)
        gl.glDrawArrays(self._drawMode, 0, self.xVboData.size)

        gl.glDisable(gl.GL_LINE_SMOOTH)


def _distancesFromArrays(xData, yData):
    deltas = numpy.dstack((
        numpy.ediff1d(xData, to_begin=numpy.float32(0.)),
        numpy.ediff1d(yData, to_begin=numpy.float32(0.))))[0]
    return numpy.cumsum(numpy.sqrt(numpy.sum(deltas ** 2, axis=1)))


# points ######################################################################

DIAMOND, CIRCLE, SQUARE, PLUS, X_MARKER, POINT, PIXEL, ASTERISK = \
    'd', 'o', 's', '+', 'x', '.', ',', '*'

H_LINE, V_LINE = '_', '|'


class _Points2D(object):
    MARKERS = (DIAMOND, CIRCLE, SQUARE, PLUS, X_MARKER, POINT, PIXEL, ASTERISK,
               H_LINE, V_LINE)

    _LINEAR, _LOG10_X, _LOG10_Y, _LOG10_X_Y = 0, 1, 2, 3

    _SHADERS = {
        'vertexTransforms': {
            _LINEAR: """
        vec4 transformXY(float x, float y) {
            return vec4(x, y, 0.0, 1.0);
        }
        """,
            _LOG10_X: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(oneOverLog10 * log(x), y, 0.0, 1.0);
        }
        """,
            _LOG10_Y: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(x, oneOverLog10 * log(y), 0.0, 1.0);
        }
        """,
            _LOG10_X_Y: """
        const float oneOverLog10 = 0.43429448190325176;

        vec4 transformXY(float x, float y) {
            return vec4(oneOverLog10 * log(x),
                        oneOverLog10 * log(y),
                        0.0, 1.0);
        }
        """
        },
        'vertex': """
    #version 120

    uniform mat4 matrix;
    uniform int transform;
    uniform float size;
    attribute float xPos;
    attribute float yPos;
    attribute vec4 color;

    varying vec4 vColor;

    %s

    void main(void) {
        gl_Position = matrix * transformXY(xPos, yPos);
        vColor = color;
        gl_PointSize = size;
    }
    """,

        'fragmentSymbols': {
            DIAMOND: """
        float alphaSymbol(vec2 coord, float size) {
            vec2 centerCoord = abs(coord - vec2(0.5, 0.5));
            float f = centerCoord.x + centerCoord.y;
            return clamp(size * (0.5 - f), 0.0, 1.0);
        }
        """,
            CIRCLE: """
        float alphaSymbol(vec2 coord, float size) {
            float radius = 0.5;
            float r = distance(coord, vec2(0.5, 0.5));
            return clamp(size * (radius - r), 0.0, 1.0);
        }
        """,
            SQUARE: """
        float alphaSymbol(vec2 coord, float size) {
            return 1.0;
        }
        """,
            PLUS: """
        float alphaSymbol(vec2 coord, float size) {
            vec2 d = abs(size * (coord - vec2(0.5, 0.5)));
            if (min(d.x, d.y) < 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        """,
            X_MARKER: """
        float alphaSymbol(vec2 coord, float size) {
            vec2 pos = floor(size * coord) + 0.5;
            vec2 d_x = abs(pos.x + vec2(- pos.y, pos.y - size));
            if (min(d_x.x, d_x.y) <= 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        """,
            ASTERISK: """
        float alphaSymbol(vec2 coord, float size) {
            /* Combining +, x and circle */
            vec2 d_plus = abs(size * (coord - vec2(0.5, 0.5)));
            vec2 pos = floor(size * coord) + 0.5;
            vec2 d_x = abs(pos.x + vec2(- pos.y, pos.y - size));
            if (min(d_plus.x, d_plus.y) < 0.5) {
                return 1.0;
            } else if (min(d_x.x, d_x.y) <= 0.5) {
                float r = distance(coord, vec2(0.5, 0.5));
                return clamp(size * (0.5 - r), 0.0, 1.0);
            } else {
                return 0.0;
            }
        }
        """,
            H_LINE: """
        float alphaSymbol(vec2 coord, float size) {
            float dy = abs(size * (coord.y - 0.5));
            if (dy < 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        """,
            V_LINE: """
        float alphaSymbol(vec2 coord, float size) {
            float dx = abs(size * (coord.x - 0.5));
            if (dx < 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        """
        },

        'fragment': """
    #version 120

    uniform float size;

    varying vec4 vColor;

    %s

    void main(void) {
        float alpha = alphaSymbol(gl_PointCoord, size);
        if (alpha <= 0.0) {
            discard;
        } else {
            gl_FragColor = vec4(vColor.rgb, alpha * clamp(vColor.a, 0.0, 1.0));
        }
    }
    """
    }

    _programs = {}

    def __init__(self, xVboData=None, yVboData=None, colorVboData=None,
                 marker=SQUARE, color=(0., 0., 0., 1.), size=7):
        self.color = color
        self._marker = None
        self.marker = marker
        self._size = 1
        self.size = size

        self.xVboData = xVboData
        self.yVboData = yVboData
        self.colorVboData = colorVboData
        self.useColorVboData = colorVboData is not None

    @property
    def marker(self):
        return self._marker

    @marker.setter
    def marker(self, marker):
        if marker in _MPL_NONES:
            self._marker = None
            self.render = self._renderNone
        else:
            assert marker in self.MARKERS
            self._marker = marker
            self.render = self._renderMarkers

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        # try:
        #    sizeRange = self._sizeRange
        # except AttributeError:
        #    sizeRange = gl.glGetFloatv(gl.GL_POINT_SIZE_RANGE)
        #    # Shared among contexts, this should be enough..
        #    _Points2D._sizeRange = sizeRange
        # assert size >= sizeRange[0] and size <= sizeRange[1]
        self._size = size

    @classmethod
    def _getProgram(cls, transform, marker):
        """On-demand shader program creation."""
        if marker == PIXEL:
            marker = SQUARE
        elif marker == POINT:
            marker = CIRCLE
        try:
            prgm = cls._programs[(transform, marker)]
        except KeyError:
            vertShdr = cls._SHADERS['vertex'] % \
                cls._SHADERS['vertexTransforms'][transform]
            fragShdr = cls._SHADERS['fragment'] % \
                cls._SHADERS['fragmentSymbols'][marker]
            prgm = Program(vertShdr, fragShdr, attrib0='xPos')

            cls._programs[(transform, marker)] = prgm
        return prgm

    @classmethod
    def init(cls):
        version = gl.glGetString(gl.GL_VERSION)
        majorVersion = int(version[0])
        assert majorVersion >= 2
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        if majorVersion >= 3:  # OpenGL 3
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    def _renderNone(self, matrix, isXLog, isYLog):
        pass

    render = _renderNone

    def _renderMarkers(self, matrix, isXLog, isYLog):
        if isXLog:
            transform = self._LOG10_X_Y if isYLog else self._LOG10_X
        else:
            transform = self._LOG10_Y if isYLog else self._LINEAR

        prog = self._getProgram(transform, self.marker)
        prog.use()
        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE, matrix)
        if self.marker == PIXEL:
            size = 1
        elif self.marker == POINT:
            size = math.ceil(0.5 * self.size) + 1  # Mimic Matplotlib point
        else:
            size = self.size
        gl.glUniform1f(prog.uniforms['size'], size)
        # gl.glPointSize(self.size)

        cAttrib = prog.attributes['color']
        if self.useColorVboData and self.colorVboData is not None:
            gl.glEnableVertexAttribArray(cAttrib)
            self.colorVboData.setVertexAttrib(cAttrib)
        else:
            gl.glDisableVertexAttribArray(cAttrib)
            gl.glVertexAttrib4f(cAttrib, *self.color)

        xAttrib = prog.attributes['xPos']
        gl.glEnableVertexAttribArray(xAttrib)
        self.xVboData.setVertexAttrib(xAttrib)

        yAttrib = prog.attributes['yPos']
        gl.glEnableVertexAttribArray(yAttrib)
        self.yVboData.setVertexAttrib(yAttrib)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.xVboData.size)

        gl.glUseProgram(0)


# error bars ##################################################################

class _ErrorBars(object):
    """Display errors bars.

    This is using its own VBO as opposed to fill/points/lines.
    There is no picking on error bars.
    As is, there is no way to update data and errors, but it handles
    log scales by removing data <= 0 and clipping error bars to positive
    range.

    It uses 2 vertices per error bars and uses :class:`_Lines2D` to
    render error bars and :class:`_Points2D` to render the ends.
    """

    def __init__(self, xData, yData, xError, yError,
                 xMin, yMin,
                 color=(0., 0., 0., 1.)):
        """Initialization.

        :param numpy.ndarray xData: X coordinates of the data.
        :param numpy.ndarray yData: Y coordinates of the data.
        :param xError: The absolute error on the X axis.
        :type xError: A float, or a numpy.ndarray of float32.
                      If it is an array, it can either be a 1D array of
                      same length as the data or a 2D array with 2 rows
                      of same length as the data: row 0 for negative errors,
                      row 1 for positive errors.
        :param yError: The absolute error on the Y axis.
        :type yError: A float, or a numpy.ndarray of float32. See xError.
        :param float xMin: The min X value already computed by GLPlotCurve2D.
        :param float yMin: The min Y value already computed by GLPlotCurve2D.
        :param color: The color to use for both lines and ending points.
        :type color: tuple of 4 floats
        """
        self._attribs = None
        self._isXLog, self._isYLog = False, False
        self._xMin, self._yMin = xMin, yMin

        if xError is not None or yError is not None:
            assert len(xData) == len(yData)
            self._xData = numpy.array(
                xData, order='C', dtype=numpy.float32, copy=False)
            self._yData = numpy.array(
                yData, order='C', dtype=numpy.float32, copy=False)

            # This also works if xError, yError is a float/int
            self._xError = numpy.array(
                xError, order='C', dtype=numpy.float32, copy=False)
            self._yError = numpy.array(
                yError, order='C', dtype=numpy.float32, copy=False)
        else:
            self._xData, self._yData = None, None
            self._xError, self._yError = None, None

        self._lines = _Lines2D(None, None, color=color, drawMode=gl.GL_LINES)
        self._xErrPoints = _Points2D(None, None, color=color, marker=V_LINE)
        self._yErrPoints = _Points2D(None, None, color=color, marker=H_LINE)

    def _positiveValueFilter(self, onlyXPos, onlyYPos):
        """Filter data (x, y) and errors (xError, yError) to remove
        negative and null data values on required axis (onlyXPos, onlyYPos).

        Returned arrays might be NOT contiguous.

        :return: Filtered xData, yData, xError and yError arrays.
        """
        if ((not onlyXPos or self._xMin > 0.) and
                (not onlyYPos or self._yMin > 0.)):
            # No need to filter, all values are > 0 on log axes
            return self._xData, self._yData, self._xError, self._yError

        _logger.warning(
            'Removing values <= 0 of curve with error bars on a log axis.')

        x, y = self._xData, self._yData
        xError, yError = self._xError, self._yError

        # First remove negative data
        if onlyXPos and onlyYPos:
            mask = (x > 0.) & (y > 0.)
        elif onlyXPos:
            mask = x > 0.
        else:  # onlyYPos
            mask = y > 0.
        x, y = x[mask], y[mask]

        # Remove corresponding values from error arrays
        if xError is not None and xError.size != 1:
            if len(xError.shape) == 1:
                xError = xError[mask]
            else:  # 2 rows
                xError = xError[:, mask]
        if yError is not None and yError.size != 1:
            if len(yError.shape) == 1:
                yError = yError[mask]
            else:  # 2 rows
                yError = yError[:, mask]

        return x, y, xError, yError

    def _buildVertices(self, isXLog, isYLog):
        """Generates error bars vertices according to log scales."""
        xData, yData, xError, yError = self._positiveValueFilter(
            isXLog, isYLog)

        nbLinesPerDataPts = 1 if xError is not None else 0
        nbLinesPerDataPts += 1 if yError is not None else 0

        nbDataPts = len(xData)

        # interleave coord+error, coord-error.
        # xError vertices first if any, then yError vertices if any.
        xCoords = numpy.empty(nbDataPts * nbLinesPerDataPts * 2,
                              dtype=numpy.float32)
        yCoords = numpy.empty(nbDataPts * nbLinesPerDataPts * 2,
                              dtype=numpy.float32)

        if xError is not None:  # errors on the X axis
            if len(xError.shape) == 2:
                xErrorMinus, xErrorPlus = xError[0], xError[1]
            else:
                # numpy arrays of len 1 or len(xData)
                xErrorMinus, xErrorPlus = xError, xError

            # Interleave vertices for xError
            endXError = 2 * nbDataPts
            xCoords[0:endXError-1:2] = xData + xErrorPlus

            minValues = xData - xErrorMinus
            if isXLog:
                # Clip min bounds to positive value
                minValues[minValues <= 0] = FLOAT32_MINPOS
            xCoords[1:endXError:2] = minValues

            yCoords[0:endXError-1:2] = yData
            yCoords[1:endXError:2] = yData
        else:
            endXError = 0

        if yError is not None:  # errors on the Y axis
            if len(yError.shape) == 2:
                yErrorMinus, yErrorPlus = yError[0], yError[1]
            else:
                # numpy arrays of len 1 or len(yData)
                yErrorMinus, yErrorPlus = yError, yError

            # Interleave vertices for yError
            xCoords[endXError::2] = xData
            xCoords[endXError+1::2] = xData
            yCoords[endXError::2] = yData + yErrorPlus
            minValues = yData - yErrorMinus
            if isYLog:
                # Clip min bounds to positive value
                minValues[minValues <= 0] = FLOAT32_MINPOS
            yCoords[endXError+1::2] = minValues

        return xCoords, yCoords

    def prepare(self, isXLog, isYLog):
        if self._xData is None:
            return

        if self._isXLog != isXLog or self._isYLog != isYLog:
            # Log state has changed
            self._isXLog, self._isYLog = isXLog, isYLog

            self.discard()  # discard existing VBOs

        if self._attribs is None:
            xCoords, yCoords = self._buildVertices(isXLog, isYLog)

            xAttrib, yAttrib = vertexBuffer((xCoords, yCoords))
            self._attribs = xAttrib, yAttrib

            self._lines.xVboData, self._lines.yVboData = xAttrib, yAttrib

            # Set xError points using the same VBO as lines
            self._xErrPoints.xVboData = xAttrib.copy()
            self._xErrPoints.xVboData.size //= 2
            self._xErrPoints.yVboData = yAttrib.copy()
            self._xErrPoints.yVboData.size //= 2

            # Set yError points using the same VBO as lines
            self._yErrPoints.xVboData = xAttrib.copy()
            self._yErrPoints.xVboData.size //= 2
            self._yErrPoints.xVboData.offset += (xAttrib.itemsize *
                                                 xAttrib.size // 2)
            self._yErrPoints.yVboData = yAttrib.copy()
            self._yErrPoints.yVboData.size //= 2
            self._yErrPoints.yVboData.offset += (yAttrib.itemsize *
                                                 yAttrib.size // 2)

    def render(self, matrix, isXLog, isYLog):
        if self._attribs is not None:
            self._lines.render(matrix, isXLog, isYLog)
            self._xErrPoints.render(matrix, isXLog, isYLog)
            self._yErrPoints.render(matrix, isXLog, isYLog)

    def discard(self):
        if self._attribs is not None:
            self._lines.xVboData, self._lines.yVboData = None, None
            self._xErrPoints.xVboData, self._xErrPoints.yVboData = None, None
            self._yErrPoints.xVboData, self._yErrPoints.yVboData = None, None
            self._attribs[0].vbo.discard()
            self._attribs = None


# curves ######################################################################

def _proxyProperty(*componentsAttributes):
    """Create a property to access an attribute of attribute(s).
    Useful for composition.
    Supports multiple components this way:
    getter returns the first found, setter sets all
    """
    def getter(self):
        for compName, attrName in componentsAttributes:
            try:
                component = getattr(self, compName)
            except AttributeError:
                pass
            else:
                return getattr(component, attrName)

    def setter(self, value):
        for compName, attrName in componentsAttributes:
            component = getattr(self, compName)
            setattr(component, attrName, value)
    return property(getter, setter)


class GLPlotCurve2D(object):
    def __init__(self, xData, yData, colorData=None,
                 xError=None, yError=None,
                 lineStyle=None, lineColor=None,
                 lineWidth=None, lineDashPeriod=None,
                 marker=None, markerColor=None, markerSize=None,
                 fillColor=None):
        self._isXLog = False
        self._isYLog = False
        self.xData, self.yData, self.colorData = xData, yData, colorData

        if fillColor is not None:
            self.fill = _Fill2D(color=fillColor)
        else:
            self.fill = None

        # Compute x bounds
        if xError is None:
            result = min_max(xData, min_positive=True)
            self.xMin = result.minimum
            self.xMinPos = result.min_positive
            self.xMax = result.maximum
        else:
            # Takes the error into account
            if hasattr(xError, 'shape') and len(xError.shape) == 2:
                xErrorPlus, xErrorMinus = xError[0], xError[1]
            else:
                xErrorPlus, xErrorMinus = xError, xError
            result = min_max(xData - xErrorMinus, min_positive=True)
            self.xMin = result.minimum
            self.xMinPos = result.min_positive
            self.xMax = (xData + xErrorPlus).max()

        # Compute y bounds
        if yError is None:
            result = min_max(yData, min_positive=True)
            self.yMin = result.minimum
            self.yMinPos = result.min_positive
            self.yMax = result.maximum
        else:
            # Takes the error into account
            if hasattr(yError, 'shape') and len(yError.shape) == 2:
                yErrorPlus, yErrorMinus = yError[0], yError[1]
            else:
                yErrorPlus, yErrorMinus = yError, yError
            result = min_max(yData - yErrorMinus, min_positive=True)
            self.yMin = result.minimum
            self.yMinPos = result.min_positive
            self.yMax = (yData + yErrorPlus).max()

        self._errorBars = _ErrorBars(xData, yData, xError, yError,
                                     self.xMin, self.yMin)

        kwargs = {'style': lineStyle}
        if lineColor is not None:
            kwargs['color'] = lineColor
        if lineWidth is not None:
            kwargs['width'] = lineWidth
        if lineDashPeriod is not None:
            kwargs['dashPeriod'] = lineDashPeriod
        self.lines = _Lines2D(**kwargs)

        kwargs = {'marker': marker}
        if markerColor is not None:
            kwargs['color'] = markerColor
        if markerSize is not None:
            kwargs['size'] = markerSize
        self.points = _Points2D(**kwargs)

    xVboData = _proxyProperty(('lines', 'xVboData'), ('points', 'xVboData'))

    yVboData = _proxyProperty(('lines', 'yVboData'), ('points', 'yVboData'))

    colorVboData = _proxyProperty(('lines', 'colorVboData'),
                                  ('points', 'colorVboData'))

    useColorVboData = _proxyProperty(('lines', 'useColorVboData'),
                                     ('points', 'useColorVboData'))

    distVboData = _proxyProperty(('lines', 'distVboData'))

    lineStyle = _proxyProperty(('lines', 'style'))

    lineColor = _proxyProperty(('lines', 'color'))

    lineWidth = _proxyProperty(('lines', 'width'))

    lineDashPeriod = _proxyProperty(('lines', 'dashPeriod'))

    marker = _proxyProperty(('points', 'marker'))

    markerColor = _proxyProperty(('points', 'color'))

    markerSize = _proxyProperty(('points', 'size'))

    @classmethod
    def init(cls):
        _Lines2D.init()
        _Points2D.init()

    @staticmethod
    def _logFilterData(x, y, color=None, xLog=False, yLog=False):
        # Copied from Plot.py
        if xLog and yLog:
            idx = numpy.nonzero((x > 0) & (y > 0))[0]
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
        elif yLog:
            idx = numpy.nonzero(y > 0)[0]
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
        elif xLog:
            idx = numpy.nonzero(x > 0)[0]
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
        else:
            idx = None

        if idx is not None and isinstance(color, numpy.ndarray):
            colors = numpy.zeros((x.size, 4), color.dtype)
            colors[:, 0] = color[idx, 0]
            colors[:, 1] = color[idx, 1]
            colors[:, 2] = color[idx, 2]
            colors[:, 3] = color[idx, 3]
        else:
            colors = color
        return x, y, colors

    def prepare(self, isXLog, isYLog):
        # init only supports updating isXLog, isYLog
        xData, yData, colorData = self.xData, self.yData, self.colorData

        if self._isXLog != isXLog or self._isYLog != isYLog:
            # Log state has changed
            self._isXLog, self._isYLog = isXLog, isYLog

            # Check if data <= 0. with log scale
            if (isXLog and self.xMin <= 0.) or (isYLog and self.yMin <= 0.):
                # Filtering data is needed
                xData, yData, colorData = self._logFilterData(
                    self.xData, self.yData, self.colorData,
                    self._isXLog, self._isYLog)

                self.discard()  # discard existing VBOs

        if self.xVboData is None:
            xAttrib, yAttrib, cAttrib, dAttrib = None, None, None, None
            if self.lineStyle in (DASHED, DASHDOT, DOTTED):
                dists = _distancesFromArrays(xData, yData)
                if self.colorData is None:
                    xAttrib, yAttrib, dAttrib = vertexBuffer(
                        (xData, yData, dists),
                        prefix=(1, 1, 0), suffix=(1, 1, 0))
                else:
                    xAttrib, yAttrib, cAttrib, dAttrib = vertexBuffer(
                        (xData, yData, colorData, dists),
                        prefix=(1, 1, 0, 0), suffix=(1, 1, 0, 0))
            elif self.colorData is None:
                xAttrib, yAttrib = vertexBuffer(
                    (xData, yData), prefix=(1, 1), suffix=(1, 1))
            else:
                xAttrib, yAttrib, cAttrib = vertexBuffer(
                    (xData, yData, colorData), prefix=(1, 1, 0))

            # Shrink VBO
            self.xVboData = xAttrib.copy()
            self.xVboData.size -= 2
            self.xVboData.offset += xAttrib.itemsize

            self.yVboData = yAttrib.copy()
            self.yVboData.size -= 2
            self.yVboData.offset += yAttrib.itemsize

            if cAttrib is not None and colorData.dtype.kind == 'u':
                cAttrib.normalization = True  # Normalize uint to [0, 1]
            self.colorVboData = cAttrib
            self.useColorVboData = cAttrib is not None
            self.distVboData = dAttrib

            if self.fill is not None:
                xData = xData.reshape(xData.size, 1)
                zero = numpy.array((1e-32,), dtype=self.yData.dtype)

                # Add one point before data: (x0, 0.)
                xAttrib.vbo.update(xData[0], xAttrib.offset,
                                   xData[0].itemsize)
                yAttrib.vbo.update(zero, yAttrib.offset, zero.itemsize)

                # Add one point after data: (xN, 0.)
                xAttrib.vbo.update(xData[-1],
                                   xAttrib.offset +
                                   (xAttrib.size - 1) * xAttrib.itemsize,
                                   xData[-1].itemsize)
                yAttrib.vbo.update(zero,
                                   yAttrib.offset +
                                   (yAttrib.size - 1) * yAttrib.itemsize,
                                   zero.itemsize)

                self.fill.xFillVboData = xAttrib
                self.fill.yFillVboData = yAttrib
                self.fill.xMin, self.fill.yMin = self.xMin, self.yMin
                self.fill.xMax, self.fill.yMax = self.xMax, self.yMax

        self._errorBars.prepare(isXLog, isYLog)

    def render(self, matrix, isXLog, isYLog):
        self.prepare(isXLog, isYLog)
        if self.fill is not None:
            self.fill.render(matrix, isXLog, isYLog)
        self._errorBars.render(matrix, isXLog, isYLog)
        self.lines.render(matrix, isXLog, isYLog)
        self.points.render(matrix, isXLog, isYLog)

    def discard(self):
        if self.xVboData is not None:
            self.xVboData.vbo.discard()

        self.xVboData = None
        self.yVboData = None
        self.colorVboData = None
        self.distVboData = None

        self._errorBars.discard()

    def pick(self, xPickMin, yPickMin, xPickMax, yPickMax):
        """Perform picking on the curve according to its rendering.

        The picking area is [xPickMin, xPickMax], [yPickMin, yPickMax].

        In case a segment between 2 points with indices i, i+1 is picked,
        only its lower index end point (i.e., i) is added to the result.
        In case an end point with index i is picked it is added to the result,
        and the segment [i-1, i] is not tested for picking.

        :return: The indices of the picked data
        :rtype: list of int
        """
        if (self.marker is None and self.lineStyle is None) or \
                self.xMin > xPickMax or xPickMin > self.xMax or \
                self.yMin > yPickMax or yPickMin > self.yMax:
            # Note: With log scale the bounding box is too large if
            # some data <= 0.
            return None

        elif self.lineStyle is not None:
            # Using Cohen-Sutherland algorithm for line clipping
            codes = ((self.yData > yPickMax) << 3) | \
                    ((self.yData < yPickMin) << 2) | \
                    ((self.xData > xPickMax) << 1) | \
                    (self.xData < xPickMin)

            # Add all points that are inside the picking area
            indices = numpy.nonzero(codes == 0)[0].tolist()

            # Segment that might cross the area with no end point inside it
            segToTestIdx = numpy.nonzero((codes[:-1] != 0) &
                                         (codes[1:] != 0) &
                                         ((codes[:-1] & codes[1:]) == 0))[0]

            TOP, BOTTOM, RIGHT, LEFT = (1 << 3), (1 << 2), (1 << 1), (1 << 0)

            for index in segToTestIdx:
                if index not in indices:
                    x0, y0 = self.xData[index], self.yData[index]
                    x1, y1 = self.xData[index + 1], self.yData[index + 1]
                    code1 = codes[index + 1]

                    # check for crossing with horizontal bounds
                    # y0 == y1 is a never event:
                    # => pt0 and pt1 in same vertical area are not in segToTest
                    if code1 & TOP:
                        x = x0 + (x1 - x0) * (yPickMax - y0) / (y1 - y0)
                    elif code1 & BOTTOM:
                        x = x0 + (x1 - x0) * (yPickMin - y0) / (y1 - y0)
                    else:
                        x = None  # No horizontal bounds intersection test

                    if x is not None and xPickMin <= x <= xPickMax:
                        # Intersection
                        indices.append(index)

                    else:
                        # check for crossing with vertical bounds
                        # x0 == x1 is a never event (see remark for y)
                        if code1 & RIGHT:
                            y = y0 + (y1 - y0) * (xPickMax - x0) / (x1 - x0)
                        elif code1 & LEFT:
                            y = y0 + (y1 - y0) * (xPickMin - x0) / (x1 - x0)
                        else:
                            y = None  # No vertical bounds intersection test

                        if y is not None and yPickMin <= y <= yPickMax:
                            # Intersection
                            indices.append(index)

            indices.sort()

        else:
            indices = numpy.nonzero((self.xData >= xPickMin) &
                                    (self.xData <= xPickMax) &
                                    (self.yData >= yPickMin) &
                                    (self.yData <= yPickMax))[0].tolist()

        return indices
