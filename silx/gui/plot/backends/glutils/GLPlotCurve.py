# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2020 European Synchrotron Radiation Facility
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

from __future__ import division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


import math
import logging

import numpy

from silx.math.combo import min_max

from ...._glutils import gl
from ...._glutils import Program, vertexBuffer, VertexBufferAttrib
from .GLSupport import buildFillMaskIndices, mat4Identity, mat4Translate


_logger = logging.getLogger(__name__)


_MPL_NONES = None, 'None', '', ' '
"""Possible values for None"""


def _notNaNSlices(array, length=1):
    """Returns slices of none NaN values in the array.

    :param numpy.ndarray array: 1D array from which to get slices
    :param int length: Slices shorter than length gets discarded
    :return: Array of (start, end) slice indices
    :rtype: numpy.ndarray
    """
    isnan = numpy.isnan(numpy.array(array, copy=False).reshape(-1))
    notnan = numpy.logical_not(isnan)
    start = numpy.where(numpy.logical_and(isnan[:-1], notnan[1:]))[0] + 1
    if notnan[0]:
        start = numpy.append(0, start)
    end = numpy.where(numpy.logical_and(notnan[:-1], isnan[1:]))[0] + 1
    if notnan[-1]:
        end = numpy.append(end, len(array))
    slices = numpy.transpose((start, end))
    if length > 1:
        # discard slices with less than length values
        slices = slices[numpy.diff(slices, axis=1).ravel() >= length]
    return slices


# fill ########################################################################

class _Fill2D(object):
    """Object rendering curve filling as polygons

    :param numpy.ndarray xData: X coordinates of points
    :param numpy.ndarray yData: Y coordinates of points
    :param float baseline: Y value of the 'bottom' of the fill.
        0 for linear Y scale, -38 for log Y scale
    :param List[float] color: RGBA color as 4 float in [0, 1]
    :param List[float] offset: Translation of coordinates (ox, oy)
    """

    _PROGRAM = Program(
        vertexShader="""
        #version 120

        uniform mat4 matrix;
        attribute float xPos;
        attribute float yPos;

        void main(void) {
            gl_Position = matrix * vec4(xPos, yPos, 0.0, 1.0);
        }
        """,
        fragmentShader="""
        #version 120

        uniform vec4 color;

        void main(void) {
            gl_FragColor = color;
        }
        """,
        attrib0='xPos')

    def __init__(self, xData=None, yData=None,
                 baseline=0,
                 color=(0., 0., 0., 1.),
                 offset=(0., 0.)):
        self.xData = xData
        self.yData = yData
        self._xFillVboData = None
        self._yFillVboData = None
        self.color = color
        self.offset = offset

        # Offset baseline
        self.baseline = baseline - self.offset[1]

    def prepare(self):
        """Rendering preparation: build indices and bounding box vertices"""
        if (self._xFillVboData is None and
                self.xData is not None and self.yData is not None):

            # Get slices of not NaN values longer than 1 element
            isnan = numpy.logical_or(numpy.isnan(self.xData), numpy.isnan(self.yData))
            notnan = numpy.logical_not(isnan)
            start = numpy.where(numpy.logical_and(isnan[:-1], notnan[1:]))[0] + 1
            if notnan[0]:
                start = numpy.append(0, start)
            end = numpy.where(numpy.logical_and(notnan[:-1], isnan[1:]))[0] + 1
            if notnan[-1]:
                end = numpy.append(end, len(isnan))
            slices = numpy.transpose((start, end))
            # discard slices with less than length values
            slices = slices[numpy.diff(slices, axis=1).reshape(-1) >= 2]

            # Number of points: slice + 2 * leading and trailing points
            # Twice leading and trailing points to produce degenerated triangles
            nbPoints = numpy.sum(numpy.diff(slices, axis=1)) * 2 + 4 * len(slices)
            points = numpy.empty((nbPoints, 2), dtype=numpy.float32)

            offset = 0
            # invert baseline for filling
            new_y_data = numpy.append(self.yData, self.baseline)
            for start, end in slices:
                # Duplicate first point for connecting degenerated triangle
                points[offset:offset+2] = self.xData[start], new_y_data[start]

                # 2nd point of the polygon is last point
                points[offset+2] = self.xData[start], self.baseline[start]

                indices = numpy.append(numpy.arange(start, end),
                                       numpy.arange(len(self.xData) + end-1, len(self.xData) + start-1, -1))
                indices = indices[buildFillMaskIndices(len(indices))]

                points[offset+3:offset+3+len(indices), 0] = self.xData[indices % len(self.xData)]
                points[offset+3:offset+3+len(indices), 1] = new_y_data[indices]

                # Duplicate last point for connecting degenerated triangle
                points[offset+3+len(indices)] = points[offset+3+len(indices)-1]

                offset += len(indices) + 4

            self._xFillVboData, self._yFillVboData = vertexBuffer(points.T)

    def render(self, matrix):
        """Perform rendering

        :param numpy.ndarray matrix: 4x4 transform matrix to use
        """
        self.prepare()

        if self._xFillVboData is None:
            return  # Nothing to display

        self._PROGRAM.use()

        gl.glUniformMatrix4fv(
            self._PROGRAM.uniforms['matrix'], 1, gl.GL_TRUE,
            numpy.dot(matrix,
                      mat4Translate(*self.offset)).astype(numpy.float32))

        gl.glUniform4f(self._PROGRAM.uniforms['color'], *self.color)

        xPosAttrib = self._PROGRAM.attributes['xPos']
        yPosAttrib = self._PROGRAM.attributes['yPos']

        gl.glEnableVertexAttribArray(xPosAttrib)
        self._xFillVboData.setVertexAttrib(xPosAttrib)

        gl.glEnableVertexAttribArray(yPosAttrib)
        self._yFillVboData.setVertexAttrib(yPosAttrib)

        # Prepare fill mask
        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glStencilMask(1)
        gl.glStencilFunc(gl.GL_ALWAYS, 1, 1)
        gl.glStencilOp(gl.GL_INVERT, gl.GL_INVERT, gl.GL_INVERT)
        gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
        gl.glDepthMask(gl.GL_FALSE)

        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, self._xFillVboData.size)

        gl.glStencilFunc(gl.GL_EQUAL, 1, 1)
        # Reset stencil while drawing
        gl.glStencilOp(gl.GL_ZERO, gl.GL_ZERO, gl.GL_ZERO)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
        gl.glDepthMask(gl.GL_TRUE)

        # Draw directly in NDC
        gl.glUniformMatrix4fv(self._PROGRAM.uniforms['matrix'], 1, gl.GL_TRUE,
                              mat4Identity().astype(numpy.float32))

        # NDC vertices
        gl.glVertexAttribPointer(
            xPosAttrib, 1, gl.GL_FLOAT, gl.GL_FALSE, 0,
            numpy.array((-1., -1., 1., 1.), dtype=numpy.float32))
        gl.glVertexAttribPointer(
            yPosAttrib, 1, gl.GL_FLOAT, gl.GL_FALSE, 0,
            numpy.array((-1., 1., -1., 1.), dtype=numpy.float32))

        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        gl.glDisable(gl.GL_STENCIL_TEST)

    def discard(self):
        """Release VBOs"""
        if self._xFillVboData is not None:
            self._xFillVboData.vbo.discard()

        self._xFillVboData = None
        self._yFillVboData = None


# line ########################################################################

SOLID, DASHED, DASHDOT, DOTTED = '-', '--', '-.', ':'


class GLLines2D(object):
    """Object rendering curve as a polyline

    :param xVboData: X coordinates VBO
    :param yVboData: Y coordinates VBO
    :param colorVboData: VBO of colors
    :param distVboData: VBO of distance along the polyline
    :param str style: Line style in: '-', '--', '-.', ':'
    :param List[float] color: RGBA color as 4 float in [0, 1]
    :param float width: Line width
    :param float dashPeriod: Period of dashes
    :param drawMode: OpenGL drawing mode
    :param List[float] offset: Translation of coordinates (ox, oy)
    """

    STYLES = SOLID, DASHED, DASHDOT, DOTTED
    """Supported line styles"""

    _SOLID_PROGRAM = Program(
        vertexShader="""
        #version 120

        uniform mat4 matrix;
        attribute float xPos;
        attribute float yPos;
        attribute vec4 color;

        varying vec4 vColor;

        void main(void) {
            gl_Position = matrix * vec4(xPos, yPos, 0., 1.) ;
            vColor = color;
        }
        """,
        fragmentShader="""
        #version 120

        varying vec4 vColor;

        void main(void) {
            gl_FragColor = vColor;
        }
        """,
        attrib0='xPos')

    # Limitation: Dash using an estimate of distance in screen coord
    # to avoid computing distance when viewport is resized
    # results in inequal dashes when viewport aspect ratio is far from 1
    _DASH_PROGRAM = Program(
        vertexShader="""
        #version 120

        uniform mat4 matrix;
        uniform vec2 halfViewportSize;
        attribute float xPos;
        attribute float yPos;
        attribute vec4 color;
        attribute float distance;

        varying float vDist;
        varying vec4 vColor;

        void main(void) {
            gl_Position = matrix * vec4(xPos, yPos, 0., 1.);
            //Estimate distance in pixels
            vec2 probe = vec2(matrix * vec4(1., 1., 0., 0.)) *
                         halfViewportSize;
            float pixelPerDataEstimate = length(probe)/sqrt(2.);
            vDist = distance * pixelPerDataEstimate;
            vColor = color;
        }
        """,
        fragmentShader="""
        #version 120

        /* Dashes: [0, x], [y, z]
           Dash period: w */
        uniform vec4 dash;
        uniform vec4 dash2ndColor;

        varying float vDist;
        varying vec4 vColor;

        void main(void) {
            float dist = mod(vDist, dash.w);
            if ((dist > dash.x && dist < dash.y) || dist > dash.z) {
                if (dash2ndColor.a == 0.) {
                    discard;  // Discard full transparent bg color
                } else {
                    gl_FragColor = dash2ndColor;
                }
            } else {
                gl_FragColor = vColor;
            }
        }
        """,
        attrib0='xPos')

    def __init__(self, xVboData=None, yVboData=None,
                 colorVboData=None, distVboData=None,
                 style=SOLID, color=(0., 0., 0., 1.), dash2ndColor=None,
                 width=1, dashPeriod=10., drawMode=None,
                 offset=(0., 0.)):
        if (xVboData is not None and
                not isinstance(xVboData, VertexBufferAttrib)):
            xVboData = numpy.array(xVboData, copy=False, dtype=numpy.float32)
        self.xVboData = xVboData

        if (yVboData is not None and
                not isinstance(yVboData, VertexBufferAttrib)):
            yVboData = numpy.array(yVboData, copy=False, dtype=numpy.float32)
        self.yVboData = yVboData

        # Compute distances if not given while providing numpy array coordinates
        if (isinstance(self.xVboData, numpy.ndarray) and
                isinstance(self.yVboData, numpy.ndarray) and
                distVboData is None):
            distVboData = distancesFromArrays(self.xVboData, self.yVboData)

        if (distVboData is not None and
                not isinstance(distVboData, VertexBufferAttrib)):
            distVboData = numpy.array(
                distVboData, copy=False, dtype=numpy.float32)
        self.distVboData = distVboData

        if colorVboData is not None:
            assert isinstance(colorVboData, VertexBufferAttrib)
        self.colorVboData = colorVboData
        self.useColorVboData = colorVboData is not None

        self.color = color
        self.dash2ndColor = dash2ndColor
        self.width = width
        self._style = None
        self.style = style
        self.dashPeriod = dashPeriod
        self.offset = offset

        self._drawMode = drawMode if drawMode is not None else gl.GL_LINE_STRIP

    @property
    def style(self):
        """Line style (Union[str,None])"""
        return self._style

    @style.setter
    def style(self, style):
        if style in _MPL_NONES:
            self._style = None
        else:
            assert style in self.STYLES
            self._style = style

    @classmethod
    def init(cls):
        """OpenGL context initialization"""
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

    def render(self, matrix):
        """Perform rendering

        :param numpy.ndarray matrix: 4x4 transform matrix to use
        """
        style = self.style
        if style is None:
            return

        elif style == SOLID:
            program = self._SOLID_PROGRAM
            program.use()

        else:  # DASHED, DASHDOT, DOTTED
            program = self._DASH_PROGRAM
            program.use()

            x, y, viewWidth, viewHeight = gl.glGetFloatv(gl.GL_VIEWPORT)
            gl.glUniform2f(program.uniforms['halfViewportSize'],
                           0.5 * viewWidth, 0.5 * viewHeight)

            dashPeriod = self.dashPeriod * self.width
            if self.style == DOTTED:
                dash = (0.2 * dashPeriod,
                        0.5 * dashPeriod,
                        0.7 * dashPeriod,
                        dashPeriod)
            elif self.style == DASHDOT:
                dash = (0.3 * dashPeriod,
                        0.5 * dashPeriod,
                        0.6 * dashPeriod,
                        dashPeriod)
            else:
                dash = (0.5 * dashPeriod,
                        dashPeriod,
                        dashPeriod,
                        dashPeriod)

            gl.glUniform4f(program.uniforms['dash'], *dash)

            if self.dash2ndColor is None:
                # Use fully transparent color which gets discarded in shader
                dash2ndColor = (0., 0., 0., 0.)
            else:
                dash2ndColor = self.dash2ndColor
            gl.glUniform4f(program.uniforms['dash2ndColor'], *dash2ndColor)

            distAttrib = program.attributes['distance']
            gl.glEnableVertexAttribArray(distAttrib)
            if isinstance(self.distVboData, VertexBufferAttrib):
                self.distVboData.setVertexAttrib(distAttrib)
            else:
                gl.glVertexAttribPointer(distAttrib,
                                         1,
                                         gl.GL_FLOAT,
                                         False,
                                         0,
                                         self.distVboData)

        if self.width != 1:
            gl.glEnable(gl.GL_LINE_SMOOTH)

        matrix = numpy.dot(matrix,
                           mat4Translate(*self.offset)).astype(numpy.float32)
        gl.glUniformMatrix4fv(program.uniforms['matrix'],
                              1, gl.GL_TRUE, matrix)

        colorAttrib = program.attributes['color']
        if self.useColorVboData and self.colorVboData is not None:
            gl.glEnableVertexAttribArray(colorAttrib)
            self.colorVboData.setVertexAttrib(colorAttrib)
        else:
            gl.glDisableVertexAttribArray(colorAttrib)
            gl.glVertexAttrib4f(colorAttrib, *self.color)

        xPosAttrib = program.attributes['xPos']
        gl.glEnableVertexAttribArray(xPosAttrib)
        if isinstance(self.xVboData, VertexBufferAttrib):
            self.xVboData.setVertexAttrib(xPosAttrib)
        else:
            gl.glVertexAttribPointer(xPosAttrib,
                                     1,
                                     gl.GL_FLOAT,
                                     False,
                                     0,
                                     self.xVboData)

        yPosAttrib = program.attributes['yPos']
        gl.glEnableVertexAttribArray(yPosAttrib)
        if isinstance(self.yVboData, VertexBufferAttrib):
            self.yVboData.setVertexAttrib(yPosAttrib)
        else:
            gl.glVertexAttribPointer(yPosAttrib,
                                     1,
                                     gl.GL_FLOAT,
                                     False,
                                     0,
                                     self.yVboData)

        gl.glLineWidth(self.width)
        gl.glDrawArrays(self._drawMode, 0, self.xVboData.size)

        gl.glDisable(gl.GL_LINE_SMOOTH)


def distancesFromArrays(xData, yData):
    """Returns distances between each points

    :param numpy.ndarray xData: X coordinate of points
    :param numpy.ndarray yData: Y coordinate of points
    :rtype: numpy.ndarray
    """
    deltas = numpy.dstack((
        numpy.ediff1d(xData, to_begin=numpy.float32(0.)),
        numpy.ediff1d(yData, to_begin=numpy.float32(0.))))[0]
    return numpy.cumsum(numpy.sqrt(numpy.sum(deltas ** 2, axis=1)))


# points ######################################################################

DIAMOND, CIRCLE, SQUARE, PLUS, X_MARKER, POINT, PIXEL, ASTERISK = \
    'd', 'o', 's', '+', 'x', '.', ',', '*'

H_LINE, V_LINE, HEART = '_', '|', u'\u2665'

TICK_LEFT = "tickleft"
TICK_RIGHT = "tickright"
TICK_UP = "tickup"
TICK_DOWN = "tickdown"
CARET_LEFT = "caretleft"
CARET_RIGHT = "caretright"
CARET_UP = "caretup"
CARET_DOWN = "caretdown"


class _Points2D(object):
    """Object rendering curve markers

    :param xVboData: X coordinates VBO
    :param yVboData: Y coordinates VBO
    :param colorVboData: VBO of colors
    :param str marker: Kind of symbol to use, see :attr:`MARKERS`.
    :param List[float] color: RGBA color as 4 float in [0, 1]
    :param float size: Marker size
    :param List[float] offset: Translation of coordinates (ox, oy)
    """

    MARKERS = (DIAMOND, CIRCLE, SQUARE, PLUS, X_MARKER, POINT, PIXEL, ASTERISK,
               H_LINE, V_LINE, HEART, TICK_LEFT, TICK_RIGHT, TICK_UP, TICK_DOWN,
               CARET_LEFT, CARET_RIGHT, CARET_UP, CARET_DOWN)
    """List of supported markers"""

    _VERTEX_SHADER = """
    #version 120

    uniform mat4 matrix;
    uniform int transform;
    uniform float size;
    attribute float xPos;
    attribute float yPos;
    attribute vec4 color;

    varying vec4 vColor;

    void main(void) {
        gl_Position = matrix * vec4(xPos, yPos, 0., 1.);
        vColor = color;
        gl_PointSize = size;
    }
    """

    _FRAGMENT_SHADER_SYMBOLS = {
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
        """,
            HEART: """
        float alphaSymbol(vec2 coord, float size) {
            coord = (coord - 0.5) * 2.;
            coord *= 0.75;
            coord.y += 0.25;
            float a = atan(coord.x,-coord.y)/3.141593;
            float r = length(coord);
            float h = abs(a);
            float d = (13.0*h - 22.0*h*h + 10.0*h*h*h)/(6.0-5.0*h);
            float res = clamp(r-d, 0., 1.);
            // antialiasing
            res = smoothstep(0.1, 0.001, res);
            return res;
        }
        """,
            TICK_LEFT: """
        float alphaSymbol(vec2 coord, float size) {
            coord  = size * (coord - 0.5);
            float dy = abs(coord.y);
            if (dy < 0.5 && coord.x < 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        """,
            TICK_RIGHT: """
        float alphaSymbol(vec2 coord, float size) {
            coord  = size * (coord - 0.5);
            float dy = abs(coord.y);
            if (dy < 0.5 && coord.x > -0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        """,
            TICK_UP: """
        float alphaSymbol(vec2 coord, float size) {
            coord  = size * (coord - 0.5);
            float dx = abs(coord.x);
            if (dx < 0.5 && coord.y < 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        """,
            TICK_DOWN: """
        float alphaSymbol(vec2 coord, float size) {
            coord  = size * (coord - 0.5);
            float dx = abs(coord.x);
            if (dx < 0.5 && coord.y > -0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        """,
            CARET_LEFT: """
        float alphaSymbol(vec2 coord, float size) {
            coord  = size * (coord - 0.5);
            float d = abs(coord.x) - abs(coord.y);
            if (d >= -0.1 && coord.x > 0.5) {
                return smoothstep(-0.1, 0.1, d);
            } else {
                return 0.0;
            }
        }
        """,
            CARET_RIGHT: """
        float alphaSymbol(vec2 coord, float size) {
            coord  = size * (coord - 0.5);
            float d = abs(coord.x) - abs(coord.y);
            if (d >= -0.1 && coord.x < 0.5) {
                return smoothstep(-0.1, 0.1, d);
            } else {
                return 0.0;
            }
        }
        """,
            CARET_UP: """
        float alphaSymbol(vec2 coord, float size) {
            coord  = size * (coord - 0.5);
            float d = abs(coord.y) - abs(coord.x);
            if (d >= -0.1 && coord.y > 0.5) {
                return smoothstep(-0.1, 0.1, d);
            } else {
                return 0.0;
            }
        }
        """,
            CARET_DOWN: """
        float alphaSymbol(vec2 coord, float size) {
            coord  = size * (coord - 0.5);
            float d = abs(coord.y) - abs(coord.x);
            if (d >= -0.1 && coord.y < 0.5) {
                return smoothstep(-0.1, 0.1, d);
            } else {
                return 0.0;
            }
        }
        """,
    }

    _FRAGMENT_SHADER_TEMPLATE = """
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

    _PROGRAMS = {}

    def __init__(self, xVboData=None, yVboData=None, colorVboData=None,
                 marker=SQUARE, color=(0., 0., 0., 1.), size=7,
                 offset=(0., 0.)):
        self.color = color
        self._marker = None
        self.marker = marker
        self.size = size
        self.offset = offset

        self.xVboData = xVboData
        self.yVboData = yVboData
        self.colorVboData = colorVboData
        self.useColorVboData = colorVboData is not None

    @property
    def marker(self):
        """Symbol used to display markers (str)"""
        return self._marker

    @marker.setter
    def marker(self, marker):
        if marker in _MPL_NONES:
            self._marker = None
        else:
            assert marker in self.MARKERS
            self._marker = marker

    @classmethod
    def _getProgram(cls, marker):
        """On-demand shader program creation."""
        if marker == PIXEL:
            marker = SQUARE
        elif marker == POINT:
            marker = CIRCLE

        if marker not in cls._PROGRAMS:
            cls._PROGRAMS[marker] = Program(
                vertexShader=cls._VERTEX_SHADER,
                fragmentShader=(cls._FRAGMENT_SHADER_TEMPLATE %
                                cls._FRAGMENT_SHADER_SYMBOLS[marker]),
                attrib0='xPos')

        return cls._PROGRAMS[marker]

    @classmethod
    def init(cls):
        """OpenGL context initialization"""
        version = gl.glGetString(gl.GL_VERSION)
        majorVersion = int(version[0])
        assert majorVersion >= 2
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        if majorVersion >= 3:  # OpenGL 3
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    def render(self, matrix):
        """Perform rendering

        :param numpy.ndarray matrix: 4x4 transform matrix to use
        """
        if self.marker is None:
            return

        program = self._getProgram(self.marker)
        program.use()

        matrix = numpy.dot(matrix,
                           mat4Translate(*self.offset)).astype(numpy.float32)
        gl.glUniformMatrix4fv(program.uniforms['matrix'], 1, gl.GL_TRUE, matrix)

        if self.marker == PIXEL:
            size = 1
        elif self.marker == POINT:
            size = math.ceil(0.5 * self.size) + 1  # Mimic Matplotlib point
        else:
            size = self.size
        gl.glUniform1f(program.uniforms['size'], size)
        # gl.glPointSize(self.size)

        cAttrib = program.attributes['color']
        if self.useColorVboData and self.colorVboData is not None:
            gl.glEnableVertexAttribArray(cAttrib)
            self.colorVboData.setVertexAttrib(cAttrib)
        else:
            gl.glDisableVertexAttribArray(cAttrib)
            gl.glVertexAttrib4f(cAttrib, *self.color)

        xAttrib = program.attributes['xPos']
        gl.glEnableVertexAttribArray(xAttrib)
        self.xVboData.setVertexAttrib(xAttrib)

        yAttrib = program.attributes['yPos']
        gl.glEnableVertexAttribArray(yAttrib)
        self.yVboData.setVertexAttrib(yAttrib)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.xVboData.size)

        gl.glUseProgram(0)


# error bars ##################################################################

class _ErrorBars(object):
    """Display errors bars.

    This is using its own VBO as opposed to fill/points/lines.
    There is no picking on error bars.

    It uses 2 vertices per error bars and uses :class:`GLLines2D` to
    render error bars and :class:`_Points2D` to render the ends.

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
    :param List[float] color: RGBA color as 4 float in [0, 1]
    :param List[float] offset: Translation of coordinates (ox, oy)
    """

    def __init__(self, xData, yData, xError, yError,
                 xMin, yMin,
                 color=(0., 0., 0., 1.),
                 offset=(0., 0.)):
        self._attribs = None
        self._xMin, self._yMin = xMin, yMin
        self.offset = offset

        if xError is not None or yError is not None:
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

        self._lines = GLLines2D(
            None, None, color=color, drawMode=gl.GL_LINES, offset=offset)
        self._xErrPoints = _Points2D(
            None, None, color=color, marker=V_LINE, offset=offset)
        self._yErrPoints = _Points2D(
            None, None, color=color, marker=H_LINE, offset=offset)

    def _buildVertices(self):
        """Generates error bars vertices"""
        nbLinesPerDataPts = (0 if self._xError is None else 2) + \
                            (0 if self._yError is None else 2)

        nbDataPts = len(self._xData)

        # interleave coord+error, coord-error.
        # xError vertices first if any, then yError vertices if any.
        xCoords = numpy.empty(nbDataPts * nbLinesPerDataPts * 2,
                              dtype=numpy.float32)
        yCoords = numpy.empty(nbDataPts * nbLinesPerDataPts * 2,
                              dtype=numpy.float32)

        if self._xError is not None:  # errors on the X axis
            if len(self._xError.shape) == 2:
                xErrorMinus, xErrorPlus = self._xError[0], self._xError[1]
            else:
                # numpy arrays of len 1 or len(xData)
                xErrorMinus, xErrorPlus = self._xError, self._xError

            # Interleave vertices for xError
            endXError = 4 * nbDataPts
            xCoords[0:endXError-3:4] = self._xData + xErrorPlus
            xCoords[1:endXError-2:4] = self._xData
            xCoords[2:endXError-1:4] = self._xData
            xCoords[3:endXError:4] = self._xData - xErrorMinus

            yCoords[0:endXError-3:4] = self._yData
            yCoords[1:endXError-2:4] = self._yData
            yCoords[2:endXError-1:4] = self._yData
            yCoords[3:endXError:4] = self._yData

        else:
            endXError = 0

        if self._yError is not None:  # errors on the Y axis
            if len(self._yError.shape) == 2:
                yErrorMinus, yErrorPlus = self._yError[0], self._yError[1]
            else:
                # numpy arrays of len 1 or len(yData)
                yErrorMinus, yErrorPlus = self._yError, self._yError

            # Interleave vertices for yError
            xCoords[endXError::4] = self._xData
            xCoords[endXError+1::4] = self._xData
            xCoords[endXError+2::4] = self._xData
            xCoords[endXError+3::4] = self._xData

            yCoords[endXError::4] = self._yData + yErrorPlus
            yCoords[endXError+1::4] = self._yData
            yCoords[endXError+2::4] = self._yData
            yCoords[endXError+3::4] = self._yData - yErrorMinus

        return xCoords, yCoords

    def prepare(self):
        """Rendering preparation: build indices and bounding box vertices"""
        if self._xData is None:
            return

        if self._attribs is None:
            xCoords, yCoords = self._buildVertices()

            xAttrib, yAttrib = vertexBuffer((xCoords, yCoords))
            self._attribs = xAttrib, yAttrib

            self._lines.xVboData = xAttrib
            self._lines.yVboData = yAttrib

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

    def render(self, matrix):
        """Perform rendering

        :param numpy.ndarray matrix: 4x4 transform matrix to use
        """
        self.prepare()

        if self._attribs is not None:
            self._lines.render(matrix)
            self._xErrPoints.render(matrix)
            self._yErrPoints.render(matrix)

    def discard(self):
        """Release VBOs"""
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
                 lineStyle=SOLID,
                 lineColor=(0., 0., 0., 1.),
                 lineWidth=1,
                 lineDashPeriod=20,
                 marker=SQUARE,
                 markerColor=(0., 0., 0., 1.),
                 markerSize=7,
                 fillColor=None,
                 baseline=None,
                 isYLog=False):

        self.colorData = colorData

        # Compute x bounds
        if xError is None:
            self.xMin, self.xMax = min_max(xData, min_positive=False)
        else:
            # Takes the error into account
            if hasattr(xError, 'shape') and len(xError.shape) == 2:
                xErrorMinus, xErrorPlus = xError[0], xError[1]
            else:
                xErrorMinus, xErrorPlus = xError, xError
            self.xMin = numpy.nanmin(xData - xErrorMinus)
            self.xMax = numpy.nanmax(xData + xErrorPlus)

        # Compute y bounds
        if yError is None:
            self.yMin, self.yMax = min_max(yData, min_positive=False)
        else:
            # Takes the error into account
            if hasattr(yError, 'shape') and len(yError.shape) == 2:
                yErrorMinus, yErrorPlus = yError[0], yError[1]
            else:
                yErrorMinus, yErrorPlus = yError, yError
            self.yMin = numpy.nanmin(yData - yErrorMinus)
            self.yMax = numpy.nanmax(yData + yErrorPlus)

        # Handle data offset
        if xData.itemsize > 4 or yData.itemsize > 4:  # Use normalization
            # offset data, do not offset error as it is relative
            self.offset = self.xMin, self.yMin
            self.xData = (xData - self.offset[0]).astype(numpy.float32)
            self.yData = (yData - self.offset[1]).astype(numpy.float32)

        else:  # float32
            self.offset = 0., 0.
            self.xData = xData
            self.yData = yData
        if fillColor is not None:
            def deduce_baseline(baseline):
                if baseline is None:
                    _baseline = 0
                else:
                    _baseline = baseline
                if not isinstance(_baseline, numpy.ndarray):
                    _baseline = numpy.repeat(_baseline,
                                             len(self.xData))
                if isYLog is True:
                    with numpy.errstate(divide='ignore', invalid='ignore'):
                        log_val = numpy.log10(_baseline)
                        _baseline = numpy.where(_baseline>0.0, log_val, -38)
                return _baseline

            _baseline = deduce_baseline(baseline)

            # Use different baseline depending of Y log scale
            self.fill = _Fill2D(self.xData, self.yData,
                                baseline=_baseline,
                                color=fillColor,
                                offset=self.offset)
        else:
            self.fill = None

        self._errorBars = _ErrorBars(self.xData, self.yData,
                                     xError, yError,
                                     self.xMin, self.yMin,
                                     offset=self.offset)

        self.lines = GLLines2D()
        self.lines.style = lineStyle
        self.lines.color = lineColor
        self.lines.width = lineWidth
        self.lines.dashPeriod = lineDashPeriod
        self.lines.offset = self.offset

        self.points = _Points2D()
        self.points.marker = marker
        self.points.color = markerColor
        self.points.size = markerSize
        self.points.offset = self.offset

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
        """OpenGL context initialization"""
        GLLines2D.init()
        _Points2D.init()

    def prepare(self):
        """Rendering preparation: build indices and bounding box vertices"""
        if self.xVboData is None:
            xAttrib, yAttrib, cAttrib, dAttrib = None, None, None, None
            if self.lineStyle in (DASHED, DASHDOT, DOTTED):
                dists = distancesFromArrays(self.xData, self.yData)
                if self.colorData is None:
                    xAttrib, yAttrib, dAttrib = vertexBuffer(
                        (self.xData, self.yData, dists))
                else:
                    xAttrib, yAttrib, cAttrib, dAttrib = vertexBuffer(
                        (self.xData, self.yData, self.colorData, dists))
            elif self.colorData is None:
                xAttrib, yAttrib = vertexBuffer((self.xData, self.yData))
            else:
                xAttrib, yAttrib, cAttrib = vertexBuffer(
                    (self.xData, self.yData, self.colorData))

            self.xVboData = xAttrib
            self.yVboData = yAttrib
            self.distVboData = dAttrib

            if cAttrib is not None and self.colorData.dtype.kind == 'u':
                cAttrib.normalization = True  # Normalize uint to [0, 1]
            self.colorVboData = cAttrib
            self.useColorVboData = cAttrib is not None

    def render(self, matrix, isXLog, isYLog):
        """Perform rendering

        :param numpy.ndarray matrix: 4x4 transform matrix to use
        :param bool isXLog:
        :param bool isYLog:
        """
        self.prepare()
        if self.fill is not None:
            self.fill.render(matrix)
        self._errorBars.render(matrix)
        self.lines.render(matrix)
        self.points.render(matrix)

    def discard(self):
        """Release VBOs"""
        if self.xVboData is not None:
            self.xVboData.vbo.discard()

        self.xVboData = None
        self.yVboData = None
        self.colorVboData = None
        self.distVboData = None

        self._errorBars.discard()
        if self.fill is not None:
            self.fill.discard()

    def pick(self, xPickMin, yPickMin, xPickMax, yPickMax):
        """Perform picking on the curve according to its rendering.

        The picking area is [xPickMin, xPickMax], [yPickMin, yPickMax].

        In case a segment between 2 points with indices i, i+1 is picked,
        only its lower index end point (i.e., i) is added to the result.
        In case an end point with index i is picked it is added to the result,
        and the segment [i-1, i] is not tested for picking.

        :return: The indices of the picked data
        :rtype: Union[List[int],None]
        """
        if (self.marker is None and self.lineStyle is None) or \
                self.xMin > xPickMax or xPickMin > self.xMax or \
                self.yMin > yPickMax or yPickMin > self.yMax:
            return None

        # offset picking bounds
        xPickMin = xPickMin - self.offset[0]
        xPickMax = xPickMax - self.offset[0]
        yPickMin = yPickMin - self.offset[1]
        yPickMax = yPickMax - self.offset[1]

        if self.lineStyle is not None:
            # Using Cohen-Sutherland algorithm for line clipping
            with numpy.errstate(invalid='ignore'):  # Ignore NaN comparison warnings
                codes = ((self.yData > yPickMax) << 3) | \
                    ((self.yData < yPickMin) << 2) | \
                    ((self.xData > xPickMax) << 1) | \
                    (self.xData < xPickMin)

            notNaN = numpy.logical_not(numpy.logical_or(
                numpy.isnan(self.xData), numpy.isnan(self.yData)))

            # Add all points that are inside the picking area
            indices = numpy.nonzero(
                numpy.logical_and(codes == 0, notNaN))[0].tolist()

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
            with numpy.errstate(invalid='ignore'):  # Ignore NaN comparison warnings
                indices = numpy.nonzero((self.xData >= xPickMin) &
                                        (self.xData <= xPickMax) &
                                        (self.yData >= yPickMin) &
                                        (self.yData <= yPickMax))[0].tolist()

        return tuple(indices) if len(indices) > 0 else None
