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
This module provides convenient classes and functions for OpenGL rendering.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


import numpy

from ...._glutils import gl


def buildFillMaskIndices(nIndices, dtype=None):
    """Returns triangle strip indices for rendering a filled polygon mask

    :param int nIndices: Number of points
    :param Union[numpy.dtype,None] dtype:
       If specified the dtype of the returned indices array
    :return: 1D array of indices constructing a triangle strip
    :rtype: numpy.ndarray
    """
    if dtype is None:
        if nIndices <= numpy.iinfo(numpy.uint16).max + 1:
            dtype = numpy.uint16
        else:
            dtype = numpy.uint32

    lastIndex = nIndices - 1
    splitIndex = lastIndex // 2 + 1
    indices = numpy.empty(nIndices, dtype=dtype)
    indices[::2] = numpy.arange(0, splitIndex, step=1, dtype=dtype)
    indices[1::2] = numpy.arange(lastIndex, splitIndex - 1, step=-1,
                                 dtype=dtype)
    return indices


class Shape2D(object):
    _NO_HATCH = 0
    _HATCH_STEP = 20

    def __init__(self, points, fill='solid', stroke=True,
                 fillColor=(0., 0., 0., 1.), strokeColor=(0., 0., 0., 1.),
                 strokeClosed=True):
        self.vertices = numpy.array(points, dtype=numpy.float32, copy=False)
        self.strokeClosed = strokeClosed

        self._indices = buildFillMaskIndices(len(self.vertices))

        tVertex = numpy.transpose(self.vertices)
        xMin, xMax = min(tVertex[0]), max(tVertex[0])
        yMin, yMax = min(tVertex[1]), max(tVertex[1])
        self.bboxVertices = numpy.array(((xMin, yMin), (xMin, yMax),
                                         (xMax, yMin), (xMax, yMax)),
                                        dtype=numpy.float32)
        self._xMin, self._xMax = xMin, xMax
        self._yMin, self._yMax = yMin, yMax

        self.fill = fill
        self.fillColor = fillColor
        self.stroke = stroke
        self.strokeColor = strokeColor

    @property
    def xMin(self):
        return self._xMin

    @property
    def xMax(self):
        return self._xMax

    @property
    def yMin(self):
        return self._yMin

    @property
    def yMax(self):
        return self._yMax

    def prepareFillMask(self, posAttrib):
        gl.glEnableVertexAttribArray(posAttrib)
        gl.glVertexAttribPointer(posAttrib,
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0, self.vertices)

        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glStencilMask(1)
        gl.glStencilFunc(gl.GL_ALWAYS, 1, 1)
        gl.glStencilOp(gl.GL_INVERT, gl.GL_INVERT, gl.GL_INVERT)
        gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
        gl.glDepthMask(gl.GL_FALSE)

        gl.glDrawElements(gl.GL_TRIANGLE_STRIP, len(self._indices),
                          gl.GL_UNSIGNED_SHORT, self._indices)

        gl.glStencilFunc(gl.GL_EQUAL, 1, 1)
        # Reset stencil while drawing
        gl.glStencilOp(gl.GL_ZERO, gl.GL_ZERO, gl.GL_ZERO)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
        gl.glDepthMask(gl.GL_TRUE)

    def renderFill(self, posAttrib):
        self.prepareFillMask(posAttrib)

        gl.glVertexAttribPointer(posAttrib,
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0, self.bboxVertices)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.bboxVertices))

        gl.glDisable(gl.GL_STENCIL_TEST)

    def renderStroke(self, posAttrib):
        gl.glEnableVertexAttribArray(posAttrib)
        gl.glVertexAttribPointer(posAttrib,
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0, self.vertices)
        gl.glLineWidth(1)
        drawMode = gl.GL_LINE_LOOP if self.strokeClosed else gl.GL_LINE_STRIP
        gl.glDrawArrays(drawMode, 0, len(self.vertices))

    def render(self, posAttrib, colorUnif, hatchStepUnif):
        assert self.fill in ['hatch', 'solid', None]
        if self.fill is not None:
            gl.glUniform4f(colorUnif, *self.fillColor)
            step = self._HATCH_STEP if self.fill == 'hatch' else self._NO_HATCH
            gl.glUniform1i(hatchStepUnif, step)
            self.renderFill(posAttrib)

        if self.stroke:
            gl.glUniform4f(colorUnif, *self.strokeColor)
            gl.glUniform1i(hatchStepUnif, self._NO_HATCH)
            self.renderStroke(posAttrib)


# matrix ######################################################################

def mat4Ortho(left, right, bottom, top, near, far):
    """Orthographic projection matrix (row-major)"""
    return numpy.array((
        (2./(right - left), 0., 0., -(right+left)/float(right-left)),
        (0., 2./(top - bottom), 0., -(top+bottom)/float(top-bottom)),
        (0., 0., -2./(far-near),    -(far+near)/float(far-near)),
        (0., 0., 0., 1.)), dtype=numpy.float64)


def mat4Translate(x=0., y=0., z=0.):
    """Translation matrix (row-major)"""
    return numpy.array((
        (1., 0., 0., x),
        (0., 1., 0., y),
        (0., 0., 1., z),
        (0., 0., 0., 1.)), dtype=numpy.float64)


def mat4Scale(sx=1., sy=1., sz=1.):
    """Scale matrix (row-major)"""
    return numpy.array((
        (sx, 0., 0., 0.),
        (0., sy, 0., 0.),
        (0., 0., sz, 0.),
        (0., 0., 0., 1.)), dtype=numpy.float64)


def mat4Identity():
    """Identity matrix"""
    return numpy.array((
        (1., 0., 0., 0.),
        (0., 1., 0., 0.),
        (0., 0., 1., 0.),
        (0., 0., 0., 1.)), dtype=numpy.float64)
