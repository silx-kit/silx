# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
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

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import string
import numpy

from .. import glutils
from ..glutils import gl

from . import function
from . import primitives
from . import utils


# SurfacePlot #################################################################

class SurfacePlot(primitives.Geometry):
    """A set of values arranged on a 2D regular grid.

    Visualised as a height + colormap.
    Works with a texture so data size is limited by texture size.
    """
    # Simple implementation: everything gets displayed
    # TODO automatic sub-sampling according to the extent in pixels

    # TODO lighting
    # TODO for now x and curves in [0, 1]
    # TODO for now value in [0, 1] and unique colormap
    # TODO use gl_VertexID
    # TODO double check coordinate system

    _VALUES_TEX_UNIT = 0

    _INTERNAL_FORMATS = {
        numpy.dtype(numpy.float32): gl.GL_R32F,
        # TODO add normalization to support uint8, uint16
        # Use normalized integer for unsigned int formats
        # numpy.dtype(numpy.uint16): gl.GL_R16,
        # numpy.dtype(numpy.uint8): gl.GL_R8,
    }

    _shaders = ("""
    #version 120

    attribute vec2 position;

    uniform mat4 matrix;
    uniform sampler2D valuesTex;

    varying vec2 vPosition;

    void main(void)
    {
        float value = texture2DLod(valuesTex, position, 0.0).r;
        vPosition = position;
        gl_Position = matrix * vec4(position, value, 1.0);
    }
    """,
                string.Template("""
    #version 120

    $colormapDecl

    uniform sampler2D valuesTex;

    varying vec2 vPosition;

    void main(void)
    {
        float value = texture2D(valuesTex, vPosition).r;
        gl_FragColor = $colormapCall(value);
    }
    """))

    def __init__(self, data, x=None, curves=None, strides=(1, 1),
                 colormap=None):
        """Surface plot.

        :param numpy.ndarray data: 2D array of data as float32.
        :param numpy.ndarray x: 1D array of X coordinates of the data points.
                                If None, using increasing integers.
        :param numpy.ndarray curves: 1D array of curve positions in 3D.
                                     If None, using increasing integers.
        :param strides: Sub-sampling parameter to make the grid.
        :type strides: 2-tuple of floats.
        :param dict colormap: Colormap description (See
            :meth:`PyMca5.PyMcaGraph.PlotBackend.getDefaultColormap`
            for details) or None for default.
        """
        assert len(data.shape) == 2
        assert data.dtype in self._INTERNAL_FORMATS
        self._data = data
        self._texture = None
        self._dirtyTexture = True

        if x is None:
            x = numpy.arange(data.shape[1], dtype=numpy.float32)
        assert len(x) == data.shape[1]

        if curves is None:
            curves = numpy.arange(data.shape[0], dtype=numpy.float32)
        assert len(curves) == data.shape[0]

        # Sub-sampling
        if strides[0] > 1:
            curves = curves[::strides[0]]
        if strides[1] > 1:
            x = x[::strides[1]]

        # TODO Can avoid to set-up all vertices for the grid
        # If grid is regular, compute vertex from gl_VertexID and len(x)
        # If grid is not regular, can get x and curve from 2 textures?
        # GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS >= 16 for OpenGL 3.3
        # but >= 0 for OpenGL 2.1
        grid = utils.gridVertices(curves, x, dtype=numpy.float32)
        grid.shape = -1, 2  # 2D to be understood by Geometry

        # TODO as triangles
        # tStripIndices = utils.triangleStripGridIndices(len(curves), len(x))
        # super(SurfacePlot, self).__init__(mode='triangle_strip',
        #                                  indices=tStripIndices,
        #                                  position=grid)

        indices = utils.linesGridIndices(len(curves), len(x))
        super(SurfacePlot, self).__init__(mode='lines',
                                          indices=indices,
                                          position=grid)

        if colormap is None:
            colormap = function.Colormap(
                range_=(self._data.min(), self._data.max()))
        self._colormap = colormap
        self._colormap.addListener(self._updated)

    def _bounds(self, dataBounds=False):
        bounds = super(SurfacePlot, self)._bounds(dataBounds)
        bounds[:, 2] = self._data.min(), self._data.max()
        return bounds

    def _updated(self, source, *args, **kwargs):
        if source is not self:
            self.notify(*args, **kwargs)

    @property
    def colormap(self):
        """The colormap in use."""
        return self._colormap

    def updateData(self, data):
        """Update data with data associated to surface plot.

        :param data: Data.
        :type data: 2D numpy.ndarray.
        """
        assert data.shape == self._data.shape
        assert data.dtype == self._data.dtype
        self._data = data
        self._dirtyTexture = True
        self.notify()

    def prepareGL2(self, ctx):
        super(SurfacePlot, self).prepareGL2(ctx)
        if self._dirtyTexture:
            self._dirtyTexture = False

            internalformat = self._INTERNAL_FORMATS[self._data.dtype]

            if self._texture is not None:
                self._texture.discard()

            # TODO store in GL context
            self._texture = glutils.Texture(internalformat,
                                            data=self._data,
                                            format_=gl.GL_RED,
                                            texUnit=self._VALUES_TEX_UNIT,
                                            minFilter=gl.GL_NEAREST)

    def renderGL2(self, ctx):
        fragment = self._shaders[1].substitute(
            colormapDecl=self._colormap.decl,
            colormapCall=self._colormap.call)
        prog = ctx.glCtx.prog(self._shaders[0], fragment)
        prog.use()

        self.useAttribute(prog)

        self._colormap.setupProgram(ctx, prog)

        gl.glLineWidth(1.0)

        gl.glUniformMatrix4fv(prog.uniforms['matrix'], 1, gl.GL_TRUE,
                              ctx.objectToNDC.matrix)
        gl.glUniform1i(prog.uniforms['valuesTex'], self._VALUES_TEX_UNIT)

        self._texture.bind(self._VALUES_TEX_UNIT)

        gl.glDisable(gl.GL_LINE_SMOOTH)
        self._draw()
        gl.glEnable(gl.GL_LINE_SMOOTH)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)  # Unbind texture
