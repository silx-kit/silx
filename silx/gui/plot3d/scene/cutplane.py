# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""A cut plane in a 3D texture: hackish implementation...
"""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/10/2016"

import string
import numpy

from .. import glutils
from ..glutils import gl

from .function import Colormap
from .primitives import Box, Geometry, PlaneInGroup
from . import transform, utils


class ColormapMesh3D(Geometry):
    """A 3D mesh with color from a 3D texture."""

    _shaders = ("""
    attribute vec3 position;
    attribute vec3 normal;

    uniform mat4 matrix;
    uniform mat4 transformMat;
    //uniform mat3 matrixInvTranspose;

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec3 vTexCoords;

    void main(void)
    {
        vCameraPosition = transformMat * vec4(position, 1.0);
        //vNormal = matrixInvTranspose * normalize(normal);
        vPosition = position;
        vNormal = normal;
        gl_Position = matrix * vec4(position, 1.0);
    }
    """,
                string.Template("""
    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    uniform sampler3D data;
    uniform float alpha;
    uniform vec3 dataScale;

    $colormapDecl

    $clippingDecl
    $lightingFunction

    void main(void)
    {
        float value = texture3D(data, vPosition * dataScale).r;
        vec4 color = $colormapCall(value);
        color.a = alpha;

        $clippingCall(vCameraPosition);

        gl_FragColor = $lightingCall(color, vPosition, vNormal);
    }
    """))

    def __init__(self, position, normal, data, copy=True,
                 mode='triangles', indices=None, colormap=None):
        assert mode in self._TRIANGLE_MODES
        data = numpy.array(data, copy=copy, order='C')
        assert data.ndim == 3
        self._data = data
        self._texture = None
        self._update_texture = True
        self._alpha = 1.
        self._colormap = colormap or Colormap()  # Default colormap
        self._colormap.addListener(self._cmapChanged)
        self._interpolation = 'linear'
        super(ColormapMesh3D, self).__init__(mode,
                                             indices,
                                             position=position,
                                             normal=normal)

        self.isBackfaceVisible = True

    def setData(self, data, copy=True):
        data = numpy.array(data, copy=copy, order='C')
        assert data.ndim == 3
        self._data = data
        self._update_texture = True

    def getData(self, copy=True):
        return numpy.array(self._data, copy=copy)

    @property
    def interpolation(self):
        """The texture interpolation mode: 'linear' or 'nearest'"""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        assert interpolation in ('linear', 'nearest')
        self._interpolation = interpolation
        # TODO improve, now reload texture to change filter parameters
        self._update_texture = True

    @property
    def alpha(self):
        """Transparency of the plane, float in [0, 1]"""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)

    @property
    def colormap(self):
        """The colormap used by this primitive"""
        return self._colormap

    def _cmapChanged(self, source, *args, **kwargs):
        """Broadcast colormap changes"""
        self.notify(*args, **kwargs)

    def prepareGL2(self, ctx):
        if self._texture is None or self._update_texture:
            if self._texture is not None:
                self._texture.discard()

            if self.interpolation == 'nearest':
                filter_ = gl.GL_NEAREST
            else:
                filter_ = gl.GL_LINEAR
            self._update_texture = False
            self._texture = glutils.Texture(
                gl.GL_R32F, self._data, gl.GL_RED,
                minFilter=filter_,
                magFilter=filter_,
                wrap=gl.GL_CLAMP_TO_EDGE)
        super(ColormapMesh3D, self).prepareGL2(ctx)

    def renderGL2(self, ctx):
        fragment = self._shaders[1].substitute(
            clippingDecl=ctx.clipper.fragDecl,
            clippingCall=ctx.clipper.fragCall,
            lightingFunction=ctx.viewport.light.fragmentDef,
            lightingCall=ctx.viewport.light.fragmentCall,
            colormapDecl=self.colormap.decl,
            colormapCall=self.colormap.call
            )
        program = ctx.glCtx.prog(self._shaders[0], fragment)
        program.use()

        ctx.viewport.light.setupProgram(ctx, program)
        self.colormap.setupProgram(ctx, program)

        if not self.isBackfaceVisible:
            gl.glCullFace(gl.GL_BACK)
            gl.glEnable(gl.GL_CULL_FACE)

        program.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        program.setUniformMatrix('transformMat',
                                 ctx.objectToCamera.matrix,
                                 safe=True)
        gl.glUniform1f(program.uniforms['alpha'], self._alpha)

        shape = self._data.shape
        scales = 1./shape[2], 1./shape[1], 1./shape[0]
        gl.glUniform3f(program.uniforms['dataScale'], *scales)

        gl.glUniform1i(program.uniforms['data'], self._texture.texUnit)

        ctx.clipper.setupProgram(ctx, program)

        self._texture.bind()
        self._draw(program)

        if not self.isBackfaceVisible:
            gl.glDisable(gl.GL_CULL_FACE)


class CutPlane(PlaneInGroup):
    """A cutting plane in a 3D texture"""

    def __init__(self, point=(0., 0., 0.), normal=(0., 0., 1.)):
        self._data = None
        self._mesh = None
        self._alpha = 1.
        self._colormap = Colormap()
        super(CutPlane, self).__init__(point, normal)

    def setData(self, data, copy=True):
        if data is None:
            self._data = None
            if self._mesh is not None:
                self._children.remove(self._mesh)
            self._mesh = None

        else:
            data = numpy.array(data, copy=copy, order='C')
            assert data.ndim == 3
            self._data = data
            if self._mesh is not None:
                self._mesh.setData(data, copy=False)

    def getData(self, copy=True):
        return None if self._mesh is None else self._mesh.getData(copy=copy)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)
        if self._mesh is not None:
            self._mesh.alpha = alpha

    @property
    def colormap(self):
        return self._colormap

    def prepareGL2(self, ctx):
        if self.isValid:
            if self._mesh is None and self._data is not None:
                self._mesh = ColormapMesh3D(self.contourVertices,
                                            normal=self.plane.normal,
                                            data=self._data,
                                            copy=False,
                                            mode='fan',
                                            colormap=self.colormap)
                self._mesh.alpha = self._alpha
                self._children.insert(0, self._mesh)

            if self._mesh is not None:
                if (self.contourVertices is None or
                        len(self.contourVertices) == 0):
                    self._mesh.visible = False
                else:
                    self._mesh.visible = True
                    self._mesh.setAttribute('normal', self.plane.normal)
                    self._mesh.setAttribute('position', self.contourVertices)

        super(CutPlane, self).prepareGL2(ctx)

    def renderGL2(self, ctx):
        with self.viewport.light.turnOff():
            super(CutPlane, self).renderGL2(ctx)

    def _bounds(self, dataBounds=False):
        if not dataBounds:
            vertices = self.contourVertices
            if vertices is not None:
                return numpy.array(
                    (vertices.min(axis=0), vertices.max(axis=0)),
                    dtype=numpy.float32)
            else:
                return None  # Plane in not slicing the data volume
        else:
            if self._data is None:
                return None
            else:
                depth, height, width = self._data.shape
                return numpy.array(((0., 0., 0.),
                                    (width, height, depth)),
                                   dtype=numpy.float32)

    @property
    def contourVertices(self):
        """The vertices of the contour of the plane/bounds intersection."""
        # TODO copy from PlaneInGroup, refactor all that!
        bounds = self.bounds(dataBounds=True)
        if bounds is None:
            return None  # No bounds: no vertices

        # Check if cache is valid and return it
        cachebounds, cachevertices = self._cache
        if numpy.all(numpy.equal(bounds, cachebounds)):
            return cachevertices

        # Cache is not OK, rebuild it
        boxvertices = bounds[0] + Box._vertices.copy()*(bounds[1] - bounds[0])
        lineindices = Box._lineIndices
        vertices = utils.boxPlaneIntersect(
            boxvertices, lineindices, self.plane.normal, self.plane.point)

        self._cache = bounds, vertices if len(vertices) != 0 else None

        return self._cache[1]

    # Render transforms RW, TODO refactor this!
    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, iterable):
        self._transforms.removeListener(self._transformChanged)
        if isinstance(iterable, transform.TransformList):
            # If it is a TransformList, do not create one to enable sharing.
            self._transforms = iterable
        else:
            assert hasattr(iterable, '__iter__')
            self._transforms = transform.TransformList(iterable)
        self._transforms.addListener(self._transformChanged)
