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

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import collections
import ctypes
from functools import reduce
import logging
import string

import numpy

from silx.gui.plot.Colors import rgba

from ... import _glutils
from ..._glutils import gl

from . import event
from . import core
from . import transform
from . import utils

_logger = logging.getLogger(__name__)


# Geometry ####################################################################

class Geometry(core.Elem):
    """Set of vertices with normals and colors.

    :param str mode: OpenGL drawing mode:
                     lines, line_strip, loop, triangles, triangle_strip, fan
    :param indices: Array of vertex indices or None
    :param bool copy: True (default) to copy the data, False to use as is.
    :param attributes: Provide list of attributes as extra parameters.
    """

    _ATTR_INFO = {
        'position': {'dims': (1, 2), 'lastDim': (2, 3, 4)},
        'normal': {'dims': (1, 2), 'lastDim': (3,)},
        'color': {'dims': (1, 2), 'lastDim': (3, 4)},
    }

    _MODE_CHECKS = {  # Min, Modulo
        'lines': (2, 2), 'line_strip': (2, 0), 'loop': (2, 0),
        'points': (1, 0),
        'triangles': (3, 3), 'triangle_strip': (3, 0), 'fan': (3, 0)
    }

    _MODES = {
        'lines': gl.GL_LINES,
        'line_strip': gl.GL_LINE_STRIP,
        'loop': gl.GL_LINE_LOOP,

        'points': gl.GL_POINTS,

        'triangles': gl.GL_TRIANGLES,
        'triangle_strip': gl.GL_TRIANGLE_STRIP,
        'fan': gl.GL_TRIANGLE_FAN
    }

    _LINE_MODES = 'lines', 'line_strip', 'loop'

    _TRIANGLE_MODES = 'triangles', 'triangle_strip', 'fan'

    def __init__(self, mode, indices=None, copy=True, **attributes):
        super(Geometry, self).__init__()

        self._vbos = {}  # Store current vbos
        self._unsyncAttributes = []  # Store attributes to copy to vbos
        self.__bounds = None  # Cache object's bounds

        assert mode in self._MODES
        self._mode = mode

        # Set attributes
        self._attributes = {}
        for name, data in attributes.items():
            self.setAttribute(name, data, copy=copy)

        # Set indices
        self._indices = None
        self.setIndices(indices, copy=copy)

        # More consistency checks
        mincheck, modulocheck = self._MODE_CHECKS[self._mode]
        if self._indices is not None:
            nbvertices = len(self._indices)
        else:
            nbvertices = self.nbVertices
        assert nbvertices >= mincheck
        if modulocheck != 0:
            assert (nbvertices % modulocheck) == 0

    @staticmethod
    def _glReadyArray(array, copy=True):
        """Making a contiguous array, checking float types.

        :param iterable array: array-like data to prepare for attribute
        :param bool copy: True to make a copy of the array, False to use as is
        """
        # Convert single value (int, float, numpy types) to tuple
        if not isinstance(array, collections.Iterable):
            array = (array, )

        # Makes sure it is an array
        array = numpy.array(array, copy=False)

        # Cast all float to float32
        dtype = None
        if numpy.dtype(array.dtype).kind == 'f':
            dtype = numpy.float32

        return numpy.array(array, dtype=dtype, order='C', copy=copy)

    @property
    def nbVertices(self):
        """Returns the number of vertices of current attributes.

        It returns None if there is no attributes.
        """
        for array in self._attributes.values():
            if len(array.shape) == 2:
                return len(array)
        return None

    def setAttribute(self, name, array, copy=True):
        """Set attribute with provided array.

        :param str name: The name of the attribute
        :param array: Array-like attribute data or None to remove attribute
        :param bool copy: True (default) to copy the data, False to use as is
        """
        # This triggers associated GL resources to be garbage collected
        self._vbos.pop(name, None)

        if array is None:
            self._attributes.pop(name, None)

        else:
            array = self._glReadyArray(array, copy=copy)

            if name not in self._ATTR_INFO:
                _logger.info('Not checking attibute %s dimensions', name)
            else:
                checks = self._ATTR_INFO[name]

                if (len(array.shape) == 1 and checks['lastDim'] == (1,) and
                        len(array) > 1):
                    array = array.reshape((len(array), 1))

                # Checks
                assert len(array.shape) in checks['dims'], "Attr %s" % name
                assert array.shape[-1] in checks['lastDim'], "Attr %s" % name

            # Check length against another attribute array
            # Causes problems when updating
            # nbVertices = self.nbVertices
            # if len(array.shape) == 2 and nbVertices is not None:
            #     assert len(array) == nbVertices

            self._attributes[name] = array
            if len(array.shape) == 2:  # Store this in a VBO
                self._unsyncAttributes.append(name)

            if name == 'position':  # Reset bounds
                self.__bounds = None

        self.notify()

    def getAttribute(self, name, copy=True):
        """Returns the numpy.ndarray corresponding to the name attribute.

        :param str name: The name of the attribute to get.
        :param bool copy: True to get a copy (default),
                          False to get internal array (DO NOT MODIFY)
        :return: The corresponding array or None if no corresponding attribute.
        :rtype: numpy.ndarray
        """
        attr = self._attributes.get(name, None)
        return None if attr is None else numpy.array(attr, copy=copy)

    def useAttribute(self, program, name=None):
        """Enable and bind attribute(s) for a specific program.

        This MUST be called with OpenGL context active and after prepareGL2
        has been called.

        :param GLProgram program: The program for which to set the attributes
        :param str name: The attribute name to set or None to set then all
        """
        if name is None:
            for name in program.attributes:
                self.useAttribute(program, name)

        else:
            attribute = program.attributes.get(name)
            if attribute is None:
                return

            vboattrib = self._vbos.get(name)
            if vboattrib is not None:
                gl.glEnableVertexAttribArray(attribute)
                vboattrib.setVertexAttrib(attribute)

            elif name not in self._attributes:
                gl.glDisableVertexAttribArray(attribute)

            else:
                array = self._attributes[name]
                assert array is not None

                if len(array.shape) == 1:
                    assert len(array) in (1, 2, 3, 4)
                    gl.glDisableVertexAttribArray(attribute)
                    _glVertexAttribFunc = getattr(
                        _glutils.gl, 'glVertexAttrib{}f'.format(len(array)))
                    _glVertexAttribFunc(attribute, *array)
                else:
                    # TODO As is this is a never event, remove?
                    gl.glEnableVertexAttribArray(attribute)
                    gl.glVertexAttribPointer(
                        attribute,
                        array.shape[-1],
                        _glutils.numpyToGLType(array.dtype),
                        gl.GL_FALSE,
                        0,
                        array)

    def setIndices(self, indices, copy=True):
        """Set the primitive indices to use.

        :param indices: Array-like of uint primitive indices or None to unset
        :param bool copy: True (default) to copy the data, False to use as is
        """
        # Trigger garbage collection of previous indices VBO if any
        self._vbos.pop('__indices__', None)

        if indices is None:
            self._indices = None
        else:
            indices = self._glReadyArray(indices, copy=copy).ravel()
            assert indices.dtype.name in ('uint8', 'uint16', 'uint32')
            if _logger.getEffectiveLevel() <= logging.DEBUG:
                # This might be a costy check
                assert indices.max() < self.nbVertices
            self._indices = indices

    def getIndices(self, copy=True):
        """Returns the numpy.ndarray corresponding to the indices.

        :param bool copy: True to get a copy (default),
                          False to get internal array (DO NOT MODIFY)
        :return: The primitive indices array or None if not set.
        :rtype: numpy.ndarray or None
        """
        if self._indices is None:
            return None
        else:
            return numpy.array(self._indices, copy=copy)

    def _bounds(self, dataBounds=False):
        if self.__bounds is None:
            self.__bounds = numpy.zeros((2, 3), dtype=numpy.float32)
            # Support vertex with to 2 to 4 coordinates
            positions = self._attributes['position']
            self.__bounds[0, :positions.shape[1]] = \
                numpy.nanmin(positions, axis=0)[:3]
            self.__bounds[1, :positions.shape[1]] = \
                numpy.nanmax(positions, axis=0)[:3]
            self.__bounds[numpy.isnan(self.__bounds)] = 0.  # Avoid NaNs
        return self.__bounds.copy()

    def prepareGL2(self, ctx):
        # TODO manage _vbo and multiple GL context + allow to share them !
        # TODO make one or multiple VBO depending on len(vertices),
        # TODO use a general common VBO for small amount of data
        for name in self._unsyncAttributes:
            array = self._attributes[name]
            self._vbos[name] = ctx.glCtx.makeVboAttrib(array)
        self._unsyncAttributes = []

        if self._indices is not None and '__indices__' not in self._vbos:
            vbo = ctx.glCtx.makeVbo(self._indices,
                                    usage=gl.GL_STATIC_DRAW,
                                    target=gl.GL_ELEMENT_ARRAY_BUFFER)
            self._vbos['__indices__'] = vbo

    def _draw(self, program=None, nbVertices=None):
        """Perform OpenGL draw calls.

        :param GLProgram program:
            If not None, call :meth:`useAttribute` for this program.
        :param int nbVertices:
            The number of vertices to render or None to render all vertices.
        """
        if program is not None:
            self.useAttribute(program)

        if self._indices is None:
            if nbVertices is None:
                nbVertices = self.nbVertices
            gl.glDrawArrays(self._MODES[self._mode], 0, nbVertices)
        else:
            if nbVertices is None:
                nbVertices = self._indices.size
            with self._vbos['__indices__']:
                gl.glDrawElements(self._MODES[self._mode],
                                  nbVertices,
                                  _glutils.numpyToGLType(self._indices.dtype),
                                  ctypes.c_void_p(0))


# Lines #######################################################################

class Lines(Geometry):
    """A set of segments"""
    _shaders = ("""
    attribute vec3 position;
    attribute vec3 normal;
    attribute vec4 color;

    uniform mat4 matrix;
    uniform mat4 transformMat;

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec4 vColor;

    void main(void)
    {
        gl_Position = matrix * vec4(position, 1.0);
        vCameraPosition = transformMat * vec4(position, 1.0);
        vPosition = position;
        vNormal = normal;
        vColor = color;
    }
    """,
                string.Template("""
    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec4 vColor;

    $clippingDecl
    $lightingFunction

    void main(void)
    {
        $clippingCall(vCameraPosition);
        gl_FragColor = $lightingCall(vColor, vPosition, vNormal);
    }
    """))

    def __init__(self, positions, normals=None, colors=(1., 1., 1., 1.),
                 indices=None, mode='lines', width=1.):
        if mode == 'strip':
            mode = 'line_strip'
        assert mode in self._LINE_MODES

        self._width = width
        self._smooth = True

        super(Lines, self).__init__(mode, indices,
                                    position=positions,
                                    normal=normals,
                                    color=colors)

    width = event.notifyProperty('_width', converter=float,
                                 doc="Width of the line in pixels.")

    smooth = event.notifyProperty(
        '_smooth',
        converter=bool,
        doc="Smooth line rendering enabled (bool, default: True)")

    def renderGL2(self, ctx):
        # Prepare program
        isnormals = 'normal' in self._attributes
        if isnormals:
            fraglightfunction = ctx.viewport.light.fragmentDef
        else:
            fraglightfunction = ctx.viewport.light.fragmentShaderFunctionNoop

        fragment = self._shaders[1].substitute(
            clippingDecl=ctx.clipper.fragDecl,
            clippingCall=ctx.clipper.fragCall,
            lightingFunction=fraglightfunction,
            lightingCall=ctx.viewport.light.fragmentCall)
        prog = ctx.glCtx.prog(self._shaders[0], fragment)
        prog.use()

        if isnormals:
            ctx.viewport.light.setupProgram(ctx, prog)

        gl.glLineWidth(self.width)

        prog.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        prog.setUniformMatrix('transformMat',
                              ctx.objectToCamera.matrix,
                              safe=True)

        ctx.clipper.setupProgram(ctx, prog)

        with gl.enabled(gl.GL_LINE_SMOOTH, self._smooth):
            self._draw(prog)


class DashedLines(Lines):
    """Set of dashed lines

    This MUST be defined as a set of lines (no strip or loop).
    """

    _shaders = ("""
    attribute vec3 position;
    attribute vec3 origin;
    attribute vec3 normal;
    attribute vec4 color;

    uniform mat4 matrix;
    uniform mat4 transformMat;
    uniform vec2 viewportSize;  /* Width, height of the viewport */

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec4 vColor;
    varying vec2 vOriginFragCoord;

    void main(void)
    {
        gl_Position = matrix * vec4(position, 1.0);
        vCameraPosition = transformMat * vec4(position, 1.0);
        vPosition = position;
        vNormal = normal;
        vColor = color;

        vec4 clipOrigin = matrix * vec4(origin, 1.0);
        vec4 ndcOrigin = clipOrigin / clipOrigin.w;  /* Perspective divide */
        /* Convert to same frame as gl_FragCoord: lower-left, pixel center at 0.5, 0.5 */
        vOriginFragCoord = (ndcOrigin.xy + vec2(1.0, 1.0)) * 0.5 * viewportSize + vec2(0.5, 0.5);
    }
    """,  # noqa
                string.Template("""
    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec4 vColor;
    varying vec2 vOriginFragCoord;

    uniform vec2 dash;

    $clippingDecl
    $lightingFunction

    void main(void)
    {
        /* Discard off dash fragments */
        float lineDist = distance(vOriginFragCoord, gl_FragCoord.xy);
        if (mod(lineDist, dash.x + dash.y) > dash.x) {
            discard;
        }
        $clippingCall(vCameraPosition);
        gl_FragColor = $lightingCall(vColor, vPosition, vNormal);
    }
    """))

    def __init__(self, positions, colors=(1., 1., 1., 1.),
                 indices=None, width=1.):
        self._dash = 1, 0
        super(DashedLines, self).__init__(positions=positions,
                                          colors=colors,
                                          indices=indices,
                                          mode='lines',
                                          width=width)

    @property
    def dash(self):
        """Dash of the line as a 2-tuple of lengths in pixels: (on, off)"""
        return self._dash

    @dash.setter
    def dash(self, dash):
        dash = float(dash[0]), float(dash[1])
        if dash != self._dash:
            self._dash = dash
            self.notify()

    def getPositions(self, copy=True):
        """Get coordinates of lines.

        :param bool copy: True to get a copy, False otherwise
        :returns: Coordinates of lines
        :rtype: numpy.ndarray of float32 of shape (N, 2, Ndim)
        """
        return self.getAttribute('position', copy=copy)

    def setPositions(self, positions, copy=True):
        """Set line coordinates.

        :param positions: Array of line coordinates
        :param bool copy: True to copy input array, False to use as is
        """
        self.setAttribute('position', positions, copy=copy)
        # Update line origins from given positions
        origins = numpy.array(positions, copy=True, order='C')
        origins[1::2] = origins[::2]
        self.setAttribute('origin', origins, copy=False)

    def renderGL2(self, context):
        # Prepare program
        isnormals = 'normal' in self._attributes
        if isnormals:
            fraglightfunction = context.viewport.light.fragmentDef
        else:
            fraglightfunction = \
                context.viewport.light.fragmentShaderFunctionNoop

        fragment = self._shaders[1].substitute(
            clippingDecl=context.clipper.fragDecl,
            clippingCall=context.clipper.fragCall,
            lightingFunction=fraglightfunction,
            lightingCall=context.viewport.light.fragmentCall)
        program = context.glCtx.prog(self._shaders[0], fragment)
        program.use()

        if isnormals:
            context.viewport.light.setupProgram(context, program)

        gl.glLineWidth(self.width)

        program.setUniformMatrix('matrix', context.objectToNDC.matrix)
        program.setUniformMatrix('transformMat',
                                 context.objectToCamera.matrix,
                                 safe=True)

        gl.glUniform2f(
            program.uniforms['viewportSize'], *context.viewport.size)
        gl.glUniform2f(program.uniforms['dash'], *self.dash)

        context.clipper.setupProgram(context, program)

        self._draw(program)


class Box(core.PrivateGroup):
    """Rectangular box"""

    _lineIndices = numpy.array((
        (0, 1), (1, 2), (2, 3), (3, 0),  # Lines with z=0
        (0, 4), (1, 5), (2, 6), (3, 7),  # Lines from z=0 to z=1
        (4, 5), (5, 6), (6, 7), (7, 4)),  # Lines with z=1
        dtype=numpy.uint8)

    _faceIndices = numpy.array(
        (0, 3, 1, 2, 5, 6, 4, 7, 7, 6, 6, 2, 7, 3, 4, 0, 5, 1),
        dtype=numpy.uint8)

    _vertices = numpy.array((
        # Corners with z=0
        (0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.),
        # Corners with z=1
        (0., 0., 1.), (1., 0., 1.), (1., 1., 1.), (0., 1., 1.)),
        dtype=numpy.float32)

    def __init__(self, size=(1., 1., 1.),
                 stroke=(1., 1., 1., 1.),
                 fill=(1., 1., 1., 0.)):
        super(Box, self).__init__()

        self._fill = Mesh3D(self._vertices,
                            colors=rgba(fill),
                            mode='triangle_strip',
                            indices=self._faceIndices)
        self._fill.visible = self.fillColor[-1] != 0.

        self._stroke = Lines(self._vertices,
                             indices=self._lineIndices,
                             colors=rgba(stroke),
                             mode='lines')
        self._stroke.visible = self.strokeColor[-1] != 0.
        self.strokeWidth = 1.

        self._children = [self._stroke, self._fill]

        self._size = None
        self.size = size

    @property
    def size(self):
        """Size of the box (sx, sy, sz)"""
        return self._size

    @size.setter
    def size(self, size):
        assert len(size) == 3
        size = tuple(size)
        if size != self.size:
            self._size = size
            self._fill.setAttribute(
                'position',
                self._vertices * numpy.array(size, dtype=numpy.float32))
            self._stroke.setAttribute(
                'position',
                self._vertices * numpy.array(size, dtype=numpy.float32))
            self.notify()

    @property
    def strokeSmooth(self):
        """True to draw smooth stroke, False otherwise"""
        return self._stroke.smooth

    @strokeSmooth.setter
    def strokeSmooth(self, smooth):
        smooth = bool(smooth)
        if smooth != self._stroke.smooth:
            self._stroke.smooth = smooth
            self.notify()

    @property
    def strokeWidth(self):
        """Width of the stroke (float)"""
        return self._stroke.width

    @strokeWidth.setter
    def strokeWidth(self, width):
        width = float(width)
        if width != self.strokeWidth:
            self._stroke.width = width
            self.notify()

    @property
    def strokeColor(self):
        """RGBA color of the box lines (4-tuple of float in [0, 1])"""
        return tuple(self._stroke.getAttribute('color', copy=False))

    @strokeColor.setter
    def strokeColor(self, color):
        color = rgba(color)
        if color != self.strokeColor:
            self._stroke.setAttribute('color', color)
            # Fully transparent = hidden
            self._stroke.visible = color[-1] != 0.
            self.notify()

    @property
    def fillColor(self):
        """RGBA color of the box faces (4-tuple of float in [0, 1])"""
        return tuple(self._fill.getAttribute('color', copy=False))

    @fillColor.setter
    def fillColor(self, color):
        color = rgba(color)
        if color != self.fillColor:
            self._fill.setAttribute('color', color)
            # Fully transparent = hidden
            self._fill.visible = color[-1] != 0.
            self.notify()

    @property
    def fillCulling(self):
        return self._fill.culling

    @fillCulling.setter
    def fillCulling(self, culling):
        self._fill.culling = culling


class Axes(Lines):
    """3D RGB orthogonal axes"""
    _vertices = numpy.array(((0., 0., 0.), (1., 0., 0.),
                             (0., 0., 0.), (0., 1., 0.),
                             (0., 0., 0.), (0., 0., 1.)),
                            dtype=numpy.float32)

    _colors = numpy.array(((255, 0, 0, 255), (255, 0, 0, 255),
                           (0, 255, 0, 255), (0, 255, 0, 255),
                           (0, 0, 255, 255), (0, 0, 255, 255)),
                          dtype=numpy.uint8)

    def __init__(self):
        super(Axes, self).__init__(self._vertices,
                                   colors=self._colors,
                                   width=3.)


class BoxWithAxes(Lines):
    """Rectangular box with RGB OX, OY, OZ axes

    :param color: RGBA color of the box
    """

    _vertices = numpy.array((
        # Axes corners
        (0., 0., 0.), (1., 0., 0.),
        (0., 0., 0.), (0., 1., 0.),
        (0., 0., 0.), (0., 0., 1.),
        # Box corners with z=0
        (1., 0., 0.), (1., 1., 0.), (0., 1., 0.),
        # Box corners with z=1
        (0., 0., 1.), (1., 0., 1.), (1., 1., 1.), (0., 1., 1.)),
        dtype=numpy.float32)

    _axesColors = numpy.array(((1., 0., 0., 1.), (1., 0., 0., 1.),
                               (0., 1., 0., 1.), (0., 1., 0., 1.),
                               (0., 0., 1., 1.), (0., 0., 1., 1.)),
                              dtype=numpy.float32)

    _lineIndices = numpy.array((
        (0, 1), (2, 3), (4, 5),  # Axes lines
        (6, 7), (7, 8),  # Box lines with z=0
        (6, 10), (7, 11), (8, 12),  # Box lines from z=0 to z=1
        (9, 10), (10, 11), (11, 12), (12, 9)),  # Box lines with z=1
        dtype=numpy.uint8)

    def __init__(self, color=(1., 1., 1., 1.)):
        self._color = (1., 1., 1., 1.)
        colors = numpy.ones((len(self._vertices), 4), dtype=numpy.float32)
        colors[:len(self._axesColors), :] = self._axesColors

        super(BoxWithAxes, self).__init__(self._vertices,
                                          indices=self._lineIndices,
                                          colors=colors,
                                          width=2.)
        self.color = color

    @property
    def color(self):
        """The RGBA color to use for the box: 4 float in [0, 1]"""
        return self._color

    @color.setter
    def color(self, color):
        color = rgba(color)
        if color != self._color:
            self._color = color
            colors = numpy.empty((len(self._vertices), 4), dtype=numpy.float32)
            colors[:len(self._axesColors), :] = self._axesColors
            colors[len(self._axesColors):, :] = self._color
            self.setAttribute('color', colors)  # Do the notification


class PlaneInGroup(core.PrivateGroup):
    """A plane using its parent bounds to display a contour.

    If plane is outside the bounds of its parent, it is not visible.

    Cannot set the transform attribute of this primitive.
    This primitive never has any bounds.
    """
    # TODO inherit from Lines directly?, make sure the plane remains visible?

    def __init__(self, point=(0., 0., 0.), normal=(0., 0., 1.)):
        super(PlaneInGroup, self).__init__()
        self._cache = None, None  # Store bounds, vertices
        self._outline = None

        self._color = None
        self.color = 1., 1., 1., 1.  # Set _color
        self._width = 2.

        self._plane = utils.Plane(point, normal)
        self._plane.addListener(self._planeChanged)

    def moveToCenter(self):
        """Place the plane at the center of the data, not changing orientation.
        """
        if self.parent is not None:
            bounds = self.parent.bounds(dataBounds=True)
            if bounds is not None:
                center = (bounds[0] + bounds[1]) / 2.
                _logger.debug('Moving plane to center: %s', str(center))
                self.plane.point = center

    @property
    def color(self):
        """Plane outline color (array of 4 float in [0, 1])."""
        return self._color.copy()

    @color.setter
    def color(self, color):
        self._color = numpy.array(color, copy=True, dtype=numpy.float32)
        if self._outline is not None:
            self._outline.setAttribute('color', self._color)
        self.notify()  # This is OK as Lines are rebuild for each rendering

    @property
    def width(self):
        """Width of the plane stroke in pixels"""
        return self._width

    @width.setter
    def width(self, width):
        self._width = float(width)
        if self._outline is not None:
            self._outline.width = self._width  # Sync width

    # Plane access

    @property
    def plane(self):
        """The plane parameters in the frame of the object."""
        return self._plane

    def _planeChanged(self, source):
        """Listener of plane changes: clear cache and notify listeners."""
        self._cache = None, None
        self.notify()

    # Disable some scene features

    @property
    def transforms(self):
        # Ready-only transforms to prevent using it
        return self._transforms

    def _bounds(self, dataBounds=False):
        # This is bound less as it uses the bounds of its parent.
        return None

    @property
    def contourVertices(self):
        """The vertices of the contour of the plane/bounds intersection."""
        parent = self.parent
        if parent is None:
            return None  # No parent: no vertices

        bounds = parent.bounds(dataBounds=True)
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

    @property
    def center(self):
        """The center of the plane/bounds intersection points."""
        if not self.isValid:
            return None
        else:
            return numpy.mean(self.contourVertices, axis=0)

    @property
    def isValid(self):
        """True if a contour is defined, False otherwise."""
        return self.plane.isPlane and self.contourVertices is not None

    def prepareGL2(self, ctx):
        if self.isValid:
            if self._outline is None:  # Init outline
                self._outline = Lines(self.contourVertices,
                                      mode='loop',
                                      colors=self.color)
                self._outline.width = self._width
                self._children.append(self._outline)

            # Update vertices, TODO only when necessary
            self._outline.setAttribute('position', self.contourVertices)

            super(PlaneInGroup, self).prepareGL2(ctx)

    def renderGL2(self, ctx):
        if self.isValid:
            super(PlaneInGroup, self).renderGL2(ctx)


# Points ######################################################################

_POINTS_ATTR_INFO = Geometry._ATTR_INFO.copy()
_POINTS_ATTR_INFO.update(value={'dims': (1, 2), 'lastDim': (1,)},
                         size={'dims': (1, 2), 'lastDim': (1,)},
                         symbol={'dims': (1, 2), 'lastDim': (1,)})


class Points(Geometry):
    """A set of data points with an associated value and size."""
    _shaders = ("""
    #version 120

    attribute vec3 position;
    attribute float symbol;
    attribute float value;
    attribute float size;

    uniform mat4 matrix;
    uniform mat4 transformMat;

    uniform vec2 valRange;

    varying vec4 vCameraPosition;
    varying float vSymbol;
    varying float vNormValue;
    varying float vSize;

    void main(void)
    {
        vSymbol = symbol;

        vNormValue = clamp((value - valRange.x) / (valRange.y - valRange.x),
                           0.0, 1.0);

        bool isValueInRange = value >= valRange.x && value <= valRange.y;
        if (isValueInRange) {
            gl_Position = matrix * vec4(position, 1.0);
        } else {
            gl_Position = vec4(2.0, 0.0, 0.0, 1.0); /* Get clipped */
        }
        vCameraPosition = transformMat * vec4(position, 1.0);

        gl_PointSize = size;
        vSize = size;
    }
    """,
                string.Template("""
    #version 120

    varying vec4 vCameraPosition;
    varying float vSize;
    varying float vSymbol;
    varying float vNormValue;

    $clippinDecl

    /* Circle */
    #define SYMBOL_CIRCLE 1.0

    float alphaCircle(vec2 coord, float size) {
        float radius = 0.5;
        float r = distance(coord, vec2(0.5, 0.5));
        return clamp(size * (radius - r), 0.0, 1.0);
    }

    /* Half lines */
    #define SYMBOL_H_LINE 2.0
    #define LEFT 1.0
    #define RIGHT 2.0
    #define SYMBOL_V_LINE 3.0
    #define UP 1.0
    #define DOWN 2.0

    float alphaLine(vec2 coord, float size, float direction)
    {
        vec2 delta = abs(size * (coord - 0.5));

        if (direction == SYMBOL_H_LINE) {
            return (delta.y < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_H_LINE + LEFT) {
            return (coord.x <= 0.5 && delta.y < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_H_LINE + RIGHT) {
            return (coord.x >= 0.5 && delta.y < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_V_LINE) {
            return (delta.x < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_V_LINE + UP) {
            return (coord.y <= 0.5 && delta.x < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_V_LINE + DOWN) {
             return (coord.y >= 0.5 && delta.x < 0.5) ? 1.0 : 0.0;
        }
        return 1.0;
    }

    void main(void)
    {
        $clippingCall(vCameraPosition);

        gl_FragColor = vec4(0.5 * vNormValue + 0.5, 0.0, 0.0, 1.0);

        float alpha = 1.0;
        float symbol = floor(vSymbol);
        if (1 == 1) { //symbol == SYMBOL_CIRCLE) {
            alpha = alphaCircle(gl_PointCoord, vSize);
        }
        else if (symbol >= SYMBOL_H_LINE &&
                 symbol <= (SYMBOL_V_LINE + DOWN)) {
            alpha = alphaLine(gl_PointCoord, vSize, symbol);
        }
        if (alpha == 0.0) {
            discard;
        }
        gl_FragColor.a *= alpha;
    }
    """))

    _ATTR_INFO = _POINTS_ATTR_INFO

    # TODO Add colormap, light?

    def __init__(self, vertices, values=0., sizes=1., indices=None,
                 symbols=0.,
                 minValue=None, maxValue=None):
        super(Points, self).__init__('points', indices,
                                     position=vertices,
                                     value=values,
                                     size=sizes,
                                     symbol=symbols)

        values = self._attributes['value']
        self._minValue = values.min() if minValue is None else minValue
        self._maxValue = values.max() if maxValue is None else maxValue

    minValue = event.notifyProperty('_minValue')
    maxValue = event.notifyProperty('_maxValue')

    def renderGL2(self, ctx):
        fragment = self._shaders[1].substitute(
            clippingDecl=ctx.clipper.fragDecl,
            clippingCall=ctx.clipper.fragCall)
        prog = ctx.glCtx.prog(self._shaders[0], fragment)
        prog.use()

        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        # gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        prog.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        prog.setUniformMatrix('transformMat',
                              ctx.objectToCamera.matrix,
                              safe=True)

        ctx.clipper.setupProgram(ctx, prog)

        gl.glUniform2f(prog.uniforms['valRange'], self.minValue, self.maxValue)

        self._draw(prog)


class ColorPoints(Geometry):
    """A set of points with an associated color and size."""

    _shaders = ("""
    #version 120

    attribute vec3 position;
    attribute float symbol;
    attribute vec4 color;
    attribute float size;

    uniform mat4 matrix;
    uniform mat4 transformMat;

    varying vec4 vCameraPosition;
    varying float vSymbol;
    varying vec4 vColor;
    varying float vSize;

    void main(void)
    {
        vCameraPosition = transformMat * vec4(position, 1.0);
        vSymbol = symbol;
        vColor = color;
        gl_Position = matrix * vec4(position, 1.0);
        gl_PointSize = size;
        vSize = size;
    }
    """,
                string.Template("""
    #version 120

    varying vec4 vCameraPosition;
    varying float vSize;
    varying float vSymbol;
    varying vec4 vColor;

    $clippingDecl;

    /* Circle */
    #define SYMBOL_CIRCLE 1.0

    float alphaCircle(vec2 coord, float size) {
        float radius = 0.5;
        float r = distance(coord, vec2(0.5, 0.5));
        return clamp(size * (radius - r), 0.0, 1.0);
    }

    /* Half lines */
    #define SYMBOL_H_LINE 2.0
    #define LEFT 1.0
    #define RIGHT 2.0
    #define SYMBOL_V_LINE 3.0
    #define UP 1.0
    #define DOWN 2.0

    float alphaLine(vec2 coord, float size, float direction)
    {
        vec2 delta = abs(size * (coord - 0.5));

        if (direction == SYMBOL_H_LINE) {
            return (delta.y < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_H_LINE + LEFT) {
            return (coord.x <= 0.5 && delta.y < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_H_LINE + RIGHT) {
            return (coord.x >= 0.5 && delta.y < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_V_LINE) {
            return (delta.x < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_V_LINE + UP) {
            return (coord.y <= 0.5 && delta.x < 0.5) ? 1.0 : 0.0;
        }
        else if (direction == SYMBOL_V_LINE + DOWN) {
             return (coord.y >= 0.5 && delta.x < 0.5) ? 1.0 : 0.0;
        }
        return 1.0;
    }

    void main(void)
    {
        $clippingCall(vCameraPosition);

        gl_FragColor = vColor;

        float alpha = 1.0;
        float symbol = floor(vSymbol);
        if (1 == 1) { //symbol == SYMBOL_CIRCLE) {
            alpha = alphaCircle(gl_PointCoord, vSize);
        }
        else if (symbol >= SYMBOL_H_LINE &&
                 symbol <= (SYMBOL_V_LINE + DOWN)) {
            alpha = alphaLine(gl_PointCoord, vSize, symbol);
        }
        if (alpha == 0.0) {
            discard;
        }
        gl_FragColor.a *= alpha;
    }
    """))

    _ATTR_INFO = _POINTS_ATTR_INFO

    def __init__(self, vertices, colors=(1., 1., 1., 1.), sizes=1.,
                 indices=None, symbols=0.,
                 minValue=None, maxValue=None):
        super(ColorPoints, self).__init__('points', indices,
                                          position=vertices,
                                          color=colors,
                                          size=sizes,
                                          symbol=symbols)

    def renderGL2(self, ctx):
        fragment = self._shaders[1].substitute(
            clippingDecl=ctx.clipper.fragDecl,
            clippingCall=ctx.clipper.fragCall)
        prog = ctx.glCtx.prog(self._shaders[0], fragment)
        prog.use()

        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        # gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        prog.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        prog.setUniformMatrix('transformMat',
                              ctx.objectToCamera.matrix,
                              safe=True)

        ctx.clipper.setupProgram(ctx, prog)

        self._draw(prog)


class GridPoints(Geometry):
    # GLSL 1.30 !
    """Data points on a regular grid with an associated value and size."""
    _shaders = ("""
    #version 130

    in float value;
    in float size;

    uniform ivec3 gridDims;
    uniform mat4 matrix;
    uniform mat4 transformMat;
    uniform vec2 valRange;

    out vec4 vCameraPosition;
    out float vNormValue;

    //ivec3 coordsFromIndex(int index, ivec3 shape)
    //{
        /*Assumes that data is stored as z-major, then y, contiguous on x
        */
   //     int yxPlaneSize = shape.y * shape.x; /* nb of elem in 2d yx plane */
   //     int z = index / yxPlaneSize;
   //     int yxIndex = index - z * yxPlaneSize; /* index in 2d yx plane */
   //     int y = yxIndex / shape.x;
   //     int x = yxIndex - y * shape.x;
   //     return ivec3(x, y, z);
   // }

    ivec3 coordsFromIndex(int index, ivec3 shape)
    {
        /*Assumes that data is stored as x-major, then y, contiguous on z
        */
        int yzPlaneSize = shape.y * shape.z; /* nb of elem in 2d yz plane */
        int x = index / yzPlaneSize;
        int yzIndex = index - x * yzPlaneSize; /* index in 2d yz plane */
        int y = yzIndex / shape.z;
        int z = yzIndex - y * shape.z;
        return ivec3(x, y, z);
    }

    void main(void)
    {
        vNormValue = clamp((value - valRange.x) / (valRange.y - valRange.x),
                           0.0, 1.0);

        bool isValueInRange = value >= valRange.x && value <= valRange.y;
        if (isValueInRange) {
            /* Retrieve 3D position from gridIndex */
            vec3 coords = vec3(coordsFromIndex(gl_VertexID, gridDims));
            vec3 position = coords / max(vec3(gridDims) - 1.0, 1.0);
            gl_Position = matrix * vec4(position, 1.0);
            vCameraPosition = transformMat * vec4(position, 1.0);
        } else {
            gl_Position = vec4(2.0, 0.0, 0.0, 1.0); /* Get clipped */
            vCameraPosition = vec4(0.0, 0.0, 0.0, 0.0);
        }

        gl_PointSize = size;
    }
    """,
                string.Template("""
    #version 130

    in vec4 vCameraPosition;
    in float vNormValue;
    out vec4 fragColor;

    $clippingDecl

    void main(void)
    {
        $clippingCall(vCameraPosition);

        fragColor = vec4(0.5 * vNormValue + 0.5, 0.0, 0.0, 1.0);
    }
    """))

    _ATTR_INFO = {
        'value': {'dims': (1, 2), 'lastDim': (1,)},
        'size': {'dims': (1, 2), 'lastDim': (1,)}
    }

    # TODO Add colormap, shape?
    # TODO could also use a texture to store values

    def __init__(self, values=0., shape=None, sizes=1., indices=None,
                 minValue=None, maxValue=None):
        if isinstance(values, collections.Iterable):
            values = numpy.array(values, copy=False)

            # Test if gl_VertexID will overflow
            assert values.size < numpy.iinfo(numpy.int32).max

            self._shape = values.shape
            values = values.ravel()  # 1D to add as a 1D vertex attribute

        else:
            assert shape is not None
            self._shape = tuple(shape)

        assert len(self._shape) in (1, 2, 3)

        super(GridPoints, self).__init__('points', indices,
                                         value=values,
                                         size=sizes)

        data = self.getAttribute('value', copy=False)
        self._minValue = data.min() if minValue is None else minValue
        self._maxValue = data.max() if maxValue is None else maxValue

    minValue = event.notifyProperty('_minValue')
    maxValue = event.notifyProperty('_maxValue')

    def _bounds(self, dataBounds=False):
        # Get bounds from values shape
        bounds = numpy.zeros((2, 3), dtype=numpy.float32)
        bounds[1, :] = self._shape
        bounds[1, :] -= 1
        return bounds

    def renderGL2(self, ctx):
        fragment = self._shaders[1].substitute(
            clippingDecl=ctx.clipper.fragDecl,
            clippingCall=ctx.clipper.fragCall)
        prog = ctx.glCtx.prog(self._shaders[0], fragment)
        prog.use()

        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        # gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        prog.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        prog.setUniformMatrix('transformMat',
                              ctx.objectToCamera.matrix,
                              safe=True)

        ctx.clipper.setupProgram(ctx, prog)

        gl.glUniform3i(prog.uniforms['gridDims'],
                       self._shape[2] if len(self._shape) == 3 else 1,
                       self._shape[1] if len(self._shape) >= 2 else 1,
                       self._shape[0])

        gl.glUniform2f(prog.uniforms['valRange'], self.minValue, self.maxValue)

        self._draw(prog, nbVertices=reduce(lambda a, b: a * b, self._shape))


# Spheres #####################################################################

class Spheres(Geometry):
    """A set of spheres.

    Spheres are rendered as circles using points.
    This brings some limitations:
    - Do not support non-uniform scaling.
    - Assume the projection keeps ratio.
    - Do not render distorion by perspective projection.
    - If the sphere center is clipped, the whole sphere is not displayed.
    """
    # TODO check those links
    # Accounting for perspective projection
    # http://iquilezles.org/www/articles/sphereproj/sphereproj.htm

    # Michael Mara and Morgan McGuire.
    # 2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere
    # Journal of Computer Graphics Techniques, Vol. 2, No. 2, 2013.
    # http://jcgt.org/published/0002/02/05/paper.pdf
    # https://research.nvidia.com/publication/2d-polyhedral-bounds-clipped-perspective-projected-3d-sphere

    # TODO some issues with small scaling and regular grid or due to sampling

    _shaders = ("""
    #version 120

    attribute vec3 position;
    attribute vec4 color;
    attribute float radius;

    uniform mat4 transformMat;
    uniform mat4 projMat;
    uniform vec2 screenSize;

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec4 vColor;
    varying float vViewDepth;
    varying float vViewRadius;

    void main(void)
    {
        vCameraPosition = transformMat * vec4(position, 1.0);
        gl_Position = projMat * vCameraPosition;

        vPosition = gl_Position.xyz / gl_Position.w;

        /* From object space radius to view space diameter.
         * Do not support non-uniform scaling */
        vec4 viewSizeVector = transformMat * vec4(2.0 * radius, 0.0, 0.0, 0.0);
        float viewSize = length(viewSizeVector.xyz);

        /* Convert to pixel size at the xy center of the view space */
        vec4 projSize = projMat * vec4(0.5 * viewSize, 0.0,
                                       vCameraPosition.z, vCameraPosition.w);
        gl_PointSize = max(1.0, screenSize[0] * projSize.x / projSize.w);

        vColor = color;
        vViewRadius = 0.5 * viewSize;
        vViewDepth = vCameraPosition.z;
    }
    """,
                string.Template("""
    # version 120

    uniform mat4 projMat;

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec4 vColor;
    varying float vViewDepth;
    varying float vViewRadius;

    $clippingDecl
    $lightingFunction

    void main(void)
    {
        $clippingCall(vCameraPosition);

        /* Get normal from point coords */
        vec3 normal;
        normal.xy = 2.0 * gl_PointCoord - vec2(1.0);
        normal.y *= -1.0; /*Invert y to match NDC orientation*/
        float sqLength = dot(normal.xy, normal.xy);
        if (sqLength > 1.0) { /* Length -> out of sphere */
            discard;
        }
        normal.z = sqrt(1.0 - sqLength);

        /*Lighting performed in NDC*/
        /*TODO update this when lighting changed*/
        //XXX vec3 position = vPosition + vViewRadius * normal;
        gl_FragColor = $lightingCall(vColor, vPosition, normal);

        /*Offset depth*/
        float viewDepth = vViewDepth + vViewRadius * normal.z;
        vec2 clipZW = viewDepth * projMat[2].zw + projMat[3].zw;
        gl_FragDepth = 0.5 * (clipZW.x / clipZW.y) + 0.5;
    }
    """))

    _ATTR_INFO = {
        'position': {'dims': (2, ), 'lastDim': (2, 3, 4)},
        'radius': {'dims': (1, 2), 'lastDim': (1, )},
        'color': {'dims': (1, 2), 'lastDim': (3, 4)},
    }

    def __init__(self, positions, radius=1., colors=(1., 1., 1., 1.)):
        self.__bounds = None
        super(Spheres, self).__init__('points', None,
                                      position=positions,
                                      radius=radius,
                                      color=colors)

    def renderGL2(self, ctx):
        fragment = self._shaders[1].substitute(
            clippingDecl=ctx.clipper.fragDecl,
            clippingCall=ctx.clipper.fragCall,
            lightingFunction=ctx.viewport.light.fragmentDef,
            lightingCall=ctx.viewport.light.fragmentCall)
        prog = ctx.glCtx.prog(self._shaders[0], fragment)
        prog.use()

        ctx.viewport.light.setupProgram(ctx, prog)

        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        # gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        prog.setUniformMatrix('projMat', ctx.projection.matrix)
        prog.setUniformMatrix('transformMat',
                              ctx.objectToCamera.matrix,
                              safe=True)

        ctx.clipper.setupProgram(ctx, prog)

        gl.glUniform2f(prog.uniforms['screenSize'], *ctx.viewport.size)

        self._draw(prog)

    def _bounds(self, dataBounds=False):
        if self.__bounds is None:
            self.__bounds = numpy.zeros((2, 3), dtype=numpy.float32)
            # Support vertex with to 2 to 4 coordinates
            positions = self._attributes['position']
            radius = self._attributes['radius']
            self.__bounds[0, :positions.shape[1]] = \
                (positions - radius).min(axis=0)[:3]
            self.__bounds[1, :positions.shape[1]] = \
                (positions + radius).max(axis=0)[:3]
        return self.__bounds.copy()


# Meshes ######################################################################

class Mesh3D(Geometry):
    """A conventional 3D mesh"""

    _shaders = ("""
    attribute vec3 position;
    attribute vec3 normal;
    attribute vec4 color;

    uniform mat4 matrix;
    uniform mat4 transformMat;
    //uniform mat3 matrixInvTranspose;

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec4 vColor;

    void main(void)
    {
        vCameraPosition = transformMat * vec4(position, 1.0);
        //vNormal = matrixInvTranspose * normalize(normal);
        vPosition = position;
        vNormal = normal;
        vColor = color;
        gl_Position = matrix * vec4(position, 1.0);
    }
    """,
                string.Template("""
    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec4 vColor;

    $clippingDecl
    $lightingFunction

    void main(void)
    {
        $clippingCall(vCameraPosition);

        gl_FragColor = $lightingCall(vColor, vPosition, vNormal);
    }
    """))

    def __init__(self,
                 positions,
                 colors,
                 normals=None,
                 mode='triangles',
                 indices=None):
        assert mode in self._TRIANGLE_MODES
        super(Mesh3D, self).__init__(mode, indices,
                                     position=positions,
                                     normal=normals,
                                     color=colors)

        self._culling = None

    @property
    def culling(self):
        """Face culling (str)

        One of 'back', 'front' or None.
        """
        return self._culling

    @culling.setter
    def culling(self, culling):
        assert culling in ('back', 'front', None)
        if culling != self._culling:
            self._culling = culling
            self.notify()

    def renderGL2(self, ctx):
        isnormals = 'normal' in self._attributes
        if isnormals:
            fragLightFunction = ctx.viewport.light.fragmentDef
        else:
            fragLightFunction = ctx.viewport.light.fragmentShaderFunctionNoop

        fragment = self._shaders[1].substitute(
            clippingDecl=ctx.clipper.fragDecl,
            clippingCall=ctx.clipper.fragCall,
            lightingFunction=fragLightFunction,
            lightingCall=ctx.viewport.light.fragmentCall)
        prog = ctx.glCtx.prog(self._shaders[0], fragment)
        prog.use()

        if isnormals:
            ctx.viewport.light.setupProgram(ctx, prog)

        if self.culling is not None:
            cullFace = gl.GL_FRONT if self.culling == 'front' else gl.GL_BACK
            gl.glCullFace(cullFace)
            gl.glEnable(gl.GL_CULL_FACE)

        prog.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        prog.setUniformMatrix('transformMat',
                              ctx.objectToCamera.matrix,
                              safe=True)

        ctx.clipper.setupProgram(ctx, prog)

        self._draw(prog)

        if self.culling is not None:
            gl.glDisable(gl.GL_CULL_FACE)


# Group ######################################################################

# TODO lighting, clipping as groups?
# group composition?

class GroupDepthOffset(core.Group):
    """A group using 2-pass rendering and glDepthRange to avoid Z-fighting"""

    def __init__(self, children=(), epsilon=None):
        super(GroupDepthOffset, self).__init__(children)
        self._epsilon = epsilon
        self.isDepthRangeOn = True

    def prepareGL2(self, ctx):
        if self._epsilon is None:
            depthbits = gl.glGetInteger(gl.GL_DEPTH_BITS)
            self._epsilon = 1. / (1 << (depthbits - 1))

    def renderGL2(self, ctx):
        if self.isDepthRangeOn:
            self._renderGL2WithDepthRange(ctx)
        else:
            super(GroupDepthOffset, self).renderGL2(ctx)

    def _renderGL2WithDepthRange(self, ctx):
        # gl.glDepthFunc(gl.GL_LESS)
        with gl.enabled(gl.GL_CULL_FACE):
            gl.glCullFace(gl.GL_BACK)
            for child in self.children:
                gl.glColorMask(
                    gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDepthRange(self._epsilon, 1.)

                child.render(ctx)

                gl.glColorMask(
                    gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
                gl.glDepthMask(gl.GL_FALSE)
                gl.glDepthRange(0., 1. - self._epsilon)

                child.render(ctx)

            gl.glCullFace(gl.GL_FRONT)
            for child in reversed(self.children):
                gl.glColorMask(
                    gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDepthRange(self._epsilon, 1.)

                child.render(ctx)

                gl.glColorMask(
                    gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
                gl.glDepthMask(gl.GL_FALSE)
                gl.glDepthRange(0., 1. - self._epsilon)

                child.render(ctx)

        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthRange(0., 1.)
        # gl.glDepthFunc(gl.GL_LEQUAL)
        # TODO use epsilon for all rendering?
        # TODO issue with picking in depth buffer!


class GroupBBox(core.PrivateGroup):
    """A group displaying a bounding box around the children."""

    def __init__(self, children=(), color=(1., 1., 1., 1.)):
        super(GroupBBox, self).__init__()
        self._group = core.Group(children)

        self._boxTransforms = transform.TransformList(
            (transform.Translate(), transform.Scale()))

        self._boxWithAxes = BoxWithAxes(color)
        self._boxWithAxes.smooth = False
        self._boxWithAxes.transforms = self._boxTransforms

        self._children = [self._boxWithAxes, self._group]

    def _updateBoxAndAxes(self):
        """Update bbox and axes position and size according to children."""
        bounds = self._group.bounds(dataBounds=True)
        if bounds is not None:
            origin = bounds[0]
            scale = [(d if d != 0. else 1.) for d in bounds[1] - bounds[0]]
        else:
            origin, scale = (0., 0., 0.), (1., 1., 1.)

        self._boxTransforms[0].translation = origin
        self._boxTransforms[1].scale = scale

    def _bounds(self, dataBounds=False):
        self._updateBoxAndAxes()
        return super(GroupBBox, self)._bounds(dataBounds)

    def prepareGL2(self, ctx):
        self._updateBoxAndAxes()
        super(GroupBBox, self).prepareGL2(ctx)

    # Give access to _group children

    @property
    def children(self):
        return self._group.children

    @children.setter
    def children(self, iterable):
        self._group.children = iterable

    # Give access to box color

    @property
    def color(self):
        """The RGBA color to use for the box: 4 float in [0, 1]"""
        return self._boxWithAxes.color

    @color.setter
    def color(self, color):
        self._boxWithAxes.color = color


# Clipping Plane ##############################################################

class ClipPlane(PlaneInGroup):
    """A clipping plane attached to a box"""

    def renderGL2(self, ctx):
        super(ClipPlane, self).renderGL2(ctx)

        if self.visible:
            # Set-up clipping plane for following brothers

            # No need of perspective divide, no projection
            point = ctx.objectToCamera.transformPoint(self.plane.point,
                                                      perspectiveDivide=False)
            normal = ctx.objectToCamera.transformNormal(self.plane.normal)
            ctx.setClipPlane(point, normal)

    def postRender(self, ctx):
        if self.visible:
            # Disable clip planes
            ctx.setClipPlane()
