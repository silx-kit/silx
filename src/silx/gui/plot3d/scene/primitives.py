# /*##########################################################################
#
# Copyright (c) 2015-2021 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"

try:
    from collections import abc
except ImportError:  # Python2 support
    import collections as abc
import ctypes
from functools import reduce
import logging
import string

import numpy

from silx.gui.colors import rgba

from ... import _glutils
from ..._glutils import gl

from . import event
from . import core
from . import transform
from . import utils
from .function import Colormap

_logger = logging.getLogger(__name__)


# Geometry ####################################################################

class Geometry(core.Elem):
    """Set of vertices with normals and colors.

    :param str mode: OpenGL drawing mode:
                     lines, line_strip, loop, triangles, triangle_strip, fan
    :param indices: Array of vertex indices or None
    :param bool copy: True (default) to copy the data, False to use as is.
    :param str attrib0: Name of the attribute that MUST be an array.
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

    def __init__(self,
                 mode,
                 indices=None,
                 copy=True,
                 attrib0='position',
                 **attributes):
        super(Geometry, self).__init__()

        self._attrib0 = str(attrib0)

        self._vbos = {}  # Store current vbos
        self._unsyncAttributes = []  # Store attributes to copy to vbos
        self.__bounds = None  # Cache object's bounds
        # Attribute names defining the object bounds
        self.__boundsAttributeNames = (self._attrib0,)

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

        if nbvertices != 0:
            assert nbvertices >= mincheck
            if modulocheck != 0:
                assert (nbvertices % modulocheck) == 0

    @property
    def drawMode(self):
        """Kind of primitive to render, in :attr:`_MODES` (str)"""
        return self._mode

    @staticmethod
    def _glReadyArray(array, copy=True):
        """Making a contiguous array, checking float types.

        :param iterable array: array-like data to prepare for attribute
        :param bool copy: True to make a copy of the array, False to use as is
        """
        # Convert single value (int, float, numpy types) to tuple
        if not isinstance(array, abc.Iterable):
            array = (array, )

        # Makes sure it is an array
        array = numpy.array(array, copy=False)

        dtype = None
        if array.dtype.kind == 'f' and array.dtype.itemsize != 4:
            # Cast  to float32
            _logger.info('Cast array to float32')
            dtype = numpy.float32
        elif array.dtype.itemsize > 4:
            # Cast (u)int64 to (u)int32
            if array.dtype.kind == 'i':
                _logger.info('Cast array to int32')
                dtype = numpy.int32
            elif array.dtype.kind == 'u':
                _logger.info('Cast array to uint32')
                dtype = numpy.uint32

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

    @property
    def attrib0(self):
        """Attribute name that MUST be an array (str)"""
        return self._attrib0

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
                _logger.debug('Not checking attribute %s dimensions', name)
            else:
                checks = self._ATTR_INFO[name]

                if (array.ndim == 1 and checks['lastDim'] == (1,) and
                        len(array) > 1):
                    array = array.reshape((len(array), 1))

                # Checks
                assert array.ndim in checks['dims'], "Attr %s" % name
                assert array.shape[-1] in checks['lastDim'], "Attr %s" % name

            # Makes sure attrib0 is considered as an array of values
            if name == self.attrib0 and array.ndim == 1:
                array.shape = 1, -1

            # Check length against another attribute array
            # Causes problems when updating
            # nbVertices = self.nbVertices
            # if array.ndim == 2 and nbVertices is not None:
            #     assert len(array) == nbVertices

            self._attributes[name] = array
            if array.ndim == 2:  # Store this in a VBO
                self._unsyncAttributes.append(name)

            if name in self.boundsAttributeNames:  # Reset bounds
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

                if array.ndim == 1:
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
        self.notify()

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

    @property
    def boundsAttributeNames(self):
        """Tuple of attribute names defining the bounds of the object.

        Attributes name are taken in the given order to compute the
        (x, y, z) the bounding box, e.g.::

          geometry.boundsAttributeNames = 'position'
          geometry.boundsAttributeNames = 'x', 'y', 'z'
        """
        return self.__boundsAttributeNames

    @boundsAttributeNames.setter
    def boundsAttributeNames(self, names):
        self.__boundsAttributeNames = tuple(str(name) for name in names)
        self.__bounds = None
        self.notify()

    def _bounds(self, dataBounds=False):
        if self.__bounds is None:
            if len(self.boundsAttributeNames) == 0:
                return None  # No bounds

            self.__bounds = numpy.zeros((2, 3), dtype=numpy.float32)

            # Coordinates defined in one or more attributes
            index = 0
            for name in self.boundsAttributeNames:
                if index == 3:
                    _logger.error("Too many attributes defining bounds")
                    break

                attribute = self._attributes[name]
                assert attribute.ndim in (1, 2)
                if attribute.ndim == 1:  # Single value
                    min_ = attribute
                    max_ = attribute
                elif len(attribute) > 0:  # Array of values, compute min/max
                    min_ = numpy.nanmin(attribute, axis=0)
                    max_ = numpy.nanmax(attribute, axis=0)
                else:
                    min_, max_ = numpy.zeros((2, attribute.shape[1]), dtype=numpy.float32)

                toCopy = min(len(min_), 3-index)
                if toCopy != len(min_):
                    _logger.error("Attribute defining bounds"
                                  " has too many dimensions")

                self.__bounds[0, index:index+toCopy] = min_[:toCopy]
                self.__bounds[1, index:index+toCopy] = max_[:toCopy]

                index += toCopy

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

    $sceneDecl
    $lightingFunction

    void main(void)
    {
        $scenePreCall(vCameraPosition);
        gl_FragColor = $lightingCall(vColor, vPosition, vNormal);
        $scenePostCall(vCameraPosition);
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
            sceneDecl=ctx.fragDecl,
            scenePreCall=ctx.fragCallPre,
            scenePostCall=ctx.fragCallPost,
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

        ctx.setupProgram(prog)

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

    $sceneDecl
    $lightingFunction

    void main(void)
    {
        $scenePreCall(vCameraPosition);

        /* Discard off dash fragments */
        float lineDist = distance(vOriginFragCoord, gl_FragCoord.xy);
        if (mod(lineDist, dash.x + dash.y) > dash.x) {
            discard;
        }
        gl_FragColor = $lightingCall(vColor, vPosition, vNormal);

        $scenePostCall(vCameraPosition);
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
            sceneDecl=context.fragDecl,
            scenePreCall=context.fragCallPre,
            scenePostCall=context.fragCallPost,
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

        context.setupProgram(program)

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

    def __init__(self,  stroke=(1., 1., 1., 1.), fill=(1., 1., 1., 0.)):
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

        self._size = 1., 1., 1.

    @classmethod
    def getLineIndices(cls, copy=True):
        """Returns 2D array of Box lines indices

        :param copy: True (default) to get a copy,
                     False to get internal array (Do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(cls._lineIndices, copy=copy)

    @classmethod
    def getVertices(cls, copy=True):
        """Returns 2D array of Box corner coordinates.

        :param copy: True (default) to get a copy,
                     False to get internal array (Do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(cls._vertices, copy=copy)

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
        self._size = 1., 1., 1.

    @property
    def size(self):
        """Size of the axes (sx, sy, sz)"""
        return self._size

    @size.setter
    def size(self, size):
        assert len(size) == 3
        size = tuple(size)
        if size != self.size:
            self._size = size
            self.setAttribute(
                'position',
                self._vertices * numpy.array(size, dtype=numpy.float32))
            self.notify()


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
        self._size = 1., 1., 1.
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

    @property
    def size(self):
        """Size of the axes (sx, sy, sz)"""
        return self._size

    @size.setter
    def size(self, size):
        assert len(size) == 3
        size = tuple(size)
        if size != self.size:
            self._size = size
            self.setAttribute(
                'position',
                self._vertices * numpy.array(size, dtype=numpy.float32))
            self.notify()


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
        self._strokeVisible = True

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

    @property
    def strokeVisible(self):
        """Whether surrounding stroke is visible or not (bool)."""
        return self._strokeVisible

    @strokeVisible.setter
    def strokeVisible(self, visible):
        self._strokeVisible = bool(visible)
        if self._outline is not None:
            self._outline.visible = self._strokeVisible

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
        boxVertices = Box.getVertices(copy=True)
        boxVertices = bounds[0] + boxVertices * (bounds[1] - bounds[0])
        lineIndices = Box.getLineIndices(copy=False)
        vertices = utils.boxPlaneIntersect(
            boxVertices, lineIndices, self.plane.normal, self.plane.point)

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
                self._outline.visible = self._strokeVisible
                self._children.append(self._outline)

            # Update vertices, TODO only when necessary
            self._outline.setAttribute('position', self.contourVertices)

            super(PlaneInGroup, self).prepareGL2(ctx)

    def renderGL2(self, ctx):
        if self.isValid:
            super(PlaneInGroup, self).renderGL2(ctx)


class BoundedGroup(core.Group):
    """Group with data bounds"""

    _shape = None  # To provide a default value without overriding __init__

    @property
    def shape(self):
        """Data shape (depth, height, width) of this group or None"""
        return self._shape

    @shape.setter
    def shape(self, shape):
        if shape is None:
            self._shape = None
        else:
            depth, height, width = shape
            self._shape = float(depth), float(height), float(width)

    @property
    def size(self):
        """Data size (width, height, depth) of this group or None"""
        shape = self.shape
        if shape is None:
            return None
        else:
            return shape[2], shape[1], shape[0]

    @size.setter
    def size(self, size):
        if size is None:
            self.shape = None
        else:
            self.shape = size[2], size[1], size[0]

    def _bounds(self, dataBounds=False):
        if dataBounds and self.size is not None:
            return numpy.array(((0., 0., 0.), self.size),
                               dtype=numpy.float32)
        else:
            return super(BoundedGroup, self)._bounds(dataBounds)


# Points ######################################################################

class _Points(Geometry):
    """Base class to render a set of points."""

    DIAMOND = 'd'
    CIRCLE = 'o'
    SQUARE = 's'
    PLUS = '+'
    X_MARKER = 'x'
    ASTERISK = '*'
    H_LINE = '_'
    V_LINE = '|'

    SUPPORTED_MARKERS = (DIAMOND, CIRCLE, SQUARE, PLUS,
                         X_MARKER, ASTERISK, H_LINE, V_LINE)
    """List of supported markers:

    - 'd' diamond
    - 'o' circle
    - 's' square
    - '+' cross
    - 'x' x-cross
    - '*' asterisk
    - '_' horizontal line
    - '|' vertical line
    """

    _MARKER_FUNCTIONS = {
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
    }

    _shaders = (string.Template("""
    #version 120

    attribute float x;
    attribute float y;
    attribute float z;
    attribute $valueType value;
    attribute float size;

    uniform mat4 matrix;
    uniform mat4 transformMat;

    varying vec4 vCameraPosition;
    varying $valueType vValue;
    varying float vSize;

    void main(void)
    {
        vValue = value;

        vec4 positionVec4 = vec4(x, y, z, 1.0);
        gl_Position = matrix * positionVec4;
        vCameraPosition = transformMat * positionVec4;

        gl_PointSize = size;
        vSize = size;
    }
    """),
                string.Template("""
    #version 120

    varying vec4 vCameraPosition;
    varying float vSize;
    varying $valueType vValue;

    $valueToColorDecl
    $sceneDecl
    $alphaSymbolDecl

    void main(void)
    {
        $scenePreCall(vCameraPosition);

        float alpha = alphaSymbol(gl_PointCoord, vSize);

        gl_FragColor = $valueToColorCall(vValue);
        gl_FragColor.a *= alpha;
        if (gl_FragColor.a == 0.0) {
            discard;
        }

        $scenePostCall(vCameraPosition);
    }
    """))

    _ATTR_INFO = {
        'x': {'dims': (1, 2), 'lastDim': (1,)},
        'y': {'dims': (1, 2), 'lastDim': (1,)},
        'z': {'dims': (1, 2), 'lastDim': (1,)},
        'size': {'dims': (1, 2), 'lastDim': (1,)},
    }

    def __init__(self, x, y, z, value, size=1., indices=None):
        super(_Points, self).__init__('points', indices,
                                      x=x,
                                      y=y,
                                      z=z,
                                      value=value,
                                      size=size,
                                      attrib0='x')
        self.boundsAttributeNames = 'x', 'y', 'z'
        self._marker = 'o'

    @property
    def marker(self):
        """The marker symbol used to display the scatter plot (str)

        See :attr:`SUPPORTED_MARKERS` for the list of supported marker string.
        """
        return self._marker

    @marker.setter
    def marker(self, marker):
        marker = str(marker)
        assert marker in self.SUPPORTED_MARKERS
        if marker != self._marker:
            self._marker = marker
            self.notify()

    def _shaderValueDefinition(self):
        """Type definition, fragment shader declaration, fragment shader call
        """
        raise NotImplementedError(
            "This method must be implemented in subclass")

    def _renderGL2PreDrawHook(self, ctx, program):
        """Override in subclass to run code before calling gl draw"""
        pass

    def renderGL2(self, ctx):
        valueType, valueToColorDecl, valueToColorCall = \
            self._shaderValueDefinition()
        vertexShader = self._shaders[0].substitute(
            valueType=valueType)
        fragmentShader = self._shaders[1].substitute(
            sceneDecl=ctx.fragDecl,
            scenePreCall=ctx.fragCallPre,
            scenePostCall=ctx.fragCallPost,
            valueType=valueType,
            valueToColorDecl=valueToColorDecl,
            valueToColorCall=valueToColorCall,
            alphaSymbolDecl=self._MARKER_FUNCTIONS[self.marker])
        program = ctx.glCtx.prog(vertexShader, fragmentShader,
                                 attrib0=self.attrib0)
        program.use()

        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        # gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        program.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        program.setUniformMatrix('transformMat',
                                 ctx.objectToCamera.matrix,
                                 safe=True)

        ctx.setupProgram(program)

        self._renderGL2PreDrawHook(ctx, program)

        self._draw(program)


class Points(_Points):
    """A set of data points with an associated value and size."""

    _ATTR_INFO = _Points._ATTR_INFO.copy()
    _ATTR_INFO.update({'value': {'dims': (1, 2), 'lastDim': (1,)}})

    def __init__(self, x, y, z, value=0., size=1.,
                 indices=None, colormap=None):
        super(Points, self).__init__(x=x,
                                     y=y,
                                     z=z,
                                     indices=indices,
                                     size=size,
                                     value=value)

        self._colormap = colormap or Colormap()  # Default colormap
        self._colormap.addListener(self._cmapChanged)

    @property
    def colormap(self):
        """The colormap used to render the image"""
        return self._colormap

    def _cmapChanged(self, source, *args, **kwargs):
        """Broadcast colormap changes"""
        self.notify(*args, **kwargs)

    def _shaderValueDefinition(self):
        """Type definition, fragment shader declaration, fragment shader call
        """
        return 'float', self.colormap.decl, self.colormap.call

    def _renderGL2PreDrawHook(self, ctx, program):
        """Set-up colormap before calling gl draw"""
        self.colormap.setupProgram(ctx, program)


class ColorPoints(_Points):
    """A set of points with an associated color and size."""

    _ATTR_INFO = _Points._ATTR_INFO.copy()
    _ATTR_INFO.update({'value': {'dims': (1, 2), 'lastDim': (3, 4)}})

    def __init__(self, x, y, z, color=(1., 1., 1., 1.), size=1.,
                 indices=None):
        super(ColorPoints, self).__init__(x=x,
                                          y=y,
                                          z=z,
                                          indices=indices,
                                          size=size,
                                          value=color)

    def _shaderValueDefinition(self):
        """Type definition, fragment shader declaration, fragment shader call
        """
        return 'vec4', '', ''

    def setColor(self, color, copy=True):
        """Set colors

        :param color: Single RGBA color or
                      2D array of color of length number of points
        :param bool copy: True to copy colors (default),
                          False to use provided array (Do not modify!)
        """
        self.setAttribute('value', color, copy=copy)

    def getColor(self, copy=True):
        """Returns the color or array of colors of the points.

        :param copy: True to get a copy (default),
                     False to return internal array (Do not modify!)
        :return: Color or array of colors
        :rtype: numpy.ndarray
        """
        return self.getAttribute('value', copy=copy)


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
    out vec4 gl_FragColor;

    $sceneDecl

    void main(void)
    {
        $scenePreCall(vCameraPosition);

        gl_FragColor = vec4(0.5 * vNormValue + 0.5, 0.0, 0.0, 1.0);

        $scenePostCall(vCameraPosition);
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
        if isinstance(values, abc.Iterable):
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
            sceneDecl=ctx.fragDecl,
            scenePreCall=ctx.fragCallPre,
            scenePostCall=ctx.fragCallPost)
        prog = ctx.glCtx.prog(self._shaders[0], fragment)
        prog.use()

        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        gl.glEnable(gl.GL_POINT_SPRITE)  # OpenGL 2
        # gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        prog.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        prog.setUniformMatrix('transformMat',
                              ctx.objectToCamera.matrix,
                              safe=True)

        ctx.setupProgram(prog)

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

    $sceneDecl
    $lightingFunction

    void main(void)
    {
        $scenePreCall(vCameraPosition);

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

        $scenePostCall(vCameraPosition);
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
            sceneDecl=ctx.fragDecl,
            scenePreCall=ctx.fragCallPre,
            scenePostCall=ctx.fragCallPost,
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

        ctx.setupProgram(prog)

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

    $sceneDecl
    $lightingFunction

    void main(void)
    {
        $scenePreCall(vCameraPosition);

        gl_FragColor = $lightingCall(vColor, vPosition, vNormal);

        $scenePostCall(vCameraPosition);
    }
    """))

    def __init__(self,
                 positions,
                 colors,
                 normals=None,
                 mode='triangles',
                 indices=None,
                 copy=True):
        assert mode in self._TRIANGLE_MODES
        super(Mesh3D, self).__init__(mode, indices,
                                     position=positions,
                                     normal=normals,
                                     color=colors,
                                     copy=copy)

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
            sceneDecl=ctx.fragDecl,
            scenePreCall=ctx.fragCallPre,
            scenePostCall=ctx.fragCallPost,
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

        ctx.setupProgram(prog)

        self._draw(prog)

        if self.culling is not None:
            gl.glDisable(gl.GL_CULL_FACE)


class ColormapMesh3D(Geometry):
    """A 3D mesh with color computed from a colormap"""

    _shaders = ("""
    attribute vec3 position;
    attribute vec3 normal;
    attribute float value;

    uniform mat4 matrix;
    uniform mat4 transformMat;
    //uniform mat3 matrixInvTranspose;

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying float vValue;

    void main(void)
    {
        vCameraPosition = transformMat * vec4(position, 1.0);
        //vNormal = matrixInvTranspose * normalize(normal);
        vPosition = position;
        vNormal = normal;
        vValue = value;
        gl_Position = matrix * vec4(position, 1.0);
    }
    """,
                string.Template("""
    uniform float alpha;

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying float vValue;

    $colormapDecl
    $sceneDecl
    $lightingFunction

    void main(void)
    {
        $scenePreCall(vCameraPosition);

        vec4 color = $colormapCall(vValue);
        gl_FragColor = $lightingCall(color, vPosition, vNormal);
        gl_FragColor.a *= alpha;

        $scenePostCall(vCameraPosition);
    }
    """))

    def __init__(self,
                 position,
                 value,
                 colormap=None,
                 normal=None,
                 mode='triangles',
                 indices=None,
                 copy=True):
        super(ColormapMesh3D, self).__init__(mode, indices,
                                             position=position,
                                             normal=normal,
                                             value=value,
                                             copy=copy)

        self._alpha = 1.0
        self._lineWidth = 1.0
        self._lineSmooth = True
        self._culling = None
        self._colormap = colormap or Colormap()  # Default colormap
        self._colormap.addListener(self._cmapChanged)

    lineWidth = event.notifyProperty('_lineWidth', converter=float,
                                     doc="Width of the line in pixels.")

    lineSmooth = event.notifyProperty(
        '_lineSmooth',
        converter=bool,
        doc="Smooth line rendering enabled (bool, default: True)")

    alpha = event.notifyProperty(
        '_alpha', converter=float,
        doc="Transparency of the mesh, float in [0, 1]")

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

    @property
    def colormap(self):
        """The colormap used to render the image"""
        return self._colormap

    def _cmapChanged(self, source, *args, **kwargs):
        """Broadcast colormap changes"""
        self.notify(*args, **kwargs)

    def renderGL2(self, ctx):
        if 'normal' in self._attributes:
            self._renderGL2(ctx)
        else:  # Disable lighting
            with self.viewport.light.turnOff():
                self._renderGL2(ctx)

    def _renderGL2(self, ctx):
        fragment = self._shaders[1].substitute(
            sceneDecl=ctx.fragDecl,
            scenePreCall=ctx.fragCallPre,
            scenePostCall=ctx.fragCallPost,
            lightingFunction=ctx.viewport.light.fragmentDef,
            lightingCall=ctx.viewport.light.fragmentCall,
            colormapDecl=self.colormap.decl,
            colormapCall=self.colormap.call)
        program = ctx.glCtx.prog(self._shaders[0], fragment)
        program.use()

        ctx.viewport.light.setupProgram(ctx, program)
        ctx.setupProgram(program)
        self.colormap.setupProgram(ctx, program)

        if self.culling is not None:
            cullFace = gl.GL_FRONT if self.culling == 'front' else gl.GL_BACK
            gl.glCullFace(cullFace)
            gl.glEnable(gl.GL_CULL_FACE)

        program.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        program.setUniformMatrix('transformMat',
                                 ctx.objectToCamera.matrix,
                                 safe=True)
        gl.glUniform1f(program.uniforms['alpha'], self._alpha)

        if self.drawMode in self._LINE_MODES:
            gl.glLineWidth(self.lineWidth)
            with gl.enabled(gl.GL_LINE_SMOOTH, self.lineSmooth):
                self._draw(program)
        else:
            self._draw(program)

        if self.culling is not None:
            gl.glDisable(gl.GL_CULL_FACE)


# ImageData ##################################################################

class _Image(Geometry):
    """Base class for ImageData and ImageRgba"""

    _shaders = ("""
    attribute vec2 position;

    uniform mat4 matrix;
    uniform mat4 transformMat;
    uniform vec2 dataScale;

    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec3 vNormal;
    varying vec2 vTexCoords;

    void main(void)
    {
        vec4 positionVec4 = vec4(position, 0.0, 1.0);
        vCameraPosition = transformMat * positionVec4;
        vPosition = positionVec4.xyz;
        vTexCoords = dataScale * position;
        gl_Position = matrix * positionVec4;
    }
    """,
                string.Template("""
    varying vec4 vCameraPosition;
    varying vec3 vPosition;
    varying vec2 vTexCoords;
    uniform sampler2D data;
    uniform float alpha;

    $imageDecl
    $sceneDecl
    $lightingFunction

    void main(void)
    {
        $scenePreCall(vCameraPosition);

        vec4 color = imageColor(data, vTexCoords);
        color.a *= alpha;
        if (color.a == 0.) { /* Discard fully transparent pixels */
            discard;
        }

        vec3 normal = vec3(0.0, 0.0, 1.0);
        gl_FragColor = $lightingCall(color, vPosition, normal);

        $scenePostCall(vCameraPosition);
    }
    """))

    _UNIT_SQUARE = numpy.array(((0., 0.), (1., 0.), (0., 1.), (1., 1.)),
                               dtype=numpy.float32)

    def __init__(self, data, copy=True):
        super(_Image, self).__init__(mode='triangle_strip',
                                     position=self._UNIT_SQUARE)

        self._texture = None
        self._update_texture = True
        self._update_texture_filter = False
        self._data = None
        self.setData(data, copy)
        self._alpha = 1.
        self._interpolation = 'linear'

        self.isBackfaceVisible = True

    def setData(self, data, copy=True):
        assert isinstance(data, numpy.ndarray)

        if copy:
            data = numpy.array(data, copy=True)

        self._data = data
        self._update_texture = True
        # By updating the position rather than always using a unit square
        # we benefit from Geometry bounds handling
        self.setAttribute('position', self._UNIT_SQUARE * (self._data.shape[1], self._data.shape[0]))
        self.notify()

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
        self._update_texture_filter = True
        self.notify()

    @property
    def alpha(self):
        """Transparency of the image, float in [0, 1]"""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)
        self.notify()

    def _textureFormat(self):
        """Implement this method to provide texture internal format and format

        :return: 2-tuple of gl flags (internalFormat, format)
        """
        raise NotImplementedError(
            "This method must be implemented in a subclass")

    def prepareGL2(self, ctx):
        if self._texture is None or self._update_texture:
            if self._texture is not None:
                self._texture.discard()

            if self.interpolation == 'nearest':
                filter_ = gl.GL_NEAREST
            else:
                filter_ = gl.GL_LINEAR
            self._update_texture = False
            self._update_texture_filter = False
            if self._data.size == 0:
                self._texture = None
            else:
                internalFormat, format_ = self._textureFormat()
                self._texture = _glutils.Texture(
                    internalFormat,
                    self._data,
                    format_,
                    minFilter=filter_,
                    magFilter=filter_,
                    wrap=gl.GL_CLAMP_TO_EDGE)

        if self._update_texture_filter and self._texture is not None:
            self._update_texture_filter = False
            if self.interpolation == 'nearest':
                filter_ = gl.GL_NEAREST
            else:
                filter_ = gl.GL_LINEAR
            self._texture.minFilter = filter_
            self._texture.magFilter = filter_

        super(_Image, self).prepareGL2(ctx)

    def renderGL2(self, ctx):
        if self._texture is None:
            return  # Nothing to render

        with self.viewport.light.turnOff():
            self._renderGL2(ctx)

    def _renderGL2PreDrawHook(self, ctx, program):
        """Override in subclass to run code before calling gl draw"""
        pass

    def _shaderImageColorDecl(self):
        """Returns fragment shader imageColor function declaration"""
        raise NotImplementedError(
            "This method must be implemented in a subclass")

    def _renderGL2(self, ctx):
        fragment = self._shaders[1].substitute(
            sceneDecl=ctx.fragDecl,
            scenePreCall=ctx.fragCallPre,
            scenePostCall=ctx.fragCallPost,
            lightingFunction=ctx.viewport.light.fragmentDef,
            lightingCall=ctx.viewport.light.fragmentCall,
            imageDecl=self._shaderImageColorDecl()
            )
        program = ctx.glCtx.prog(self._shaders[0], fragment)
        program.use()

        ctx.viewport.light.setupProgram(ctx, program)

        if not self.isBackfaceVisible:
            gl.glCullFace(gl.GL_BACK)
            gl.glEnable(gl.GL_CULL_FACE)

        program.setUniformMatrix('matrix', ctx.objectToNDC.matrix)
        program.setUniformMatrix('transformMat',
                                 ctx.objectToCamera.matrix,
                                 safe=True)
        gl.glUniform1f(program.uniforms['alpha'], self._alpha)

        shape = self._data.shape
        gl.glUniform2f(program.uniforms['dataScale'], 1./shape[1], 1./shape[0])

        gl.glUniform1i(program.uniforms['data'], self._texture.texUnit)

        ctx.setupProgram(program)

        self._texture.bind()

        self._renderGL2PreDrawHook(ctx, program)

        self._draw(program)

        if not self.isBackfaceVisible:
            gl.glDisable(gl.GL_CULL_FACE)


class ImageData(_Image):
    """Display a 2x2 data array with a texture."""

    _imageDecl = string.Template("""
    $colormapDecl

    vec4 imageColor(sampler2D data, vec2 texCoords) {
        float value = texture2D(data, texCoords).r;
        vec4 color = $colormapCall(value);
        return color;
    }
    """)

    def __init__(self, data, copy=True, colormap=None):
        super(ImageData, self).__init__(data, copy=copy)

        self._colormap = colormap or Colormap()  # Default colormap
        self._colormap.addListener(self._cmapChanged)

    def setData(self, data, copy=True):
        data = numpy.array(data, copy=copy, order='C', dtype=numpy.float32)
        # TODO support (u)int8|16
        assert data.ndim == 2

        super(ImageData, self).setData(data, copy=False)

    @property
    def colormap(self):
        """The colormap used to render the image"""
        return self._colormap

    def _cmapChanged(self, source, *args, **kwargs):
        """Broadcast colormap changes"""
        self.notify(*args, **kwargs)

    def _textureFormat(self):
        return gl.GL_R32F, gl.GL_RED

    def _renderGL2PreDrawHook(self, ctx, program):
        self.colormap.setupProgram(ctx, program)

    def _shaderImageColorDecl(self):
        return self._imageDecl.substitute(
            colormapDecl=self.colormap.decl,
            colormapCall=self.colormap.call)


# ImageRgba ##################################################################

class ImageRgba(_Image):
    """Display a 2x2 RGBA image with a texture.

    Supports images of float in [0, 1] and uint8.
    """

    _imageDecl = """
    vec4 imageColor(sampler2D data, vec2 texCoords) {
        vec4 color = texture2D(data, texCoords);
        return color;
    }
    """

    def __init__(self, data, copy=True):
        super(ImageRgba, self).__init__(data, copy=copy)

    def setData(self, data, copy=True):
        data = numpy.array(data, copy=copy, order='C')
        assert data.ndim == 3
        assert data.shape[2] in (3, 4)
        if data.dtype.kind == 'f':
            if data.dtype != numpy.dtype(numpy.float32):
                _logger.warning("Converting image data to float32")
                data = numpy.array(data, dtype=numpy.float32, copy=False)
        else:
            assert data.dtype == numpy.dtype(numpy.uint8)

        super(ImageRgba, self).setData(data, copy=False)

    def _textureFormat(self):
        format_ = gl.GL_RGBA if self._data.shape[2] == 4 else gl.GL_RGB
        return format_, format_

    def _shaderImageColorDecl(self):
        return self._imageDecl


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


class GroupNoDepth(core.Group):
    """A group rendering its children without writing to the depth buffer

    :param bool mask: True (default) to disable writing in the depth buffer
    :param bool notest: True (default) to disable depth test
    """

    def __init__(self, children=(), mask=True, notest=True):
        super(GroupNoDepth, self).__init__(children)
        self._mask = bool(mask)
        self._notest = bool(notest)

    def renderGL2(self, ctx):
        if self._mask:
            gl.glDepthMask(gl.GL_FALSE)

        with gl.disabled(gl.GL_DEPTH_TEST, disable=self._notest):
            super(GroupNoDepth, self).renderGL2(ctx)

        if self._mask:
            gl.glDepthMask(gl.GL_TRUE)


class GroupBBox(core.PrivateGroup):
    """A group displaying a bounding box around the children."""

    def __init__(self, children=(), color=(1., 1., 1., 1.)):
        super(GroupBBox, self).__init__()
        self._group = core.Group(children)

        self._boxTransforms = transform.TransformList((transform.Translate(),))

        # Using 1 of 3 primitives to render axes and/or bounding box
        # To avoid z-fighting between axes and bounding box
        self._boxWithAxes = BoxWithAxes(color)
        self._boxWithAxes.smooth = False
        self._boxWithAxes.transforms = self._boxTransforms

        self._box = Box(stroke=color, fill=(1., 1., 1., 0.))
        self._box.strokeSmooth = False
        self._box.transforms = self._boxTransforms
        self._box.visible = False

        self._axes = Axes()
        self._axes.smooth = False
        self._axes.transforms = self._boxTransforms
        self._axes.visible = False

        self.strokeWidth = 2.

        self._children = [self._boxWithAxes, self._box, self._axes, self._group]

    def _updateBoxAndAxes(self):
        """Update bbox and axes position and size according to children."""
        bounds = self._group.bounds(dataBounds=True)
        if bounds is not None:
            origin = bounds[0]
            size = bounds[1] - bounds[0]
        else:
            origin, size = (0., 0., 0.), (1., 1., 1.)

        self._boxTransforms[0].translation = origin

        self._boxWithAxes.size = size
        self._box.size = size
        self._axes.size = size

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

    # Give access to box color and stroke width

    @property
    def color(self):
        """The RGBA color to use for the box: 4 float in [0, 1]"""
        return self._box.strokeColor

    @color.setter
    def color(self, color):
        self._box.strokeColor = color
        self._boxWithAxes.color = color

    @property
    def strokeWidth(self):
        """The width of the stroke lines in pixels (float)"""
        return self._box.strokeWidth

    @strokeWidth.setter
    def strokeWidth(self, width):
        width = float(width)
        self._box.strokeWidth = width
        self._boxWithAxes.width = width
        self._axes.width = width

    # Toggle axes visibility

    def _updateBoxAndAxesVisibility(self, axesVisible, boxVisible):
        """Update visible flags of box and axes primitives accordingly.

        :param bool axesVisible: True to display axes
        :param bool boxVisible: True to display bounding box
        """
        self._boxWithAxes.visible = boxVisible and axesVisible
        self._box.visible = boxVisible and not axesVisible
        self._axes.visible = not boxVisible and axesVisible

    @property
    def axesVisible(self):
        """Whether axes are displayed or not (bool)"""
        return self._boxWithAxes.visible or self._axes.visible

    @axesVisible.setter
    def axesVisible(self, visible):
        self._updateBoxAndAxesVisibility(axesVisible=bool(visible),
                                         boxVisible=self.boxVisible)

    @property
    def boxVisible(self):
        """Whether bounding box is displayed or not (bool)"""
        return self._boxWithAxes.visible or self._box.visible

    @boxVisible.setter
    def boxVisible(self, visible):
        self._updateBoxAndAxesVisibility(axesVisible=self.axesVisible,
                                         boxVisible=bool(visible))


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
