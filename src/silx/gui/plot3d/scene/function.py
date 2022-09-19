# /*##########################################################################
#
# Copyright (c) 2015-2020 European Synchrotron Radiation Facility
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
"""This module provides functions to add to shaders."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/07/2018"


import contextlib
import logging
import string
import numpy

from ... import _glutils
from ..._glutils import gl

from . import event
from . import utils


_logger = logging.getLogger(__name__)


class ProgramFunction(object):
    """Class providing a function to add to a GLProgram shaders.
    """

    def setupProgram(self, context, program):
        """Sets-up uniforms of a program using this shader function.

        :param RenderContext context: The current rendering context
        :param GLProgram program: The program to set-up.
                                  It MUST be in use and using this function.
        """
        pass


class Fog(event.Notifier, ProgramFunction):
    """Linear fog over the whole scene content.

    The background of the viewport is used as fog color,
    otherwise it defaults to white.
    """
    # TODO: add more controls (set fog range), add more fog modes

    _fragDecl = """
    /* (1/(far - near) or 0, near) z in [0 (camera), -inf[ */
    uniform vec2 fogExtentInfo;

    /* Color to use as fog color */
    uniform vec3 fogColor;

    vec4 fog(vec4 color, vec4 cameraPosition) {
        /* d = (pos - near) / (far - near) */
        float distance = fogExtentInfo.x * (cameraPosition.z/cameraPosition.w - fogExtentInfo.y);
        float fogFactor = clamp(distance, 0.0, 1.0);
        vec3 rgb = mix(color.rgb, fogColor, fogFactor);
        return vec4(rgb.r, rgb.g, rgb.b, color.a);
    }
    """

    _fragDeclNoop = """
    vec4 fog(vec4 color, vec4 cameraPosition) {
        return color;
    }
    """

    def __init__(self):
        super(Fog, self).__init__()
        self._isOn = True

    @property
    def isOn(self):
        """True to enable fog, False to disable (bool)"""
        return self._isOn

    @isOn.setter
    def isOn(self, isOn):
        isOn = bool(isOn)
        if self._isOn != isOn:
            self._isOn = bool(isOn)
            self.notify()

    @property
    def fragDecl(self):
        return self._fragDecl if self.isOn else self._fragDeclNoop

    @property
    def fragCall(self):
        return "fog"

    @staticmethod
    def _zExtentCamera(viewport):
        """Return (far, near) planes Z in camera coordinates.

        :param Viewport viewport:
        :return: (far, near) position in camera coords (from 0 to -inf)
        """
        # Provide scene z extent in camera coords
        bounds = viewport.camera.extrinsic.transformBounds(
            viewport.scene.bounds(transformed=True, dataBounds=True))
        return bounds[:, 2]

    def setupProgram(self, context, program):
        if not self.isOn:
            return

        far, near = context.cache(key='zExtentCamera',
                                  factory=self._zExtentCamera,
                                  viewport=context.viewport)
        extent = far - near
        gl.glUniform2f(program.uniforms['fogExtentInfo'],
                       0.9/extent if extent != 0. else 0.,
                       near)

        # Use background color as fog color
        bgColor = context.viewport.background
        if bgColor is None:
            bgColor = 1., 1., 1.
        gl.glUniform3f(program.uniforms['fogColor'], *bgColor[:3])


class ClippingPlane(ProgramFunction):
    """Description of a clipping plane and rendering.

    Convention: Clipping is performed in camera/eye space.

    :param point: Local coordinates of a point on the plane.
    :type point: numpy.ndarray-like of 3 float32
    :param normal: Local coordinates of the plane normal.
    :type normal: numpy.ndarray-like of 3 float32
    """

    _fragDecl = """
    /* Clipping plane */
    /* as rx + gy + bz + a > 0, clipping all positive */
    uniform vec4 planeEq;

    /* Position is in camera/eye coordinates */

    bool isClipped(vec4 position) {
        vec4 tmp = planeEq * position;
        float value = tmp.x + tmp.y + tmp.z + planeEq.a;
        return (value < 0.0001);
    }

    void clipping(vec4 position) {
        if (isClipped(position)) {
            discard;
        }
    }
    /* End of clipping */
    """

    _fragDeclNoop = """
    bool isClipped(vec4 position)
    {
        return false;
    }

    void clipping(vec4 position) {}
    """

    def __init__(self, point=(0., 0., 0.), normal=(0., 0., 0.)):
        self._plane = utils.Plane(point, normal)

    @property
    def plane(self):
        """Plane parameters in camera space."""
        return self._plane

    # GL2

    @property
    def fragDecl(self):
        return self._fragDecl if self.plane.isPlane else self._fragDeclNoop

    @property
    def fragCall(self):
        return "clipping"

    def setupProgram(self, context, program):
        """Sets-up uniforms of a program using this shader function.

        :param RenderContext context: The current rendering context
        :param GLProgram program: The program to set-up.
                                  It MUST be in use and using this function.
        """
        if self.plane.isPlane:
            gl.glUniform4f(program.uniforms['planeEq'], *self.plane.parameters)


class DirectionalLight(event.Notifier, ProgramFunction):
    """Description of a directional Phong light.

    :param direction: The direction of the light or None to disable light
    :type direction: ndarray of 3 floats or None
    :param ambient: RGB ambient light
    :type ambient: ndarray of 3 floats in [0, 1], default: (1., 1., 1.)
    :param diffuse: RGB diffuse light parameter
    :type diffuse: ndarray of 3 floats in [0, 1], default: (0., 0., 0.)
    :param specular: RGB specular light parameter
    :type specular: ndarray of 3 floats in [0, 1], default: (1., 1., 1.)
    :param int shininess: The shininess of the material for specular term,
                          default: 0 which disables specular component.
    """

    fragmentShaderFunction = """
    /* Lighting */
    struct DLight {
        vec3 lightDir; // Direction of light in object space
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        float shininess;
        vec3 viewPos; // Camera position in object space
    };

    uniform DLight dLight;

    vec4 lighting(vec4 color, vec3 position, vec3 normal)
    {
        normal = normalize(normal);
        // 1-sided
        float nDotL = max(0.0, dot(normal, - dLight.lightDir));

        // 2-sided
        //float nDotL = dot(normal, - dLight.lightDir);
        //if (nDotL < 0.) {
        //    nDotL = - nDotL;
        //    normal = - normal;
        //}

        float specFactor = 0.;
        if (dLight.shininess > 0. && nDotL > 0.) {
            vec3 reflection = reflect(dLight.lightDir, normal);
            vec3 viewDir = normalize(dLight.viewPos - position);
            specFactor = max(0.0, dot(reflection, viewDir));
            if (specFactor > 0.) {
                specFactor = pow(specFactor, dLight.shininess);
            }
        }

        vec3 enlightedColor = color.rgb * (dLight.ambient +
                                           dLight.diffuse * nDotL) +
                                           dLight.specular * specFactor;

        return vec4(enlightedColor.rgb, color.a);
    }
    /* End of Lighting */
    """

    fragmentShaderFunctionNoop = """
    vec4 lighting(vec4 color, vec3 position, vec3 normal)
    {
        return color;
    }
    """

    def __init__(self, direction=None,
                 ambient=(1., 1., 1.), diffuse=(0., 0., 0.),
                 specular=(1., 1., 1.), shininess=0):
        super(DirectionalLight, self).__init__()
        self._direction = None
        self.direction = direction  # Set _direction
        self._isOn = True
        self._ambient = ambient
        self._diffuse = diffuse
        self._specular = specular
        self._shininess = shininess

    ambient = event.notifyProperty('_ambient')
    diffuse = event.notifyProperty('_diffuse')
    specular = event.notifyProperty('_specular')
    shininess = event.notifyProperty('_shininess')

    @property
    def isOn(self):
        """True if light is on, False otherwise."""
        return self._isOn and self._direction is not None

    @isOn.setter
    def isOn(self, isOn):
        self._isOn = bool(isOn)

    @contextlib.contextmanager
    def turnOff(self):
        """Context manager to temporary turn off lighting during rendering.

        >>> with light.turnOff():
        ...     # Do some rendering without lighting
        """
        wason = self._isOn
        self._isOn = False
        yield
        self._isOn = wason

    @property
    def direction(self):
        """The direction of the light, or None if light is not on."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        if direction is None:
            self._direction = None
        else:
            assert len(direction) == 3
            direction = numpy.array(direction, dtype=numpy.float32, copy=True)
            norm = numpy.linalg.norm(direction)
            assert norm != 0
            self._direction = direction / norm
        self.notify()

    # GL2

    @property
    def fragmentDef(self):
        """Definition to add to fragment shader"""
        if self.isOn:
            return self.fragmentShaderFunction
        else:
            return self.fragmentShaderFunctionNoop

    @property
    def fragmentCall(self):
        """Function name to call in fragment shader"""
        return "lighting"

    def setupProgram(self, context, program):
        """Sets-up uniforms of a program using this shader function.

        :param RenderContext context: The current rendering context
        :param GLProgram program: The program to set-up.
                                  It MUST be in use and using this function.
        """
        if self.isOn and self._direction is not None:
            # Transform light direction from camera space to object coords
            lightdir = context.objectToCamera.transformDir(
                self._direction, direct=False)
            lightdir /= numpy.linalg.norm(lightdir)

            gl.glUniform3f(program.uniforms['dLight.lightDir'], *lightdir)

            # Convert view position to object coords
            viewpos = context.objectToCamera.transformPoint(
                numpy.array((0., 0., 0., 1.), dtype=numpy.float32),
                direct=False,
                perspectiveDivide=True)[:3]
            gl.glUniform3f(program.uniforms['dLight.viewPos'], *viewpos)

            gl.glUniform3f(program.uniforms['dLight.ambient'], *self.ambient)
            gl.glUniform3f(program.uniforms['dLight.diffuse'], *self.diffuse)
            gl.glUniform3f(program.uniforms['dLight.specular'], *self.specular)
            gl.glUniform1f(program.uniforms['dLight.shininess'],
                           self.shininess)


class Colormap(event.Notifier, ProgramFunction):

    _declTemplate = string.Template("""
    uniform sampler2D cmap_texture;
    uniform int cmap_normalization;
    uniform float cmap_parameter;
    uniform float cmap_min;
    uniform float cmap_oneOverRange;
    uniform vec4 nancolor;

    const float oneOverLog10 = 0.43429448190325176;

    vec4 colormap(float value) {
        float data = value; /* Keep original input value for isnan test */

        if (cmap_normalization == 1) { /* Log10 mapping */
            if (value > 0.0) {
                value = clamp(cmap_oneOverRange *
                              (oneOverLog10 * log(value) - cmap_min),
                              0.0, 1.0);
            } else {
                value = 0.0;
            }
        } else if (cmap_normalization == 2) { /* Sqrt mapping */
            if (value > 0.0) {
                value = clamp(cmap_oneOverRange * (sqrt(value) - cmap_min),
                              0.0, 1.0);
            } else {
                value = 0.0;
            }
        } else if (cmap_normalization == 3) { /*Gamma correction mapping*/
            value = pow(
                clamp(cmap_oneOverRange * (value - cmap_min), 0.0, 1.0),
                cmap_parameter);
        } else if (cmap_normalization == 4) { /* arcsinh mapping */
            /* asinh = log(x + sqrt(x*x + 1) for compatibility with GLSL 1.20 */
            value = clamp(cmap_oneOverRange * (log(value + sqrt(value*value + 1.0)) - cmap_min), 0.0, 1.0);
        } else { /* Linear mapping */
            value = clamp(cmap_oneOverRange * (value - cmap_min), 0.0, 1.0);
        }

        $discard

        vec4 color;
        if (data != data) { /* isnan alternative for compatibility with GLSL 1.20 */
            color = nancolor;
        } else {
            color = texture2D(cmap_texture, vec2(value, 0.5));
        }
        return color;
    }
    """)

    _discardCode = """
        if (value == 0.) {
            discard;
        }
    """

    call = "colormap"

    NORMS = 'linear', 'log', 'sqrt', 'gamma', 'arcsinh'
    """Tuple of supported normalizations."""

    _COLORMAP_TEXTURE_UNIT = 1
    """Texture unit to use for storing the colormap"""

    def __init__(self, colormap=None, norm='linear', gamma=0., range_=(1., 10.)):
        """Shader function to apply a colormap to a value.

        :param colormap: RGB(A) color look-up table (default: gray)
        :param colormap: numpy.ndarray of numpy.uint8 of dimension Nx3 or Nx4
        :param str norm: Normalization to apply: see :attr:`NORMS`.
        :param float gamma: Gamma normalization parameter
        :param range_: Range of value to map to the colormap.
        :type range_: 2-tuple of float (begin, end).
        """
        super(Colormap, self).__init__()

        # Init privates to default
        self._colormap = None
        self._norm = 'linear'
        self._gamma = -1.
        self._range = 1., 10.
        self._displayValuesBelowMin = True
        self._nancolor = numpy.array((1., 1., 1., 0.), dtype=numpy.float32)

        self._texture = None
        self._textureToDiscard = None

        if colormap is None:
            # default colormap
            colormap = numpy.empty((256, 3), dtype=numpy.uint8)
            colormap[:] = numpy.arange(256,
                                       dtype=numpy.uint8)[:, numpy.newaxis]

        # Set to values through properties to perform asserts and updates
        self.colormap = colormap
        self.norm = norm
        self.gamma = gamma
        self.range_ = range_

    @property
    def decl(self):
        """Source code of the function declaration"""
        return self._declTemplate.substitute(
            discard="" if self.displayValuesBelowMin else self._discardCode)

    @property
    def colormap(self):
        """Color look-up table to use."""
        return numpy.array(self._colormap, copy=True)

    @colormap.setter
    def colormap(self, colormap):
        colormap = numpy.array(colormap, copy=True)
        assert colormap.ndim == 2
        assert colormap.shape[1] in (3, 4)
        self._colormap = colormap

        if self._texture is not None and self._texture.name is not None:
            self._textureToDiscard = self._texture

        data = numpy.empty(
            (16, self._colormap.shape[0], self._colormap.shape[1]),
            dtype=self._colormap.dtype)
        data[:] = self._colormap

        format_ = gl.GL_RGBA if data.shape[-1] == 4 else gl.GL_RGB

        self._texture = _glutils.Texture(
            format_, data, format_,
            texUnit=self._COLORMAP_TEXTURE_UNIT,
            minFilter=gl.GL_NEAREST,
            magFilter=gl.GL_NEAREST,
            wrap=gl.GL_CLAMP_TO_EDGE)

        self.notify()

    @property
    def nancolor(self):
        """RGBA color to use for Not-A-Number values as 4 float in [0., 1.]"""
        return self._nancolor

    @nancolor.setter
    def nancolor(self, color):
        color = numpy.clip(numpy.array(color, dtype=numpy.float32), 0., 1.)
        assert color.ndim == 1
        assert len(color) == 4
        if not numpy.array_equal(self._nancolor, color):
            self._nancolor = color
            self.notify()

    @property
    def norm(self):
        """Normalization to use for colormap mapping.

        One of 'linear' (the default), 'log' for log10 mapping or 'sqrt'.
        Invalid values (e.g., negative values with 'log' or 'sqrt') are mapped to 0.
        """
        return self._norm

    @norm.setter
    def norm(self, norm):
        if norm != self._norm:
            assert norm in self.NORMS
            self._norm = norm
            if norm in ('log', 'sqrt'):
                self.range_ = self.range_  # To test for positive range_
            self.notify()

    @property
    def gamma(self):
        """Gamma correction normalization parameter (float >= 0.)"""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        if gamma != self._gamma:
            assert gamma >= 0.
            self._gamma = gamma
            self.notify()

    @property
    def range_(self):
        """Range of values to map to the colormap.

        2-tuple of floats: (begin, end).
        The begin value is mapped to the origin of the colormap and the
        end value is mapped to the other end of the colormap.
        The colormap is reversed if begin > end.
        """
        return self._range

    @range_.setter
    def range_(self, range_):
        assert len(range_) == 2
        range_ = float(range_[0]), float(range_[1])

        if self.norm == 'log' and (range_[0] <= 0. or range_[1] <= 0.):
            _logger.warning(
                "Log normalization and negative range: updating range.")
            minPos = numpy.finfo(numpy.float32).tiny
            range_ = max(range_[0], minPos), max(range_[1], minPos)
        elif self.norm == 'sqrt' and (range_[0] < 0. or range_[1] < 0.):
            _logger.warning(
                "Sqrt normalization and negative range: updating range.")
            range_ = max(range_[0], 0.), max(range_[1], 0.)

        if range_ != self._range:
            self._range = range_
            self.notify()

    @property
    def displayValuesBelowMin(self):
        """True to display values below colormap min, False to discard them.
        """
        return self._displayValuesBelowMin

    @displayValuesBelowMin.setter
    def displayValuesBelowMin(self, displayValuesBelowMin):
        displayValuesBelowMin = bool(displayValuesBelowMin)
        if self._displayValuesBelowMin != displayValuesBelowMin:
            self._displayValuesBelowMin = displayValuesBelowMin
            self.notify()

    def setupProgram(self, context, program):
        """Sets-up uniforms of a program using this shader function.

        :param RenderContext context: The current rendering context
        :param GLProgram program: The program to set-up.
                                  It MUST be in use and using this function.
        """
        self.prepareGL2(context)  # TODO see how to handle

        self._texture.bind()

        gl.glUniform1i(program.uniforms['cmap_texture'],
                       self._texture.texUnit)

        min_, max_ = self.range_
        param = 0.
        if self._norm == 'log':
            min_, max_ = numpy.log10(min_), numpy.log10(max_)
            normID = 1
        elif self._norm == 'sqrt':
            min_, max_ = numpy.sqrt(min_), numpy.sqrt(max_)
            normID = 2
        elif self._norm == 'gamma':
            # Keep min_, max_ as is
            param = self._gamma
            normID = 3
        elif self._norm == 'arcsinh':
            min_, max_ = numpy.arcsinh(min_), numpy.arcsinh(max_)
            normID = 4
        else:  # Linear
            normID = 0

        gl.glUniform1i(program.uniforms['cmap_normalization'], normID)
        gl.glUniform1f(program.uniforms['cmap_parameter'], param)
        gl.glUniform1f(program.uniforms['cmap_min'], min_)
        gl.glUniform1f(program.uniforms['cmap_oneOverRange'],
                       (1. / (max_ - min_)) if max_ != min_ else 0.)
        gl.glUniform4f(program.uniforms['nancolor'], *self._nancolor)

    def prepareGL2(self, context):
        if self._textureToDiscard is not None:
            self._textureToDiscard.discard()
            self._textureToDiscard = None

        self._texture.prepare()
