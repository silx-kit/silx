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
"""This module provides functions to add to shaders."""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "08/11/2016"


import contextlib
import logging
import numpy

from ..glutils import gl

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
    # TODO use colors for out-of-bound values, for <=0 with log, for nan
    # TODO texture-based colormap

    decl = """
    #define CMAP_GRAY   0
    #define CMAP_R_GRAY 1
    #define CMAP_RED    2
    #define CMAP_GREEN  3
    #define CMAP_BLUE   4
    #define CMAP_TEMP   5

    uniform struct {
        int id;
        bool isLog;
        float min;
        float oneOverRange;
    } cmap;

    const float oneOverLog10 = 0.43429448190325176;

    vec4 colormap(float value) {
        if (cmap.isLog) { /* Log10 mapping */
            if (value > 0.0) {
                value = clamp(cmap.oneOverRange *
                              (oneOverLog10 * log(value) - cmap.min),
                              0.0, 1.0);
            } else {
                value = 0.0;
            }
        } else { /* Linear mapping */
            value = clamp(cmap.oneOverRange * (value - cmap.min), 0.0, 1.0);
        }

        if (cmap.id == CMAP_GRAY) {
            return vec4(value, value, value, 1.0);
        }
        else if (cmap.id == CMAP_R_GRAY) {
            float invValue = 1.0 - value;
            return vec4(invValue, invValue, invValue, 1.0);
        }
        else if (cmap.id == CMAP_RED) {
            return vec4(value, 0.0, 0.0, 1.0);
        }
        else if (cmap.id == CMAP_GREEN) {
            return vec4(0.0, value, 0.0, 1.0);
        }
        else if (cmap.id == CMAP_BLUE) {
            return vec4(0.0, 0.0, value, 1.0);
        }
        else if (cmap.id == CMAP_TEMP) {
            //red: 0.5->0.75: 0->1
            //green: 0.->0.25: 0->1; 0.75->1.: 1->0
            //blue: 0.25->0.5: 1->0
            return vec4(
                clamp(4.0 * value - 2.0, 0.0, 1.0),
                1.0 - clamp(4.0 * abs(value - 0.5) - 1.0, 0.0, 1.0),
                1.0 - clamp(4.0 * value - 1.0, 0.0, 1.0),
                1.0);
        }
        else {
            /* Unknown colormap */
            return vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
    """

    call = "colormap"

    _COLORMAPS = {
        'gray': 0,
        'reversed gray': 1,
        'red': 2,
        'green': 3,
        'blue': 4,
        'temperature': 5
    }

    COLORMAPS = tuple(_COLORMAPS.keys())
    """Tuple of supported colormap names."""

    NORMS = 'linear', 'log'
    """Tuple of supported normalizations."""

    def __init__(self, name='gray', norm='linear', range_=(1., 10.)):
        """Shader function to apply a colormap to a value.

        :param str name: Name of the colormap.
        :param str norm: Normalization to apply: 'linear' (default) or 'log'.
        :param range_: Range of value to map to the colormap.
        :type range_: 2-tuple of float (begin, end).
        """
        super(Colormap, self).__init__()

        # Init privates to default
        self._name, self._norm, self._range = 'gray', 'linear', (1., 10.)

        # Set to param values through properties to go through asserts
        self.name = name
        self.norm = norm
        self.range_ = range_

    @property
    def name(self):
        """Name of the colormap in use."""
        return self._name

    @name.setter
    def name(self, name):
        if name != self._name:
            assert name in self.COLORMAPS
            self._name = name
            self.notify()

    @property
    def norm(self):
        """Normalization to use for colormap mapping.

        Either 'linear' (the default) or 'log' for log10 mapping.
        With 'log' normalization, values <= 0. are set to 1. (i.e. log == 0)
        """
        return self._norm

    @norm.setter
    def norm(self, norm):
        if norm != self._norm:
            assert norm in self.NORMS
            self._norm = norm
            if norm == 'log':
                self.range_ = self.range_  # To test for positive range_
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
            _logger.warn(
                "Log normalization and negative range: updating range.")
            minPos = numpy.finfo(numpy.float32).tiny
            range_ = max(range_[0], minPos), max(range_[1], minPos)

        if range_ != self._range:
            self._range = range_
            self.notify()

    def setupProgram(self, context, program):
        """Sets-up uniforms of a program using this shader function.

        :param RenderContext context: The current rendering context
        :param GLProgram program: The program to set-up.
                                  It MUST be in use and using this function.
        """
        gl.glUniform1i(program.uniforms['cmap.id'], self._COLORMAPS[self.name])
        gl.glUniform1i(program.uniforms['cmap.isLog'], self._norm == 'log')

        min_, max_ = self.range_
        if self._norm == 'log':
            min_, max_ = numpy.log10(min_), numpy.log10(max_)

        gl.glUniform1f(program.uniforms['cmap.min'], min_)
        gl.glUniform1f(program.uniforms['cmap.oneOverRange'],
                       (1. / (max_ - min_)) if max_ != min_ else 0.)
