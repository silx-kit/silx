# /*##########################################################################
#
# Copyright (c) 2014-2019 European Synchrotron Radiation Facility
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
"""This module provides a class to handle shader program compilation."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import logging
import weakref

import numpy

from . import Context, gl

_logger = logging.getLogger(__name__)


class Program(object):
    """Wrap OpenGL shader program.

    The program is compiled lazily (i.e., at first program :meth:`use`).
    When the program is compiled, it stores attributes and uniforms locations.
    So, attributes and uniforms must be used after :meth:`use`.

    This object supports multiple OpenGL contexts.

    :param str vertexShader: The source of the vertex shader.
    :param str fragmentShader: The source of the fragment shader.
    :param str attrib0:
        Attribute's name to bind to position 0 (default: 'position').
        On certain platform, this attribute MUST be active and with an
        array attached to it in order for the rendering to occur....
    """

    def __init__(self, vertexShader, fragmentShader,
                 attrib0='position'):
        self._vertexShader = vertexShader
        self._fragmentShader = fragmentShader
        self._attrib0 = attrib0
        self._programs = weakref.WeakKeyDictionary()

    @staticmethod
    def _compileGL(vertexShader, fragmentShader, attrib0):
        program = gl.glCreateProgram()

        gl.glBindAttribLocation(program, 0, attrib0.encode('ascii'))

        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertex, vertexShader)
        gl.glCompileShader(vertex)
        if gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            raise RuntimeError(gl.glGetShaderInfoLog(vertex))
        gl.glAttachShader(program, vertex)
        gl.glDeleteShader(vertex)

        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment, fragmentShader)
        gl.glCompileShader(fragment)
        if gl.glGetShaderiv(fragment,
                            gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            raise RuntimeError(gl.glGetShaderInfoLog(fragment))
        gl.glAttachShader(program, fragment)
        gl.glDeleteShader(fragment)

        gl.glLinkProgram(program)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            raise RuntimeError(gl.glGetProgramInfoLog(program))

        attributes = {}
        for index in range(gl.glGetProgramiv(program,
                                             gl.GL_ACTIVE_ATTRIBUTES)):
            name = gl.glGetActiveAttrib(program, index)[0]
            namestr = name.decode('ascii')
            attributes[namestr] = gl.glGetAttribLocation(program, name)

        uniforms = {}
        for index in range(gl.glGetProgramiv(program, gl.GL_ACTIVE_UNIFORMS)):
            name = gl.glGetActiveUniform(program, index)[0]
            namestr = name.decode('ascii')
            uniforms[namestr] = gl.glGetUniformLocation(program, name)

        return program, attributes, uniforms

    def _getProgramInfo(self):
        glcontext = Context.getCurrent()
        if glcontext not in self._programs:
            raise RuntimeError(
                "Program was not compiled for current OpenGL context.")
        return self._programs[glcontext]

    @property
    def attributes(self):
        """Vertex attributes names and locations as a dict of {str: int}.

        WARNING:
        Read-only usage.
        To use only with a valid OpenGL context and after :meth:`use`
        has been called for this context.
        """
        return self._getProgramInfo()[1]

    @property
    def uniforms(self):
        """Program uniforms names and locations as a dict of {str: int}.

        WARNING:
        Read-only usage.
        To use only with a valid OpenGL context and after :meth:`use`
        has been called for this context.
        """
        return self._getProgramInfo()[2]

    @property
    def program(self):
        """OpenGL id of the program.

        WARNING:
        To use only with a valid OpenGL context and after :meth:`use`
        has been called for this context.
        """
        return self._getProgramInfo()[0]

    # def discard(self):
    #    pass  # Not implemented yet

    def use(self):
        """Make use of the program, compiling it if necessary"""
        glcontext = Context.getCurrent()

        if glcontext not in self._programs:
            self._programs[glcontext] = self._compileGL(
                self._vertexShader,
                self._fragmentShader,
                self._attrib0)

        if _logger.getEffectiveLevel() <= logging.DEBUG:
            gl.glValidateProgram(self.program)
            if gl.glGetProgramiv(
                    self.program, gl.GL_VALIDATE_STATUS) != gl.GL_TRUE:
                _logger.debug('Cannot validate program: %s',
                              gl.glGetProgramInfoLog(self.program))

        gl.glUseProgram(self.program)

    def setUniformMatrix(self, name, value, transpose=True, safe=False):
        """Wrap glUniformMatrix[2|3|4]fv

        :param str name: The name of the uniform.
        :param value: The 2D matrix (or the array of matrices, 3D).
                      Matrices are 2x2, 3x3 or 4x4.
        :type value: numpy.ndarray with 2 or 3 dimensions of float32
        :param bool transpose: Whether to transpose (True, default) or not.
        :param bool safe: False: raise an error if no uniform with this name;
                          True: silently ignores it.

        :raises KeyError: if no uniform corresponds to name.
        """
        assert value.dtype == numpy.float32

        shape = value.shape
        assert len(shape) in (2, 3)
        assert shape[-1] in (2, 3, 4)
        assert shape[-1] == shape[-2]  # As in OpenGL|ES 2.0

        location = self.uniforms.get(name)
        if location is not None:
            count = 1 if len(shape) == 2 else shape[0]
            transpose = gl.GL_TRUE if transpose else gl.GL_FALSE

            if shape[-1] == 2:
                gl.glUniformMatrix2fv(location, count, transpose, value)
            elif shape[-1] == 3:
                gl.glUniformMatrix3fv(location, count, transpose, value)
            elif shape[-1] == 4:
                gl.glUniformMatrix4fv(location, count, transpose, value)

        elif not safe:
            raise KeyError('No uniform: %s' % name)
