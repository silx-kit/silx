# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This module provides a class to handle shader program compilation.
"""


# import ######################################################################

from ctypes import c_float
import warnings

import numpy

from .gl import *  # noqa
from .GLContext import getGLContext


# utils #######################################################################

def _glGetActiveAttrib(program, index):
    """Wrap PyOpenGL glGetActiveAttrib as for glGetActiveUniform
    """
    bufSize = glGetProgramiv(program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH)
    length = GLsizei()
    size = GLint()
    type_ = GLenum()
    name = (GLchar * bufSize)()

    glGetActiveAttrib(program, index, bufSize, length, size, type_, name)
    return name.value, size.value, type_.value


# GLProgram ###################################################################

class GLProgram(object):
    """Wrap OpenGL shader program.

    The program is compiled lazily (i.e., at first program :meth:`use`).
    When the program is compiled, it stores attributes and uniforms locations.
    So, attributes and uniforms must be used after :meth:`use`.

    This object supports multiple OpenGL contexts.
    """

    def __init__(self, vertexShaderSrc, fragmentShaderSrc):
        """Create the object handling a shader program.

        :param str vertexShaderSrc: The source of the vertex shader.
        :param str fragmentShaderSrc: The source of the fragment shader.
        """
        self._vertexShaderSrc = vertexShaderSrc
        self._fragmentShaderSrc = fragmentShaderSrc
        self._programs = {}

    @staticmethod
    def _compileGL(vertexShaderSrc, fragmentShaderSrc):
        program = glCreateProgram()

        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, vertexShaderSrc)
        glCompileShader(vertexShader)
        if glGetShaderiv(vertexShader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(vertexShader))
        glAttachShader(program, vertexShader)
        glDeleteShader(vertexShader)

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, fragmentShaderSrc)
        glCompileShader(fragmentShader)
        if glGetShaderiv(fragmentShader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(fragmentShader))
        glAttachShader(program, fragmentShader)
        glDeleteShader(fragmentShader)

        glLinkProgram(program)
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(program))

        glValidateProgram(program)
        if glGetProgramiv(program, GL_VALIDATE_STATUS) != GL_TRUE:
            warnings.warn(
                'Cannot validate program: ' + glGetProgramInfoLog(program),
                RuntimeWarning)

        attributes = {}
        for index in range(glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES)):
            name = _glGetActiveAttrib(program, index)[0]
            nameStr = name.decode('ascii')
            attributes[nameStr] = glGetAttribLocation(program, name)

        uniforms = {}
        for index in range(glGetProgramiv(program, GL_ACTIVE_UNIFORMS)):
            name = glGetActiveUniform(program, index)[0]
            nameStr = name.decode('ascii')
            uniforms[nameStr] = glGetUniformLocation(program, name)

        return program, attributes, uniforms

    def _getProgramInfo(self):
        glContext = getGLContext()
        if glContext not in self._programs:
            raise RuntimeError(
                "Program was not compiled for current OpenGL context.")
        return self._programs[glContext]

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

    def discard(self):
        pass  # Not implemented yet

    def __del__(self):
        self.discard()

    def use(self):
        glContext = getGLContext()

        if glContext not in self._programs:
            self._programs[glContext] = self._compileGL(
                self._vertexShaderSrc, self._fragmentShaderSrc)

        glUseProgram(self.program)

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
            transpose = GL_TRUE if transpose else GL_FALSE

            if shape[-1] == 2:
                glUniformMatrix2fv(location, count, transpose, value)
            elif shape[-1] == 3:
                glUniformMatrix3fv(location, count, transpose, value)
            elif shape[-1] == 4:
                glUniformMatrix4fv(location, count, transpose, value)

        elif not safe:
            raise KeyError('No uniform: %s' % name)


# main ########################################################################

if __name__ == "__main__":
    import sys
    try:
        from PyQt4.QtGui import QApplication
        from PyQt4.QtOpenGL import QGLWidget
    except ImportError:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtOpenGL import QGLWidget

    # TODO a better test example
    class Test(QGLWidget):
        _vertexShaderSrc = """
            attribute vec2 position;

            void main(void) {
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """

        _fragmentShaderSrc = """
            uniform vec4 color;

            void main(void) {
                gl_FragColor = color;
            }
            """

        def initializeGL(self):
            glClearColor(1., 1., 1., 0.)

            self.glProgram = Program(self._vertexShaderSrc,
                                     self._fragmentShaderSrc)

        def paintGL(self):
            glClear(GL_COLOR_BUFFER_BIT)

            self.glProgram.use()
            print("Attributes: {0}".format(self.glProgram.attributes))
            print("Uniforms: {0}".format(self.glProgram.uniforms))

            w, h = 128, 128
            data = (c_float * (w * h * 3))()
            for i in range(w * h):
                data[3*i] = i/float(w*h)
                data[3*i+1] = i/float(w*h)
                data[3*i+2] = i/float(w*h)

            glUniform4f(self.glProgram.uniforms['color'], 1., 0., 0., 1.)

            positions = (c_float * (4 * 2))(
                0., 0.,   1., 0.,   0., 1.,   1., 1.)
            glEnableVertexAttribArray(self.glProgram.attributes['position'])
            glVertexAttribPointer(self.glProgram.attributes['position'],
                                  2,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  0, positions)

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        def resizeGL(self, w, h):
            glViewport(0, 0, w, h)

    app = QApplication([])
    widget = Test()
    widget.show()
    sys.exit(app.exec_())
