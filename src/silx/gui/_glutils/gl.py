# /*##########################################################################
#
# Copyright (c) 2014-2022 European Synchrotron Radiation Facility
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
"""This module loads PyOpenGL and provides a namespace for OpenGL."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


from contextlib import contextmanager as _contextmanager
from ctypes import c_uint
import logging
import sys

import numpy as np
from packaging.version import Version
import OpenGL

_logger = logging.getLogger(__name__)

# Set the following to true for debugging
if _logger.getEffectiveLevel() <= logging.DEBUG:
    _logger.debug("Enabling PyOpenGL debug flags")
    OpenGL.ERROR_LOGGING = True
    OpenGL.ERROR_CHECKING = True
    OpenGL.ERROR_ON_COPY = True
else:
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_ON_COPY = False

if sys.version_info >= (3, 12) and Version(OpenGL.__version__) <= Version("3.1.7"):
    # Python3.12 patch: see https://github.com/mcfletch/pyopengl/pull/100
    OpenGL.FormatHandler.by_name("ctypesparameter").check.append("_ctypes.CArgObject")


import OpenGL.GL as _GL  # noqa: E402
from OpenGL.GL import *  # noqa
import OpenGL.platform  # noqa: E402

# Extentions core in OpenGL 3
from OpenGL.GL.ARB import framebuffer_object as _FBO  # noqa: E402
from OpenGL.GL.ARB.framebuffer_object import *  # noqa
from OpenGL.GL.ARB.texture_rg import GL_R32F, GL_R16F  # noqa
from OpenGL.GL.ARB.texture_rg import GL_R16, GL_R8  # noqa

# PyOpenGL 3.0.1 does not define it
try:
    _GL.GLchar
except NameError:
    from ctypes import c_char

    GLchar = c_char


def getPlatform() -> str | None:
    """Returns the name of the PyOpenGL class handling the platform.

    E.g., GLXPlatform, EGLPlatform
    """
    try:
        platform = OpenGL.platform.PLATFORM
    except AttributeError:
        return None
    return platform.__class__.__name__


def getVersion() -> tuple:
    """Returns the GL version as tuple of integers.

    Raises:
        ValueError: If the version returned by the driver is not supported
    """
    try:
        desc = _GL.glGetString(_GL.GL_VERSION)
        if isinstance(desc, bytes):
            desc = desc.decode("ascii")
        version = desc.split(" ", 1)[0]
        return tuple([int(i) for i in version.split(".")])
    except Exception as e:
        raise ValueError("GL version not properly formatted") from e


def testGL() -> bool:
    """Test if required OpenGL version and extensions are available.

    This MUST be run with an active OpenGL context.
    """
    version = getVersion()
    major, minor = version[0], version[1]
    if major < 2 or (major == 2 and minor < 1):
        _logger.error("OpenGL version >=2.1 required, running with %s" % version)
        return False

    if major == 2:
        from OpenGL.GL.ARB.framebuffer_object import glInitFramebufferObjectARB
        from OpenGL.GL.ARB.texture_rg import glInitTextureRgARB

        if not glInitFramebufferObjectARB():
            _logger.error("OpenGL GL_ARB_framebuffer_object extension required!")
            return False

        if not glInitTextureRgARB():
            _logger.error("OpenGL GL_ARB_texture_rg extension required!")
            return False

    return True


# Additional setup
if hasattr(_GL.glget, "addGLGetConstant"):
    _GL.glget.addGLGetConstant(_FBO.GL_FRAMEBUFFER_BINDING, (1,))


@_contextmanager
def enabled(capacity, enable=True):
    """Context manager enabling an OpenGL capacity.

    This is not checking the current state of the capacity.

    :param capacity: The OpenGL capacity enum to enable/disable
    :param bool enable:
        True (default) to enable during context, False to disable
    """
    if bool(enable) == _GL.glGetBoolean(capacity):
        # Already in the right state: noop
        yield
    elif enable:
        _GL.glEnable(capacity)
        yield
        _GL.glDisable(capacity)
    else:
        _GL.glDisable(capacity)
        yield
        _GL.glEnable(capacity)


def disabled(capacity, disable=True):
    """Context manager disabling an OpenGL capacity.

    This is not checking the current state of the capacity.

    :param capacity: The OpenGL capacity enum to disable/enable
    :param bool disable:
        True (default) to disable during context, False to enable
    """
    return enabled(capacity, not disable)


# Additional OpenGL wrapping


def glGetActiveAttrib(program, index):
    """Wrap PyOpenGL glGetActiveAttrib"""
    bufsize = _GL.glGetProgramiv(program, _GL.GL_ACTIVE_ATTRIBUTE_MAX_LENGTH)
    length = _GL.GLsizei()
    size = _GL.GLint()
    type_ = _GL.GLenum()
    name = (GLchar * bufsize)()

    _GL.glGetActiveAttrib(program, index, bufsize, length, size, type_, name)
    return name.value, size.value, type_.value


def glGetActiveUniform(program, index):
    """Wrap PyOpenGL glGetActiveUniform"""
    name, size, type_ = _GL.glGetActiveUniform(program, index)
    if isinstance(name, np.ndarray):
        name = name.tobytes().rstrip(b"\000")
    return name, size, type_


def glDeleteRenderbuffers(buffers):
    if not hasattr(buffers, "__len__"):  # Support single int argument
        buffers = [buffers]
    length = len(buffers)
    _FBO.glDeleteRenderbuffers(length, (c_uint * length)(*buffers))


def glDeleteFramebuffers(buffers):
    if not hasattr(buffers, "__len__"):  # Support single int argument
        buffers = [buffers]
    length = len(buffers)
    _FBO.glDeleteFramebuffers(length, (c_uint * length)(*buffers))


def glDeleteBuffers(buffers):
    if not hasattr(buffers, "__len__"):  # Support single int argument
        buffers = [buffers]
    length = len(buffers)
    _GL.glDeleteBuffers(length, (c_uint * length)(*buffers))


def glDeleteTextures(textures):
    if not hasattr(textures, "__len__"):  # Support single int argument
        textures = [textures]
    length = len(textures)
    _GL.glDeleteTextures((c_uint * length)(*textures))
