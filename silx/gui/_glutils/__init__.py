# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2017 European Synchrotron Radiation Facility
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
"""This package provides utility functions to handle OpenGL resources.

The :mod:`gl` module provides a wrapper to OpenGL based on PyOpenGL.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


# OpenGL convenient functions
from .OpenGLWidget import OpenGLWidget  # noqa
from .Context import getGLContext, setGLContextGetter  # noqa
from .FramebufferTexture import FramebufferTexture  # noqa
from .Program import Program  # noqa
from .Texture import Texture  # noqa
from .VertexBuffer import VertexBuffer, VertexBufferAttrib, vertexBuffer  # noqa
from .utils import sizeofGLType, isSupportedGLType, numpyToGLType  # noqa
