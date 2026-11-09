# /*##########################################################################
#
# Copyright (c) 2014-2021 European Synchrotron Radiation Facility
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
"""This module provides conversion functions between OpenGL and numpy types."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "10/01/2017"

import numpy

from OpenGL.constants import BYTE_SIZES as _BYTE_SIZES
from OpenGL.constants import ARRAY_TO_GL_TYPE_MAPPING as _ARRAY_TO_GL_TYPE_MAPPING


def sizeofGLType(type_):
    """Returns the size in bytes of an element of type `type_`"""
    return _BYTE_SIZES[type_]


def isSupportedGLType(type_):
    """Test if a numpy type or dtype can be converted to a GL type."""
    return numpy.dtype(type_).char in _ARRAY_TO_GL_TYPE_MAPPING


def numpyToGLType(type_):
    """Returns the GL type corresponding the provided numpy type or dtype."""
    return _ARRAY_TO_GL_TYPE_MAPPING[numpy.dtype(type_).char]
