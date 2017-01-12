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
"""This module provides conversion functions between OpenGL and numpy types.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "10/01/2017"

from . import gl
import numpy


_GL_TYPE_SIZES = {
    gl.GL_FLOAT: 4,
    gl.GL_BYTE: 1,
    gl.GL_SHORT: 2,
    gl.GL_INT: 4,
    gl.GL_UNSIGNED_BYTE: 1,
    gl.GL_UNSIGNED_SHORT: 2,
    gl.GL_UNSIGNED_INT: 4,
}


def sizeofGLType(type_):
    """Returns the size in bytes of an element of type `type_`"""
    return _GL_TYPE_SIZES[type_]


_TYPE_CONVERTER = {
    numpy.dtype(numpy.float32): gl.GL_FLOAT,
    numpy.dtype(numpy.int8): gl.GL_BYTE,
    numpy.dtype(numpy.int16): gl.GL_SHORT,
    numpy.dtype(numpy.int32): gl.GL_INT,
    numpy.dtype(numpy.uint8): gl.GL_UNSIGNED_BYTE,
    numpy.dtype(numpy.uint16): gl.GL_UNSIGNED_SHORT,
    numpy.dtype(numpy.uint32): gl.GL_UNSIGNED_INT,
}


def isSupportedGLType(type_):
    """Test if a numpy type or dtype can be converted to a GL type."""
    return numpy.dtype(type_) in _TYPE_CONVERTER


def numpyToGLType(type_):
    """Returns the GL type corresponding the provided numpy type or dtype."""
    return _TYPE_CONVERTER[numpy.dtype(type_)]
