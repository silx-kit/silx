# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
This module provides a class managing a vertex buffer
"""

# import ######################################################################

from .gl import *  # noqa
from ctypes import c_void_p, c_uint
import numpy as np


# VBO #########################################################################

class VertexBuffer(object):
    # OpenGL|ES 2.0 subset:
    _USAGES = GL_STREAM_DRAW, GL_STATIC_DRAW, GL_DYNAMIC_DRAW
    _TARGETS = GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER

    def __init__(self, data=None, sizeInBytes=None,
                 usage=None, target=None):
        if usage is None:
            usage = GL_STATIC_DRAW
        assert usage in self._USAGES

        if target is None:
            target = GL_ARRAY_BUFFER
        assert target in self._TARGETS

        self._target = target

        self._vboId = glGenBuffers(1)
        self.bind()
        if data is None:
            assert sizeInBytes is not None
            self._size = sizeInBytes
            glBufferData(self._target,
                         self._size,
                         c_void_p(0),
                         usage)
        else:
            assert isinstance(data, np.ndarray) and data.flags['C_CONTIGUOUS']
            if sizeInBytes is not None:
                assert sizeInBytes <= data.nbytes

            self._size = sizeInBytes or data.nbytes
            glBufferData(self._target,
                         self._size,
                         data,
                         usage)

        glBindBuffer(self._target, 0)

    @property
    def vboId(self):
        """OpenGL Vertex Buffer Object ID
        :type: int
        """
        try:
            return self._vboId
        except AttributeError:
            raise RuntimeError("No OpenGL buffer resource, \
                               discard has already been called")

    @property
    def size(self):
        """Size in bytes of the Vertex Buffer Object
        :type: int
        """
        try:
            return self._size
        except AttributeError:
            raise RuntimeError("No OpenGL buffer resource, \
                               discard has already been called")

    def bind(self):
        glBindBuffer(self._target, self.vboId)

    def update(self, data, offsetInBytes=0, sizeInBytes=None):
        assert isinstance(data, np.ndarray) and data.flags['C_CONTIGUOUS']
        if sizeInBytes is None:
            sizeInBytes = data.nbytes
        assert offsetInBytes + sizeInBytes <= self.size
        with self:
            glBufferSubData(self._target, offsetInBytes, sizeInBytes, data)

    def discard(self):
        if hasattr(self, '_vboId'):
            if bool(glDeleteBuffers):  # Test for __del__
                glDeleteBuffers(1, (c_uint * 1)(self._vboId))
            del self._vboId
            del self._size

    def __del__(self):
        self.discard()

    # with statement

    def __enter__(self):
        self.bind()

    def __exit__(self, excType, excValue, traceback):
        glBindBuffer(self._target, 0)


# VBOAttrib ###################################################################

class VBOAttrib(object):
    """Describes data stored in a VBO
    """

    _GL_TYPES = GL_UNSIGNED_BYTE, GL_FLOAT, GL_INT

    def __init__(self, vbo, type_,
                 size, dimension=1,
                 offset=0, stride=0):
        """
        :param VertexBuffer vbo: The VBO storing the data
        :param int type_: The OpenGL type of the data
        :param int size: The number of data elements stored in the VBO
        :param int dimension: The number of type_ element(s) in [1, 4]
        :param int offset: Start offset of data in the VBO
        :param int stride: Data stride in the VBO
        """
        self.vbo = vbo
        assert type_ in self._GL_TYPES
        self.type_ = type_
        self.size = size
        assert dimension >= 1 and dimension <= 4
        self.dimension = dimension
        self.offset = offset
        self.stride = stride

    @property
    def itemSize(self):
        """Size of a VBO element in bytes"""
        return self.dimension * sizeofGLType(self.type_)

    def setVertexAttrib(self, attrib):
        with self.vbo:
            glVertexAttribPointer(attrib,
                                  self.dimension,
                                  self.type_,
                                  GL_FALSE,
                                  self.stride,
                                  c_void_p(self.offset))

    def copy(self):
        return VBOAttrib(self.vbo,
                         self.type_,
                         self.size,
                         self.dimension,
                         self.offset,
                         self.stride)


def createVBOFromArrays(arrays, prefix=None, suffix=None, usage=None):
    """
    Create a single VBO from multiple 1D or 2D numpy arrays
    It is possible to reserve memory before and after each array in the VBO

    :param arrays: Arrays of data to store
    :type arrays: Iterable of numpy.ndarray
    :param prefix: If given, number of elements to reserve before each array
    :type prefix: Iterable of int or None
    :param suffix: If given, number of elements to reserve after each array
    :type suffix: Iterable of int or None
    :param int usage: VBO usage hint or None for default
    :returns: List of VBOAttrib objects sharing the same VBO
    """
    info = []
    vboSize = 0

    if prefix is None:
        prefix = (0,) * len(arrays)
    if suffix is None:
        suffix = (0,) * len(arrays)

    for data, pre, post in zip(arrays, prefix, suffix):
        shape = data.shape
        assert len(shape) <= 2
        type_ = numpyToGLType(data.dtype)
        size = shape[0] + pre + post
        dimension = 1 if len(shape) == 1 else shape[1]
        sizeInBytes = size * dimension * sizeofGLType(type_)
        sizeInBytes = 4 * (((sizeInBytes) + 3) >> 2)  # 4 bytes alignment
        copyOffset = vboSize + pre * dimension * sizeofGLType(type_)
        info.append((data, type_, size, dimension,
                     vboSize, sizeInBytes, copyOffset))
        vboSize += sizeInBytes

    vbo = VertexBuffer(sizeInBytes=vboSize, usage=usage)

    result = []
    for data, type_, size, dimension, offset, sizeInBytes, copyOffset in info:
        copySize = data.shape[0] * dimension * sizeofGLType(type_)
        vbo.update(data, offsetInBytes=copyOffset, sizeInBytes=copySize)
        result.append(VBOAttrib(vbo, type_, size, dimension, offset, 0))
    return result
