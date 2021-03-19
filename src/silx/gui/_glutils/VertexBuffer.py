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
"""This module provides a class managing an OpenGL vertex buffer."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "10/01/2017"


import logging
from ctypes import c_void_p
import numpy

from . import gl
from .utils import numpyToGLType, sizeofGLType


_logger = logging.getLogger(__name__)


class VertexBuffer(object):
    """Object handling an OpenGL vertex buffer object

    :param data: Data used to fill the vertex buffer
    :type data: numpy.ndarray or None
    :param int size: Size in bytes of the buffer or None for data size
    :param usage: OpenGL vertex buffer expected usage pattern:
        GL_STREAM_DRAW, GL_STATIC_DRAW (default) or GL_DYNAMIC_DRAW
    :param target: Target buffer:
        GL_ARRAY_BUFFER (default) or GL_ELEMENT_ARRAY_BUFFER
    """
    # OpenGL|ES 2.0 subset:
    _USAGES = gl.GL_STREAM_DRAW, gl.GL_STATIC_DRAW, gl.GL_DYNAMIC_DRAW
    _TARGETS = gl.GL_ARRAY_BUFFER, gl.GL_ELEMENT_ARRAY_BUFFER

    def __init__(self,
                 data=None,
                 size=None,
                 usage=None,
                 target=None):
        if usage is None:
            usage = gl.GL_STATIC_DRAW
        assert usage in self._USAGES

        if target is None:
            target = gl.GL_ARRAY_BUFFER
        assert target in self._TARGETS

        self._target = target
        self._usage = usage

        self._name = gl.glGenBuffers(1)
        self.bind()

        if data is None:
            assert size is not None
            self._size = size
            gl.glBufferData(self._target,
                            self._size,
                            c_void_p(0),
                            self._usage)
        else:
            data = numpy.array(data, copy=False, order='C')
            if size is not None:
                assert size <= data.nbytes

            self._size = size or data.nbytes
            gl.glBufferData(self._target,
                            self._size,
                            data,
                            self._usage)

        gl.glBindBuffer(self._target, 0)

    @property
    def target(self):
        """The target buffer of the vertex buffer"""
        return self._target

    @property
    def usage(self):
        """The expected usage of the vertex buffer"""
        return self._usage

    @property
    def name(self):
        """OpenGL Vertex Buffer object name (int)"""
        if self._name is not None:
            return self._name
        else:
            raise RuntimeError("No OpenGL buffer resource, \
                               discard has already been called")

    @property
    def size(self):
        """Size in bytes of the Vertex Buffer Object (int)"""
        if self._size is not None:
            return self._size
        else:
            raise RuntimeError("No OpenGL buffer resource, \
                               discard has already been called")

    def bind(self):
        """Bind the vertex buffer"""
        gl.glBindBuffer(self._target, self.name)

    def update(self, data, offset=0, size=None):
        """Update vertex buffer content.

        :param numpy.ndarray data: The data to put in the vertex buffer
        :param int offset: Offset in bytes in the buffer where to put the data
        :param int size: If provided, size of data to copy
        """
        data = numpy.array(data, copy=False, order='C')
        if size is None:
            size = data.nbytes
        assert offset + size <= self.size
        with self:
            gl.glBufferSubData(self._target, offset, size, data)

    def discard(self):
        """Delete the vertex buffer"""
        if self._name is not None:
            gl.glDeleteBuffers(self._name)
            self._name = None
            self._size = None
        else:
            _logger.warning("Discard has already been called")

    # with statement

    def __enter__(self):
        self.bind()

    def __exit__(self, exctype, excvalue, traceback):
        gl.glBindBuffer(self._target, 0)


class VertexBufferAttrib(object):
    """Describes data stored in a vertex buffer

    Convenient class to store info for glVertexAttribPointer calls

    :param VertexBuffer vbo: The vertex buffer storing the data
    :param int type_: The OpenGL type of the data
    :param int size: The number of data elements stored in the VBO
    :param int dimension: The number of `type_` element(s) in [1, 4]
    :param int offset: Start offset of data in the vertex buffer
    :param int stride: Data stride in the vertex buffer
    """

    _GL_TYPES = gl.GL_UNSIGNED_BYTE, gl.GL_FLOAT, gl.GL_INT

    def __init__(self,
                 vbo,
                 type_,
                 size,
                 dimension=1,
                 offset=0,
                 stride=0,
                 normalization=False):
        self.vbo = vbo
        assert type_ in self._GL_TYPES
        self.type_ = type_
        self.size = size
        assert 1 <= dimension <= 4
        self.dimension = dimension
        self.offset = offset
        self.stride = stride
        self.normalization = bool(normalization)

    @property
    def itemsize(self):
        """Size in bytes of a vertex buffer element (int)"""
        return self.dimension * sizeofGLType(self.type_)

    itemSize = itemsize  # Backward compatibility

    def setVertexAttrib(self, attribute):
        """Call glVertexAttribPointer with objects information"""
        normalization = gl.GL_TRUE if self.normalization else gl.GL_FALSE
        with self.vbo:
            gl.glVertexAttribPointer(attribute,
                                     self.dimension,
                                     self.type_,
                                     normalization,
                                     self.stride,
                                     c_void_p(self.offset))

    def copy(self):
        return VertexBufferAttrib(self.vbo,
                                  self.type_,
                                  self.size,
                                  self.dimension,
                                  self.offset,
                                  self.stride,
                                  self.normalization)


def vertexBuffer(arrays, prefix=None, suffix=None, usage=None):
    """Create a single vertex buffer from multiple 1D or 2D numpy arrays.

    It is possible to reserve memory before and after each array in the VBO

    :param arrays: Arrays of data to store
    :type arrays: Iterable of numpy.ndarray
    :param prefix: If given, number of elements to reserve before each array
    :type prefix: Iterable of int or None
    :param suffix: If given, number of elements to reserve after each array
    :type suffix: Iterable of int or None
    :param int usage: vertex buffer expected usage or None for default
    :returns: List of VertexBufferAttrib objects sharing the same vertex buffer
    """
    info = []
    vbosize = 0

    if prefix is None:
        prefix = (0,) * len(arrays)
    if suffix is None:
        suffix = (0,) * len(arrays)

    for data, pre, post in zip(arrays, prefix, suffix):
        data = numpy.array(data, copy=False, order='C')
        shape = data.shape
        assert len(shape) <= 2
        type_ = numpyToGLType(data.dtype)
        size = shape[0] + pre + post
        dimension = 1 if len(shape) == 1 else shape[1]
        sizeinbytes = size * dimension * sizeofGLType(type_)
        sizeinbytes = 4 * ((sizeinbytes + 3) >> 2)  # 4 bytes alignment
        copyoffset = vbosize + pre * dimension * sizeofGLType(type_)
        info.append((data, type_, size, dimension,
                     vbosize, sizeinbytes, copyoffset))
        vbosize += sizeinbytes

    vbo = VertexBuffer(size=vbosize, usage=usage)

    result = []
    for data, type_, size, dimension, offset, sizeinbytes, copyoffset in info:
        copysize = data.shape[0] * dimension * sizeofGLType(type_)
        vbo.update(data, offset=copyoffset, size=copysize)
        result.append(
            VertexBufferAttrib(vbo, type_, size, dimension, offset, 0))
    return result
