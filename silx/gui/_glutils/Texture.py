# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2018 European Synchrotron Radiation Facility
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
"""This module provides a class wrapping OpenGL 2D and 3D texture."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "04/10/2016"


import collections
from ctypes import c_void_p
import logging

import numpy

from . import gl, utils


_logger = logging.getLogger(__name__)


class Texture(object):
    """Base class to wrap OpenGL 2D and 3D texture

    :param internalFormat: OpenGL texture internal format
    :param data: The data to copy to the texture or None for an empty texture
    :type data: numpy.ndarray or None
    :param format_: Input data format if different from internalFormat
    :param shape: If data is None, shape of the texture
                  (height, width) or (depth, height, width)
    :type shape: List[int]
    :param int texUnit: The texture unit to use
    :param minFilter: OpenGL texture minimization filter (default: GL_NEAREST)
    :param magFilter: OpenGL texture magnification filter (default: GL_LINEAR)
    :param wrap: Texture wrap mode for dimensions: (t, s) or (r, t, s)
                 If a single value is provided, it used for all dimensions.
    :type wrap: OpenGL wrap mode or 2 or 3-tuple of wrap mode
    """

    def __init__(self, internalFormat, data=None, format_=None,
                 shape=None, texUnit=0,
                 minFilter=None, magFilter=None, wrap=None):

        self._internalFormat = internalFormat
        if format_ is None:
            format_ = self.internalFormat

        if data is None:
            assert shape is not None
        else:
            assert shape is None
            data = numpy.array(data, copy=False, order='C')
            if format_ != gl.GL_RED:
                shape = data.shape[:-1]  # Last dimension is channels
            else:
                shape = data.shape

        assert len(shape) in (2, 3)
        self._shape = tuple(shape)
        self._ndim = len(shape)

        self.texUnit = texUnit

        self._name = gl.glGenTextures(1)
        self.bind(self.texUnit)

        self._minFilter = None
        self.minFilter = minFilter if minFilter is not None else gl.GL_NEAREST

        self._magFilter = None
        self.magFilter = magFilter if magFilter is not None else gl.GL_LINEAR

        if wrap is not None:
            if not isinstance(wrap, collections.Iterable):
                wrap = [wrap] * self.ndim

            assert len(wrap) == self.ndim

            gl.glTexParameter(self.target,
                              gl.GL_TEXTURE_WRAP_S,
                              wrap[-1])
            gl.glTexParameter(self.target,
                              gl.GL_TEXTURE_WRAP_T,
                              wrap[-2])
            if self.ndim == 3:
                gl.glTexParameter(self.target,
                                  gl.GL_TEXTURE_WRAP_R,
                                  wrap[0])

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        # This are the defaults, useless to set if not modified
        # gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)
        # gl.glPixelStorei(gl.GL_UNPACK_SKIP_PIXELS, 0)
        # gl.glPixelStorei(gl.GL_UNPACK_SKIP_ROWS, 0)
        # gl.glPixelStorei(gl.GL_UNPACK_IMAGE_HEIGHT, 0)
        # gl.glPixelStorei(gl.GL_UNPACK_SKIP_IMAGES, 0)

        if data is None:
            data = c_void_p(0)
            type_ = gl.GL_UNSIGNED_BYTE
        else:
            type_ = utils.numpyToGLType(data.dtype)

        if self.ndim == 2:
            _logger.debug(
                'Creating 2D texture shape: (%d, %d),'
                ' internal format: %s, format: %s, type: %s',
                self.shape[0], self.shape[1],
                str(self.internalFormat), str(format_), str(type_))

            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                self.internalFormat,
                self.shape[1],
                self.shape[0],
                0,
                format_,
                type_,
                data)
        else:
            _logger.debug(
                'Creating 3D texture shape: (%d, %d, %d),'
                ' internal format: %s, format: %s, type: %s',
                self.shape[0], self.shape[1], self.shape[2],
                str(self.internalFormat), str(format_), str(type_))

            gl.glTexImage3D(
                gl.GL_TEXTURE_3D,
                0,
                self.internalFormat,
                self.shape[2],
                self.shape[1],
                self.shape[0],
                0,
                format_,
                type_,
                data)

        gl.glBindTexture(self.target, 0)

    @property
    def target(self):
        """OpenGL target type of this texture"""
        return gl.GL_TEXTURE_2D if self.ndim == 2 else gl.GL_TEXTURE_3D

    @property
    def ndim(self):
        """The number of dimensions: 2 or 3"""
        return self._ndim

    @property
    def internalFormat(self):
        """Texture internal format"""
        return self._internalFormat

    @property
    def shape(self):
        """Shape of the texture: (height, width) or (depth, height, width)"""
        return self._shape

    @property
    def name(self):
        """OpenGL texture name"""
        if self._name is not None:
            return self._name
        else:
            raise RuntimeError(
                "No OpenGL texture resource, discard has already been called")

    @property
    def minFilter(self):
        """Minifying function parameter (GL_TEXTURE_MIN_FILTER)"""
        return self._minFilter

    @minFilter.setter
    def minFilter(self, minFilter):
        if minFilter != self.minFilter:
            self._minFilter = minFilter
            self.bind()
            gl.glTexParameter(self.target,
                              gl.GL_TEXTURE_MIN_FILTER,
                              self.minFilter)

    @property
    def magFilter(self):
        """Magnification function parameter (GL_TEXTURE_MAG_FILTER)"""
        return self._magFilter

    @magFilter.setter
    def magFilter(self, magFilter):
        if magFilter != self.magFilter:
            self._magFilter = magFilter
            self.bind()
            gl.glTexParameter(self.target,
                              gl.GL_TEXTURE_MAG_FILTER,
                              self.magFilter)

    def discard(self):
        """Delete associated OpenGL texture"""
        if self._name is not None:
            gl.glDeleteTextures(self._name)
            self._name = None
        else:
            _logger.warning("Discard as already been called")

    def bind(self, texUnit=None):
        """Bind the texture to a texture unit.

        :param int texUnit: The texture unit to use
        """
        if texUnit is None:
            texUnit = self.texUnit
        gl.glActiveTexture(gl.GL_TEXTURE0 + texUnit)
        gl.glBindTexture(self.target, self.name)

    # with statement

    def __enter__(self):
        self.bind()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.texUnit)
        gl.glBindTexture(self.target, 0)

    def update(self,
               format_,
               data,
               offset=(0, 0, 0),
               texUnit=None):
        """Update the content of the texture.

        Texture is not resized, so data must fit into texture with the
        given offset.

        :param format_: The OpenGL format of the data
        :param data: The data to use to update the texture
        :param offset: The offset in the texture where to copy the data
        :type offset: List[int]
        :param int texUnit:
            The texture unit to use (default: the one provided at init)
        """
        data = numpy.array(data, copy=False, order='C')

        assert data.ndim == self.ndim
        assert len(offset) >= self.ndim
        for i in range(self.ndim):
            assert offset[i] + data.shape[i] <= self.shape[i]

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        # This are the defaults, useless to set if not modified
        # gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)
        # gl.glPixelStorei(gl.GL_UNPACK_SKIP_PIXELS, 0)
        # gl.glPixelStorei(gl.GL_UNPACK_SKIP_ROWS, 0)
        # gl.glPixelStorei(gl.GL_UNPACK_IMAGE_HEIGHT, 0)
        # gl.glPixelStorei(gl.GL_UNPACK_SKIP_IMAGES, 0)

        self.bind(texUnit)

        type_ = utils.numpyToGLType(data.dtype)

        if self.ndim == 2:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D,
                               0,
                               offset[1],
                               offset[0],
                               data.shape[1],
                               data.shape[0],
                               format_,
                               type_,
                               data)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        else:
            gl.glTexSubImage3D(gl.GL_TEXTURE_3D,
                               0,
                               offset[2],
                               offset[1],
                               offset[0],
                               data.shape[2],
                               data.shape[1],
                               data.shape[0],
                               format_,
                               type_,
                               data)
            gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
