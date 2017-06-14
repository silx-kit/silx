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
"""Association of a texture and a framebuffer object for off-screen rendering.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"


import logging

from . import gl
from .Texture import Texture


_logger = logging.getLogger(__name__)


class FramebufferTexture(object):
    """Framebuffer with a texture.

    Aimed at off-screen rendering to texture.

    :param internalFormat: OpenGL texture internal format
    :param shape: Shape (height, width) of the framebuffer and texture
    :type shape: 2-tuple of int
    :param stencilFormat: Stencil renderbuffer format
    :param depthFormat: Depth renderbuffer format
    :param kwargs: Extra arguments for :class:`Texture` constructor
    """

    _PACKED_FORMAT = gl.GL_DEPTH24_STENCIL8, gl.GL_DEPTH_STENCIL

    def __init__(self,
                 internalFormat,
                 shape,
                 stencilFormat=gl.GL_DEPTH24_STENCIL8,
                 depthFormat=gl.GL_DEPTH24_STENCIL8,
                 **kwargs):

        self._texture = Texture(internalFormat, shape=shape, **kwargs)

        self._previousFramebuffer = 0  # Used by with statement

        self._name = gl.glGenFramebuffers(1)

        with self:  # Bind FBO
            # Attachments
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,
                                      gl.GL_COLOR_ATTACHMENT0,
                                      gl.GL_TEXTURE_2D,
                                      self._texture.name,
                                      0)

            height, width = self._texture.shape

            if stencilFormat is not None:
                self._stencilId = gl.glGenRenderbuffers(1)
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._stencilId)
                gl.glRenderbufferStorage(gl.GL_RENDERBUFFER,
                                         stencilFormat,
                                         width, height)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER,
                                             gl.GL_STENCIL_ATTACHMENT,
                                             gl.GL_RENDERBUFFER,
                                             self._stencilId)
            else:
                self._stencilId = None

            if depthFormat is not None:
                if self._stencilId and depthFormat in self._PACKED_FORMAT:
                    self._depthId = self._stencilId
                else:
                    self._depthId = gl.glGenRenderbuffers(1)
                    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depthId)
                    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER,
                                             depthFormat,
                                             width, height)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER,
                                             gl.GL_DEPTH_ATTACHMENT,
                                             gl.GL_RENDERBUFFER,
                                             self._depthId)
            else:
                self._depthId = None

            assert (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) ==
                gl.GL_FRAMEBUFFER_COMPLETE)

    @property
    def shape(self):
        """Shape of the framebuffer (height, width)"""
        return self._texture.shape

    @property
    def texture(self):
        """The texture this framebuffer is rendering to.

        The life-cycle of the texture is managed by this object"""
        return self._texture

    @property
    def name(self):
        """OpenGL name of the framebuffer"""
        if self._name is not None:
            return self._name
        else:
            raise RuntimeError("No OpenGL framebuffer resource, \
                               discard has already been called")

    def bind(self):
        """Bind this framebuffer for rendering"""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.name)

    # with statement

    def __enter__(self):
        self._previousFramebuffer = gl.glGetInteger(gl.GL_FRAMEBUFFER_BINDING)
        self.bind()

    def __exit__(self, exctype, excvalue, traceback):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._previousFramebuffer)
        self._previousFramebuffer = None

    def discard(self):
        """Delete associated OpenGL resources including texture"""
        if self._name is not None:
            gl.glDeleteFramebuffers(self._name)
            self._name = None

            if self._stencilId is not None:
                gl.glDeleteRenderbuffers(self._stencilId)
                if self._stencilId == self._depthId:
                    self._depthId = None
                self._stencilId = None
            if self._depthId is not None:
                gl.glDeleteRenderbuffers(self._depthId)
                self._depthId = None

            self._texture.discard()  # Also discard the texture
        else:
            _logger.warning("Discard has already been called")
