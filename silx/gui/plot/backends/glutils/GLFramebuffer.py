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
# ############################################################################*/
"""
This module provides a texture associated to a framebuffer object for
off-screen rendering
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/04/2017"


from ...._glutils.gl import *  # noqa

from .GLTexture import Texture2D


# framebuffer #################################################################

class FBOTexture(Texture2D):
    """Texture with FBO aimed at off-screen rendering to texture"""
    _PACKED_FORMAT = GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL

    def __init__(self, internalFormat, width, height,
                 stencilFormat=GL_DEPTH24_STENCIL8,
                 depthFormat=GL_DEPTH24_STENCIL8, **kwargs):
        super(FBOTexture, self).__init__(internalFormat, width, height,
                                         **kwargs)

        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # Attachments
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self.tid, 0)

        if stencilFormat is not None:
            self._stencilId = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self._stencilId)
            glRenderbufferStorage(GL_RENDERBUFFER, stencilFormat,
                                  self.width, self.height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
                                      GL_RENDERBUFFER, self._stencilId)

        if depthFormat is not None:
            if self._stencilId and depthFormat in self._PACKED_FORMAT:
                self._depthId = self._stencilId
            else:
                self._depthId = glGenRenderbuffers(1)
                glBindRenderbuffer(GL_RENDERBUFFER, self._depthId)
                glRenderbufferStorage(GL_RENDERBUFFER, depthFormat,
                                      self.width, self.height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER, self._depthId)

        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == \
            GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    @property
    def fbo(self):
        try:
            return self._fbo
        except AttributeError:
            raise RuntimeError("No OpenGL framebuffer resource, \
                               discard has already been called")

    def bindFBO(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

    # with statement
    def __enter__(self):
        self.bindFBO()

    def __exit__(self, excType, excValue, traceback):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def discard(self):
        if hasattr(self, '_fbo'):
            if bool(glDeleteFramebuffers):  # Test for __del__
                glDeleteFramebuffers(self._fbo)
            del self._fbo
        if hasattr(self, '_stencilId'):
            if bool(glDeleteRenderbuffers):  # Test for __del__
                glDeleteRenderbuffers(self._stencilId)
            if self._stencilId == getattr(self, '_depthId', -1):
                del self._depthId
            del self._stencilId
        if hasattr(self, '_depthId'):
            if bool(glDeleteRenderbuffers):  # Test for __del__
                glDeleteRenderbuffers(self._depthId)
            del self._depthId
        super(FBOTexture, self).discard()

    def __del__(self):
        self.discard()
