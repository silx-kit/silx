# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""This module provides a class for Viewports rendering on the screen.

The :class:`Window` renders a list of Viewports in the current framebuffer.
The rendering can be performed in an off-screen framebuffer that is only
updated when the scene has changed and not each time Qt is requiring a repaint.

The :class:`Context` and :class:`ContextGL2` represent the operating system
OpenGL context and handle OpenGL resources.
"""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "10/01/2017"


import weakref
import numpy

from ..._glutils import gl
from ... import _glutils

from . import event


class Context(object):
    """Correspond to an operating system OpenGL context.

    User should NEVER use an instance of this class beyond the method
    it is passed to as an argument (i.e., do not keep a reference to it).

    :param glContextHandle: System specific OpenGL context handle.
    """

    def __init__(self, glContextHandle):
        self._context = glContextHandle
        self._isCurrent = False
        self._devicePixelRatio = 1.0

    @property
    def isCurrent(self):
        """Whether this OpenGL context is the current one or not."""
        return self._isCurrent

    def setCurrent(self, isCurrent=True):
        """Set the state of the OpenGL context to reflect OpenGL state.

        This should not be called from the scene graph, only in the
        wrapper that handle the OpenGL context to reflect its state.

        :param bool isCurrent: The state of the system OpenGL context.
        """
        self._isCurrent = bool(isCurrent)

    @property
    def devicePixelRatio(self):
        """Ratio between device and device independent pixels (float)

        This is useful for font rendering.
        """
        return self._devicePixelRatio

    @devicePixelRatio.setter
    def devicePixelRatio(self, ratio):
        assert ratio > 0
        self._devicePixelRatio = float(ratio)

    def __enter__(self):
        self.setCurrent(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.setCurrent(False)

    @property
    def glContext(self):
        """The handle to the OpenGL context provided by the system."""
        return self._context

    def cleanGLGarbage(self):
        """This is releasing OpenGL resource that are no longer used."""
        pass


class ContextGL2(Context):
    """Handle a system GL2 context.

    User should NEVER use an instance of this class beyond the method
    it is passed to as an argument (i.e., do not keep a reference to it).

    :param glContextHandle: System specific OpenGL context handle.
    """
    def __init__(self, glContextHandle):
        super(ContextGL2, self).__init__(glContextHandle)

        self._programs = {}  # GL programs already compiled
        self._vbos = {}  # GL Vbos already set
        self._vboGarbage = []  # Vbos waiting to be discarded

    # programs

    def prog(self, vertexShaderSrc, fragmentShaderSrc, attrib0='position'):
        """Cache program within context.

        WARNING: No clean-up.

        :param str vertexShaderSrc: Vertex shader source code
        :param str fragmentShaderSrc: Fragment shader source code
        :param str attrib0:
            Attribute's name to bind to position 0 (default: 'position').
            On some platform, this attribute MUST be active and with an
            array attached to it in order for the rendering to occur....
        """
        assert self.isCurrent
        key = vertexShaderSrc, fragmentShaderSrc, attrib0
        program = self._programs.get(key, None)
        if program is None:
            program = _glutils.Program(
                vertexShaderSrc, fragmentShaderSrc, attrib0=attrib0)
            self._programs[key] = program
        return program

    # VBOs

    def makeVbo(self, data=None, sizeInBytes=None,
                usage=None, target=None):
        """Create a VBO in this context with the data.

        Current limitations:

        - One array per VBO
        - Do not support sharing VertexBuffer across VboAttrib

        Automatically discards the VBO when the returned
        :class:`VertexBuffer` istance is deleted.

        :param numpy.ndarray data: 2D array of data to store in VBO or None.
        :param int sizeInBytes: Size of the VBO or None.
                                It should be <= data.nbytes if both are given.
        :param usage: OpenGL usage define in VertexBuffer._USAGES.
        :param target: OpenGL target in VertexBuffer._TARGETS.
        :return: The VertexBuffer created in this context.
        """
        assert self.isCurrent
        vbo = _glutils.VertexBuffer(data, sizeInBytes, usage, target)
        vboref = weakref.ref(vbo, self._deadVbo)
        # weakref is hashable as far as target is
        self._vbos[vboref] = vbo.name
        return vbo

    def makeVboAttrib(self, data, usage=None, target=None):
        """Create a VBO from data and returns the associated VBOAttrib.

        Automatically discards the VBO when the returned
        :class:`VBOAttrib` istance is deleted.

        :param numpy.ndarray data: 2D array of data to store in VBO or None.
        :param usage: OpenGL usage define in VertexBuffer._USAGES.
        :param target: OpenGL target in VertexBuffer._TARGETS.
        :returns: A VBOAttrib instance created in this context.
        """
        assert self.isCurrent
        vbo = self.makeVbo(data, usage=usage, target=target)

        assert len(data.shape) <= 2
        dimension = 1 if len(data.shape) == 1 else data.shape[1]

        return _glutils.VertexBufferAttrib(
            vbo,
            type_=_glutils.numpyToGLType(data.dtype),
            size=data.shape[0],
            dimension=dimension,
            offset=0,
            stride=0)

    def _deadVbo(self, vboRef):
        """Callback handling dead VBOAttribs."""
        vboid = self._vbos.pop(vboRef)
        if self.isCurrent:
            # Direct delete if context is active
            gl.glDeleteBuffers(vboid)
        else:
            # Deferred VBO delete if context is not active
            self._vboGarbage.append(vboid)

    def cleanGLGarbage(self):
        """Delete OpenGL resources that are pending for destruction.

        This requires the associated OpenGL context to be active.
        This is meant to be called before rendering.
        """
        assert self.isCurrent
        if self._vboGarbage:
            vboids = self._vboGarbage
            gl.glDeleteBuffers(vboids)
            self._vboGarbage = []


class Window(event.Notifier):
    """OpenGL Framebuffer where to render viewports

    :param str mode: Rendering mode to use:

        - 'direct' to render everything for each render call
        - 'framebuffer' to cache viewport rendering in a texture and
          update the texture only when needed.
    """

    _position = numpy.array(((-1., -1., 0., 0.),
                             (1., -1., 1., 0.),
                             (-1., 1., 0., 1.),
                             (1., 1., 1., 1.)),
                            dtype=numpy.float32)

    _shaders = ("""
        attribute vec4 position;
        varying vec2 textureCoord;

        void main(void) {
            gl_Position = vec4(position.x, position.y, 0., 1.);
            textureCoord = position.zw;
        }
        """,
                """
        uniform sampler2D texture;
        varying vec2 textureCoord;

        void main(void) {
            gl_FragColor = texture2D(texture, textureCoord);
            gl_FragColor.a = 1.0;
        }
        """)

    def __init__(self, mode='framebuffer'):
        super(Window, self).__init__()
        self._dirty = True
        self._size = 0, 0
        self._contexts = {}  # To map system GL context id to Context objects
        self._viewports = event.NotifierList()
        self._viewports.addListener(self._updated)
        self._framebufferid = 0
        self._framebuffers = {}  # Cache of framebuffers

        assert mode in ('direct', 'framebuffer')
        self._isframebuffer = mode == 'framebuffer'

    @property
    def dirty(self):
        """True if this object or any attached viewports is dirty."""
        for viewport in self._viewports:
            if viewport.dirty:
                return True
        return self._dirty

    @property
    def size(self):
        """Size (width, height) of the window in pixels"""
        return self._size

    @size.setter
    def size(self, size):
        w, h = size
        size = int(w), int(h)
        if size != self._size:
            self._size = size
            self._dirty = True
            self.notify()

    @property
    def shape(self):
        """Shape (height, width) of the window in pixels.

        This is a convenient wrapper to the reverse of size.
        """
        return self._size[1], self._size[0]

    @shape.setter
    def shape(self, shape):
        self.size = shape[1], shape[0]

    @property
    def viewports(self):
        """List of viewports to render in the corresponding framebuffer"""
        return self._viewports

    @viewports.setter
    def viewports(self, iterable):
        self._viewports.removeListener(self._updated)
        self._viewports = event.NotifierList(iterable)
        self._viewports.addListener(self._updated)
        self._updated(self)

    def _updated(self, source, *args, **kwargs):
        self._dirty = True
        self.notify(*args, **kwargs)

    framebufferid = property(lambda self: self._framebufferid,
                             doc="Framebuffer ID used to perform rendering")

    def grab(self, glcontext):
        """Returns the raster of the scene as an RGB numpy array

        :returns: OpenGL scene RGB bitmap
                  as an array of dimension (height, width, 3)
        :rtype: numpy.ndarray of uint8
        """
        height, width = self.shape
        image = numpy.empty((height, width, 3), dtype=numpy.uint8)

        previousFramebuffer = gl.glGetInteger(gl.GL_FRAMEBUFFER_BINDING)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebufferid)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadPixels(
            0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, image)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, previousFramebuffer)

        # glReadPixels gives bottom to top,
        # while images are stored as top to bottom
        image = numpy.flipud(image)

        return numpy.array(image, copy=False, order='C')

    def render(self, glcontext, devicePixelRatio):
        """Perform the rendering of attached viewports

        :param glcontext: System identifier of the OpenGL context
        :param float devicePixelRatio:
            Ratio between device and device-independent pixels
        """
        if glcontext not in self._contexts:
            self._contexts[glcontext] = ContextGL2(glcontext)  # New context

        with self._contexts[glcontext] as context:
            context.devicePixelRatio = devicePixelRatio
            if self._isframebuffer:
                self._renderWithOffscreenFramebuffer(context)
            else:
                self._renderDirect(context)

        self._dirty = False

    def _renderDirect(self, context):
        """Perform the direct rendering of attached viewports

        :param Context context: Object wrapping OpenGL context
        """
        for viewport in self._viewports:
            viewport.framebuffer = self.framebufferid
            viewport.render(context)
            viewport.resetDirty()

    def _renderWithOffscreenFramebuffer(self, context):
        """Renders viewports in a texture and render this texture on screen.

        The texture is updated only if viewport or size has changed.

        :param ContextGL2 context: Object wrappign OpenGL context
        """
        if self.dirty or context not in self._framebuffers:
            # Need to redraw framebuffer content

            if (context not in self._framebuffers or
                    self._framebuffers[context].shape != self.shape):
                # Need to rebuild framebuffer

                if context in self._framebuffers:
                    self._framebuffers[context].discard()

                fbo = _glutils.FramebufferTexture(gl.GL_RGBA,
                                                  shape=self.shape,
                                                  minFilter=gl.GL_NEAREST,
                                                  magFilter=gl.GL_NEAREST,
                                                  wrap=gl.GL_CLAMP_TO_EDGE)
                self._framebuffers[context] = fbo
                self._framebufferid = fbo.name

            # Render in framebuffer
            with self._framebuffers[context]:
                self._renderDirect(context)

        # Render framebuffer texture to screen
        fbo = self._framebuffers[context]
        height, width = fbo.shape

        program = context.prog(*self._shaders)
        program.use()

        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        # gl.glScissor(0, 0, width, height)
        gl.glClearColor(0., 0., 0., 0.)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUniform1i(program.uniforms['texture'], fbo.texture.texUnit)
        gl.glEnableVertexAttribArray(program.attributes['position'])
        gl.glVertexAttribPointer(program.attributes['position'],
                                 4,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 0,
                                 self._position)
        fbo.texture.bind()
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self._position))
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
