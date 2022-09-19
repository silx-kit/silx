# /*##########################################################################
#
# Copyright (c) 2015-2019 European Synchrotron Radiation Facility
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
"""This module provides a class to control a viewport on the rendering window.

The :class:`Viewport` describes a Viewport rendering a scene.
The attribute :attr:`scene` is the root group of the scene tree.
:class:`RenderContext` handles the current state during rendering.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import string
import numpy

from silx.gui.colors import rgba

from ..._glutils import gl

from . import camera
from . import event
from . import transform
from .function import DirectionalLight, ClippingPlane, Fog


class RenderContext(object):
    """Handle a current rendering context.

    An instance of this class is passed to rendering method through
    the scene during render.

    User should NEVER use an instance of this class beyond the method
    it is passed to as an argument (i.e., do not keep a reference to it).

    :param Viewport viewport: The viewport doing the rendering.
    :param Context glContext: The operating system OpenGL context in use.
    """

    _FRAGMENT_SHADER_SRC = string.Template("""
        void scene_post(vec4 cameraPosition) {
            gl_FragColor = $fogCall(gl_FragColor, cameraPosition);
        }
        """)

    def __init__(self, viewport, glContext):
        self._viewport = viewport
        self._glContext = glContext
        self._transformStack = [viewport.camera.extrinsic]
        self._clipPlane = ClippingPlane(normal=(0., 0., 0.))

        # cache
        self.__cache = {}

    def cache(self, key, factory, *args, **kwargs):
        """Lazy-loading cache to store values in the context for rendering

        :param key: The key to retrieve
        :param factory: A callback taking args and kwargs as arguments
            and returning the value to store.
        :return: The stored or newly allocated value
        """
        if key not in self.__cache:
            self.__cache[key] = factory(*args, **kwargs)
        return self.__cache[key]

    @property
    def viewport(self):
        """Viewport doing the current rendering"""
        return self._viewport

    @property
    def glCtx(self):
        """The OpenGL context in use"""
        return self._glContext

    @property
    def objectToCamera(self):
        """The current transform from object to camera coords.

        Do not modify.
        """
        return self._transformStack[-1]

    @property
    def projection(self):
        """Projection transform.

        Do not modify.
        """
        return self.viewport.camera.intrinsic

    @property
    def objectToNDC(self):
        """The transform from object to NDC (this includes projection).

        Do not modify.
        """
        return transform.StaticTransformList(
            (self.projection, self.objectToCamera))

    def pushTransform(self, transform_, multiply=True):
        """Push a :class:`Transform` on the transform stack.

        :param Transform transform_: The transform to add to the stack.
        :param bool multiply:
            True (the default) to multiply with the top of the stack,
            False to push the transform as is without multiplication.
        """
        if multiply:
            assert len(self._transformStack) >= 1
            transform_ = transform.StaticTransformList(
                (self._transformStack[-1], transform_))

        self._transformStack.append(transform_)

    def popTransform(self):
        """Pop the transform on top of the stack.

        :return: The Transform that is popped from the stack.
        """
        assert len(self._transformStack) > 1
        return self._transformStack.pop()

    @property
    def clipper(self):
        """The current clipping plane (ClippingPlane)"""
        return self._clipPlane

    def setClipPlane(self, point=(0., 0., 0.), normal=(0., 0., 0.)):
        """Set the clipping plane to use

        For now only handles a single clipping plane.

        :param point: A point of the plane
        :type point: 3-tuple of float
        :param normal: Normal vector of the plane or (0, 0, 0) for no clipping
        :type normal: 3-tuple of float
        """
        self._clipPlane = ClippingPlane(point, normal)

    def setupProgram(self, program):
        """Sets-up uniforms of a program using the context shader functions.

        :param GLProgram program: The program to set-up.
                                  It MUST be in use and using the context function.
        """
        self.clipper.setupProgram(self, program)
        self.viewport.fog.setupProgram(self, program)

    @property
    def fragDecl(self):
        """Fragment shader declaration for scene shader functions"""
        return '\n'.join((
            self.clipper.fragDecl,
            self.viewport.fog.fragDecl,
            self._FRAGMENT_SHADER_SRC.substitute(
                fogCall=self.viewport.fog.fragCall)))

    @property
    def fragCallPre(self):
        """Fragment shader call for scene shader functions (to do first)

        It takes the camera position (vec4) as argument.
        """
        return self.clipper.fragCall

    @property
    def fragCallPost(self):
        """Fragment shader call for scene shader functions (to do last)

        It takes the camera position (vec4) as argument.
        """
        return "scene_post"


class Viewport(event.Notifier):
    """Rendering a single scene through a camera in part of a framebuffer.

    :param int framebuffer: The framebuffer ID this viewport is rendering into
    """

    def __init__(self, framebuffer=0):
        from . import Group  # Here to avoid cyclic import
        super(Viewport, self).__init__()
        self._dirty = True
        self._origin = 0, 0
        self._size = 1, 1
        self._framebuffer = int(framebuffer)
        self.scene = Group()  # The stuff to render, add overlaid scenes?
        self.scene._setParent(self)
        self.scene.addListener(self._changed)
        self._background = 0., 0., 0., 1.
        self._camera = camera.Camera(fovy=30., near=1., far=100.,
                                     position=(0., 0., 12.))
        self._camera.addListener(self._changed)
        self._transforms = transform.TransformList([self._camera])

        self._light = DirectionalLight(direction=(0., 0., -1.),
                                       ambient=(0.3, 0.3, 0.3),
                                       diffuse=(0.7, 0.7, 0.7))
        self._light.addListener(self._changed)
        self._fog = Fog()
        self._fog.isOn = False
        self._fog.addListener(self._changed)

    @property
    def transforms(self):
        """Proxy of camera transforms.

        Do not modify the list.
        """
        return self._transforms

    def _changed(self, *args, **kwargs):
        """Callback handling scene updates"""
        self._dirty = True
        self.notify()

    @property
    def dirty(self):
        """True if scene is dirty and needs redisplay."""
        return self._dirty

    def resetDirty(self):
        """Mark the scene as not being dirty.

        To call after rendering.
        """
        self._dirty = False

    @property
    def background(self):
        """Viewport's background color (4-tuple of float in [0, 1] or None)

        The background color is used to clear to viewport.
        If None, the viewport is not cleared
        """
        return self._background

    @background.setter
    def background(self, color):
        if color is not None:
            color = rgba(color)
        if self._background != color:
            self._background = color
            self._changed()

    @property
    def camera(self):
        """The camera used to render the scene."""
        return self._camera

    @property
    def light(self):
        """The light used to render the scene."""
        return self._light

    @property
    def fog(self):
        """The fog function used to render the scene"""
        return self._fog

    @property
    def origin(self):
        """Origin (ox, oy) of the viewport in pixels"""
        return self._origin

    @origin.setter
    def origin(self, origin):
        ox, oy = origin
        origin = int(ox), int(oy)
        if origin != self._origin:
            self._origin = origin
            self._changed()

    @property
    def size(self):
        """Size (width, height) of the viewport in pixels"""
        return self._size

    @size.setter
    def size(self, size):
        w, h = size
        size = int(w), int(h)
        if size != self._size:
            self._size = size

            self.camera.intrinsic.size = size
            self._changed()

    @property
    def shape(self):
        """Shape (height, width) of the viewport in pixels.

        This is a convenient wrapper to the inverse of size.
        """
        return self._size[1], self._size[0]

    @shape.setter
    def shape(self, shape):
        self.size = shape[1], shape[0]

    @property
    def framebuffer(self):
        """The framebuffer ID this viewport is rendering into (int)"""
        return self._framebuffer

    @framebuffer.setter
    def framebuffer(self, framebuffer):
        self._framebuffer = int(framebuffer)

    def render(self, glContext):
        """Perform the rendering of the viewport

        :param Context glContext: The context used for rendering"""
        # Get a chance to run deferred delete
        glContext.cleanGLGarbage()

        # OpenGL set-up: really need to be done once
        ox, oy = self.origin
        w, h = self.size
        gl.glViewport(ox, oy, w, h)

        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(ox, oy, w, h)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthRange(0., 1.)

        # gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        # gl.glPolygonOffset(1., 1.)

        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_LINE_SMOOTH)

        if self.background is None:
            gl.glClear(gl.GL_STENCIL_BUFFER_BIT |
                       gl.GL_DEPTH_BUFFER_BIT)
        else:
            gl.glClearColor(*self.background)

            # Prepare OpenGL
            gl.glClear(gl.GL_COLOR_BUFFER_BIT |
                       gl.GL_STENCIL_BUFFER_BIT |
                       gl.GL_DEPTH_BUFFER_BIT)

        ctx = RenderContext(self, glContext)
        self.scene.render(ctx)
        self.scene.postRender(ctx)

    def adjustCameraDepthExtent(self):
        """Update camera depth extent to fit the scene bounds.

        Only near and far planes are updated.
        The scene might still not be fully visible
        (e.g., if spanning behind the viewpoint with perspective projection).
        """
        bounds = self.scene.bounds(transformed=True)
        if bounds is None:
             bounds = numpy.array(((0., 0., 0.), (1., 1., 1.)),
                                  dtype=numpy.float32)
        bounds = self.camera.extrinsic.transformBounds(bounds)

        if isinstance(self.camera.intrinsic, transform.Perspective):
            # This needs to be reworked
            zbounds = - bounds[:, 2]
            zextent = max(numpy.fabs(zbounds[0] - zbounds[1]), 0.0001)
            near = max(zextent / 1000., 0.95 * zbounds[1])
            far = max(near + 0.1, 1.05 * zbounds[0])

            self.camera.intrinsic.setDepthExtent(near, far)
        elif isinstance(self.camera.intrinsic, transform.Orthographic):
            # Makes sure z bounds are included
            border = max(abs(bounds[:, 2]))
            self.camera.intrinsic.setDepthExtent(-border, border)
        else:
            raise RuntimeError('Unsupported camera', self.camera.intrinsic)

    def resetCamera(self):
        """Change camera to have the whole scene in the viewing frustum.

        It updates the camera position and depth extent.
        Camera sight direction and up are not affected.
        """
        bounds = self.scene.bounds(transformed=True)
        if bounds is None:
            bounds = numpy.array(((0., 0., 0.), (1., 1., 1.)),
                                 dtype=numpy.float32)
        self.camera.resetCamera(bounds)

    def orbitCamera(self, direction, angle=1.):
        """Rotate the camera around center of the scene.

        :param str direction: Direction of movement relative to image plane.
                              In: 'up', 'down', 'left', 'right'.
        :param float angle: he angle in degrees of the rotation.
        """
        bounds = self.scene.bounds(transformed=True)
        if bounds is None:
             bounds = numpy.array(((0., 0., 0.), (1., 1., 1.)),
                                  dtype=numpy.float32)
        center = 0.5 * (bounds[0] + bounds[1])
        self.camera.orbit(direction, center, angle)

    def moveCamera(self, direction, step=0.1):
        """Move the camera relative to the image plane.

        :param str direction: Direction relative to image plane.
                              One of: 'up', 'down', 'left', 'right',
                              'forward', 'backward'.
        :param float step: The ratio of data to step for each pan.
        """
        bounds = self.scene.bounds(transformed=True)
        if bounds is None:
             bounds = numpy.array(((0., 0., 0.), (1., 1., 1.)),
                                  dtype=numpy.float32)
        bounds = self.camera.extrinsic.transformBounds(bounds)
        center = 0.5 * (bounds[0] + bounds[1])
        ndcCenter = self.camera.intrinsic.transformPoint(
            center, perspectiveDivide=True)

        step *= 2.  # NDC has size 2

        if direction == 'up':
            ndcCenter[1] -= step
        elif direction == 'down':
            ndcCenter[1] += step

        elif direction == 'right':
            ndcCenter[0] -= step
        elif direction == 'left':
            ndcCenter[0] += step

        elif direction == 'forward':
            ndcCenter[2] += step
        elif direction == 'backward':
            ndcCenter[2] -= step

        else:
            raise ValueError('Unsupported direction: %s' % direction)

        newCenter = self.camera.intrinsic.transformPoint(
            ndcCenter, direct=False, perspectiveDivide=True)

        self.camera.move(direction, numpy.linalg.norm(newCenter - center))

    def windowToNdc(self, winX, winY, checkInside=True):
        """Convert position from window to normalized device coordinates.

        If window coordinates are int, they are moved half a pixel
        to be positioned at the center of pixel.

        :param winX: X window coord, origin left.
        :param winY: Y window coord, origin top.
        :param bool checkInside: If True, returns None if position is
                                 outside viewport.
        :return: (x, y) Normalize device coordinates in [-1, 1] or None.
                 Origin center, x to the right, y goes upward.
        """
        ox, oy = self._origin
        width, height = self.size

        # If int, move it to the center of pixel
        if isinstance(winX, int):
            winX += 0.5
        if isinstance(winY, int):
            winY += 0.5

        x, y = winX - ox, winY - oy

        if checkInside and (x < 0. or x > width or y < 0. or y > height):
            return None  # Out of viewport

        ndcx = 2. * x / float(width) - 1.
        ndcy = 1. - 2. * y / float(height)
        return ndcx, ndcy

    def ndcToWindow(self, ndcX, ndcY, checkInside=True):
        """Convert position from normalized device coordinates (NDC) to window.

        :param float ndcX: X NDC coord.
        :param float ndcY: Y NDC coord.
        :param bool checkInside: If True, returns None if position is
                                 outside viewport.
        :return: (x, y) window coordinates or None.
                 Origin top-left, x to the right, y goes downward.
        """
        if (checkInside and
                (ndcX < -1. or ndcX > 1. or ndcY < -1. or ndcY > 1.)):
            return None  # Outside viewport

        ox, oy = self._origin
        width, height = self.size

        winx = ox + width * 0.5 * (ndcX + 1.)
        winy = oy + height * 0.5 * (1. - ndcY)
        return winx, winy

    def _pickNdcZGL(self, x, y, offset=0):
        """Retrieve depth from depth buffer and return corresponding NDC Z.

        :param int x: In pixels in window coordinates, origin left.
        :param int y: In pixels in window coordinates, origin top.
        :param int offset: Number of pixels to look at around the given pixel

        :return: Normalize device Z coordinate of depth in [-1, 1]
                 or None if outside viewport.
        :rtype: float or None
        """
        ox, oy = self._origin
        width, height = self.size

        x = int(x)
        y = height - int(y)  # Invert y coord

        if x < ox or x > ox + width or y < oy or y > oy + height:
            # Outside viewport
            return None

        # Get depth from depth buffer in [0., 1.]
        # Bind used framebuffer to get depth
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer)

        if offset == 0:  # Fast path
            # glReadPixels is not GL|ES friendly
            depth = gl.glReadPixels(
                x, y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)[0]
        else:
            offset = abs(int(offset))
            size = 2*offset + 1
            depthPatch = gl.glReadPixels(
                x - offset, y - offset,
                size, size,
                gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
            depthPatch = depthPatch.ravel()  # Work in 1D

            # TODO cache sortedIndices to avoid computing it each time
            # Compute distance of each pixels to the center of the patch
            offsetToCenter = numpy.arange(- offset, offset + 1, dtype=numpy.float32) ** 2
            sqDistToCenter = numpy.add.outer(offsetToCenter, offsetToCenter)

            # Use distance to center to sort values from the patch
            sortedIndices = numpy.argsort(sqDistToCenter.ravel())
            sortedValues = depthPatch[sortedIndices]

            # Take first depth that is not 1 in the sorted values
            hits = sortedValues[sortedValues != 1.]
            depth = 1. if len(hits) == 0 else hits[0]

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Z in NDC in [-1., 1.]
        return float(depth) * 2. - 1.

    def _getXZYGL(self, x, y):
        ndc = self.windowToNdc(x, y)
        if ndc is None:
            return None  # Outside viewport
        ndcz = self._pickNdcZGL(x, y)
        ndcpos = numpy.array((ndc[0], ndc[1], ndcz, 1.), dtype=numpy.float32)

        camerapos = self.camera.intrinsic.transformPoint(
            ndcpos, direct=False, perspectiveDivide=True)

        scenepos = self.camera.extrinsic.transformPoint(camerapos,
                                                        direct=False)
        return scenepos[:3]

    def pick(self, x, y):
        pass
        # ndcX, ndcY = self.windowToNdc(x, y)
        # ndcNearPt = ndcX, ndcY, -1.
        # ndcFarPT = ndcX, ndcY, 1.
