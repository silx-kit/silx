# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
"""This module provides a Qt widget embedding an OpenGL scene."""

from __future__ import absolute_import

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "26/01/2017"


import logging

from silx.gui import qt
from silx.gui.plot.Colors import rgba
from silx.gui.plot3d import Plot3DActions
from .._utils import convertArrayToQImage

from .._glutils import gl
from .scene import interaction, primitives, transform
from . import scene

import numpy


_logger = logging.getLogger(__name__)


class _OverviewViewport(scene.Viewport):
    """A scene displaying the orientation of the data in another scene.

    :param Camera camera: The camera to track.
    """

    def __init__(self, camera=None):
        super(_OverviewViewport, self).__init__()
        self.size = 100, 100

        self.scene.transforms = [transform.Scale(2.5, 2.5, 2.5)]

        axes = primitives.Axes()
        self.scene.children.append(axes)

        if camera is not None:
            camera.addListener(self._cameraChanged)

    def _cameraChanged(self, source):
        """Listen to camera in other scene for transformation updates.

        Sync the overview camera to point in the same direction
        but from a sphere centered on origin.
        """
        position = -12. * source.extrinsic.direction
        self.camera.extrinsic.position = position

        self.camera.extrinsic.setOrientation(
            source.extrinsic.direction, source.extrinsic.up)


class Plot3DWidget(qt.QGLWidget):
    """QGLWidget with a 3D viewport and an overview."""

    def __init__(self, parent=None):
        if not qt.QGLFormat.hasOpenGL():  # Check if any OpenGL is available
            raise RuntimeError(
                'OpenGL is not available on this platform: 3D disabled')

        self._devicePixelRatio = 1.0  # Store GL canvas/QWidget ratio
        self._isOpenGL21 = False
        self._firstRender = True

        format_ = qt.QGLFormat()
        format_.setRgba(True)
        format_.setDepth(False)
        format_.setStencil(False)
        format_.setVersion(2, 1)
        format_.setDoubleBuffer(True)

        super(Plot3DWidget, self).__init__(format_, parent)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

        self.setFocusPolicy(qt.Qt.StrongFocus)
        self._copyAction = Plot3DActions.CopyAction(parent=self, plot3d=self)
        self.addAction(self._copyAction)

        self._updating = False  # True if an update is requested

        # Main viewport
        self.viewport = scene.Viewport()
        self.viewport.background = 0.2, 0.2, 0.2, 1.

        sceneScale = transform.Scale(1., 1., 1.)
        self.viewport.scene.transforms = [sceneScale,
                                          transform.Translate(0., 0., 0.)]

        # Overview area
        self.overview = _OverviewViewport(self.viewport.camera)

        self.setBackgroundColor((0.2, 0.2, 0.2, 1.))

        # Window describing on screen area to render
        self.window = scene.Window(mode='framebuffer')
        self.window.viewports = [self.viewport, self.overview]

        self.eventHandler = interaction.CameraControl(
            self.viewport, orbitAroundCenter=False,
            mode='position', scaleTransform=sceneScale,
            selectCB=None)

        self.viewport.addListener(self._redraw)

    def setProjection(self, projection):
        """Change the projection in use.

        :param str projection: In 'perspective', 'orthographic'.
        """
        if projection == 'orthographic':
            projection = transform.Orthographic(size=self.viewport.size)
        elif projection == 'perspective':
            projection = transform.Perspective(fovy=30.,
                                               size=self.viewport.size)
        else:
            raise RuntimeError('Unsupported projection: %s' % projection)

        self.viewport.camera.intrinsic = projection
        self.viewport.resetCamera()

    def getProjection(self):
        """Return the current camera projection mode as a str.

        See :meth:`setProjection`
        """
        projection = self.viewport.camera.intrinsic
        if isinstance(projection, transform.Orthographic):
            return 'orthographic'
        elif isinstance(projection, transform.Perspective):
            return 'perspective'
        else:
            raise RuntimeError('Unknown projection in use')

    def setBackgroundColor(self, color):
        """Set the background color of the OpenGL view.

        :param color: RGB color of the isosurface: name, #RRGGBB or RGB values
        :type color:
            QColor, str or array-like of 3 or 4 float in [0., 1.] or uint8
        """
        color = rgba(color)
        self.viewport.background = color
        self.overview.background = color[0]*0.5, color[1]*0.5, color[2]*0.5, 1.

    def getBackgroundColor(self):
        """Returns the RGBA background color (QColor)."""
        return qt.QColor.fromRgbF(*self.viewport.background)

    def centerScene(self):
        """Position the center of the scene at the center of rotation."""
        self.viewport.resetCamera()

    def resetZoom(self, face='front'):
        """Reset the camera position to a default.

        :param str face: The direction the camera is looking at:
                         side, front, back, top, bottom, right, left.
                         Default: front.
        """
        self.viewport.camera.extrinsic.reset(face=face)
        self.centerScene()

    def _redraw(self, source=None):
        """Viewport listener to require repaint"""
        if not self._updating and self.viewport.dirty:
            self._updating = True  # Mark that an update is requested
            self.update()  # Queued repaint (i.e., asynchronous)

    def sizeHint(self):
        return qt.QSize(400, 300)

    def initializeGL(self):
        # Check if OpenGL2 is available
        versionflags = self.format().openGLVersionFlags()
        self._isOpenGL21 = bool(versionflags & qt.QGLFormat.OpenGL_Version_2_1)
        if not self._isOpenGL21:
            _logger.error(
                '3D rendering is disabled: OpenGL 2.1 not available')

            messageBox = qt.QMessageBox(parent=self)
            messageBox.setIcon(qt.QMessageBox.Critical)
            messageBox.setWindowTitle('Error')
            messageBox.setText('3D rendering is disabled.\n\n'
                               'Reason: OpenGL 2.1 is not available.')
            messageBox.addButton(qt.QMessageBox.Ok)
            messageBox.setWindowModality(qt.Qt.WindowModal)
            messageBox.setAttribute(qt.Qt.WA_DeleteOnClose)
            messageBox.show()

    def paintGL(self):
        # In case paintGL is called by the system and not through _redraw,
        # Mark as updating.
        self._updating = True

        if hasattr(self, 'windowHandle'):  # Qt 5
            devicePixelRatio = self.windowHandle().devicePixelRatio()
            if devicePixelRatio != self._devicePixelRatio:
                # Move window from one screen to another one
                self._devicePixelRatio = devicePixelRatio
                # Resize  might not be called, so call it explicitly
                self.resizeGL(int(self.width() * devicePixelRatio),
                              int(self.height() * devicePixelRatio))

        if not self._isOpenGL21:
            # Cannot render scene, just clear the color buffer.
            ox, oy = self.viewport.origin
            w, h = self.viewport.size
            gl.glViewport(ox, oy, w, h)

            gl.glClearColor(*self.viewport.background)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        else:
            # Update near and far planes only if viewport needs refresh
            if self.viewport.dirty:
                self.viewport.adjustCameraDepthExtent()

            self.window.render(self.context(), self._devicePixelRatio)

        if self._firstRender:  # TODO remove this ugly hack
            self._firstRender = False
            self.centerScene()
        self._updating = False

    def resizeGL(self, width, height):
        self.window.size = width, height
        self.viewport.size = width, height
        overviewWidth, overviewHeight = self.overview.size
        self.overview.origin = width - overviewWidth, height - overviewHeight

    def grabGL(self):
        """Renders the OpenGL scene into a numpy array

        :returns: OpenGL scene RGB rasterization
        :rtype: QImage
        """
        if not self._isOpenGL21:
            _logger.error('OpenGL 2.1 not available, cannot save OpenGL image')
            height, width = self.window.shape
            image = numpy.zeros((height, width, 3), dtype=numpy.uint8)

        else:
            self.makeCurrent()
            image = self.window.grab(qt.QGLContext.currentContext())

        return convertArrayToQImage(image)

    def wheelEvent(self, event):
        xpixel = event.x() * self._devicePixelRatio
        ypixel = event.y() * self._devicePixelRatio
        if hasattr(event, 'delta'):  # Qt4
            angle = event.delta() / 8.
        else:  # Qt5
            angle = event.angleDelta().y() / 8.
        event.accept()

        if angle != 0:
            self.makeCurrent()
            self.eventHandler.handleEvent('wheel', xpixel, ypixel, angle)

    def keyPressEvent(self, event):
        keycode = event.key()
        # No need to accept QKeyEvent

        converter = {
            qt.Qt.Key_Left: 'left',
            qt.Qt.Key_Right: 'right',
            qt.Qt.Key_Up: 'up',
            qt.Qt.Key_Down: 'down'
        }
        direction = converter.get(keycode, None)
        if direction is not None:
            if event.modifiers() == qt.Qt.ControlModifier:
                self.viewport.camera.rotate(direction)
            elif event.modifiers() == qt.Qt.ShiftModifier:
                self.viewport.moveCamera(direction)
            else:
                self.viewport.orbitCamera(direction)

        else:
            # Key not handled, call base class implementation
            super(Plot3DWidget, self).keyPressEvent(event)

    # Mouse events #
    _MOUSE_BTNS = {1: 'left', 2: 'right', 4: 'middle'}

    def mousePressEvent(self, event):
        xpixel = event.x() * self._devicePixelRatio
        ypixel = event.y() * self._devicePixelRatio
        btn = self._MOUSE_BTNS[event.button()]
        event.accept()

        self.makeCurrent()
        self.eventHandler.handleEvent('press', xpixel, ypixel, btn)

    def mouseMoveEvent(self, event):
        xpixel = event.x() * self._devicePixelRatio
        ypixel = event.y() * self._devicePixelRatio
        event.accept()

        self.makeCurrent()
        self.eventHandler.handleEvent('move', xpixel, ypixel)

    def mouseReleaseEvent(self, event):
        xpixel = event.x() * self._devicePixelRatio
        ypixel = event.y() * self._devicePixelRatio
        btn = self._MOUSE_BTNS[event.button()]
        event.accept()

        self.makeCurrent()
        self.eventHandler.handleEvent('release', xpixel, ypixel, btn)
