# /*##########################################################################
#
# Copyright (c) 2015-2022 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"


import enum
import logging

from silx.gui import qt
from silx.gui.colors import rgba
from . import actions

from ...utils.enum import Enum as _Enum
from ..utils.image import convertArrayToQImage

from .. import _glutils as glu
from .scene import interaction, primitives, transform
from . import scene

import numpy


_logger = logging.getLogger(__name__)


class _OverviewViewport(scene.Viewport):
    """A scene displaying the orientation of the data in another scene.

    :param Camera camera: The camera to track.
    """

    _SIZE = 100
    """Size in pixels of the overview square"""

    def __init__(self, camera=None):
        super(_OverviewViewport, self).__init__()
        self.size = self._SIZE, self._SIZE
        self.background = None  # Disable clear

        self.scene.transforms = [transform.Scale(2.5, 2.5, 2.5)]

        # Add a point to draw the background (in a group with depth mask)
        backgroundPoint = primitives.ColorPoints(
            x=0., y=0., z=0.,
            color=(1., 1., 1., 0.5),
            size=self._SIZE)
        backgroundPoint.marker = 'o'
        noDepthGroup = primitives.GroupNoDepth(mask=True, notest=True)
        noDepthGroup.children.append(backgroundPoint)
        self.scene.children.append(noDepthGroup)

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


class Plot3DWidget(glu.OpenGLWidget):
    """OpenGL widget with a 3D viewport and an overview."""

    sigInteractiveModeChanged = qt.Signal()
    """Signal emitted when the interactive mode has changed
    """

    sigStyleChanged = qt.Signal(str)
    """Signal emitted when the style of the scene has changed

    It provides the updated property.
    """

    sigSceneClicked = qt.Signal(float, float)
    """Signal emitted when the scene is clicked with the left mouse button.

    It provides the (x, y) clicked mouse position in logical widget pixel coordinates.
    """

    @enum.unique
    class FogMode(_Enum):
        """Different mode to render the scene with fog"""

        NONE = 'none'
        """No fog effect"""

        LINEAR = 'linear'
        """Linear fog through the whole scene"""

    def __init__(self, parent=None, f=qt.Qt.Widget):
        self._firstRender = True

        super(Plot3DWidget, self).__init__(
            parent,
            alphaBufferSize=8,
            depthBufferSize=0,
            stencilBufferSize=0,
            version=(2, 1),
            f=f)

        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

        self.setFocusPolicy(qt.Qt.StrongFocus)
        self._copyAction = actions.io.CopyAction(parent=self, plot3d=self)
        self.addAction(self._copyAction)

        self._updating = False  # True if an update is requested

        # Main viewport
        self.viewport = scene.Viewport()

        self._sceneScale = transform.Scale(1., 1., 1.)
        self.viewport.scene.transforms = [self._sceneScale,
                                          transform.Translate(0., 0., 0.)]

        # Overview area
        self.overview = _OverviewViewport(self.viewport.camera)

        self.setBackgroundColor((0.2, 0.2, 0.2, 1.))

        # Window describing on screen area to render
        self._window = scene.Window(mode='framebuffer')
        self._window.viewports = [self.viewport, self.overview]
        self._window.addListener(self._redraw)

        self.eventHandler = None
        self.setInteractiveMode('rotate')

    def __clickHandler(self, *args):
        """Handle interaction state machine click"""
        x, y = args[0][:2]
        # Convert from device pixel to logical pixel unit
        devicePixelRatio = self.getDevicePixelRatio()
        self.sigSceneClicked.emit(x / devicePixelRatio, y / devicePixelRatio)

    def setInteractiveMode(self, mode):
        """Set the interactive mode.

        :param str mode: The interactive mode: 'rotate', 'pan' or None
        """
        if mode == self.getInteractiveMode():
            return

        if mode is None:
            self.eventHandler = None

        elif mode == 'rotate':
            self.eventHandler = interaction.RotateCameraControl(
                self.viewport,
                orbitAroundCenter=False,
                mode='position',
                scaleTransform=self._sceneScale,
                selectCB=self.__clickHandler)

        elif mode == 'pan':
            self.eventHandler = interaction.PanCameraControl(
                self.viewport,
                orbitAroundCenter=False,
                mode='position',
                scaleTransform=self._sceneScale,
                selectCB=self.__clickHandler)

        elif isinstance(mode, interaction.StateMachine):
            self.eventHandler = mode

        else:
            raise ValueError('Unsupported interactive mode %s', str(mode))

        if (self.eventHandler is not None and
                qt.QApplication.keyboardModifiers() & qt.Qt.ControlModifier):
            self.eventHandler.handleEvent('keyPress', qt.Qt.Key_Control)

        self.sigInteractiveModeChanged.emit()

    def getInteractiveMode(self):
        """Returns the interactive mode in use.

        :rtype: str
        """
        if self.eventHandler is None:
            return None
        if isinstance(self.eventHandler, interaction.RotateCameraControl):
            return 'rotate'
        elif isinstance(self.eventHandler, interaction.PanCameraControl):
            return 'pan'
        else:
            return None

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
        if color != self.viewport.background:
            self.viewport.background = color
            self.sigStyleChanged.emit('backgroundColor')

    def getBackgroundColor(self):
        """Returns the RGBA background color (QColor)."""
        return qt.QColor.fromRgbF(*self.viewport.background)

    def setFogMode(self, mode):
        """Set the kind of fog to use for the whole scene.

        :param Union[str,FogMode] mode: The mode to use
        :raise ValueError: If mode is not supported
        """
        mode = self.FogMode.from_value(mode)
        if mode != self.getFogMode():
            self.viewport.fog.isOn = mode is self.FogMode.LINEAR
            self.sigStyleChanged.emit('fogMode')

    def getFogMode(self):
        """Returns the kind of fog in use

        :return: The kind of fog in use
        :rtype: FogMode
        """
        if self.viewport.fog.isOn:
            return self.FogMode.LINEAR
        else:
            return self.FogMode.NONE

    def isOrientationIndicatorVisible(self):
        """Returns True if the orientation indicator is displayed.

        :rtype: bool
        """
        return self.overview in self._window.viewports

    def setOrientationIndicatorVisible(self, visible):
        """Set the orientation indicator visibility.

        :param bool visible: True to show
        """
        visible = bool(visible)
        if visible != self.isOrientationIndicatorVisible():
            if visible:
                self._window.viewports = [self.viewport, self.overview]
            else:
                self._window.viewports = [self.viewport]
            self.sigStyleChanged.emit('orientationIndicatorVisible')

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
        if not self._updating:
            self._updating = True  # Mark that an update is requested
            self.update()  # Queued repaint (i.e., asynchronous)

    def sizeHint(self):
        return qt.QSize(400, 300)

    def initializeGL(self):
        pass

    def paintGL(self):
        # In case paintGL is called by the system and not through _redraw,
        # Mark as updating.
        self._updating = True

        # Update near and far planes only if viewport needs refresh
        if self.viewport.dirty:
            self.viewport.adjustCameraDepthExtent()

        self._window.render(self.context(), self.getDevicePixelRatio())

        if self._firstRender:  # TODO remove this ugly hack
            self._firstRender = False
            self.centerScene()
        self._updating = False

    def resizeGL(self, width, height):
        width *= self.getDevicePixelRatio()
        height *= self.getDevicePixelRatio()
        self._window.size = width, height
        self.viewport.size = self._window.size
        overviewWidth, overviewHeight = self.overview.size
        self.overview.origin = width - overviewWidth, height - overviewHeight

    def grabGL(self):
        """Renders the OpenGL scene into a numpy array

        :returns: OpenGL scene RGB rasterization
        :rtype: QImage
        """
        if not self.isValid():
            _logger.error('OpenGL 2.1 not available, cannot save OpenGL image')
            height, width = self._window.shape
            image = numpy.zeros((height, width, 3), dtype=numpy.uint8)

        else:
            self.makeCurrent()
            image = self._window.grab(self.context())

        return convertArrayToQImage(image)

    def wheelEvent(self, event):
        x, y = qt.getMouseEventPosition(event)
        xpixel = x * self.getDevicePixelRatio()
        ypixel = y * self.getDevicePixelRatio()
        angle = event.angleDelta().y() / 8.
        event.accept()

        if self.eventHandler is not None and angle != 0 and self.isValid():
            self.makeCurrent()
            self.eventHandler.handleEvent('wheel', xpixel, ypixel, angle)

    def keyPressEvent(self, event):
        keyCode = event.key()
        # No need to accept QKeyEvent

        converter = {
            qt.Qt.Key_Left: 'left',
            qt.Qt.Key_Right: 'right',
            qt.Qt.Key_Up: 'up',
            qt.Qt.Key_Down: 'down'
        }
        direction = converter.get(keyCode, None)
        if direction is not None:
            if event.modifiers() == qt.Qt.ControlModifier:
                self.viewport.camera.rotate(direction)
            elif event.modifiers() == qt.Qt.ShiftModifier:
                self.viewport.moveCamera(direction)
            else:
                self.viewport.orbitCamera(direction)

        else:
            if (keyCode == qt.Qt.Key_Control and
                    self.eventHandler is not None and
                    self.isValid()):
                self.eventHandler.handleEvent('keyPress', keyCode)

            # Key not handled, call base class implementation
            super(Plot3DWidget, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Catch Ctrl key release"""
        keyCode = event.key()
        if (keyCode == qt.Qt.Key_Control and
                self.eventHandler is not None and
                self.isValid()):
            self.eventHandler.handleEvent('keyRelease', keyCode)
        super(Plot3DWidget, self).keyReleaseEvent(event)

    # Mouse events #
    _MOUSE_BTNS = {
        qt.Qt.LeftButton: 'left',
        qt.Qt.RightButton: 'right',
        qt.Qt.MiddleButton: 'middle',
    }

    def mousePressEvent(self, event):
        x, y = qt.getMouseEventPosition(event)
        xpixel = x * self.getDevicePixelRatio()
        ypixel = y * self.getDevicePixelRatio()
        btn = self._MOUSE_BTNS[event.button()]
        event.accept()

        if self.eventHandler is not None and self.isValid():
            self.makeCurrent()
            self.eventHandler.handleEvent('press', xpixel, ypixel, btn)

    def mouseMoveEvent(self, event):
        x, y = qt.getMouseEventPosition(event)
        xpixel = x * self.getDevicePixelRatio()
        ypixel = y * self.getDevicePixelRatio()
        event.accept()

        if self.eventHandler is not None and self.isValid():
            self.makeCurrent()
            self.eventHandler.handleEvent('move', xpixel, ypixel)

    def mouseReleaseEvent(self, event):
        x, y = qt.getMouseEventPosition(event)
        xpixel = x * self.getDevicePixelRatio()
        ypixel = y * self.getDevicePixelRatio()
        btn = self._MOUSE_BTNS[event.button()]
        event.accept()

        if self.eventHandler is not None and self.isValid():
            self.makeCurrent()
            self.eventHandler.handleEvent('release', xpixel, ypixel, btn)
