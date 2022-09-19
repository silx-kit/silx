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
"""This module provides interaction to plug on the scene graph."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/07/2016"

import logging
import numpy

from silx.gui import qt
from silx.gui.plot.Interaction import \
    StateMachine, State, LEFT_BTN, RIGHT_BTN  # , MIDDLE_BTN

from . import transform


_logger = logging.getLogger(__name__)


class ClickOrDrag(StateMachine):
    """Click or drag interaction for a given button.

    """
    #TODO: merge this class with silx.gui.plot.Interaction.ClickOrDrag

    DRAG_THRESHOLD_SQUARE_DIST = 5 ** 2

    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == self.machine.button:
                self.goto('clickOrDrag', x, y)
                return True

    class ClickOrDrag(State):
        def enterState(self, x, y):
            self.initPos = x, y

        enter = enterState  # silx v.0.3 support, remove when 0.4 out

        def onMove(self, x, y):
            dx = (x - self.initPos[0]) ** 2
            dy = (y - self.initPos[1]) ** 2
            if (dx ** 2 + dy ** 2) >= self.machine.DRAG_THRESHOLD_SQUARE_DIST:
                self.goto('drag', self.initPos, (x, y))

        def onRelease(self, x, y, btn):
            if btn == self.machine.button:
                self.machine.click(x, y)
                self.goto('idle')

    class Drag(State):
        def enterState(self, initPos, curPos):
            self.initPos = initPos
            self.machine.beginDrag(*initPos)
            self.machine.drag(*curPos)

        enter = enterState  # silx v.0.3 support, remove when 0.4 out

        def onMove(self, x, y):
            self.machine.drag(x, y)

        def onRelease(self, x, y, btn):
            if btn == self.machine.button:
                self.machine.endDrag(self.initPos, (x, y))
                self.goto('idle')

    def __init__(self, button=LEFT_BTN):
        self.button = button
        states = {
            'idle': ClickOrDrag.Idle,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        super(ClickOrDrag, self).__init__(states, 'idle')

    def click(self, x, y):
        """Called upon a left or right button click.
        To override in a subclass.
        """
        pass

    def beginDrag(self, x, y):
        """Called at the beginning of a drag gesture with left button
        pressed.
        To override in a subclass.
        """
        pass

    def drag(self, x, y):
        """Called on mouse moved during a drag gesture.
        To override in a subclass.
        """
        pass

    def endDrag(self, x, y):
        """Called at the end of a drag gesture when the left button is
        released.
        To override in a subclass.
        """
        pass


class CameraSelectRotate(ClickOrDrag):
    """Camera rotation using an arcball-like interaction."""

    def __init__(self, viewport, orbitAroundCenter=True, button=RIGHT_BTN,
                 selectCB=None):
        self._viewport = viewport
        self._orbitAroundCenter = orbitAroundCenter
        self._selectCB = selectCB
        self._reset()
        super(CameraSelectRotate, self).__init__(button)

    def _reset(self):
        self._origin, self._center = None, None
        self._startExtrinsic = None

    def click(self, x, y):
        if self._selectCB is not None:
            ndcZ = self._viewport._pickNdcZGL(x, y)
            position = self._viewport._getXZYGL(x, y)
            # This assume no object lie on the far plane
            # Alternative, change the depth range so that far is < 1
            if ndcZ != 1. and position is not None:
                self._selectCB((x, y, ndcZ), position)

    def beginDrag(self, x, y):
        centerPos = None
        if not self._orbitAroundCenter:
            # Try to use picked object position as center of rotation
            ndcZ = self._viewport._pickNdcZGL(x, y)
            if ndcZ != 1.:
                # Hit an object, use picked point as center
                centerPos = self._viewport._getXZYGL(x, y)  # Can return None

        if centerPos is None:
            # Not using picked position, use scene center
            bounds = self._viewport.scene.bounds(transformed=True)
            centerPos = 0.5 * (bounds[0] + bounds[1])

        self._center = transform.Translate(*centerPos)
        self._origin = x, y
        self._startExtrinsic = self._viewport.camera.extrinsic.copy()

    def drag(self, x, y):
        if self._center is None:
            return

        dx, dy = self._origin[0] - x, self._origin[1] - y

        if dx == 0 and dy == 0:
            direction = self._startExtrinsic.direction
            up = self._startExtrinsic.up
            position = self._startExtrinsic.position
        else:
            minsize = min(self._viewport.size)
            distance = numpy.sqrt(dx ** 2 + dy ** 2)
            angle = distance / minsize * numpy.pi

            # Take care of y inversion
            direction = dx * self._startExtrinsic.side - \
                dy * self._startExtrinsic.up
            direction /= numpy.linalg.norm(direction)
            axis = numpy.cross(direction, self._startExtrinsic.direction)
            axis /= numpy.linalg.norm(axis)

            # Orbit start camera with current angle and axis
            # Rotate viewing direction
            rotation = transform.Rotate(numpy.degrees(angle), *axis)
            direction = rotation.transformDir(self._startExtrinsic.direction)
            up = rotation.transformDir(self._startExtrinsic.up)

            # Rotate position around center
            trlist = transform.StaticTransformList((
                self._center,
                rotation,
                self._center.inverse()))
            position = trlist.transformPoint(self._startExtrinsic.position)

        camerapos = self._viewport.camera.extrinsic
        camerapos.setOrientation(direction, up)
        camerapos.position = position

    def endDrag(self, x, y):
        self._reset()


class CameraSelectPan(ClickOrDrag):
    """Picking on click and pan camera on drag."""

    def __init__(self, viewport, button=LEFT_BTN, selectCB=None):
        self._viewport = viewport
        self._selectCB = selectCB
        self._lastPosNdc = None
        super(CameraSelectPan, self).__init__(button)

    def click(self, x, y):
        if self._selectCB is not None:
            ndcZ = self._viewport._pickNdcZGL(x, y)
            position = self._viewport._getXZYGL(x, y)
            # This assume no object lie on the far plane
            # Alternative, change the depth range so that far is < 1
            if ndcZ != 1. and position is not None:
                self._selectCB((x, y, ndcZ), position)

    def beginDrag(self, x, y):
        ndc = self._viewport.windowToNdc(x, y)
        ndcZ = self._viewport._pickNdcZGL(x, y)
        # ndcZ is the panning plane
        if ndc is not None and ndcZ is not None:
            self._lastPosNdc = numpy.array((ndc[0], ndc[1], ndcZ, 1.),
                                           dtype=numpy.float32)
        else:
            self._lastPosNdc = None

    def drag(self, x, y):
        if self._lastPosNdc is not None:
            ndc = self._viewport.windowToNdc(x, y)
            if ndc is not None:
                ndcPos = numpy.array((ndc[0], ndc[1], self._lastPosNdc[2], 1.),
                                     dtype=numpy.float32)

                # Convert last and current NDC positions to scene coords
                scenePos = self._viewport.camera.transformPoint(
                    ndcPos, direct=False, perspectiveDivide=True)
                lastScenePos = self._viewport.camera.transformPoint(
                    self._lastPosNdc, direct=False, perspectiveDivide=True)

                # Get translation in scene coords
                translation = scenePos[:3] - lastScenePos[:3]
                self._viewport.camera.extrinsic.position -= translation

                # Store for next drag
                self._lastPosNdc = ndcPos

    def endDrag(self, x, y):
        self._lastPosNdc = None


class CameraWheel(object):
    """StateMachine like class, just handling wheel events."""

    # TODO choose scale of motion? Translation or Scale?
    def __init__(self, viewport, mode='center', scaleTransform=None):
        assert mode in ('center', 'position', 'scale')
        self._viewport = viewport
        if mode == 'center':
            self._zoomTo = self._zoomToCenter
        elif mode == 'position':
            self._zoomTo = self._zoomToPosition
        elif mode == 'scale':
            self._zoomTo = self._zoomByScale
            self._scale = scaleTransform
        else:
            raise ValueError('Unsupported mode: %s' % mode)

    def handleEvent(self, eventName, *args, **kwargs):
        if eventName == 'wheel':
            return self._zoomTo(*args, **kwargs)

    def _zoomToCenter(self, x, y, angleInDegrees):
        """Zoom to center of display.

        Only works with perspective camera.
        """
        direction = 'forward' if angleInDegrees > 0 else 'backward'
        self._viewport.camera.move(direction)
        return True

    def _zoomToPositionAbsolute(self, x, y, angleInDegrees):
        """Zoom while keeping pixel under mouse invariant.

        Only works with perspective camera.
        """
        ndc = self._viewport.windowToNdc(x, y)
        if ndc is not None:
            near = numpy.array((ndc[0], ndc[1], -1., 1.), dtype=numpy.float32)

            nearscene = self._viewport.camera.transformPoint(
                near, direct=False, perspectiveDivide=True)

            far = numpy.array((ndc[0], ndc[1], 1., 1.), dtype=numpy.float32)
            farscene = self._viewport.camera.transformPoint(
                far, direct=False, perspectiveDivide=True)

            dirscene = farscene[:3] - nearscene[:3]
            dirscene /= numpy.linalg.norm(dirscene)

            if angleInDegrees < 0:
                dirscene *= -1.

            # TODO which scale
            self._viewport.camera.extrinsic.position += dirscene
        return True

    def _zoomToPosition(self, x, y, angleInDegrees):
        """Zoom while keeping pixel under mouse invariant."""
        projection = self._viewport.camera.intrinsic
        extrinsic = self._viewport.camera.extrinsic

        if isinstance(projection, transform.Perspective):
            # For perspective projection, move camera
            ndc = self._viewport.windowToNdc(x, y)
            if ndc is not None:
                ndcz = self._viewport._pickNdcZGL(x, y)

                position = numpy.array((ndc[0], ndc[1], ndcz),
                                       dtype=numpy.float32)
                positionscene = self._viewport.camera.transformPoint(
                    position, direct=False, perspectiveDivide=True)

                camtopos = extrinsic.position - positionscene

                step = 0.2 * (1. if angleInDegrees < 0 else -1.)
                extrinsic.position += step * camtopos

        elif isinstance(projection, transform.Orthographic):
            # For orthographic projection, change projection borders
            ndcx, ndcy = self._viewport.windowToNdc(x, y, checkInside=False)

            step = 0.2 * (1. if angleInDegrees < 0 else -1.)

            dx = (ndcx + 1) / 2.
            stepwidth = step * (projection.right - projection.left)
            left = projection.left - dx * stepwidth
            right = projection.right + (1. - dx) * stepwidth

            dy = (ndcy + 1) / 2.
            stepheight = step * (projection.top - projection.bottom)
            bottom = projection.bottom - dy * stepheight
            top = projection.top + (1. - dy) * stepheight

            projection.setClipping(left, right, bottom, top)

        else:
            raise RuntimeError('Unsupported camera', projection)
        return True

    def _zoomByScale(self, x, y, angleInDegrees):
        """Zoom by scaling scene (do not keep pixel under mouse invariant)."""
        scalefactor = 1.1
        if angleInDegrees < 0.:
            scalefactor = 1. / scalefactor
        self._scale.scale = scalefactor * self._scale.scale

        self._viewport.adjustCameraDepthExtent()
        return True


class FocusManager(StateMachine):
    """Manages focus across multiple event handlers

    On press an event handler can acquire focus.
    By default it looses focus when all buttons are released.
    """
    class Idle(State):
        def onPress(self, x, y, btn):
            for eventHandler in self.machine.currentEventHandler:
                requestFocus = eventHandler.handleEvent('press', x, y, btn)
                if requestFocus:
                    self.goto('focus', eventHandler, btn)
                    break

        def _processEvent(self, *args):
            for eventHandler in self.machine.currentEventHandler:
                consumeEvent = eventHandler.handleEvent(*args)
                if consumeEvent:
                    break

        def onMove(self, x, y):
            self._processEvent('move', x, y)

        def onRelease(self, x, y, btn):
            self._processEvent('release', x, y, btn)

        def onWheel(self, x, y, angle):
            self._processEvent('wheel', x, y, angle)

    class Focus(State):
        def enterState(self, eventHandler, btn):
            self.eventHandler = eventHandler
            self.focusBtns = {btn}  # Set

        enter = enterState  # silx v.0.3 support, remove when 0.4 out

        def onPress(self, x, y, btn):
            self.focusBtns.add(btn)
            self.eventHandler.handleEvent('press', x, y, btn)

        def onMove(self, x, y):
            self.eventHandler.handleEvent('move', x, y)

        def onRelease(self, x, y, btn):
            self.focusBtns.discard(btn)
            requestfocus = self.eventHandler.handleEvent('release', x, y, btn)
            if len(self.focusBtns) == 0 and not requestfocus:
                self.goto('idle')

        def onWheel(self, x, y, angleInDegrees):
            self.eventHandler.handleEvent('wheel', x, y, angleInDegrees)

    def __init__(self, eventHandlers=(), ctrlEventHandlers=None):
        self.defaultEventHandlers = eventHandlers
        self.ctrlEventHandlers = ctrlEventHandlers
        self.currentEventHandler = self.defaultEventHandlers

        states = {
            'idle': FocusManager.Idle,
            'focus': FocusManager.Focus
        }
        super(FocusManager, self).__init__(states, 'idle')

    def onKeyPress(self, key):
        if key == qt.Qt.Key_Control and self.ctrlEventHandlers is not None:
            self.currentEventHandler = self.ctrlEventHandlers

    def onKeyRelease(self, key):
        if key == qt.Qt.Key_Control:
            self.currentEventHandler = self.defaultEventHandlers

    def cancel(self):
        for handler in self.currentEventHandler:
            handler.cancel()


class RotateCameraControl(FocusManager):
    """Combine wheel and rotate state machine for left button
    and pan when ctrl is pressed
    """
    def __init__(self, viewport,
                 orbitAroundCenter=False,
                 mode='center', scaleTransform=None,
                 selectCB=None):
        handlers = (CameraWheel(viewport, mode, scaleTransform),
                    CameraSelectRotate(
                        viewport, orbitAroundCenter, LEFT_BTN, selectCB))
        ctrlHandlers = (CameraWheel(viewport, mode, scaleTransform),
                        CameraSelectPan(viewport, LEFT_BTN, selectCB))
        super(RotateCameraControl, self).__init__(handlers, ctrlHandlers)


class PanCameraControl(FocusManager):
    """Combine wheel, selectPan and rotate state machine for left button
    and rotate when ctrl is pressed"""
    def __init__(self, viewport,
                 orbitAroundCenter=False,
                 mode='center', scaleTransform=None,
                 selectCB=None):
        handlers = (CameraWheel(viewport, mode, scaleTransform),
                    CameraSelectPan(viewport, LEFT_BTN, selectCB))
        ctrlHandlers = (CameraWheel(viewport, mode, scaleTransform),
                        CameraSelectRotate(
                            viewport, orbitAroundCenter, LEFT_BTN, selectCB))
        super(PanCameraControl, self).__init__(handlers, ctrlHandlers)


class CameraControl(FocusManager):
    """Combine wheel, selectPan and rotate state machine."""
    def __init__(self, viewport,
                 orbitAroundCenter=False,
                 mode='center', scaleTransform=None,
                 selectCB=None):
        handlers = (CameraWheel(viewport, mode, scaleTransform),
                    CameraSelectPan(viewport, LEFT_BTN, selectCB),
                    CameraSelectRotate(
                        viewport, orbitAroundCenter, RIGHT_BTN, selectCB))
        super(CameraControl, self).__init__(handlers)


class PlaneRotate(ClickOrDrag):
    """Plane rotation using arcball interaction.

    Arcball ref.:
    Ken Shoemake. ARCBALL: A user interface for specifying three-dimensional
    orientation using a mouse. In Proc. GI '92. (1992). pp. 151-156.
    """

    def __init__(self, viewport, plane, button=RIGHT_BTN):
        self._viewport = viewport
        self._plane = plane
        self._reset()
        super(PlaneRotate, self).__init__(button)

    def _reset(self):
        self._beginNormal, self._beginCenter = None, None

    def click(self, x, y):
        pass  # No interaction

    @staticmethod
    def _sphereUnitVector(radius, center, position):
        """Returns the unit vector of the projection of position on a sphere.

        It assumes an orthographic projection.
        For perspective projection, it gives an approximation, but it
        simplifies computations and results in consistent arcball control
        in control space.

        All parameters must be in screen coordinate system
        (either pixels or normalized coordinates).

        :param float radius: The radius of the sphere.
        :param center: (x, y) coordinates of the center.
        :param position: (x, y) coordinates of the cursor position.
        :return: Unit vector.
        :rtype: numpy.ndarray of 3 floats.
        """
        center, position = numpy.array(center), numpy.array(position)

        # Normalize x and y on a unit circle
        spherecoords = (position - center) / float(radius)
        squarelength = numpy.sum(spherecoords ** 2)

        # Project on the unit sphere and compute z coordinates
        if squarelength > 1.0:  # Outside sphere: project
            spherecoords /= numpy.sqrt(squarelength)
            zsphere = 0.0
        else:  # In sphere: compute z
            zsphere = numpy.sqrt(1. - squarelength)

        spherecoords = numpy.append(spherecoords, zsphere)
        return spherecoords

    def beginDrag(self, x, y):
        # Makes sure the point defining the plane is at the center as
        # it will be the center of rotation (as rotation is applied to normal)
        self._plane.plane.point = self._plane.center

        # Store the plane normal
        self._beginNormal = self._plane.plane.normal

        _logger.debug(
            'Begin arcball, plane center %s', str(self._plane.center))

        # Do the arcball on the screen
        radius = min(self._viewport.size)
        if self._plane.center is None:
            self._beginCenter = None

        else:
            center = self._plane.objectToNDCTransform.transformPoint(
                self._plane.center, perspectiveDivide=True)
            self._beginCenter = self._viewport.ndcToWindow(
                center[0], center[1], checkInside=False)

            self._startVector = self._sphereUnitVector(
                radius, self._beginCenter, (x, y))

    def drag(self, x, y):
        if self._beginCenter is None:
            return

        # Compute rotation: this is twice the rotation of the arcball
        radius = min(self._viewport.size)
        currentvector = self._sphereUnitVector(
            radius, self._beginCenter, (x, y))
        crossprod = numpy.cross(self._startVector, currentvector)
        dotprod = numpy.dot(self._startVector, currentvector)

        quaternion = numpy.append(crossprod, dotprod)
        # Rotation was computed with Y downward, but apply in NDC, invert Y
        quaternion[1] *= -1.

        rotation = transform.Rotate()
        rotation.quaternion = quaternion

        # Convert to NDC, rotate, convert back to object
        normal = self._plane.objectToNDCTransform.transformNormal(
            self._beginNormal)
        normal = rotation.transformNormal(normal)
        normal = self._plane.objectToNDCTransform.transformNormal(
            normal, direct=False)
        self._plane.plane.normal = normal

    def endDrag(self, x, y):
        self._reset()


class PlanePan(ClickOrDrag):
    """Pan a plane along its normal on drag."""

    def __init__(self, viewport, plane, button=LEFT_BTN):
        self._plane = plane
        self._viewport = viewport
        self._beginPlanePoint = None
        self._beginPos = None
        self._dragNdcZ = 0.
        super(PlanePan, self).__init__(button)

    def click(self, x, y):
        pass

    def beginDrag(self, x, y):
        ndc = self._viewport.windowToNdc(x, y)
        ndcZ = self._viewport._pickNdcZGL(x, y)
        # ndcZ is the panning plane
        if ndc is not None and ndcZ is not None:
            ndcPos = numpy.array((ndc[0], ndc[1], ndcZ, 1.),
                                 dtype=numpy.float32)
            scenePos = self._viewport.camera.transformPoint(
                ndcPos, direct=False, perspectiveDivide=True)
            self._beginPos = self._plane.objectToSceneTransform.transformPoint(
                scenePos, direct=False)
            self._dragNdcZ = ndcZ
        else:
            self._beginPos = None
            self._dragNdcZ = 0.

        self._beginPlanePoint = self._plane.plane.point

    def drag(self, x, y):
        if self._beginPos is not None:
            ndc = self._viewport.windowToNdc(x, y)
            if ndc is not None:
                ndcPos = numpy.array((ndc[0], ndc[1], self._dragNdcZ, 1.),
                                     dtype=numpy.float32)

                # Convert last and current NDC positions to scene coords
                scenePos = self._viewport.camera.transformPoint(
                    ndcPos, direct=False, perspectiveDivide=True)
                curPos = self._plane.objectToSceneTransform.transformPoint(
                    scenePos, direct=False)

                # Get translation in scene coords
                translation = curPos[:3] - self._beginPos[:3]

                newPoint = self._beginPlanePoint + translation

                # Keep plane point in bounds
                bounds = self._plane.parent.bounds(dataBounds=True)
                if bounds is not None:
                    newPoint = numpy.clip(
                        newPoint, a_min=bounds[0], a_max=bounds[1])

                    # Only update plane if it is in some bounds
                    self._plane.plane.point = newPoint

    def endDrag(self, x, y):
        self._beginPlanePoint = None


class PlaneControl(FocusManager):
    """Combine wheel, selectPan and rotate state machine for plane control."""
    def __init__(self, viewport, plane,
                 mode='center', scaleTransform=None):
        handlers = (CameraWheel(viewport, mode, scaleTransform),
                    PlanePan(viewport, plane, LEFT_BTN),
                    PlaneRotate(viewport, plane, RIGHT_BTN))
        super(PlaneControl, self).__init__(handlers)


class PanPlaneRotateCameraControl(FocusManager):
    """Combine wheel, pan plane and camera rotate state machine."""
    def __init__(self, viewport, plane,
                 mode='center', scaleTransform=None):
        handlers = (CameraWheel(viewport, mode, scaleTransform),
                    PlanePan(viewport, plane, LEFT_BTN),
                    CameraSelectRotate(viewport,
                                       orbitAroundCenter=False,
                                       button=RIGHT_BTN))
        super(PanPlaneRotateCameraControl, self).__init__(handlers)


class PanPlaneZoomOnWheelControl(FocusManager):
    """Combine zoom on wheel and pan plane state machines."""
    def __init__(self, viewport, plane,
                 mode='center',
                 orbitAroundCenter=False,
                 scaleTransform=None):
        handlers = (CameraWheel(viewport, mode, scaleTransform),
                    PlanePan(viewport, plane, LEFT_BTN))
        ctrlHandlers = (CameraWheel(viewport, mode, scaleTransform),
                        CameraSelectRotate(
                            viewport, orbitAroundCenter, LEFT_BTN))
        super(PanPlaneZoomOnWheelControl, self).__init__(handlers, ctrlHandlers)
