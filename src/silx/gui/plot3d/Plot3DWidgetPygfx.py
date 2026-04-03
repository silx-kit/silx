"""pygfx-based 3D rendering widget, replacement for Plot3DWidget."""

import logging
import math

import numpy

from .. import qt
from ..colors import rgba

_logger = logging.getLogger(__name__)


def _look_at_quaternion(eye, target, up=(0, 1, 0)):
    """Compute quaternion (x, y, z, w) for camera at eye looking at target."""
    eye = numpy.asarray(eye, dtype=numpy.float64)
    target = numpy.asarray(target, dtype=numpy.float64)
    up = numpy.asarray(up, dtype=numpy.float64)

    forward = target - eye
    fwd_len = numpy.linalg.norm(forward)
    if fwd_len < 1e-10:
        return (0.0, 0.0, 0.0, 1.0)
    forward = forward / fwd_len

    right = numpy.cross(forward, up)
    right_len = numpy.linalg.norm(right)
    if right_len < 1e-6:
        alt_up = (
            numpy.array([1.0, 0, 0])
            if abs(forward[1]) > 0.9
            else numpy.array([0, 1.0, 0])
        )
        right = numpy.cross(forward, alt_up)
        right = right / numpy.linalg.norm(right)
    else:
        right = right / right_len

    up_actual = numpy.cross(right, forward)

    # Rotation matrix: camera local X=right, Y=up, -Z=forward
    R = numpy.zeros((3, 3))
    R[:, 0] = right
    R[:, 1] = up_actual
    R[:, 2] = -forward

    return _mat3_to_quat(R)


def _mat3_to_quat(m):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]

    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return (float(x), float(y), float(z), float(w))


class _StubLight:
    """Stub light for pygfx backend, compatible with _DirectionalLightProxy."""

    direction = (0, -1, -1)

    def addListener(self, callback):
        pass


class _ExtrinsicProxy:
    """Proxy for camera extrinsic with reset(face=) API."""

    _FACE_DIRECTIONS = {
        "front": numpy.array([0.0, 0.0, 1.0]),
        "back": numpy.array([0.0, 0.0, -1.0]),
        "right": numpy.array([1.0, 0.0, 0.0]),
        "left": numpy.array([-1.0, 0.0, 0.0]),
        "top": numpy.array([0.0, 1.0, 0.001]),
        "bottom": numpy.array([0.0, -1.0, 0.001]),
        "side": numpy.array([1.0, 1.0, 1.0]),
    }

    def __init__(self, widget):
        self._widget = widget

    def reset(self, face="front"):
        """Reset camera to a predefined viewpoint."""
        camera = self._widget._camera
        scene = self._widget._scene

        direction = self._FACE_DIRECTIONS.get(
            face, self._FACE_DIRECTIONS["front"]
        ).copy()
        direction = direction / numpy.linalg.norm(direction)

        center = self._widget._getSceneCenter()

        # First show_object to get proper distance
        camera.show_object(scene)
        pos = numpy.array(camera.local.position, dtype=numpy.float64)
        distance = max(numpy.linalg.norm(pos - center), 1.0)

        # Reposition camera
        new_pos = center + direction * distance
        camera.local.position = tuple(new_pos)

        # Set rotation to look at center
        up = (
            (0, 0, -1)
            if face == "top"
            else (0, 0, 1) if face == "bottom" else (0, 1, 0)
        )
        quat = _look_at_quaternion(new_pos, center, up)
        camera.local.rotation = quat

        # Recreate controller to pick up new camera state
        gfx = self._widget._gfx
        self._widget._controller = gfx.OrbitController(
            camera, register_events=self._widget._renderer
        )


class _CameraProxy:
    """Proxy for camera with extrinsic.reset(face=) API."""

    def __init__(self, widget):
        self.extrinsic = _ExtrinsicProxy(widget)


class _ViewportProxy:
    """Proxy providing viewport-like API for pygfx widget."""

    def __init__(self, widget):
        self._widget = widget
        self.camera = _CameraProxy(widget)
        self.light = _StubLight()

    def orbitCamera(self, direction, angle=1.0):
        """Rotate camera around scene center.

        :param str direction: 'up', 'down', 'left', 'right'
        :param float angle: Rotation angle in degrees
        """
        camera = self._widget._camera
        pos = numpy.array(camera.local.position, dtype=numpy.float64)
        center = self._widget._getSceneCenter()
        rel = pos - center
        distance = numpy.linalg.norm(rel)
        if distance < 1e-6:
            return

        rad = math.radians(angle)

        if direction in ("left", "right"):
            sign = 1.0 if direction == "left" else -1.0
            c, s = math.cos(sign * rad), math.sin(sign * rad)
            new_rel = numpy.array(
                [
                    rel[0] * c + rel[2] * s,
                    rel[1],
                    -rel[0] * s + rel[2] * c,
                ]
            )
        elif direction in ("up", "down"):
            sign = 1.0 if direction == "up" else -1.0
            c, s = math.cos(sign * rad), math.sin(sign * rad)
            new_rel = numpy.array(
                [
                    rel[0],
                    rel[1] * c - rel[2] * s,
                    rel[1] * s + rel[2] * c,
                ]
            )
        else:
            return

        new_pos = center + new_rel
        camera.local.position = tuple(new_pos)

        # Update rotation to look at center
        quat = _look_at_quaternion(new_pos, center)
        camera.local.rotation = quat


class Plot3DWidgetPygfx(qt.QWidget):
    """3D scene widget using pygfx/wgpu for rendering.

    Drop-in replacement for Plot3DWidget with the same public API.
    """

    sigStyleChanged = qt.Signal(str)
    sigInteractiveModeChanged = qt.Signal()
    sigSceneClicked = qt.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        import pygfx as gfx
        from rendercanvas.qt import QRenderWidget

        self._gfx = gfx

        # Layout
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Render widget
        self._renderWidget = QRenderWidget(self)
        self._renderWidget.set_update_mode("continuous", max_fps=60)
        layout.addWidget(self._renderWidget)

        # Renderer
        self._renderer = gfx.WgpuRenderer(self._renderWidget)

        # Scene
        self._scene = gfx.Scene()

        # Camera
        self._camera = gfx.PerspectiveCamera(fov=50)
        self._projection = "perspective"

        # Lights: ambient + directional (attached to camera)
        ambient = gfx.AmbientLight(intensity=0.4)
        self._scene.add(ambient)

        directional = gfx.DirectionalLight(intensity=0.8)
        self._camera.add(directional)
        self._scene.add(self._camera)

        # Controller
        self._controller = gfx.OrbitController(
            self._camera, register_events=self._renderer
        )
        self._interactiveMode = "rotate"

        # Background
        self._backgroundColor = (0.2, 0.2, 0.25, 1.0)
        bg = gfx.BackgroundMaterial(gfx.Color(*self._backgroundColor))
        self._background = gfx.Background(None, bg)
        self._scene.add(self._background)

        # Data group (items added here)
        self._dataGroup = gfx.Group()
        self._scene.add(self._dataGroup)

        # Viewport proxy for toolbar/action compatibility
        self.viewport = _ViewportProxy(self)

        # Connect render loop
        self._renderWidget.request_draw(self._animate)

    def _animate(self):
        """Render callback."""
        self._renderer.render(self._scene, self._camera)
        self._renderWidget.request_draw(self._animate)

    def _getSceneCenter(self):
        """Estimate the center of the scene data."""
        try:
            bbox = self._dataGroup.get_world_bounding_box()
            if bbox is not None:
                mn = numpy.array(bbox[0])
                mx = numpy.array(bbox[1])
                if numpy.all(numpy.isfinite(mn)) and numpy.all(numpy.isfinite(mx)):
                    return (mn + mx) / 2
        except (AttributeError, Exception):
            pass
        return numpy.array([0.0, 0.0, 0.0])

    # --- Background color ---

    def setBackgroundColor(self, color):
        """Set the background color.

        :param color: RGBA color
        """
        color = rgba(color)
        self._backgroundColor = color

        gfx = self._gfx
        self._background.material = gfx.BackgroundMaterial(gfx.Color(*color))

    def getBackgroundColor(self):
        """Return the background color.

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._backgroundColor)

    # --- Projection ---

    def setProjection(self, projection):
        """Set the projection mode.

        :param str projection: 'perspective' or 'orthographic'
        """
        gfx = self._gfx
        if projection == "orthographic" and self._projection != "orthographic":
            self._projection = "orthographic"
            self._camera = gfx.OrthographicCamera()
            self._controller = gfx.OrbitController(
                self._camera, register_events=self._renderer
            )
            self._scene.add(self._camera)
        elif projection == "perspective" and self._projection != "perspective":
            self._projection = "perspective"
            self._camera = gfx.PerspectiveCamera(fov=50)
            self._controller = gfx.OrbitController(
                self._camera, register_events=self._renderer
            )
            self._scene.add(self._camera)

    def getProjection(self):
        """Return the current projection mode.

        :rtype: str
        """
        return self._projection

    # --- Interactive mode ---

    def setInteractiveMode(self, mode):
        """Set the interactive mode.

        :param str mode: 'rotate', 'pan', or None
        """
        mode = mode if mode else "rotate"
        if mode != self._interactiveMode:
            self._interactiveMode = mode
            self.sigInteractiveModeChanged.emit()

    def getInteractiveMode(self):
        """Return the current interactive mode.

        :rtype: str
        """
        return self._interactiveMode

    # --- View control ---

    def centerScene(self):
        """Center the camera on the scene."""
        self._camera.show_object(self._scene)

    def resetZoom(self, face="front"):
        """Reset camera to a preset view.

        :param str face: The face to show ('front', 'back', etc.)
        """
        self._camera.show_object(self._scene)

    # --- Screenshot ---

    def grabGL(self):
        """Render the scene and return a QImage.

        :returns: RGBA image as QImage
        :rtype: QImage
        """
        try:
            snapshot = self._renderer.snapshot()
            arr = numpy.ascontiguousarray(numpy.asarray(snapshot))
            h, w = arr.shape[:2]
            image = qt.QImage(arr.data, w, h, w * 4, qt.QImage.Format_RGBA8888)
            # copy() to own the data (detach from numpy buffer)
            return image.copy()
        except (AttributeError, Exception):
            # No render has occurred yet
            return qt.QImage()

    # --- Device pixel ratio ---

    def getDevicePixelRatio(self):
        """Return the device pixel ratio.

        :rtype: float
        """
        return self.devicePixelRatioF()

    # --- Fog (no-op for pygfx) ---

    def setFogMode(self, mode):
        pass

    def getFogMode(self):
        return None

    # --- Light mode (no-op, always has lights) ---

    def setLightMode(self, mode):
        pass

    def getLightMode(self):
        return "directional"

    # --- Orientation indicator (no-op for now) ---

    def setOrientationIndicatorVisible(self, visible):
        pass

    def isOrientationIndicatorVisible(self):
        return False

    # --- Valid check ---

    def isValid(self):
        return True
