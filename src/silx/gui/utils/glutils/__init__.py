# /*##########################################################################
#
# Copyright (c) 2020-2023 European Synchrotron Radiation Facility
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
"""This module provides the :func:`isOpenGLAvailable` utility function.
"""
from __future__ import annotations


import os
import sys
import subprocess
from silx.gui import qt


class _isOpenGLAvailableResult:
    """Store result of checking OpenGL availability.

    It provides a `status` boolean attribute storing the result of the check and
    an `error` string attribute storting the possible error message.
    """

    def __init__(self, error: str = "", status: bool = False):
        self.__error = str(error)
        self.__status = bool(status)

    status = property(lambda self: self.__status, doc="True if OpenGL is working")
    error = property(lambda self: self.__error, doc="Error message")

    def __bool__(self):
        return self.status

    def __repr__(self):
        return f'<_isOpenGLAvailableResult: {self.status}, "{self.error}">'


def _runtimeOpenGLCheck(
    version: tuple[int, int],
    shareOpenGLContexts: bool,
) -> _isOpenGLAvailableResult:
    """Run OpenGL check in a subprocess.

    This is done by starting a subprocess that displays a Qt OpenGL widget.

    :param version:
        The minimal required OpenGL version as a 2-tuple (major, minor).
    :param shareOpenGLContexts:
        True to test the `QApplication` with `AA_ShareOpenGLContexts`.
    :return: Result status and error message
    """
    major, minor = str(version[0]), str(version[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([os.path.abspath(p) for p in sys.path])

    cmd = [sys.executable, "-s", "-S", __file__, major, minor]
    if shareOpenGLContexts:
        cmd.append("--shareOpenGLContexts")

    try:
        output = subprocess.check_output(cmd, env=env, timeout=2)
    except subprocess.TimeoutExpired:
        error = "Qt OpenGL widget hang"
        if sys.platform.startswith("linux"):
            error += ":\nIf connected remotely, GLX forwarding might be disabled."
        return _isOpenGLAvailableResult(error)
    except subprocess.CalledProcessError as e:
        return _isOpenGLAvailableResult(
            f"Qt OpenGL widget error: retcode={e.returncode}, error={e.output}"
        )

    return _isOpenGLAvailableResult(output.decode(), status=True)


_runtimeCheckCache = {}  # Cache runtime check results: {version: result}


def isOpenGLAvailable(
    version: tuple[int, int] = (2, 1),
    runtimeCheck: bool = True,
    shareOpenGLContexts: bool = False,
) -> _isOpenGLAvailableResult:
    """Check if OpenGL is available through Qt and actually working.

    After some basic tests, this is done by starting a subprocess that
    displays a Qt OpenGL widget.

    :param version:
        The minimal required OpenGL version as a 2-tuple (major, minor).
        Default: (2, 1)
    :param runtimeCheck:
        True (default) to run the test creating a Qt OpenGL widget in a subprocess,
        False to avoid this check.
    :param shareOpenGLContexts:
        True to test the `QApplication` with `AA_ShareOpenGLContexts`.
        This only can be checked with `runtimeCheck` enabled.
        Default is false.
    :return: A result object that evaluates to True if successful and
        which has a `status` boolean attribute (True if successful) and
        an `error` string attribute that is not empty if `status` is False.
    """
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY", ""):
        # On Linux and no DISPLAY available (e.g., ssh without -X)
        return _isOpenGLAvailableResult("DISPLAY environment variable not set")

    # Check pyopengl availability
    try:
        from silx.gui._glutils import gl
    except ImportError:
        return _isOpenGLAvailableResult(
            "Cannot import OpenGL wrapper: pyopengl is not installed"
        )

    # Pre checks for Qt < 5.4
    if not hasattr(qt, "QOpenGLWidget"):
        if not qt.HAS_OPENGL:
            return _isOpenGLAvailableResult(f"{qt.BINDING}.QtOpenGL not available")

        if (
            qt.BINDING == "PyQt5"
            and qt.QApplication.instance()
            and not qt.QGLFormat.hasOpenGL()
        ):
            # qt.QGLFormat.hasOpenGL MUST be called with a QApplication created
            # so this is only checked if the QApplication is already created
            return _isOpenGLAvailableResult("Qt reports OpenGL not available")

    # Check compatibility between Qt platform and pyopengl selected platform
    qt_qpa_platform = qt.QGuiApplication.platformName()
    pyopengl_platform = gl.getPlatform()
    if (qt_qpa_platform == "wayland" and pyopengl_platform != "EGLPlatform") or (
        qt_qpa_platform == "xcb" and pyopengl_platform != "GLXPlatform"
    ):
        return _isOpenGLAvailableResult(
            f"Qt platform '{qt_qpa_platform}' is not compatible with PyOpenGL platform '{pyopengl_platform}'"
        )

    keyCache = version, shareOpenGLContexts
    if keyCache in _runtimeCheckCache:  # Use cache
        return _runtimeCheckCache[keyCache]

    if not runtimeCheck:
        return _isOpenGLAvailableResult(status=True)

    # Run test in subprocess
    result = _runtimeOpenGLCheck(version, shareOpenGLContexts)
    _runtimeCheckCache[keyCache] = result
    return result


if __name__ == "__main__":
    from silx.gui._glutils import OpenGLWidget
    from silx.gui._glutils import gl
    import argparse

    class _TestOpenGLWidget(OpenGLWidget):
        """Widget checking that OpenGL is indeed available

        :param version: (major, minor) minimum OpenGL version
        """

        def __init__(self, version: tuple[int, int]):
            super(_TestOpenGLWidget, self).__init__(
                alphaBufferSize=0,
                depthBufferSize=0,
                stencilBufferSize=0,
                version=version,
            )

        def paintEvent(self, event):
            super(_TestOpenGLWidget, self).paintEvent(event)

            # Check once paint has been done
            app = qt.QApplication.instance()
            if not self.isValid():
                print("OpenGL widget is not valid")
                app.exit(1)
            else:
                qt.QTimer.singleShot(100, app.quit)

        def paintGL(self):
            gl.glClearColor(1.0, 0.0, 0.0, 0.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    parser = argparse.ArgumentParser()
    parser.add_argument("major")
    parser.add_argument("minor")
    parser.add_argument("--shareOpenGLContexts", action="store_true")

    args = parser.parse_args(args=sys.argv[1:])

    if args.shareOpenGLContexts:
        qt.QCoreApplication.setAttribute(qt.Qt.AA_ShareOpenGLContexts)
    app = qt.QApplication([])
    window = qt.QMainWindow(
        flags=qt.Qt.Popup
        | qt.Qt.FramelessWindowHint
        | qt.Qt.NoDropShadowWindowHint
        | qt.Qt.WindowStaysOnTopHint
    )
    window.setAttribute(qt.Qt.WA_ShowWithoutActivating)
    window.move(0, 0)
    window.resize(3, 3)
    widget = _TestOpenGLWidget(version=(args.major, args.minor))
    window.setCentralWidget(widget)
    window.setWindowOpacity(0.04)
    window.show()

    qt.QTimer.singleShot(1000, app.quit)
    sys.exit(app.exec())
