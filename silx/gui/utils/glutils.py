# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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

import os
import sys
import subprocess
from silx.gui import qt


class _isOpenGLAvailableResult:
    """Store result of checking OpenGL availability.

    It provides a `status` boolean attribute storing the result of the check and
    an `error` string attribute storting the possible error message.
    """

    def __init__(self, status=True, error=''):
        self.__status = bool(status)
        self.__error = str(error)

    status = property(lambda self: self.__status, doc="True if OpenGL is working")
    error = property(lambda self: self.__error, doc="Error message")

    def __bool__(self):
        return self.status

    def __repr__(self):
        return '<_isOpenGLAvailableResult: %s, "%s">' % (self.status, self.error)


def _runtimeOpenGLCheck(version):
    """Run OpenGL check in a subprocess.

    This is done by starting a subprocess that displays a Qt OpenGL widget.

    :param List[int] version:
        The minimal required OpenGL version as a 2-tuple (major, minor).
        Default: (2, 1)
    :return: An error string that is empty if no error occured
    :rtype: str
    """
    major, minor = str(version[0]), str(version[1])
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [os.path.abspath(p) for p in sys.path])

    try:
        error = subprocess.check_output(
            [sys.executable, __file__, major, minor],
            env=env,
            timeout=2)
    except subprocess.TimeoutExpired:
        status = False
        error = "Qt OpenGL widget hang"
        if sys.platform.startswith('linux'):
            error += ':\nIf connected remotely, GLX forwarding might be disabled.'
    except subprocess.CalledProcessError as e:
        status = False
        error = "Qt OpenGL widget error: retcode=%d, error=%s" % (e.returncode, e.output)
    else:
        status = True
        error = error.decode()
    return _isOpenGLAvailableResult(status, error)


_runtimeCheckCache = {}  # Cache runtime check results: {version: result}


def isOpenGLAvailable(version=(2, 1), runtimeCheck=True):
    """Check if OpenGL is available through Qt and actually working.

    After some basic tests, this is done by starting a subprocess that
    displays a Qt OpenGL widget.

    :param List[int] version:
        The minimal required OpenGL version as a 2-tuple (major, minor).
        Default: (2, 1)
    :param bool runtimeCheck:
        True (default) to run the test creating a Qt OpenGL widgt in a subprocess,
        False to avoid this check.
    :return: A result object that evaluates to True if successful and
        which has a `status` boolean attribute (True if successful) and
        an `error` string attribute that is not empty if `status` is False.
    """
    error = ''

    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
        # On Linux and no DISPLAY available (e.g., ssh without -X)
        error = 'DISPLAY environment variable not set'

    else:
        # Check pyopengl availability
        try:
            import silx.gui._glutils.gl  # noqa
        except ImportError:
            error = "Cannot import OpenGL wrapper: pyopengl is not installed"
        else:
            # Pre checks for Qt < 5.4
            if not hasattr(qt, 'QOpenGLWidget'):
                if not qt.HAS_OPENGL:
                    error = '%s.QtOpenGL not available' % qt.BINDING

                elif qt.QApplication.instance() and not qt.QGLFormat.hasOpenGL():
                    # qt.QGLFormat.hasOpenGL MUST be called with a QApplication created
                    # so this is only checked if the QApplication is already created
                    error = 'Qt reports OpenGL not available'

    result = _isOpenGLAvailableResult(error == '', error)

    if result:  # No error so far, runtime check
        if version in _runtimeCheckCache:  # Use cache
            result = _runtimeCheckCache[version]
        elif runtimeCheck:  # Run test in subprocess
            result = _runtimeOpenGLCheck(version)
            _runtimeCheckCache[version] = result

    return result


if __name__ == "__main__":
    from silx.gui._glutils import OpenGLWidget
    from silx.gui._glutils import gl
    import argparse

    class _TestOpenGLWidget(OpenGLWidget):
        """Widget checking that OpenGL is indeed available

        :param List[int] version: (major, minor) minimum OpenGL version
        """

        def __init__(self, version):
            super(_TestOpenGLWidget, self).__init__(
                alphaBufferSize=0,
                depthBufferSize=0,
                stencilBufferSize=0,
                version=version)

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
            gl.glClearColor(1., 0., 0., 0.)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)


    parser = argparse.ArgumentParser()
    parser.add_argument('major')
    parser.add_argument('minor')

    args = parser.parse_args(args=sys.argv[1:])

    app = qt.QApplication([])
    window = qt.QMainWindow(flags=
        qt.Qt.Window |
        qt.Qt.FramelessWindowHint |
        qt.Qt.NoDropShadowWindowHint |
        qt.Qt.WindowStaysOnTopHint)
    window.setAttribute(qt.Qt.WA_ShowWithoutActivating)
    window.move(0, 0)
    window.resize(3, 3)
    widget = _TestOpenGLWidget(version=(args.major, args.minor))
    window.setCentralWidget(widget)
    window.setWindowOpacity(0.04)
    window.show()

    qt.QTimer.singleShot(1000, app.quit)
    sys.exit(app.exec_())
