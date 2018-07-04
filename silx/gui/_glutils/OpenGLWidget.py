# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This package provides a compatibility layer for OpenGL widget.

It provides a compatibility layer for Qt OpenGL widget used in silx
across Qt<=5.3 QtOpenGL.QGLWidget and QOpenGLWidget.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "26/07/2017"


import logging
import sys

from .. import qt
from .._glutils import gl


_logger = logging.getLogger(__name__)


# Probe OpenGL availability and widget
ERROR = ''  # Error message from probing Qt OpenGL support
_BaseOpenGLWidget = None  # Qt OpenGL widget to use

if hasattr(qt, 'QOpenGLWidget'):  # PyQt>=5.4
    _logger.info('Using QOpenGLWidget')
    _BaseOpenGLWidget = qt.QOpenGLWidget

elif not qt.HAS_OPENGL:  # QtOpenGL not installed
    ERROR = '%s.QtOpenGL not available' % qt.BINDING

elif qt.QApplication.instance() and not qt.QGLFormat.hasOpenGL():
    # qt.QGLFormat.hasOpenGL MUST be called with a QApplication created
    # so this is only checked if the QApplication is already created
    ERROR = 'Qt reports OpenGL not available'

else:
    _logger.info('Using QGLWidget')
    _BaseOpenGLWidget = qt.QGLWidget


# Internal class wrapping Qt OpenGL widget
if _BaseOpenGLWidget is None:
    _logger.error('OpenGL-based widget disabled: %s', ERROR)
    _OpenGLWidget = None

else:
    class _OpenGLWidget(_BaseOpenGLWidget):
        """Wrapper over QOpenGLWidget and QGLWidget"""

        sigOpenGLContextError = qt.Signal(str)
        """Signal emitted when an OpenGL context error is detected at runtime.

        It provides the error reason as a str.
        """

        def __init__(self, parent,
                     alphaBufferSize=0,
                     depthBufferSize=24,
                     stencilBufferSize=8,
                     version=(2, 0),
                     f=qt.Qt.WindowFlags()):
            # True if using QGLWidget, False if using QOpenGLWidget
            self.__legacy = not hasattr(qt, 'QOpenGLWidget')

            self.__devicePixelRatio = 1.0
            self.__requestedOpenGLVersion = int(version[0]), int(version[1])
            self.__isValid = False

            if self.__legacy:  # QGLWidget
                format_ = qt.QGLFormat()
                format_.setAlphaBufferSize(alphaBufferSize)
                format_.setAlpha(alphaBufferSize != 0)
                format_.setDepthBufferSize(depthBufferSize)
                format_.setDepth(depthBufferSize != 0)
                format_.setStencilBufferSize(stencilBufferSize)
                format_.setStencil(stencilBufferSize != 0)
                format_.setVersion(*self.__requestedOpenGLVersion)
                format_.setDoubleBuffer(True)

                super(_OpenGLWidget, self).__init__(format_, parent, None, f)

            else:  # QOpenGLWidget
                super(_OpenGLWidget, self).__init__(parent, f)

                format_ = qt.QSurfaceFormat()
                format_.setAlphaBufferSize(alphaBufferSize)
                format_.setDepthBufferSize(depthBufferSize)
                format_.setStencilBufferSize(stencilBufferSize)
                format_.setVersion(*self.__requestedOpenGLVersion)
                format_.setSwapBehavior(qt.QSurfaceFormat.DoubleBuffer)
                self.setFormat(format_)

            # Enable receiving mouse move events when no buttons are pressed
            self.setMouseTracking(True)


        def getDevicePixelRatio(self):
            """Returns the ratio device-independent / device pixel size

            It should be either 1.0 or 2.0.

            :return: Scale factor between screen and Qt units
            :rtype: float
            """
            return self.__devicePixelRatio

        def getRequestedOpenGLVersion(self):
            """Returns the requested OpenGL version.

            :return: (major, minor)
            :rtype: 2-tuple of int"""
            return self.__requestedOpenGLVersion

        def getOpenGLVersion(self):
            """Returns the available OpenGL version.

            :return: (major, minor)
            :rtype: 2-tuple of int"""
            if self.__legacy:  # QGLWidget
                supportedVersion = 0, 0

                # Go through all OpenGL version flags checking support
                flags = self.format().openGLVersionFlags()
                for version in ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                                (2, 0), (2, 1),
                                (3, 0), (3, 1), (3, 2), (3, 3),
                                (4, 0)):
                    versionFlag = getattr(qt.QGLFormat,
                                          'OpenGL_Version_%d_%d' % version)
                    if not versionFlag & flags:
                        break
                    supportedVersion = version
                return supportedVersion

            else:  # QOpenGLWidget
                return self.format().version()

        # QOpenGLWidget methods

        def isValid(self):
            """Returns True if OpenGL is available.

            This adds extra checks to Qt isValid method.

            :rtype: bool
            """
            return self.__isValid and super(_OpenGLWidget, self).isValid()

        def defaultFramebufferObject(self):
            """Returns the framebuffer object handle.

            See :meth:`QOpenGLWidget.defaultFramebufferObject`
            """
            if self.__legacy:  # QGLWidget
                return 0
            else:  # QOpenGLWidget
                return super(_OpenGLWidget, self).defaultFramebufferObject()

        # *GL overridden methods

        def initializeGL(self):
            parent = self.parent()
            if parent is None:
                _logger.error('_OpenGLWidget has no parent')
                return

            # Check OpenGL version
            if self.getOpenGLVersion() >= self.getRequestedOpenGLVersion():
                version = gl.glGetString(gl.GL_VERSION)
                if version:
                    self.__isValid = True
                else:
                    errMsg = 'OpenGL not available'
                    if sys.platform.startswith('linux'):
                        errMsg += ': If connected remotely, ' \
                                  'GLX forwarding might be disabled.'
                    _logger.error(errMsg)
                    self.sigOpenGLContextError.emit(errMsg)
                    self.__isValid = False

            else:
                errMsg = 'OpenGL %d.%d not available' % \
                         self.getRequestedOpenGLVersion()
                _logger.error('OpenGL widget disabled: %s', errMsg)
                self.sigOpenGLContextError.emit(errMsg)
                self.__isValid = False

            if self.isValid():
                parent.initializeGL()

        def paintGL(self):
            parent = self.parent()
            if parent is None:
                _logger.error('_OpenGLWidget has no parent')
                return

            if qt.BINDING in ('PyQt5', 'PySide2'):
                devicePixelRatio = self.window().windowHandle().devicePixelRatio()

                if devicePixelRatio != self.getDevicePixelRatio():
                    # Update devicePixelRatio and call resizeOpenGL
                    # as resizeGL is not always called.
                    self.__devicePixelRatio = devicePixelRatio
                    self.makeCurrent()
                    parent.resizeGL(self.width(), self.height())

            if self.isValid():
                parent.paintGL()

        def resizeGL(self, width, height):
            parent = self.parent()
            if parent is None:
                _logger.error('_OpenGLWidget has no parent')
                return

            if self.isValid():
                # Call parent resizeGL with device-independent pixel unit
                # This works over both QGLWidget and QOpenGLWidget
                parent.resizeGL(self.width(), self.height())


class OpenGLWidget(qt.QWidget):
    """OpenGL widget wrapper over QGLWidget and QOpenGLWidget

    This wrapper API implements a subset of QOpenGLWidget API.
    The constructor takes a different set of arguments.
    Methods returning object like :meth:`context` returns either
    QGL* or QOpenGL* objects.

    :param parent: Parent widget see :class:`QWidget`
    :param int alphaBufferSize:
        Size in bits of the alpha channel (default: 0).
        Set to 0 to disable alpha channel.
    :param int depthBufferSize:
        Size in bits of the depth buffer (default: 24).
        Set to 0 to disable depth buffer.
    :param int stencilBufferSize:
        Size in bits of the stencil buffer (default: 8).
        Set to 0 to disable stencil buffer
    :param version: Requested OpenGL version (default: (2, 0)).
    :type version: 2-tuple of int
    :param f: see :class:`QWidget`
    """

    def __init__(self, parent=None,
                 alphaBufferSize=0,
                 depthBufferSize=24,
                 stencilBufferSize=8,
                 version=(2, 0),
                 f=qt.Qt.WindowFlags()):
        super(OpenGLWidget, self).__init__(parent, f)

        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        if _OpenGLWidget is None:
            self.__openGLWidget = None
            label = self._createErrorQLabel(ERROR)
            self.layout().addWidget(label)

        else:
            self.__openGLWidget = _OpenGLWidget(
                parent=self,
                alphaBufferSize=alphaBufferSize,
                depthBufferSize=depthBufferSize,
                stencilBufferSize=stencilBufferSize,
                version=version,
                f=f)
            # Async connection need, otherwise issue when hiding OpenGL
            # widget while doing the rendering..
            self.__openGLWidget.sigOpenGLContextError.connect(
                self._handleOpenGLInitError, qt.Qt.QueuedConnection)
            self.layout().addWidget(self.__openGLWidget)

    @staticmethod
    def _createErrorQLabel(error):
        """Create QLabel displaying error message in place of OpenGL widget

        :param str error: The error message to display"""
        label = qt.QLabel()
        label.setText('OpenGL-based widget disabled:\n%s' % error)
        label.setAlignment(qt.Qt.AlignCenter)
        label.setWordWrap(True)
        return label

    def _handleOpenGLInitError(self, error):
        """Handle runtime errors in OpenGL widget"""
        if self.__openGLWidget is not None:
            self.__openGLWidget.setVisible(False)
            self.__openGLWidget.setParent(None)
            self.__openGLWidget = None

            label = self._createErrorQLabel(error)
            self.layout().addWidget(label)

    # Additional API

    def getDevicePixelRatio(self):
        """Returns the ratio device-independent / device pixel size

        It should be either 1.0 or 2.0.

        :return: Scale factor between screen and Qt units
        :rtype: float
        """
        if self.__openGLWidget is None:
            return 1.
        else:
            return self.__openGLWidget.getDevicePixelRatio()

    def getOpenGLVersion(self):
        """Returns the available OpenGL version.

        :return: (major, minor)
        :rtype: 2-tuple of int"""
        if self.__openGLWidget is None:
            return 0, 0
        else:
            return self.__openGLWidget.getOpenGLVersion()

    # QOpenGLWidget API

    def isValid(self):
        """Returns True if OpenGL with the requested version is available.

        :rtype: bool
        """
        if self.__openGLWidget is None:
            return False
        else:
            return self.__openGLWidget.isValid()

    def context(self):
        """Return Qt OpenGL context object or None.

        See :meth:`QOpenGLWidget.context` and :meth:`QGLWidget.context`
        """
        if self.__openGLWidget is None:
            return None
        else:
            return self.__openGLWidget.context()

    def defaultFramebufferObject(self):
        """Returns the framebuffer object handle.

        See :meth:`QOpenGLWidget.defaultFramebufferObject`
        """
        if self.__openGLWidget is None:
            return 0
        else:
            return self.__openGLWidget.defaultFramebufferObject()

    def makeCurrent(self):
        """Make the underlying OpenGL widget's context current.

        See :meth:`QOpenGLWidget.makeCurrent`
        """
        if self.__openGLWidget is not None:
            self.__openGLWidget.makeCurrent()

    def update(self):
        """Async update of the OpenGL widget.

        See :meth:`QOpenGLWidget.update`
        """
        if self.__openGLWidget is not None:
            self.__openGLWidget.update()

    # QOpenGLWidget API to override

    def initializeGL(self):
        """Override to implement OpenGL initialization."""
        pass

    def paintGL(self):
        """Override to implement OpenGL rendering."""
        pass

    def resizeGL(self, width, height):
        """Override to implement resize of OpenGL framebuffer.

        :param int width: Width in device-independent pixels
        :param int height: Height in device-independent pixels
        """
        pass
