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
__date__ = "09/05/2017"


import logging

from .. import qt
from . import gl


_logger = logging.getLogger(__name__)


# TODO different modes for message box, set color of clear...
# TODO pop-up once when using fallback QWidget


if qt.BINDING == 'PyQt5' and hasattr(qt, 'QOpenGLWidget'):
    # PyQt>=5.4
    _logger.info('Using QOpenGLWidget')
    _BaseOpenGLWidget = qt.QOpenGLWidget
    _BASE_WIDGET = 'QOpenGLWidget'

elif qt.HAS_OPENGL and (
        not qt.QApplication.instance() or qt.QGLFormat.hasOpenGL()):
    # Using QtOpenGL.QGLwidget
    # qt.QGLFormat.hasOpenGL MUST be called with a QApplication created
    # so this is only checked if the QApplication is already created
    _logger.info('Using QGLWidget')
    _BaseOpenGLWidget = qt.QGLWidget
    _BASE_WIDGET = 'QGLWidget'

else: # No OpenGL widget available, fallback to a dummy widget
    if not qt.HAS_OPENGL:
         _logger.error(
            'QtOpenGL is not available: OpenGL-based widget disabled')
    elif qt.QApplication.instance() and not qt.QGLFormat.hasOpenGL():
        # qt.QGLFormat.hasOpenGL MUST be called with a QApplication created
        # so this is only checked if the QApplication is already created
        _logger.error(
            'OpenGL is not available: OpenGL-based widget disabled')
    else:
        _logger.error('OpenGL-based widget disabled')

    _logger.info('Using QWidget')
    _BaseOpenGLWidget = qt.QWidget
    _BASE_WIDGET = ''


class OpenGLWidget(_BaseOpenGLWidget):
    """OpenGL widget wrapper over QGLWidget and QOpenGLWidget

    This wrapper API follows QOpenGLWidget API as much as possible.
    The constructor takes a different set of arguments.
    Methods to override to implement rendering are named differently:

    - :meth:`initializeOpenGL` instead of :meth:`initializeGL`,
    - :meth:`paintOpenGL` instead of :meth:`paintGL` and
    - :meth:`resizeOpenGL` instead of :meth:`resizeGL`.

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

    BASE_WIDGET = _BASE_WIDGET
    """Name of the underlying OpenGL widget"""

    # Only display no OpenGL pop-up once for all widgets.
    _noOpenGLErrorMessageDisplayed = False

    def __init__(self, parent=None,
                 alphaBufferSize=0,
                 depthBufferSize=24,
                 stencilBufferSize=8,
                 version=(2, 0),
                 f=qt.Qt.WindowFlags()):
        self.__devicePixelRatio = 1.0
        self.__requestedOpenGLVersion = tuple(version)
        self.__requestedOpenGLVersionAvailable = False

        if self.BASE_WIDGET == 'QOpenGLWidget':
            super(OpenGLWidget, self).__init__(parent, f)

            format_ = qt.QSurfaceFormat()
            format_.setAlphaBufferSize(alphaBufferSize)
            format_.setDepthBufferSize(depthBufferSize)
            format_.setStencilBufferSize(stencilBufferSize)
            format_.setVersion(*self.__requestedOpenGLVersion)
            format_.setSwapBehavior(qt.QSurfaceFormat.DoubleBuffer)
            self.setFormat(format_)

        elif self.BASE_WIDGET == 'QGLWidget':
            format_ = qt.QGLFormat()
            format_.setAlphaBufferSize(alphaBufferSize)
            format_.setAlpha(alphaBufferSize != 0)
            format_.setDepthBufferSize(depthBufferSize)
            format_.setDepth(depthBufferSize != 0)
            format_.setStencilBufferSize(stencilBufferSize)
            format_.setStencil(stencilBufferSize != 0)
            format_.setVersion(*self.__requestedOpenGLVersion)
            format_.setDoubleBuffer(True)

            super(OpenGLWidget, self).__init__(format_, parent, None, f)
        else:  # Fallback
            super(OpenGLWidget, self).__init__(parent, f)

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

    def isRequestedOpenGLVersionAvailable(self):
        """Returns True if requested OpenGL version is available.

        :rtype: bool
        """
        return self.__requestedOpenGLVersionAvailable

    def getOpenGLVersion(self):
        """Returns the available OpenGL version.

        :return: (major, minor)
        :rtype: 2-tuple of int"""
        if self.BASE_WIDGET == 'QOpenGLWidget':
            return self.format().version()
        elif self.BASE_WIDGET == 'QGLWidget':
            supportedVersion = 0, 0

            # Go through all OpenGL version flags checking support
            flags = self.format().openGLVersionFlags()
            for version in ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                            (2, 0), (2, 1),
                            (3, 0), (3, 1), (3, 2), (3, 3),
                            (4, 0)):
                versionFlag = getattr(qt.QGLFormat, 'OpenGL_Version_%d_%d' % version)
                if not versionFlag & flags:
                    break
                supportedVersion = version
            return supportedVersion
        else:
            return 0, 0

    def defaultFramebufferObject(self):
        """Returns the framebuffer object handle.

        See :meth:`QOpenGLWidget.defaultFramebufferObject`
        """
        if self.BASE_WIDGET == 'QOpenGLWidget':
            return super(OpenGLWidget, self).defaultFramebufferObject()
        elif self.BASE_WIDGET == 'QGLWidget':
            return 0
        else:
            return 0

    # Method useful for no QtOpenGL widgets

    def showEvent(self, event):
        """Handle show events for the no OpenGL fallback to display an error
        """
        if not self.BASE_WIDGET and not self._noOpenGLErrorMessageDisplayed:
            self.__class__._noOpenGLErrorMessageDisplayed = True
            messageBox = qt.QMessageBox(parent=self)
            messageBox.setIcon(qt.QMessageBox.Critical)
            messageBox.setWindowTitle('Error')
            messageBox.setText('OpenGL widgets disabled.\n\n'
                               'Reason: QtOpenGL widgets not available.')
            messageBox.addButton(qt.QMessageBox.Ok)
            messageBox.setWindowModality(qt.Qt.WindowModal)
            messageBox.setAttribute(qt.Qt.WA_DeleteOnClose)
            messageBox.show()

    def makeCurrent(self):
        """Make this widget's OpenGL context current.

        See :meth:`QopenGLWidget.makeCurrent`
        """
        # Here to provide a fallback in case OpenGL widget is not available
        if self.BASE_WIDGET:
            return super(OpenGLWidget, self).makeCurrent()

    # Implementation of *GL methods

    def initializeGL(self):
        # Check OpenGL version
        self.__requestedOpenGLVersionAvailable = \
            self.getOpenGLVersion() >= self.getRequestedOpenGLVersion()

        if not self.isRequestedOpenGLVersionAvailable():
            _logger.error(
                'OpenGL widget disabled: OpenGL %d.%d not available' %
                self.getRequestedOpenGLVersion())

            if not self._noOpenGLErrorMessageDisplayed:
                self.__class__._noOpenGLErrorMessageDisplayed = True
                messageBox = qt.QMessageBox(parent=self)
                messageBox.setIcon(qt.QMessageBox.Critical)
                messageBox.setWindowTitle('Error')
                messageBox.setText('OpenGL widgets disabled.\n\n'
                                   'Reason: OpenGL %d.%d is not available.' %
                                   self.getRequestedOpenGLVersion())
                messageBox.addButton(qt.QMessageBox.Ok)
                messageBox.setWindowModality(qt.Qt.WindowModal)
                messageBox.setAttribute(qt.Qt.WA_DeleteOnClose)
                messageBox.show()

        else:
            self.initializeOpenGL()

    def paintGL(self):
        if qt.BINDING == 'PyQt5':
            devicePixelRatio = self.window().windowHandle().devicePixelRatio()

            if devicePixelRatio != self.getDevicePixelRatio():
                # Update devicePixelRatio and call resizeOpenGL
                # as resizeGL is not always called.
                self.__devicePixelRatio = devicePixelRatio
                self.makeCurrent()
                self.resizeOpenGL(self.width(), self.height())

        if not self.isRequestedOpenGLVersionAvailable():
            # Requested OpenGL version not available, just clear the color buffer.
            gl.glViewport(0,
                          0,
                          int(self.width() * self.getDevicePixelRatio()),
                          int(self.height() * self.getDevicePixelRatio()))

            gl.glClearColor(0., 0., 0., 1.)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        else:
            self.paintOpenGL()

    def resizeGL(self, width, height):
        if self.isRequestedOpenGLVersionAvailable():
            # Call resizeOpenGL with device-independent pixel unit
            # This works over both QGLWidget and QOpenGLWidget
            self.resizeOpenGL(self.width(), self.height())

    # API to override, replacing *GL methods

    def initializeOpenGL(self):
        """Override to implement equivalent of initializeGL."""
        pass

    def paintOpenGL(self):
        """Override to implement equivalent of paintGL."""
        pass

    def resizeOpenGL(self, width, height):
        """Override to implement equivalent of resizeGL.

        :param int width: Width in device-independent pixels
        :param int height: Height in device-independent pixels
        """
        pass
