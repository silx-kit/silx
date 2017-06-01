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
import numpy

from .. import qt
from .. import _utils

from . import gl


_logger = logging.getLogger(__name__)



if qt.BINDING == 'PyQt5' and hasattr(qt, 'QOpenGLWidget'):
    # PyQt>=5.4
    _logger.info('Using QOpenGLWidget')
    _BaseOpenGLWidget = qt.QOpenGLWidget
    _BASE_WIDGET = 'QOpenGLWidget'
    _ERROR_MSG = ''

elif qt.HAS_OPENGL and (
        not qt.QApplication.instance() or qt.QGLFormat.hasOpenGL()):
    # Using QtOpenGL.QGLwidget
    # qt.QGLFormat.hasOpenGL MUST be called with a QApplication created
    # so this is only checked if the QApplication is already created
    _logger.info('Using QGLWidget')
    _BaseOpenGLWidget = qt.QGLWidget
    _BASE_WIDGET = 'QGLWidget'
    _ERROR_MSG = ''

else: # No OpenGL widget available, fallback to a dummy widget
    _ERROR_MSG = 'OpenGL-based widget disabled'
    if not qt.HAS_OPENGL:
        _ERROR_MSG += ':\n%s.QtOpenGL not available' % qt.BINDING
    elif qt.QApplication.instance() and not qt.QGLFormat.hasOpenGL():
        # qt.QGLFormat.hasOpenGL MUST be called with a QApplication created
        # so this is only checked if the QApplication is already created
        _ERROR_MSG += ':\nOpenGL not available'

    _logger.error(_ERROR_MSG)
    _logger.info('Using QLabel')
    _BaseOpenGLWidget = qt.QLabel
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

    def __init__(self, parent=None,
                 alphaBufferSize=0,
                 depthBufferSize=24,
                 stencilBufferSize=8,
                 version=(2, 0),
                 f=qt.Qt.WindowFlags()):
        self.__devicePixelRatio = 1.0
        self.__requestedOpenGLVersion = tuple(version)
        self.__requestedOpenGLVersionAvailable = False
        self.__errorImage = None

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
            self.setText(_ERROR_MSG)
            self.setAlignment(qt.Qt.AlignCenter)
            self.setWordWrap(True)

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

        if self.isRequestedOpenGLVersionAvailable():
            self.initializeOpenGL()
        else:
            _logger.error(
                'OpenGL widget disabled: OpenGL %d.%d not available' %
                self.getRequestedOpenGLVersion())

    def paintGL(self):
        if qt.BINDING == 'PyQt5':
            devicePixelRatio = self.window().windowHandle().devicePixelRatio()

            if devicePixelRatio != self.getDevicePixelRatio():
                # Update devicePixelRatio and call resizeOpenGL
                # as resizeGL is not always called.
                self.__devicePixelRatio = devicePixelRatio
                self.makeCurrent()
                self.resizeOpenGL(self.width(), self.height())

        if self.isRequestedOpenGLVersionAvailable():
            self.paintOpenGL()
        else:
            # Requested OpenGL version not available.
            gl.glViewport(0,
                          0,
                          int(self.width() * self.getDevicePixelRatio()),
                          int(self.height() * self.getDevicePixelRatio()))

            bgColor = self.palette().color(qt.QPalette.Window)
            gl.glClearColor(bgColor.redF(),
                            bgColor.greenF(),
                            bgColor.blueF(),
                            bgColor.alphaF())
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            if self.width() != 0 and self.height() != 0:
                if self.__errorImage is None:  # Update background image
                    devicePixelRatio = self.getDevicePixelRatio()

                    image = qt.QImage(
                        self.width() * devicePixelRatio,
                        self.height() * devicePixelRatio,
                        qt.QImage.Format_RGB32)
                    image.fill(self.palette().color(qt.QPalette.Window))
                    if hasattr(image, 'setDevicePixelRatio'):  # Qt5
                        image.setDevicePixelRatio(devicePixelRatio)

                    painter = qt.QPainter()
                    painter.begin(image)
                    painter.setPen(self.palette().color(qt.QPalette.WindowText))
                    painter.setFont(self.font())
                    painter.drawText(0, 0, self.width(), self.height(),
                                     qt.Qt.AlignCenter | qt.Qt.TextWordWrap,
                                     'OpenGL-based widget disabled:\n'
                                     'OpenGL %d.%d is not available.' %
                                     self.getRequestedOpenGLVersion())
                    painter.end()

                    self.__errorImage = numpy.flipud(
                        _utils.convertQImageToArray(image))

                height, width = self.__errorImage.shape[:2]
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                gl.glRasterPos2f(-1., -1.)
                gl.glDrawPixels(width,
                                height,
                                gl.GL_RGB,
                                gl.GL_UNSIGNED_BYTE,
                                self.__errorImage)

    def resizeGL(self, width, height):
        if self.isRequestedOpenGLVersionAvailable():
            # Call resizeOpenGL with device-independent pixel unit
            # This works over both QGLWidget and QOpenGLWidget
            self.resizeOpenGL(self.width(), self.height())
        else:
            self.__errorImage = None  # Dirty flag the error image

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
