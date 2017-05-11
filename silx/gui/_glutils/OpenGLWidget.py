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
from .._glutils import gl


_logger = logging.getLogger(__name__)


# TODO setFormat
# TODO convert openGLVersionFlags to version
# TODO split _OpenGLWidgetBase and OpenGL2.1 specific stuff
# TODO different modes for message box, set color of clear...
# TODO handle OpenGLWidget = None: raise exception or handle elsewhere...
class _OpenGLWidgetBase(object):
    """Base class for OpenGL widget wrapper over QGLWidget and QOpenGLWidget

    This wrapper API follows QOpenGLWidget API as much as possible.
    Methods to override to implement rendering are named differently:

    - :meth:`initializeOpenGL` instead of :meth:`initializeGL`,
    - :meth:`paintOpenGL` instead of :meth:`paintGL` and
    - :meth:`resizeOpenGL` instead of :meth:`resizeGL`.
    """

    def __init__(self):
        self._devicePixelRatio = 1.0
        self._isOpenGL21 = False

    def getDevicePixelRatio(self):
        """Returns the ratio device-independent / device pixel size

        It should be either 1.0 or 2.0.

        :return: Scale factor between screen and Qt units
        :rtype: float
        """
        return self._devicePixelRatio

    def _updateDevicePixelRatio(self):
        """Run in :meth:`paintGL` to update devicePixelRatio value.
        """
        pass

    def isOpenGL2_1(self):
        """Returns whether OpenGL 2.1 is available or not.

        This is only valid after the OpenGL context has been initialised.

        :return: True if OpenGL2.1 is supported, False otherwise
        :rtype: bool
        """
        return self._isOpenGL21

    def _checkOpenGL2_1(self):
        """Override to implement the check of OpenGL version.

         This is run in :meth:`initializeGL`.

        :return: True if OpenGL 2.1 is available, False otherwise
        :rtype: bool
        """
        return False

    # Implementation of *GL methods

    def initializeGL(self):
        # Check if OpenGL2.1 is available
        self._isOpenGL21 = self._checkOpenGL2_1()

        if not self.isOpenGL2_1():
            _logger.error(
                'OpenGL widget disabled: OpenGL 2.1 not available')

            messageBox = qt.QMessageBox(parent=self)
            messageBox.setIcon(qt.QMessageBox.Critical)
            messageBox.setWindowTitle('Error')
            messageBox.setText('OpenGL widget disabled.\n\n'
                               'Reason: OpenGL 2.1 is not available.')
            messageBox.addButton(qt.QMessageBox.Ok)
            messageBox.setWindowModality(qt.Qt.WindowModal)
            messageBox.setAttribute(qt.Qt.WA_DeleteOnClose)
            messageBox.show()

        self.initializeOpenGL()

    def paintGL(self):
        self._updateDevicePixelRatio()

        if not self.isOpenGL2_1():
            # Cannot render scene, just clear the color buffer.
            gl.glViewport(0,
                          0,
                          self.width() * self.getDevicePixelRatio(),
                          self.height() * self.getDevicePixelRatio())

            gl.glClearColor(0., 0., 0., 1.)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        else:
            self.paintOpenGL()

    def resizeGL(self, width, height):
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


class _OpenGLWidgetQt5(_OpenGLWidgetBase):
    """Base class for OpenGL widget wrapper for PyQt5"""

    def _updateDevicePixelRatio(self):
        devicePixelRatio = self.context().screen().devicePixelRatio()
        if devicePixelRatio != self.getDevicePixelRatio():
            # Update devicePixelRatio and call resizeOpenGL
            # as resizeGL is not always called.
            self._devicePixelRatio = devicePixelRatio
            self.makeCurrent()
            self.resizeOpenGL(self.width(), self.height())


if qt.BINDING == 'PyQt5' and hasattr(qt, 'QOpenGLWidget'):  # PyQt>=5.4
    class OpenGLWidget(qt.QOpenGLWidget, _OpenGLWidgetQt5):

        def __init__(self, parent=None, f=qt.Qt.WindowFlags()):
            _OpenGLWidgetQt5.__init__(self)
            qt.QOpenGLWidget.__init__(self, parent, f)

        def _checkOpenGL2_1(self):
            return self.format().version() >= (2, 1)

elif qt.HAS_OPENGL:  # Using QtOpenGL.QGLwidget

    if not qt.QGLFormat.hasOpenGL():  # Check if any OpenGL is available
        _logger.error(
            'OpenGL is not available on this platform')
        OpenGLWidget = None

    else:
        class _QGLWidget(qt.QGLWidget):
            """Class with QGLWidget method shared for Qt4 and Qt5"""
            def _checkOpenGL2_1(self):
                versionFlags = self.format().openGLVersionFlags()
                return bool(versionFlags & qt.QGLFormat.OpenGL_Version_2_1)

            def defaultFramebufferObject(self):
                """Returns the framebuffer object handle = 0

                Compatibility with QOpenGLWidget
                """
                return 0

        if qt.BINDING == 'PyQt5':
            class OpenGLWidget(_QGLWidget, _OpenGLWidgetQt5):
                def __init__(self, parent=None, f=qt.Qt.WindowFlags()):
                    _OpenGLWidgetQt5.__init__(self)
                    _QGLWidget.__init__(self, parent, None, f)

        else:  # Qt4
            class OpenGLWidget(_QGLWidget, _OpenGLWidgetBase):
                def __init__(self, parent=None, f=qt.Qt.WindowFlags()):
                    _OpenGLWidgetBase.__init__(self)
                    _QGLWidget.__init__(self, parent, None, f)

else:
    _logger.error('QtOpenGL is not available!')
    OpenGLWidget = None


# Set docstring
if OpenGLWidget is not None:
    OpenGLWidget.__doc__ = _OpenGLWidgetBase.__doc__
