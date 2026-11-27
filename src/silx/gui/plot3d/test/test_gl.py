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
# ###########################################################################*/
"""Test OpenGL"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "10/08/2017"


import logging
import pytest

from silx.gui._glutils import gl, OpenGLWidget
from silx.gui.utils.testutils import TestCaseQt
from silx.gui import qt

_logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("use_opengl")
class TestOpenGL(TestCaseQt):
    """Tests of OpenGL widget."""

    class OpenGLWidgetLogger(OpenGLWidget):
        """Widget logging information of available OpenGL version"""

        def __init__(self):
            self._dump = False
            super().__init__(version=(1, 0))

        def paintOpenGL(self):
            """Perform the rendering and logging"""
            if not self._dump:
                self._dump = True
                _logger.info("OpenGL info:")
                _logger.info(
                    "\tQt OpenGL context version: %d.%d", *self.getOpenGLVersion()
                )
                _logger.info("\tGL_VERSION: %s" % gl.glGetString(gl.GL_VERSION))
                _logger.info(
                    "\tGL_SHADING_LANGUAGE_VERSION: %s"
                    % gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
                )
                _logger.debug("\tGL_EXTENSIONS: %s" % gl.glGetString(gl.GL_EXTENSIONS))

            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    class MouseMoveLogger(OpenGLWidget):
        """Widget logging the position of mouse move events"""

        def __init__(self):
            super().__init__()
            self.setMouseTracking(True)
            self.positions = []

        def mouseMoveEvent(self, event):
            self.positions.append(qt.getMouseEventPosition(event))

    def testMousePosition(self):
        """Mouse events reach the widget at their sub-pixel position"""
        if qt.BINDING == "PyQt5":
            self.skipTest("Qt5 mouse event positions are integers")
        widget = self.MouseMoveLogger()
        widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        widget.resize(100, 100)
        widget.show()
        self.qWaitForWindowExposed(widget)

        # Let Qt pick the receiver as for an event from the window system
        window = widget.windowHandle()
        position = qt.QPointF(10.5, 20.25)
        event = qt.QMouseEvent(
            qt.QEvent.MouseMove,
            position,
            window.mapToGlobal(position),
            qt.Qt.NoButton,
            qt.Qt.NoButton,
            qt.Qt.NoModifier,
        )
        widget.positions.clear()
        qt.QApplication.sendEvent(window, event)
        positions = widget.positions
        widget.close()

        self.assertEqual(positions, [(10.5, 20.25)])

    def testOpenGL(self):
        """Log OpenGL version using an OpenGLWidget"""
        super().setUp()
        widget = self.OpenGLWidgetLogger()
        widget.show()
        widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.qWaitForWindowExposed(widget)
        widget.close()
