# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""Tests for BoxLayoutDockWidget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2018"

import unittest

from silx.gui.widgets.BoxLayoutDockWidget import BoxLayoutDockWidget
from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt


class TestBoxLayoutDockWidget(TestCaseQt):
    """Tests for BoxLayoutDockWidget"""

    def setUp(self):
        """Create and show a main window"""
        self.window = qt.QMainWindow()
        self.qWaitForWindowExposed(self.window)

    def tearDown(self):
        """Delete main window"""
        self.window.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.window.close()
        del self.window
        self.qapp.processEvents()

    def test(self):
        """Test update of layout direction according to dock area"""
        # Create a widget with a QBoxLayout
        layout = qt.QBoxLayout(qt.QBoxLayout.LeftToRight)
        layout.addWidget(qt.QLabel('First'))
        layout.addWidget(qt.QLabel('Second'))
        widget = qt.QWidget()
        widget.setLayout(layout)

        # Add it to a BoxLayoutDockWidget
        dock = BoxLayoutDockWidget()
        dock.setWidget(widget)

        self.window.addDockWidget(qt.Qt.BottomDockWidgetArea, dock)
        self.qapp.processEvents()
        self.assertEqual(layout.direction(), qt.QBoxLayout.LeftToRight)

        self.window.addDockWidget(qt.Qt.LeftDockWidgetArea, dock)
        self.qapp.processEvents()
        self.assertEqual(layout.direction(), qt.QBoxLayout.TopToBottom)
