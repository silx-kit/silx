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
"""Tests for FlowLayout"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/08/2018"

import unittest

from silx.gui.widgets.FlowLayout import FlowLayout
from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt


class TestFlowLayout(TestCaseQt):
    """Tests for FlowLayout"""

    def setUp(self):
        """Create and show a widget"""
        self.widget = qt.QWidget()
        self.widget.show()
        self.qWaitForWindowExposed(self.widget)

    def tearDown(self):
        """Delete widget"""
        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.widget.close()
        del self.widget
        self.qapp.processEvents()

    def test(self):
        """Basic tests"""
        layout = FlowLayout()
        self.widget.setLayout(layout)

        layout.addWidget(qt.QLabel('first'))
        layout.addWidget(qt.QLabel('second'))
        self.assertEqual(layout.count(), 2)

        layout.setHorizontalSpacing(10)
        self.assertEqual(layout.horizontalSpacing(), 10)
        layout.setVerticalSpacing(5)
        self.assertEqual(layout.verticalSpacing(), 5)
